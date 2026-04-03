## vllm accelerate version
## For evaluating the tool-calling version of RL and SFT

import asyncio
import json
import os
import gc
import torch
from tqdm.asyncio import tqdm
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

from main_infer_tool_assistant import get_user_prompt, check_tool_call_at_end, chemcotbench_loading, call_chem_api
from get_tool_sft_trajectory import sft_trajectory_builder

# --- 辅助函数：实时追加写入 ---
def append_to_jsonl(data, file_path):
    """每生成一个样本就调用一次，安全且高效"""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        

MAX_MODEL_LEN = 20480

# --- 核心推理逻辑 ---
async def process_single_task(engine, tokenizer, sample, my_system_prompt, sampling_params):
    """处理单个样本的完整生命周期"""
    test_sample = sample
    sample_task = test_sample['extra_info']['task']
    sample_subtask = test_sample['extra_info'].get('subtask') or sample_task
    
    # 构造初始 Prompt
    user_prompt = get_user_prompt(
        task_type=sample_subtask, 
        meta_info=test_sample['extra_info'], 
        question=test_sample['user_prompt']
    )
    current_history = [
        {'role': 'system', 'content': my_system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    while True:
        model_input_text = tokenizer.apply_chat_template(
            current_history, add_generation_prompt=True, tokenize=False
        )
        current_tokens = tokenizer(model_input_text, return_tensors="pt").input_ids.shape[1]
        if current_tokens >= MAX_MODEL_LEN:
            print(f"Warning: Reached max length limit ({current_tokens} tokens). Forcing break.")
            current_history.append({"role": "assistant", "content": "Error: Context length exceeded limit."})
            break
        
        # 异步提交给 vLLM 引擎
        request_id = f"req-{hash(str(model_input_text)) + hash(str(sample_task))}"
        results_generator = engine.generate(model_input_text, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        response_text = final_output.outputs[0].text
        finish_reason = final_output.outputs[0].finish_reason

        # 更新历史
        if current_history[-1]['role'] == 'assistant':
            current_history[-1]['content'] += response_text
        else:
            current_history.append({'role': 'assistant', 'content': response_text})

        # 检查是否完成 (EOS)
        if finish_reason == "stop" and not any(s in response_text for s in sampling_params.stop):
            break

        # 工具调用判定
        cleaned_text = response_text.strip()
        if cleaned_text.endswith('</tool_call>'):
            tool_info_dict = check_tool_call_at_end(response_text)
            if tool_info_dict:
                tool_name = tool_info_dict['name']
                # {'SMILES1': KK, 'SMILES2': MM} --> KK;MM
                query_key = list()
                for key in tool_info_dict['arguments'].keys():
                    query_key.append(key)
                query = tool_info_dict['arguments'][query_key[0]]
                if len(query_key) > 1:
                    for key in query_key[1:]:
                        query += f";{tool_info_dict['arguments'][key]}"   
                
                api_result = call_chem_api(tool_name, query)
                api_text_result = api_result.json()['result']
                result_str = f"<result> {api_text_result} </result>."
                current_history.append({"role": "tool", "content": result_str})
            else:
                current_history.append({"role": "tool", "content": "<result> tool_call error </result>"})
        else:
            continue # 未完待续

    # 返回最终格式，方便直接保存
    return {'current_history': current_history, 'extra_info': test_sample['extra_info']}


async def run_evaluation_for_step(step):  
    root_path = "/cto_labs/lihao/chem_reason"
    
    ## SFT model evaluation  
    # step = 300
    # model_name = 'chemcot-tool-coldstart-rxn-qwen-7B-coldstart-55'
    # model_path=f"{root_path}/ChemSearch/search_saves/verl_checkpoints/{model_name}/global_step_{step}/"
    # output_file=f'{root_path}/ChemSearch/search_saves/results/tool_sft_results/{model_name}/vllm_temp_v2.json'
    
    ## RL model evaluation
    model_name = 'ChemAgent-Qwen-7B-with-RXN'
    model_path=f"{root_path}/ChemSearch/verl/checkpoints/{model_name}/chemagent-epoch-3-lr-1e-6/global_step_{step}/actor/huggingface/"
    output_file=f'{root_path}/ChemSearch/search_saves/results/tool_rl_results/{model_name}-epoch3/model_{step}_epoch_3_1e-6.json'
    
    ## Batch evaluation for non-tool & tool models
    # model_name = "chemcot-fullsft-tool-rxn-qwen-7B"
    # model_path = f"{root_path}/ChemSearch/search_saves/verl_checkpoints/{model_name}/global_step_{step}/"
    # output_file = f'{root_path}/ChemSearch/search_saves/results/ablations/{model_name}/model_{step}.json'
    
    # model_name = "chemcot-tool-coldstart-rxn-qwen-7B-coldstart-55-murcko-82"
    # model_path = f"{root_path}/ChemSearch/search_saves/verl_checkpoints/coldstart_models/{model_name}/global_step_{step}/"
    # output_file = f'{root_path}/ChemSearch/search_saves/results/tool_sft_results/{model_name}/model_{step}.json'
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        enable_prefix_caching=True, # 核心优化：1500个任务共享系统提示词
        trust_remote_code=True,
        gpu_memory_utilization=0.8, # 留出20%给API处理和系统
        max_model_len=20480,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. Prepare Data
    test_list = chemcotbench_loading(dataset_path=f"{root_path}/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/chemcotbench_with_rxn.parquet")
    my_data_builder = sft_trajectory_builder()
    my_system_prompt = my_data_builder._get_stage4_system_prompt()
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["</tool_call>", " </tool_call>", "</tool_call>\n", " </tool_call>\n", "</tool_call>\n\n", " </tool_call>\n\n"],
        include_stop_str_in_output=True
    )

    # 3. 提交所有任务到事件循环
    tasks = [
        process_single_task(engine, tokenizer, sample, my_system_prompt, sampling_params) 
        for sample in test_list
    ]

    # 4. 关键：实时监听完成事件并保存
    print(f"Starting inference... Saving to {output_file}")
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
        try:
            sample_result = await coro
            # 每出一个结果，立刻写入硬盘（JSONL 格式）
            append_to_jsonl(sample_result, output_file)
        except Exception as e:
            print(f"Task failed with error: {e}")
            
    del engine
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    await asyncio.sleep(2) # 给系统一点喘息时间

async def main():
    steps_to_eval = [i for i in range(240, 1860, 60)]
    for step in steps_to_eval:
        try:
            await run_evaluation_for_step(step)
        except Exception as e:
            print(f"Failed to evaluate step {step}: {e}")
            continue # fail in some steps

if __name__ == "__main__":
    asyncio.run(main())