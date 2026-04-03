# 单机多卡的对qwen-7B模型的读取
# 只对他进行推理，读入一个test_dataset能够输出每个batch的predict sentence
# 用于评测 verl-sft 后的chemcot模型, format-accuracy究竟如何

import re
import os
import gc
import json

import torch
import wandb
# import evaluate
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from tqdm import tqdm
import pandas as pd

# --- 配置 ---
MODEL_ID = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-v2-qwen-2.5-7B-instruct/global_step_{}/"
TESTDATA_PATH = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-sft/test_sft.parquet"
# WandB 项目配置
WANDB_PROJECT_NAME = "chemcot-sft-v2-qwen-2.5-7B-instruct"
# 推理配置
BATCH_SIZE = 32  # 每个GPU的批量大小
MAX_NEW_TOKENS = 1024 # 生成文本的最大长度

# prompts_and_references = [
#     {"prompt": "用中文写一句关于人工智能的诗:", "reference": "智海无涯泛舟楫，码绘未来启新篇。"},
#     {"prompt": "Translate to Chinese: The quick brown fox jumps over the lazy dog.", "reference": "敏捷的棕色狐狸跳过了懒狗。"},
#     {"prompt": "What is the capital of France?", "reference": "The capital of France is Paris."},
#     {"prompt": "解释一下什么是黑洞:", "reference": "黑洞是时空展现出极端引力的区域，以至于任何粒子、甚至光这样的电磁辐射都无法从中逃逸。"},
#     {"prompt": "用Python写一个冒泡排序函数:", "reference": "def bubble_sort(arr): n = len(arr); for i in range(n): for j in range(0, n-i-1): if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]"},
#     {"prompt": "Translate to Chinese: Hello, world!", "reference": "你好，世界！"},
#     {"prompt": "续写这个故事：在一个遥远的星系，有一颗会唱歌的星球...", "reference": "它的歌声能穿越时空，吸引着宇宙中孤独的旅行者。"},
#     {"prompt": "中国的首都是哪里？", "reference": "中国的首都是北京。"},
# ]

def get_dataloader(tokenizer):
    # --- 准备数据集和 DataLoader ---
    prompts_and_references = list()
    
    dataframe = pd.read_parquet(TESTDATA_PATH)
    for i in range(len(dataframe)):
        prompts_and_references.append({
            "prompt": dataframe.iloc[i]['question'],
            "reference": dataframe.iloc[i]['answer'],
        })
    
    test_dataset = Dataset.from_list(prompts_and_references)

    def tokenize_function(examples):
        # 对prompt进行分词
        return tokenizer(examples["prompt"], padding="longest", truncation=True, max_length=512)

    tokenized_dataset = test_dataset.map(tokenize_function, batched=True)
    # 移除原始文本列，只保留模型需要的ID
    tokenized_dataset = tokenized_dataset.remove_columns(["prompt"])
    
    # *** FIX: 明确设置数据集格式，确保DataLoader输出PyTorch Tensors ***
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # 创建 DataLoader
    test_dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE)
    
    return test_dataloader

def evaluate_step_structure(text):
    """
    评测字符串是否符合 Step-1 → Step-2 → ... → Final Answer: 的结构
    参数: text (str): 要评测的文本
    返回: dict: {'is_step_valid': False, 'is_final_answer': False, 'step_count': 0}
    """
    final_results = {'is_step_valid': False, 'is_final_answer': False, 'step_count': 0}
    # 检查是否为空
    if not text.strip():
        return {'is_step_valid': False, 'is_final_answer': False, 'step_count': 0}
    
    pattern = r"Step-(\d+)"
    matches = re.finditer(pattern, text)

    results = dict(step_num_list=[], step_pos_list=[], step_str=[])
    for match in matches:
        step_number = match.group(1)  # 提取编号
        start_pos = match.start()     # 起始位置
        results['step_str'].append(f"Step-{step_number}")
        results['step_num_list'].append(step_number)
        results['step_pos_list'].append(start_pos)
        
    step_len = len(results['step_num_list'])
    tmp_num_list = [str(i) for i in range(1, step_len+1)]
    final_results['step_count'] = step_len
    
    if tmp_num_list == results['step_num_list'] and tmp_num_list != []:
        final_results['is_step_valid'] = True
        
    match = re.search(r"Final Answer:", text, re.IGNORECASE)
    if match and len(results['step_pos_list'])>0:
        if match.start() > results['step_pos_list'][-1]:
            final_results['is_final_answer'] = True
    elif match:
        if match: final_results['is_final_answer'] = True
        
    match = re.search(r"Output:", text, re.IGNORECASE)
    if match and len(results['step_pos_list'])>0:
        if match.start() > results['step_pos_list'][-1]:
            final_results['is_final_answer'] = True
    elif match:
        if match: final_results['is_final_answer'] = True
    
    match = re.search(r"Final Output:", text, re.IGNORECASE)
    if match and len(results['step_pos_list'])>0:
        if match.start() > results['step_pos_list'][-1]:
            final_results['is_final_answer'] = True
    elif match:
        if match: final_results['is_final_answer'] = True
    
    return final_results

def main():
    # --- 1. 初始化 Accelerator ---
    # Accelerator 会自动处理设备分配（将模型和数据移动到正确的GPU）
    # `log_with="wandb"` 会自动初始化wandb，并只在主进程中记录
    accelerator = Accelerator(log_with="wandb")

    # --- 2. 初始化 WandB (仅在主进程) ---
    if accelerator.is_main_process:
        print("Initializing WandB on the main process...")
        wandb.init(project=WANDB_PROJECT_NAME, config={
            # "model_id": MODEL_ID,
            "batch_size_per_gpu": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_gpus": accelerator.num_processes,
        })
        print("WandB initialized successfully.")

    # --- 3. 加载 Tokenizer 和模型 ---
    # Tokenizer 在所有进程中加载
    print(f"Process {accelerator.process_index}: Loading tokenizer for {MODEL_ID.format(str(10))}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID.format(10), trust_remote_code=True, padding_side='left')
    
    # 修复Qwen tokenizer可能没有pad_token的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    test_dataloader = get_dataloader(tokenizer)
    
    for step in range(16, 21):
    # for step in range(1, 3):
        model_name = str(int(step*10))
        # 模型会根据Accelerator的设备映射自动分配到各个GPU上
        print(f"Process {accelerator.process_index}: Loading model from {MODEL_ID.format(model_name)}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID.format(model_name),
            torch_dtype=torch.bfloat16, # 使用 bfloat16 加速并减少显存占用
            device_map={"": accelerator.device}, # 关键！让Accelerator决定模型应该加载到哪个设备
            trust_remote_code=True
        )
        
        # 将模型设置为评估模式
        model.eval()
        print(f"Process {accelerator.process_index}: Model loaded on {accelerator.device}")

        # --- 5. 使用 accelerator.prepare() 包装模型和数据加载器 ---
        # 这是关键步骤，Accelerator会处理分布式数据加载
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        
        all_predictions = []
        all_references = []
        
        print(f"Process {accelerator.process_index}: Starting inference loop...")
        # --- 6. 推理循环 ---
        for batch in tqdm(test_dataloader, desc=f"Inference on Process {accelerator.process_index}", disable=not accelerator.is_local_main_process):
            # 从batch中获取输入和参考文本
            # 注意：tokenized_dataset中已经没有"reference"列了，我们需要从原始数据集中获取
            # 为了简化，我们在这里直接使用原始列表。在更复杂的场景中，你可能需要将ID和原始文本一起传递。
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            with torch.no_grad():
                # 生成文本。`generate`函数会自动处理多GPU上的操作。
                generated_token_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # 使用accelerator.gather_for_metrics来收集所有GPU上的生成结果
            # 这可以确保即使在不同GPU上batch size不同也能正确处理
            gathered_ids = accelerator.gather_for_metrics(generated_token_ids)

            # 在主进程上进行解码和存储
            if accelerator.is_main_process:
                # 解码时跳过特殊token，并清理空格
                preds = tokenizer.batch_decode(gathered_ids, skip_special_tokens=True)
                all_predictions.extend(preds)

        # --- 7. 收集所有参考文本到主进程 ---
        if accelerator.is_main_process:
            # all_references = [item['reference'] for item in prompts_and_references]
            # 确保预测和参考的数量一致
            all_predictions = all_predictions
            json.dump(
                all_predictions, 
                open(f"/cto_labs/lihao/chem_reason/ChemSearch/search_sft/results/sft_format/result_step_{model_name}.json", "w"),
                indent=4
            )

        # 等待所有进程完成
        accelerator.wait_for_everyone()

        # --- 8. 评估和记录 (仅在主进程) ---
        if accelerator.is_main_process:  
            step_acc, final_answer_acc = 0, 0     
            for i in range(len(all_predictions)):
                pred_text = all_predictions[i]
                pred_eval = evaluate_step_structure(text=pred_text)
                if pred_eval['is_step_valid']: step_acc += 1
                if pred_eval['is_final_answer']: final_answer_acc += 1
                
            wandb.log({
                "step_acc": step_acc/len(all_predictions), 
                "final_answer_acc": final_answer_acc/len(all_predictions),
            })

        ## clean the cuda memory
        print(torch.cuda.memory_summary())
        del model  # 1. 删除模型引用
        torch.cuda.empty_cache()  # 2. 清空CUDA缓存
        gc.collect()  # 3. 调用Python垃圾回收
        
    print("\nResults logged to WandB successfully.")
    wandb.finish()

if __name__ == "__main__":
    main()