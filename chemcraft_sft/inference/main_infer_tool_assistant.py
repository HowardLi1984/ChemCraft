## 
from tqdm import tqdm
import pandas as pd

import requests
import torch
import re
import ast
import sys
sys.path.append("../../search_saves/datasets/chemcot/cold_start_v2")

import transformers
from transformers import StoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from main_infer_moledit import update_json_file
from get_tool_sft_trajectory import sft_trajectory_builder

def call_chem_api(function_name: str, query: str):
    # payload = ChemFunctionRequest(function_name=function_name, query=query)
    payload = {
        "function_name": function_name,
        "query": query
    }
    
    response = requests.post(
        "http://127.0.0.1:8080/chem_function", 
        json=payload
    )
    return response

def get_user_prompt(task_type, meta_info, question):
    user_prompt = ""
    if task_type in ['qed', 'logp', 'solubility', 'jnk', 'gsk', 'drd']:
        user_prompt = f"You are a chemical assistent. Given the source molecule: {meta_info['molecule']}, Optimize the Molecule to improve its {task_type} property while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>."
    elif task_type == 'add':
        user_prompt = f"Modify the molecule {meta_info['molecule']} by adding a {meta_info['added_group']}. Output: <answer> SMILES of the modified molecule <answer>."
    elif task_type == 'delete':
        user_prompt = f"Please remove a {meta_info['removed_group']} from the molecule {meta_info['molecule']}. Output: <answer> SMILES of the modified molecule <answer>."
    elif task_type == "sub":
        user_prompt = f"Modify the molecule {meta_info['molecule']} by substituting a {meta_info['removed_group']} with a {meta_info['added_group']}. Output: <answer> SMILES of the modified molecule <answer>."
    elif task_type == "equivalence":
        user_prompt = f"You are a chemical assistent. Given two molecule SMILES, Please Determine whether these two Molecules are the same. Input: Molecule A: {meta_info['molecule']}, Molecule B: {meta_info['molecule2']}. Output: <answer> Yes/No </answer>."
    elif task_type == 'fg_count':
        user_prompt = f"You are a chemical assistent. Giving you an Input Molecule: {meta_info['molecule']} and a Fragment name: {meta_info['functional_group']}, help me count the number of the fragment in the Molecule. Output: <answer> count-number </answer>."
    elif task_type == "Murcko_scaffold":
        user_prompt = f"You are a chemical assistent. Please predict the Largest Murcko scaffold in the Molecule. Input molecule: {meta_info['molecule']}. Output: <answer> Scaffold SMILES </answer>. Definition: The Murcko scaffold is obtained by removing all side chains, functional groups, and exocyclic modifications, leaving only the ring systems and connecting bonds."
    elif task_type == "ring_count":
        user_prompt = f"You are a chemical assistent. Giving you an Input Molecule: {meta_info['molecule']} and a Ring Structure: {meta_info['ring']}, help me count the number of ring structure in the Molecule. Output: <answer> count-number </answer>."
    elif task_type == "ring_system_scaffold":
        user_prompt = f"You are a chemical assistent. Please Determine whether the ring_system_scaffold is in the Molecule. Input Molecule: {meta_info['molecule']},  the Ring System Scaffold: {meta_info['ring_system_scaffold']}. Output: <answer> Yes/No </answer>."
    else:
        user_prompt = question
    
    return user_prompt

def model_loading(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.eval() 
    return model, tokenizer


def dataset_loading(dataset_path):
    test_dataset = list()
    dataframe = pd.read_parquet(dataset_path)
    for i in range(len(dataframe)):
        test_dataset.append({
            "system_prompt": dataframe.iloc[i]['system_prompt'],
            "user_prompt": dataframe.iloc[i]['user_prompt'],
            "answer_assistant_tool": dataframe.iloc[i]['answer_assistant_tool'],
            "extra_info": dataframe.iloc[i]['extra_info'],
        })
    return test_dataset

def chemcotbench_loading(dataset_path):
    test_dataset = list()
    dataframe = pd.read_parquet(dataset_path)
    for i in range(len(dataframe)):
        test_dataset.append({
            "user_prompt": dataframe.iloc[i]['user_prompt'],
            "extra_info": dataframe.iloc[i]['extra_info'],
        })
    return test_dataset

curr_eos = [151645, 151643] # for Qwen2.5 series models

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def check_tool_call_at_end(response_text):
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = list(re.finditer(pattern, response_text, re.DOTALL))
    if not matches: return None

    last_match = matches[-1]
    if last_match.end() != len(response_text): return None

    dict_string = last_match.group(1).strip()
    string_to_parse = ""
    
    # 这个正则模式会查找 "'name':" 后面跟着可选的空格，然后是一个单引号, 如果这种情况, 修改为'name'形式
    is_quoted_pattern = r"'name':\s*'" 
    if re.search(is_quoted_pattern, dict_string):
        string_to_parse = dict_string
    else:
        string_to_parse = re.sub(r"('name':)\s*(.*?)\s*,", r"\1 '\2',", dict_string)
        
    try:
        extracted_dict = ast.literal_eval(string_to_parse)
        print(f"extracted_dict: ", extracted_dict)
        if isinstance(extracted_dict, dict) and 'name' in extracted_dict.keys() and 'arguments' in extracted_dict.keys():
            return extracted_dict
        else:
            return None # 如果解析出来的不是字典，也返回None
    except (ValueError, SyntaxError):
        return None  # 如果字符串格式错误，无法被解析，则返回None

def inference_with_tools(model, tokenizer, prompt: str, max_total_tokens=20000):
    current_history = prompt
    
    ## init the stopping criteria from Search-R1
    target_sequences = ["</tool_call>", " </tool_call>", "</tool_call>\n", " </tool_call>\n", "</tool_call>\n\n", " </tool_call>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    with torch.no_grad():
        while True:
            model_input_text = tokenizer.apply_chat_template(
                current_history,
                add_generation_prompt=True, # 添加助手的起始提示
                tokenize=False,
            )
            input_ids = tokenizer(model_input_text, return_tensors="pt").input_ids.to(model.device)
            if input_ids.shape[1] > max_total_tokens:
                print(f"Warning: Context length ({input_ids.shape[1]}) exceeded the limit ({max_total_tokens}). Stopping generation.")
                break

            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=False,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, # 明确指定结束标记
            )

            generated_tokens = outputs[0][input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            if tokenizer.eos_token_id == generated_tokens[-1]:
                ## Find [EOS] Token, finish reasoning
                if current_history[-1]['role'] == 'assistant': 
                    current_history[-1]['content'] += response_text
                else:
                    current_history.append({'role': 'assistant', 'content': response_text})
                break
            
            current_history.append({"role": "assistant", "content": response_text})
            
            ## 1. 移除response_text最后的空格以及\n, 判定(2) 如果有</tool_call>
            cleaned_text = response_text.strip()
            if not cleaned_text.endswith('</tool_call>'):
                # (1) 如果后面没有</tool_call>, 说明content不够长自动停的
                # 由于之前已经把response_text存在current_history了, 所以可以直接continue
                continue

            else:
                # check tool-calling tokens
                tool_info_dict = check_tool_call_at_end(response_text)
                if tool_info_dict: # locate a true tool
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
                    result_str = "<result> tool_call error, use correct format: '<tool_call>{'name': 'Your_tool_name', 'arguments':{}</tool_call>' </result>"
                    current_history.append({"role": "tool", "content": result_str})
            
    return current_history

def main_inference_chemcotdataset():
    # Inference Qwen-Chem-Agent with only SFT on ChemCoTDataset
    # qwen_model, qwen_tokenizer = model_loading(model_path="/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-tool-coldstart-qwen-7B-full-tool-assistant/global_step_1260/")
    # epoch = 900
    # short_name = "7B"
    # model_name = 'chemcot-tool-coldstart-qwen-7B-full-tool-assistant'
    
    epoch = 500
    short_name = "14B"
    model_name = 'chemcot-tool-coldstart-qwen-14B-full-tool-assistant'
    
    qwen_model, qwen_tokenizer = model_loading(model_path=f"/cto_labs/lihao/chem_reason/ChemSearch/search_saves/verl_checkpoints/{model_name}/global_step_{epoch}/")
    
    test_list = dataset_loading(dataset_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/test_tool_full.parquet")
    
    for i in tqdm(range(len(test_list)), desc=f"SFT-Test-{short_name}"):
        test_sample = test_list[i]
        init_input = [
            {'role': 'system', 'content': test_sample['system_prompt']},
            {'role': 'user', 'content': test_sample['user_prompt']},
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        ]
        current_history = inference_with_tools(model=qwen_model, tokenizer=qwen_tokenizer, prompt=init_input)
        
        sample_result = {'current_history': current_history, 'extra_info': test_sample['extra_info']}
        update_json_file(
            sample_result, 
            file_name=f'/cto_labs/lihao/chem_reason/ChemSearch/search_saves/results/tool_sft_results/{model_name}/sft_{epoch}.json'
        )
        
def main_inference_rl_chemcotbench():
    epoch = 300
    short_name = "7B"
    model_name = "ChemAgent-Qwen-7B-format-reward-v2"
    root_path = "/cto_labs/lihao/chem_reason"
    
    qwen_model, qwen_tokenizer = model_loading(model_path=f"{root_path}/ChemSearch/verl/checkpoints/{model_name}/test-for-qwenchemagent/global_step_{epoch}/actor/huggingface")
    test_list = chemcotbench_loading(dataset_path=f"{root_path}/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/chemcotbench.parquet")
    
    my_data_builder = sft_trajectory_builder()
    my_system_prompt = my_data_builder._get_stage4_system_prompt()
    for i in tqdm(range(len(test_list)), desc=f"ChemCoTBench-{short_name}"):
        test_sample = test_list[i]
        sample_task, sample_subtask = test_sample['extra_info']['task'], test_sample['extra_info']['subtask']
        if sample_subtask == None:
            sample_subtask = sample_task
        user_prompt = get_user_prompt(sample_subtask, test_sample['extra_info'])
        init_input = [
            {'role': 'system', 'content': my_system_prompt},
            {'role': 'user', 'content': user_prompt},
            # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        ]
        # print(init_input)
        current_history = inference_with_tools(model=qwen_model, tokenizer=qwen_tokenizer, prompt=init_input)
        
        sample_result = {'current_history': current_history, 'extra_info': test_sample['extra_info']}
        update_json_file(
            sample_result, 
            file_name=f'{root_path}/ChemSearch/search_saves/results/tool_rl_results/{model_name}/chemcotbench_{epoch}.json'
        )

def main_inference_chemcotbench():
    # Inference Qwen-Chem-Agent with only SFT on ChemCoTBench
    # epoch = 125
    # short_name = "7B"
    # model_name = 'chemcot-tool-coldstart-7B-for-rl'
    # root_path = "/cto_labs/lihao/chem_reason"
    
    # epoch = 500
    # short_name = "14B"
    # model_name = 'chemcot-tool-coldstart-qwen-14B-full-tool-assistant-murcko'
    # root_path = "/cto_labs/lihao/chem_reason"
    
    epoch = 300
    short_name = "7B"
    model_name = 'chemcot-tool-coldstart-rxn-qwen-7B-coldstart-55'
    root_path = "/cto_labs/lihao/chem_reason"
    
    qwen_model, qwen_tokenizer = model_loading(model_path=f"{root_path}/ChemSearch/search_saves/verl_checkpoints/{model_name}/global_step_{epoch}/")
    test_list = chemcotbench_loading(dataset_path=f"{root_path}/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/chemcotbench_with_rxn.parquet")
    
    my_data_builder = sft_trajectory_builder()
    my_system_prompt = my_data_builder._get_stage4_system_prompt()
    for i in tqdm(range(len(test_list)), desc=f"ChemCoTBench-{short_name}"):
        test_sample = test_list[i]
        sample_task, sample_subtask = test_sample['extra_info']['task'], test_sample['extra_info']['subtask']
        if sample_subtask == None:
            sample_subtask = sample_task
        user_prompt = get_user_prompt(task_type=sample_subtask, meta_info=test_sample['extra_info'], question=test_sample['user_prompt'])
        init_input = [
            {'role': 'system', 'content': my_system_prompt},
            {'role': 'user', 'content': user_prompt},
            # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        ]
        # print(init_input)
        current_history = inference_with_tools(model=qwen_model, tokenizer=qwen_tokenizer, prompt=init_input)
        
        sample_result = {'current_history': current_history, 'extra_info': test_sample['extra_info']}
        update_json_file(
            sample_result, 
            file_name=f'{root_path}/ChemSearch/search_saves/results/tool_sft_results/{model_name}/chemcotbench_{epoch}_full_gt.json'
        )

if __name__ == "__main__":
    # main_inference_chemcotdataset()
    main_inference_chemcotbench()
    
    # main_inference_rl_chemcotbench()