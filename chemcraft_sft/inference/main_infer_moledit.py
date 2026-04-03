## 针对ChemCoTDataset-SFT之后的QWen-Instruct模型(1.5B, 7B, 14B, 32B), 评测在mol-edit的add, delete, sub三个任务上的性能 
import os
import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

def tranform_str_to_json(str_input):
    ## 假如LLM输出的是类似json的字符串, 我需要设定一个逻辑, 把字符串重新转换成json
    ## o1-mini的感觉, 是要移除字符串里面的\n，并且把所有的\"都改成 "
    if "</think>\n\n" in str_input:
        str_input = str_input.split("</think>\n\n")[-1]
    
    if "```json\n" in str_input:
        str_input = str_input.split("```json\n")[1]
        str_input = str_input.replace("\n```", '')
    
    unescaped_str = str_input.replace('\n    ', '').replace('\n', '').replace('\"', '"')
    try:
        json_obj = json.loads(unescaped_str)
        return json_obj
    except json.JSONDecodeError as e:
        return None

def update_json_file(info_dict, file_name='data.json'):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(info_dict)
    
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def get_model(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.5
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return model, tokenizer, generation_config

def task_specific_system_content(taskname, promptname):  
    if promptname in ["chem_sft", "raw"]: 
        if taskname == "add":   
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: <answer> Modified SMILES </answer>.
            Your response must contains the step-by-step reasoning
                Step-1: molecule_analysis: [your reasoning] Analyze the functional groups and other components within the molecule,
                Step-2: function_group_introduce_strategy: [your reasoning] Determine how and at which site the new group can be most reasonably added,
                Step-3: feasibility_analysis: [your reasoning] Assess the chemical viability of the proposed modification,
            The answer MUST BE Output: <answer> Modified SMILES </answer>.
            """
        
        elif taskname == "delete":
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me DELETE a specified functional group and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: <answer> Modified SMILES </answer>.
            Your response must contains the step-by-step reasoning
                Step-1: molecule_analysis: [your reasoning] Analyze the functional groups and other components within the molecule,
                Step-2: functional_group_identification: [your reasoning] Locate the functional group position and analyse,
                Step-3: delete_strategy: [your reasoning] Determine how and at which site the functional group can be most reasonably deleted,
                Step-4: feasibility_analysis: [your reasoning] Assess the chemical viability of the proposed modification,
            The answer MUST BE Output: <answer> Modified SMILES </answer>.
            """
        
        elif taskname == "sub":
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me ADD and DELETE specified functional groups and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Names. Output: <answer> Modified SMILES </answer>.
            Your response must contains the step-by-step reasoning, 
                Step-1: molecule_analysis: [your reasoning] Analyze the functional groups and other components within the molecule,
                Step-2: functional_group_identification: [your reasoning] Locate the functional group position and analyse,
                Step-3: add_strategy: [your reasoning] Determine how and at which site the new group can be most reasonably added,
                Step-4: delete_strategy: [your reasoning] Determine how and at which site the functional group can be most reasonably deleted,
                Step-5: feasibility_analysis: [your reasoning] Assess the chemical viability of the proposed modification,
            The answer MUST BE Output: <answer> Modified SMILES </answer>.
            """
    
    return system_content
    

def llm_predict(llm_info, mol_info, system_content, user_content, taskname, saving_path):
    (llm_model, llm_tokenizer, llm_generate_config) = llm_info
    prompt = f"""<|im_start|>system
    {system_content}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    """

    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            generation_config=llm_generate_config
        )
    
    output_ids = outputs[0][len(inputs["input_ids"][0]):]
    response = llm_tokenizer.decode(output_ids, skip_special_tokens=True)
    
    mol_info['task'] = taskname
    mol_info['sft_results'] = response
    json_str = json.dumps(mol_info, indent=4, ensure_ascii=False) 
    
    update_json_file(
        info_dict = mol_info,
        file_name=saving_path
    )
    
    
def get_mol_edit(taskname, modelname, promptname, llm_info):
    ## 生成deepseek在mol-understanding benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    assert taskname in ['add', 'delete', 'sub']
    assert promptname in ['raw', 'cot_template', 'cot_groundtruth', 'chem_sft']
    print(f"*** Task={taskname}, Model={modelname}, Prompt={promptname}")

    task_info_path = dict(
        add="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_edit/1-instruct_to_edit/add_cot.json",
        delete="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_edit/1-instruct_to_edit/delete_cot.json",
        sub="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_edit/1-instruct_to_edit/sub_cot.json"
    )
    
    task_saving_path = dict(
        add=f"../results/qwen_sft_results/mol_edit/add/cot_results_{modelname}_{promptname}.json",
        delete=f"../results/qwen_sft_results/mol_edit/delete/cot_results_{modelname}_{promptname}.json",
        sub=f"../results/qwen_sft_results/mol_edit/sub/cot_results_{modelname}_{promptname}.json",
    )
    
    file_path = task_info_path[taskname]
    mol_infos = json.load(open(file_path, "r"))
    
    if os.path.exists(task_saving_path[taskname]):
        done_info = json.load(open(task_saving_path[taskname], "r"))
        mol_infos = mol_infos[len(done_info):]
    
    for i in tqdm(range(len(mol_infos)), desc=f"{taskname}-{modelname}-{promptname}"):
        system_content = task_specific_system_content(taskname, promptname)
        
        if taskname == "add":
            smiles, added_group = mol_infos[i]['molecule'], mol_infos[i]['added_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to add: {added_group}." 
        elif taskname == "delete":
            smiles, removed_group = mol_infos[i]['molecule'], mol_infos[i]['removed_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}." 
        elif taskname == "sub":
            smiles, added_group, removed_group = mol_infos[i]['molecule'], mol_infos[i]['added_group'], mol_infos[i]['removed_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}, Functional Group to add: {added_group}."
        
        llm_predict(llm_info=llm_info, mol_info=mol_infos[i], system_content=system_content, user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])

if __name__ == "__main__":
    # model_list = ["qwen2.5_1.5b", "qwen2.5_7b", "qwen2.5_14b", "qwen2.5_1.5b_distill", "qwen2.5_7b_distill", "qwen2.5_14b_distill"]
    # model_list = ["qwen2.5_1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
    
    # model_list = ["qwen2.5_32b"]
    # prompt_list = ['chem_sft']

    model_list = ["s1.1_32b"]
    prompt_list = ['raw']
    
    model_sft_dict = {
        "qwen2.5_3b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-3B-instruct/global_step_90",
        "qwen2.5_7b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-7B-instruct/global_step_90",
        "qwen2.5_14b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-14B-instruct/global_step_90",
        "qwen2.5_32b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-32B-instruct/global_step_90",
        "s1.1_32b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/s1.1-32B",
    }
    
    for model_name in model_list:
        for prompt_type in prompt_list:
            llm_model, llm_tokenizer, llm_generate_config = get_model(model_path=model_sft_dict[model_name])
            get_mol_edit(taskname="add", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))
            get_mol_edit(taskname="delete", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))
            get_mol_edit(taskname="sub", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))