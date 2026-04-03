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
    if taskname == "Murcko_scaffold":  
        system_content = f"""
        You are a chemistry expert performing Murcko scaffold extraction. Input: a molecule's SMILES string. Output: <answer> the Murcko scaffold in SMILES format </answer>.
        Definition: The Murcko scaffold is obtained by removing all side chains, functional groups, and exocyclic modifications, leaving only the ring systems and connecting bonds.
        Your response must contains the step-by-step reasoning
            Step-1: input_structure: original input structure,
            Step-2: core_ring_identification: [your reasoning] describe the identified core rings in the input Molecule,
            Step-3: scaffold_analysis: [your reasoning] describe the murcko scaffold,
            Step-4: matching_analysis": [your reasoning] matching the scaffold with the molecule,
        The answer MUST BE Output: <answer> Modified SMILES </answer>.
        """
        
    elif taskname == "ring_count":
        system_content = f"""
        You are a chemical assistent. Giving you an Input Molecule and a Ring Structure, help me count the number of ring structure in the Molecule. Output: <answer> count-number </answer>.
        Your response must contains the step-by-step reasoning
            Step-1: input_structure: original input structure,
            Step-2: ring_identification: [your reasoning] describe the ring strutures in the input Molecule,
            Step-3: counting: [your reasoning] count the matching rings in the input molecule.
        The answer MUST BE Output: <answer> count-number </answer>.
        """
        
    elif taskname == "ring_system_scaffold":
        system_content = f"""
        You are a chemical assistent. Please Determine whether the ring_system_scaffold is in the Molecule. Input: a molecule's SMILES string, a Ring System Scaffold. Output: <answer> Yes/No </answer>.
        Definition: The ring system scaffold consists of one or more cyclic (ring-shaped) molecular structures
        Your response must contains the step-by-step reasoning
            Step-1: input_structure: original input structure,
            Step-2: molecule_structure_analysis: [your reasoning] describe the structure of the input Molecule,
            Step-3: scaffold_analysis: [your reasoning] describe the ring system scaffold,
            Step-4: matching_analysis: [your reasoning] matching the scaffold with the molecule
        The answer MUST BE Output: <answer> Yes/No </answer>.
        """
    
    elif taskname == "fg_samples":
        system_content = f"""
        You are a chemical assistent. Giving you an Input Molecule and a Fragment name and SMILES, help me count the number of the fragment in the Molecule. Output: <answer> count-number </answer>.
        Your response must contains the step-by-step reasoning
            Step-1: fragment_structure: [your reasoning] fragment structure analysis,
            Step-1: matching_analysis: [your reasoning] describe and match the input Molecule with the fragment,
        The answer MUST BE Output: <answer> count-number </answer>.
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
    
    
def predict_mol_understanding(taskname, modelname, promptname, llm_info):
    ## 生成deepseek在mol-understanding benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    assert taskname in ['fg_samples', 'Murcko_scaffold', 'ring_count', 'ring_system_scaffold']
    assert promptname in ['raw', 'cot_template', 'cot_groundtruth', 'chem_sft']
    print(f"*** Task={taskname}, Model={modelname}, Prompt={promptname}")

    task_info_path = dict(
        fg_samples="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_understanding/1-fg_detect/fg_count_new.json",
        Murcko_scaffold="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_understanding/2-frag_detect/Murcko_scaffold_new.json",
        ring_count="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_understanding/2-frag_detect/ring_count_new.json",
        ring_system_scaffold="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/mol_understanding/2-frag_detect/ring_system_scaffold_new.json",
    )
    
    task_saving_path = dict(
        fg_samples=f"../results/qwen_sft_results/mol_understanding/fg_samples/cot_results_{modelname}_{promptname}.json",
        Murcko_scaffold=f"../results/qwen_sft_results/mol_understanding/frag_detect_murcko/cot_results_{modelname}_{promptname}.json",
        ring_count=f"../results/qwen_sft_results/mol_understanding/frag_detect_ring_count/cot_results_{modelname}_{promptname}.json",
        ring_system_scaffold=f"../results/qwen_sft_results/mol_understanding/frag_detect_ring_system/cot_results_{modelname}_{promptname}.json",
    )
    
    file_path = task_info_path[taskname]
    mol_infos = json.load(open(file_path, "r"))
    
    if os.path.exists(task_saving_path[taskname]):
        done_info = json.load(open(task_saving_path[taskname], "r"))
        mol_infos = mol_infos[len(done_info):]
    
    for i in tqdm(range(len(mol_infos)), desc=f"{taskname}-{modelname}-{promptname}"):
        system_content = task_specific_system_content(taskname, promptname)
        
        if taskname == "fg_samples":
            smiles = mol_infos[i]['smiles']
            fg_name = mol_infos[i]["fg_name"]
            fg_label = tranform_str_to_json(mol_infos[i]["meta"])["fg_label"]
            source_content = f"Input Molecule: {smiles}, Fragment SMILES: {fg_label}, Fragment Name: {fg_name}"
        elif taskname == "Murcko_scaffold":
            src_smiles = mol_infos[i]['smiles']
            source_content = f"Input Molecule: {src_smiles}." 
        elif taskname == "ring_count":
            src_smiles, ring_structure = mol_infos[i]['smiles'], mol_infos[i]['ring']
            source_content = f"Input Molecule: {src_smiles}, Ring Structure: {ring_structure}"
        elif taskname == "ring_system_scaffold":
            smiles, ring_scaffold = mol_infos[i]['smiles'], mol_infos[i]['ring_system_scaffold']
            source_content = f"Input Molecule: {smiles}, "
            source_content += f"Ring System Structure: {ring_scaffold}"
     
        llm_predict(llm_info=llm_info, mol_info=mol_infos[i], system_content=system_content, user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])

if __name__ == "__main__":
    # model_list = ["qwen2.5_1.5b", "qwen2.5_7b", "qwen2.5_14b", "qwen2.5_1.5b_distill", "qwen2.5_7b_distill", "qwen2.5_14b_distill"]
    # model_list = ["qwen2.5_1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
    
    # model_list = ["qwen2.5_32b"]
    # prompt_list = ['chem_sft']
    # model_sft_dict = {
    #     "qwen2.5_3b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-3B-instruct/global_step_90",
    #     "qwen2.5_7b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-7B-instruct/global_step_90",
    #     "qwen2.5_14b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-14B-instruct/global_step_90",
    #     "qwen2.5_32b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-32B-instruct/global_step_90",
    # }
    
    model_list = ["s1.1_32b"]
    prompt_list = ['raw']
    model_sft_dict = {
        "qwen2.5_3b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-3B-Instruct",
        "qwen2.5_7b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-7B-Instruct",
        "qwen2.5_14b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-14B-Instruct",
        "qwen2.5_32b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-32B-Instruct",
        "s1.1_32b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/s1.1-32B",
    }

    for model_name in model_list:
        for prompt_type in prompt_list:
            llm_model, llm_tokenizer, llm_generate_config = get_model(model_path=model_sft_dict[model_name])
            predict_mol_understanding(taskname="fg_samples", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))
            predict_mol_understanding(taskname="Murcko_scaffold", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))
            predict_mol_understanding(taskname="ring_count", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))
            predict_mol_understanding(taskname="ring_system_scaffold", modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))