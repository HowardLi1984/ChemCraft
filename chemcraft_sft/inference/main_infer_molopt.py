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
    prop_descrip_dict = dict(
        drd="DRD2 property (Dopamine D2 Receptor Activity)",
        jnk="JNK3 property (c-Jun N-terminal kinase 3 inhibition)",
        gsk="GSK3-beta property (Glycogen Synthase Kinase 3-beta Inhibition)",
        qed="QED property (Drug-likeness)",
        clint="Hepatic intrinsic clearance (Clint)",
        logp="Distribution coefficient (LogD)",
        solubility="compound's ability to dissolve in water (Solubility)"
    ) 
 
    system_content = f"""
    You are a chemical assistent, Optimize the Source Molecule to improve the {prop_descrip_dict[taskname]} while following a structured intermediate optimization process. Output: <answer> Modified-Molecule-SMILES </answer>.
    Your response must contains the step-by-step reasoning
        Step-1: Structural Analysis of Source Molecule: "",
        Step-2: Property Analysis: "",
        Step-3: Limitation in Source Molecule for Property: ""
        Step-4: Optimization for Source Molecule: "",
    The answer MUST BE Output: <answer> Modified-Molecule-SMILES </answer>.
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
    
    
def predict_mol_understanding(modelname, promptname, llm_info):
    ## 生成deepseek在mol-understanding benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    prop_list = ['logp', 'solubility', 'qed', 'drd', 'gsk', 'jnk']
    assert promptname in ['raw', 'cot_template', 'cot_groundtruth', 'chem_sft']
    
    for prop in prop_list:
        print(f"*** Task={prop}, Model={modelname}, Prompt={promptname}")
        mmp_file_path = f"/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/deep_mol_opt/{prop}/final_mmp_cot_iupac.json"
        task_saving_path = f"../results/qwen_sft_results/deep_mol_opt/{prop}/cot_results_{modelname}_{promptname}.json"
        mmp_infos = json.load(open(mmp_file_path, "r"))
        
        if os.path.exists(task_saving_path):
            done_info = json.load(open(task_saving_path, "r"))
            mmp_infos = mmp_infos[len(done_info):]
        
        for i in tqdm(range(len(mmp_infos)), desc=prop):
            system_content = task_specific_system_content(prop, promptname)
            source_content = f"Source Molecule: {mmp_infos[i]['src']}."
        
            llm_predict(llm_info=llm_info, mol_info=mmp_infos[i], system_content=system_content, user_content=source_content, taskname=prop, saving_path=task_saving_path)

if __name__ == "__main__":
    # model_list = ["qwen2.5_1.5b", "qwen2.5_7b", "qwen2.5_14b", "qwen2.5_1.5b_distill", "qwen2.5_7b_distill", "qwen2.5_14b_distill"]
    # model_list = ["qwen2.5_1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
    
    # model_list = ["qwen2.5_7b"]
    # prompt_list = ['chem_sft']
    
    # model_sft_dict = {
    #     "qwen2.5_7b": "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-7B-instruct/global_step_90",
    # }

    model_list = ["qwen2.5_3b", "qwen2.5_7b", "qwen2.5_14b", "qwen2.5_32b"]
    prompt_list = ['raw']
    
    model_sft_dict = {
        "qwen2.5_3b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-3B-Instruct",
        "qwen2.5_7b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-7B-Instruct",
        "qwen2.5_14b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-14B-Instruct",
        "qwen2.5_32b": "/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-32B-Instruct",
    }
    
    for model_name in model_list:
        for prompt_type in prompt_list:
            llm_model, llm_tokenizer, llm_generate_config = get_model(model_path=model_sft_dict[model_name])
            predict_mol_understanding(modelname=model_name, promptname=prompt_type, llm_info=(llm_model, llm_tokenizer, llm_generate_config))