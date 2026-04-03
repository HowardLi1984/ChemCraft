## 针对ChemCoTDataset-SFT之后的QWen-Instruct模型(1.5B, 7B, 14B, 32B), 评测在mol-edit的add, delete, sub三个任务上的性能 
import os
import json
from tqdm import tqdm

from openai import OpenAI

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
    
def task_specific_system_content(taskname, promptname):  
    if promptname == "cot_template": 
        if taskname == "add":   
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

            Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
            {{
                "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
                "function_group_introduce_strategy": "[your reasoning] Determine how and at which site the new group can be most reasonably added",
                "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
                "output": "Modified Molecule SMILES"
            }}
            DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
            """
        
        elif taskname == "delete":
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me DELETE a specified functional group and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

            Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
            {{
                "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
                "functional_group_identification": "[your reasoning] Locate the functional group position and analyse",
                "delete_strategy": "[your reasoning] Determine how and at which site the functional group can be most reasonably deleted",
                "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
                "output": "Modified Molecule SMILES"
            }}
            DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
            """
        
        elif taskname == "sub":
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me ADD and DELETE specified functional groups and output the modified SMILES sequence of the molecule. Input: Molecule SMILES string, Functional Group Names. Output: Modified Molecule SMILES string.

            Your response must contains the step-by-step reasoning, and must be directly parsable JSON format: \n
            {{
                "molecule_analysis": "[your reasoning] Analyze the functional groups and other components within the molecule",
                "functional_group_identification": "[your reasoning] Locate the functional group position and analyse",
                "add_strategy": "[your reasoning] Determine how and at which site the new group can be most reasonably added",
                "delete_strategy": "[your reasoning] Determine how and at which site the functional group can be most reasonably deleted",
                "feasibility_analysis": "[your reasoning] Assess the chemical viability of the proposed modification",
                "output": "Modified Molecule SMILES"
            }}
            DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
            """
    if promptname in ["raw", "cot_groundtruth"]:
        if promptname == "cot_groundtruth":
            sentence = "To help you, I provide you with the GROUND-TRUTH reasoning steps. FOLLOW the LEAD!!"
        else:
            sentence = ""
        
        if taskname == "add":   
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. {sentence} Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

            Your response must be directly parsable JSON format: \n
            {{
                "output": "Modified Molecule SMILES"
            }}
            DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
            """
        
        elif taskname == "delete":
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me DELETE a specified functional group and output the modified SMILES sequence of the molecule. {sentence} Input: Molecule SMILES string, Functional Group Name. Output: Modified Molecule SMILES string.

            Your response must be directly parsable JSON format: \n
            {{
                "output": "Modified Molecule SMILES"
            }}
            DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
            """
        
        elif taskname == "sub":
            system_content = f"""
            You are a chemical assistent. Given the SMILES structural formula of a molecule, help me ADD and DELETE specified functional groups and output the modified SMILES sequence of the molecule. {sentence} Input: Molecule SMILES string, Functional Group Names. Output: Modified Molecule SMILES string.

            Your response must be directly parsable JSON format: \n
            {{
                "output": "Modified Molecule SMILES"
            }}
            DO NOT output other text except for the answer. If your response includes ```json ```, regenerate it and output ONLY the pure JSON content.
            """
    
    return system_content
    


def llm_predict(llm_client, llm_name, mol_info, system_content, user_content, taskname, saving_path):
    response = llm_client.chat.completions.create(
        model=llm_name,
        messages=[
            { "role": "system", "content": system_content},
            { "role": "user", "content": user_content},
        ],
        stream=False,
    )
    content = response.choices[0].message.content
    
    try:
        content_json = json.loads(content)
    except json.JSONDecodeError as e:
        content_json = content
    
    mol_info['task'] = taskname
    mol_info['json_results'] = content_json
    json_str = json.dumps(mol_info, indent=4, ensure_ascii=False) 
    
    update_json_file(
        info_dict = mol_info,
        file_name=saving_path
    )
    
    
def get_mol_edit(taskname, modelname, promptname, llm_info):
    ## 生成deepseek在mol-understanding benchmark上的test结果, 包括raw-cot以及json格式的预测输出
    assert taskname in ['add', 'delete', 'sub']
    assert promptname in ['raw', 'cot_template', 'cot_groundtruth']
    print(f"*** Task={taskname}, Model={modelname}, Prompt={promptname}")
    
    llm_client = OpenAI(
        api_key="EMPTY", 
        base_url=f"http://192.168.{llm_info['node']}:{llm_info['port']}/v1"
    )
    
    task_info_path = dict(
        add="../dataset/mol_edit/1-instruct_to_edit/add_cot.json",
        delete="../dataset/mol_edit/1-instruct_to_edit/delete_cot.json",
        sub="../dataset/mol_edit/1-instruct_to_edit/sub_cot.json"
    )
    
    task_saving_path = dict(
        add=f"results/mol_edit/add/cot_results_{modelname}_{promptname}.json",
        delete=f"results/mol_edit/delete/cot_results_{modelname}_{promptname}.json",
        sub=f"results/mol_edit/sub/cot_results_{modelname}_{promptname}.json",
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
            if promptname == "cot_groundtruth":
                cot_gt = mol_infos[i]['json_results']
                if type(cot_gt) == str:
                    cot_gt = tranform_str_to_json(cot_gt)
                if cot_gt != None:
                    cot_gt['output'] = ""
                    source_content += json.dumps(cot_gt, indent=4, ensure_ascii=False) 
            llm_predict(llm_client=llm_client, llm_name=llm_info['llm'], mol_info=mol_infos[i], system_content=system_content, user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "delete":
            smiles, removed_group = mol_infos[i]['molecule'], mol_infos[i]['removed_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}." 
            if promptname == "cot_groundtruth":
                cot_gt = mol_infos[i]['json_results']
                if type(cot_gt) == str:
                    cot_gt = tranform_str_to_json(cot_gt)
                if cot_gt != None:
                    cot_gt['output'] = ""
                    source_content += json.dumps(cot_gt, indent=4, ensure_ascii=False)
            llm_predict(llm_client=llm_client, llm_name=llm_info['llm'], mol_info=mol_infos[i], system_content=system_content, user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        
        elif taskname == "sub":
            smiles, added_group, removed_group = mol_infos[i]['molecule'], mol_infos[i]['added_group'], mol_infos[i]['removed_group']
            source_content = f"Input Molecule: {smiles}, Functional Group to delete: {removed_group}, Functional Group to add: {added_group}."
            if promptname == "cot_groundtruth":
                cot_gt = mol_infos[i]['json_results']
                if type(cot_gt) == str:
                    cot_gt = tranform_str_to_json(cot_gt)
                if cot_gt != None:
                    cot_gt['output'] = ""
                    source_content += json.dumps(cot_gt, indent=4, ensure_ascii=False)
            llm_predict(llm_client=llm_client, llm_name=llm_info['llm'], mol_info=mol_infos[i], system_content=system_content, user_content=source_content, taskname=taskname, saving_path=task_saving_path[taskname])
        

if __name__ == "__main__":
    # model_list = ["qwen2.5_1.5b", "qwen2.5_7b", "qwen2.5_14b", "qwen2.5_1.5b_distill", "qwen2.5_7b_distill", "qwen2.5_14b_distill"]
    # model_list = ["qwen2.5_1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
    
    model_list = ["distill-1.5b", "distill-7b", "distill-14b", "distill-32b"]
    prompt_list = ['raw', 'cot_template', 'cot_groundtruth']
    
    llm_infos = {
        "qwen2.5_1.5b": dict(llm="qwen2.5-1.5B", node="81.83", port="7140"),
        "qwen2.5-7b": dict(llm="qwen2.5-7B", node="81.83", port="7141"),
        "qwen2.5-14b": dict(llm="qwen2.5-14B", node="81.80", port="7140"),
        "qwen2.5-32b": dict(llm="qwen2.5-32B", node="81.80", port="7141"),
        "distill-1.5b": dict(llm="DeepSeek-R1-Distill-Qwen-1.5B", node="81.82", port="7140"),
        "distill-7b": dict(llm="DeepSeek-R1-Distill-Qwen-7B", node="81.80", port="12000"),
        "distill-14b": dict(llm="DeepSeek-R1-Distill-Qwen-14B", node="81.80", port="12001"),
        "distill-32b": dict(llm="DeepSeek-R1-Distill-Qwen-32B", node="81.81", port="7140"),
    }
    
    for model_name in model_list:
        for prompt_type in prompt_list:
            get_mol_edit(taskname="add", modelname=model_name, promptname=prompt_type, llm_info=llm_infos[model_name])
            get_mol_edit(taskname="delete", modelname=model_name, promptname=prompt_type, llm_info=llm_infos[model_name])
            get_mol_edit(taskname="sub", modelname=model_name, promptname=prompt_type, llm_info=llm_infos[model_name])