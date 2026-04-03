## chemcotbench: 
# (1) \nYour response must be directly parsable JSON format:\n{\n    \"output\": \"Modified Molecule SMILES\"\n}\n
# (2) \nYour response must be directly parsable JSON format:\n{\n    \"Final Target Molecule\": \"SMILES\",\n}\n
# (3) \nYour response must be directly parsable JSON format:\n{\n    \"output\": \"Yes / No\"\n}\n
# (4) \nYour response must be directly parsable JSON format:\n{\n    \"Output Scaffold\": \"SMILES\"\n}\n
# (5) \nYour response must be directly parsable JSON format:\n{\n    \"count\": \"Your Answer Number\"\n}\n

## 读取chemcot-bench里面的所有任务的json
##  (1) 补全meta信息中缺乏的内容
##  (2) 在query的基础上, 生成一个short_query

import re
import os
import json
import ast

def extract_text_info(task_type, input_text):
    result = {}
    if task_type in ['qed', 'logp', 'solubility', 'jnk', 'gsk', 'drd']:
        source_match = re.search(r"Source Molecule:\s*([^\n]+\.)", input_text)
        task_match = re.search(r"improve the(.*?)while", input_text, re.DOTALL)
        
        if source_match == None or task_match == None:
            print(task_type, input_text)
            assert 1 == 0

        result['molecule'] = source_match.group(1).strip().rstrip('.')
        result['task_description'] = task_match.group(1).strip()
        
    elif task_type in ['add', 'delete', 'sub']:
        input_molecule_match = re.search(r"Input Molecule:\s*([^,]+)[,.]?", input_text)
        functional_group_matches = re.findall(
            r"Functional Group to\s+(add|delete):\s*([^,]+)[,.]?", 
            input_text, re.IGNORECASE
        )
        
        result['molecule'] = input_molecule_match.group(1).strip().rstrip('.')
        operations = [match[0].lower() for match in functional_group_matches]
        functional_groups = [match[1].strip('.') for match in functional_group_matches]
        for i in range(len(operations)):
            if operations[i] == 'add': result['added_group'] = functional_groups[i]
            elif operations[i] == 'delete': result['removed_group'] = functional_groups[i]
    
    elif task_type == "equivalence":
        molecule_a_match = re.search(r"Molecule A:\s*([^,]+)[,.]?", input_text)
        molecule_b_match = re.search(r"Molecule B:\s*([^,]+)[,.]?", input_text)
        result["molecule"] = molecule_a_match.group(1).strip().rstrip('.')
        result["molecule2"] = molecule_b_match.group(1).strip().rstrip('.')
    
    elif task_type == 'fg_count':
        molecule_match = re.search(r"Input Molecule:\s*([^,]+)[,.]?", input_text)
        frag_match = re.search(r"Fragment Name:\s*([^,]+)[,.]?", input_text)
        result["molecule"] = molecule_match.group(1).strip().rstrip('.')
        result["functional_group"] = frag_match.group(1).strip().rstrip('.')
    
    elif task_type == "Murcko_scaffold":
        molecule_match = re.search(r"Input Molecule:\s*([^,]+)[,.]?", input_text)
        result["molecule"] = molecule_match.group(1).strip().rstrip('.')
    
    elif task_type == "ring_count":
        molecule_match = re.search(r"Input Molecule:\s*([^,]+)[,.]?", input_text)
        result["molecule"] = molecule_match.group(1).strip().rstrip('.')
        ringcount_match = re.search(r"Ring Structure:\s*([^,]+)[,.]?", input_text)
        result["ring"] = ringcount_match.group(1).strip().rstrip('.')
    
    elif task_type == "ring_system_scaffold":
        molecule_match = re.search(r"Input Molecule:\s*([^,]+)[,.]?", input_text)
        result["molecule"] = molecule_match.group(1).strip().rstrip('.')
        ringsystem_match = re.search(r"Ring System Structure:\s*([^,]+)[,.]?", input_text)
        result["ring_system_scaffold"] = ringsystem_match.group(1).strip().rstrip('.')
    
    return result

def reshape(input_dir, task_list):
    all_data = list(); json_files = list()
    
    for main_task in task_list:
        main_task_path = os.path.join(input_dir, main_task)
        for filename in os.listdir(main_task_path):
            if filename.endswith(".json"):
                basename, extension = os.path.splitext(filename)
                json_files.append((basename, os.path.join(main_task_path, filename)))
    
    for task_info in json_files:
        task_data = json.load(open(task_info[1], "r"))
        print(f"Task: {task_info[0]}, Num: {len(task_data)}")
        
        result_list = list()
        for sample in task_data:
            meta_info = extract_text_info(task_type=task_info[0], input_text=sample['query'])
            original_meta = json.loads(sample['meta'])
            merged_meta = {**meta_info, **original_meta} 
            merged_meta_str = json.dumps(merged_meta)
            sample['meta'] = merged_meta_str
            result_list.append(sample)
        
        json.dump(result_list, open(task_info[1], "w"), indent=4)

import ast
from tqdm import tqdm

def rxn_reshape(source_dir, target_dir, task_list):
    ## for fs, split to fs_major_product, fs_by_product
    ## for other tasks, only reshape query
    
    for task in task_list:
        raw_info_list = json.load(open(os.path.join(source_dir, f"{task}.json"), "r"))
        if task == 'fs':
            ## fs split to fs_major_product, fs_by_product
            result_major_list, result_by_list = [], []
            for i in tqdm(range(len(raw_info_list))):
                raw_info = raw_info_list[i]
                query_major_product = raw_info['query'].split("Answer:\n{\n    \"Major Product\": ...\n    \"Byproduct(s)\": ...\n}\n")[0] + "Output: <answer> SMILES of By Product </answer>."
                query_major_product = query_major_product.replace("The answer should be a json format that includes the main product SMILES and byproduct SMILES", "The answer should be a <answer> </answer> format that includes the main product SMILES")
                
                query_by_product = raw_info['query'].split("Answer:\n{\n    \"Major Product\": ...\n    \"Byproduct(s)\": ...\n}\n")[0] + "Output: <answer> SMILES of By Product </answer>."
                query_by_product = query_by_product.replace("your task is to predict the main product SMILES", "your task is to predict the by product SMILES")
                query_by_product = query_by_product.replace("The answer should be a json format that includes the main product SMILES and byproduct SMILES", "The answer should be a <answer> </answer> format that includes the by product SMILES")
                gt_dict = ast.literal_eval(raw_info['gt'])
                
                result_major_list.append(dict(
                    id=raw_info['id'],
                    query=query_major_product,
                    gt=gt_dict['Major Product'],
                    task=raw_info['task'],
                    subtask="fs_major_product",
                    meta=raw_info['meta'],
                ))
                
                if gt_dict['Byproduct(s)'] != []:
                    result_by_list.append(dict(
                        id=raw_info['id'],
                        query=query_by_product,
                        gt=gt_dict['Byproduct(s)'],
                        task=raw_info['task'],
                        subtask="fs_by_product",
                        meta=raw_info['meta'],
                    ))
            json.dump(result_major_list, open(os.path.join(target_dir, f"fs_major_product.json"), "w"), indent=4)
            json.dump(result_by_list, open(os.path.join(target_dir, f"fs_by_product.json"), "w"), indent=4)
                
        elif task == 'mechsel':
            result_list = []
            for i in tqdm(range(len(raw_info_list))):
                raw_info = raw_info_list[i]
                new_query = raw_info['query'].replace("in JSON format:\n{\n    \"choice\": str # (e.g. 'A'/'B')\n}", 'in format, Output: <answer> A/B/C ... </answer>.')
                raw_info['query'] = new_query
                result_list.append(raw_info)
            json.dump(result_list, open(os.path.join(target_dir, f"{task}.json"), "w"), indent=4)
            
        elif task == 'nepp':
            result_list = []
            for i in tqdm(range(len(raw_info_list))):
                raw_info = raw_info_list[i]
                new_query = raw_info['query'].replace('Just return the SMILES of prediction. Your response must contains directly parsable JSON format: \n\n{\n    \"pred_smi\": str\n}', 'Output: <answer> SMILES of the prediction </answer>.')
                raw_info['query'] = new_query
                result_list.append(raw_info)
            json.dump(result_list, open(os.path.join(target_dir, f"{task}.json"), "w"), indent=4)
            
        elif task == 'rcr':
            result_list = []
            for i in tqdm(range(len(raw_info_list))):
                raw_info = raw_info_list[i]
                new_query = raw_info['query'].replace("following the JSON format:\n{\n    \"SMILES\": str\n}\n", "following the format, Output: <answer> SMILES </answer>.")
                raw_info['query'] = new_query
                result_list.append(raw_info)
            json.dump(result_list, open(os.path.join(target_dir, f"{task}.json"), "w"), indent=4)
        
        elif task == 'retro':
            result_list = []
            for i in tqdm(range(len(raw_info_list))):
                raw_info = raw_info_list[i]
                new_query = raw_info['query'].replace("Answer:\n{\n    \"Reactants\": ...\n}\n", "Output: <answer> SMILES of Reactants </answer>.")
                raw_info['query'] = new_query
                result_list.append(raw_info)
            json.dump(result_list, open(os.path.join(target_dir, f"{task}.json"), "w"), indent=4)
            
        else: assert KeyError

if __name__ == "__main__":
    ## reshaping mol_und, mol_edit, mol_opt
    # data_source = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/"
    # reshape(
    #     input_dir=data_source, 
    #     # task_list=['mol_und'],
    #     task_list=['mol_edit', 'mol_opt', 'mol_und'],
    # )
    
    old_data_source = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/old_versions/chemcotbench/reaction/"
    new_data_source = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotbench/reaction/"
    rxn_reshape(
        source_dir=old_data_source,
        target_dir=new_data_source,
        # task_list=['fs', 'mechsel', 'nepp', 'rcr', 'retro'],
        task_list=['fs'],
    )
