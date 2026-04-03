## 将 chemcotdataset 作为train, 将 chemcotbench 作为test, 生成他们的train-test-parquet
## 1. 研究 nq_search / nq_hotpot 的数据格式, 做转换
## 2. 把 chemcotbench, chemcotdataset 转换成 jsonl的形式
## 3. datasets.load() 读取 jsonl, 然后保存成 train.parquet, test.parquet


import datasets
import json
import pandas as pd
from tqdm import tqdm
    
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

def get_json():
    ## generate the train.json from chemcotdataset, and test.json from chemcotbench
    ## question, golden_answer
    ## data_source, index, split="train"/"test"
    
    chemcotdataset_path = "/cto_labs/lihao/chem_reason/opt_datacollection/dataset-large/"
    chemcotbench_path = "/cto_labs/lihao/chem_reason/opt_datacollection/dataset/"
    
    mol_edit_info = dict(
        add=dict(
            datapath=chemcotdataset_path+"mol_edit/add_cot.json",
            question_template=None,
            extra_info_keys=["Instruction","molecule","added_group","reference","task"],
            cot_key="json_results",
            template_keys=None,
        ),
        delete=dict(
            datapath=chemcotdataset_path+"mol_edit/delete_cot.json",
            question_template=None,
            extra_info_keys=["Instruction","molecule","removed_group","reference","task"],
            cot_key="json_results",
            template_keys=None,
        ),
        sub=dict(
            datapath=chemcotdataset_path+"mol_edit/sub_cot.json",
            question_template=None,
            extra_info_keys=["Instruction","molecule","added_group","removed_group","reference","task"],
            cot_key="json_results",
            template_keys=None,
        ),
    )
    
    mol_und_info = dict(
        fg_detect=dict(
            datapath=chemcotdataset_path+"mol_und/fg_detect_samples_cot.json",
            question_template="You are a chemical assistent. Giving you an Input Molecule: {} and a Fragment name: {}, help me count the number of the fragment in the Molecule. Output: <answer> count-number </answer>.",
            extra_info_keys=["smiles", "fg_name", "fg_num", "task"],
            cot_key="json_results",
            template_keys=["smiles", "fg_name"],
        ),
        murcko_scaffold=dict(
            datapath=chemcotdataset_path+"mol_und/murcko_scaffold_iupac_cot.json",
            question_template='''You are a chemical assistent. Please Determine whether the Murcko scaffold is in the Molecule. Input molecule: {}, the Largest Murcko scaffold: {}.  Output: <answer> Yes/No </answer>. Definition: The Murcko scaffold is obtained by removing all side chains, functional groups, and exocyclic modifications, leaving only the ring systems and connecting bonds.''',
            extra_info_keys=["smiles", "largest_scaffold", "type", "task"],
            cot_key="json_results",
            template_keys=["smiles", "largest_scaffold"],
        ),
        mutated=dict(
            datapath=chemcotdataset_path+"mol_und/mutated_cot.json",
            question_template='''You are a chemical assistent. Given two molecule SMILES, Please Determine whether these two Molecules are the same. Input: Molecule A: {}, Molecule B: {}. Output: <answer> Yes/No </answer>.''',
            extra_info_keys=["smiles", "mutated", "type", "task"],
            cot_key="json_results",
            template_keys=["smiles", "mutated"],
        ),
        permutated=dict(
            datapath=chemcotdataset_path+"mol_und/permutated_cot.json",
            question_template='''You are a chemical assistent. Given two molecule SMILES, Please Determine whether these two Molecules are the same. Input: Molecule A: {}, Molecule B: {}. Output: <answer> Yes/No </answer>.''',
            extra_info_keys=["smiles", "permutated", "type", "task"],
            cot_key="json_results",
            template_keys=["smiles", "permutated"],
        ),
        ring_count=dict(
            datapath=chemcotdataset_path+"mol_und/ring_count_iupac_cot.json",
            question_template='''You are a chemical assistent. Giving you an Input Molecule: {} and a Ring Structure: {}, help me count the number of ring structure in the Molecule. Output: <answer> count-number </answer>.''',
            extra_info_keys=["smiles", "ring", "count", "type", "task"],
            cot_key="json_results",
            template_keys=["smiles", "ring"],
        ),
        ring_system_scaffold=dict(
            datapath=chemcotdataset_path+"mol_und/ring_system_scaffold_iupac_cot.json",
            question_template='''You are a chemical assistent. Please Determine whether the ring_system_scaffold is in the Molecule. Input Molecule: {},  the Ring System Scaffold: {}. Output: <answer> Yes/No </answer>.''',
            extra_info_keys=["smiles", "ring_system_scaffold", "type", "task"],
            cot_key="json_results",
            template_keys=["smiles", "ring_system_scaffold"],
        ),
    )
    
    mol_opt_info = dict(
        drd=dict(
            datapath=chemcotdataset_path+"mol_opt_single/drd/final_mmp_cot.json",
            question_template='''You are a chemical assistent. Given the source molecule: {}, Optimize the Molecule to improve its DRD2 property (Dopamine D2 Receptor Activity) while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>.''',
            extra_info_keys=["src", "tgt", "src_drd2", "tgt_drd2"],
            cot_key="cot_result",
            template_keys=["src"],
        ),
        gsk=dict(
            datapath=chemcotdataset_path+"mol_opt_single/gsk/final_mmp_cot.json",
            question_template='''You are a chemical assistent. Given the source molecule: {}, Optimize the Molecule to improve its GSK3-beta property (Glycogen Synthase Kinase 3-beta Inhibition) while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>.''',
            extra_info_keys=["src", "tgt", "src_gsk", "tgt_gsk"],
            cot_key="cot_result",
            template_keys=["src"],
        ),
        jnk=dict(
            datapath=chemcotdataset_path+"mol_opt_single/jnk/final_mmp_cot.json",
            question_template='''You are a chemical assistent. Given the source molecule: {}, Optimize the Molecule to improve its JNK3 property (c-Jun N-terminal kinase 3 inhibition) while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>.''',
            extra_info_keys=["src", "tgt", "src_jnk3", "tgt_jnk3"],
            cot_key="cot_result",
            template_keys=["src"],
        ),
        qed=dict(
            datapath=chemcotdataset_path+"mol_opt_single/qed/final_mmp_cot.json",
            question_template='''You are a chemical assistent. Given the source molecule: {}, Optimize the Molecule to improve its QED property (Drug-likeness) while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>.''',
            extra_info_keys=["src", "tgt", "src_qed", "tgt_qed"],
            cot_key="cot_result",
            template_keys=["src"],
        ),
        logp=dict(
            datapath=chemcotdataset_path+"mol_opt_single/logp/final_mmp_cot.json",
            question_template='''You are a chemical assistent. Given the source molecule: {}, Optimize the Molecule to improve its LogP property (Partition coefficient between octanol and water) while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>.''',
            extra_info_keys=["src", "tgt", "src_logp", "tgt_logp"],
            cot_key="cot_result",
            template_keys=["src"],
        ),
        solubility=dict(
            datapath=chemcotdataset_path+"mol_opt_single/solubility/final_mmp_cot.json",
            question_template='''You are a chemical assistent. Given the source molecule: {}, Optimize the Molecule to improve its Solubility property (compound's ability to dissolve in water) while following a structured intermediate optimization process. Output: <answer> SMILES of the modified molecule </answer>.''',
            extra_info_keys=["src", "tgt", "src_solubility", "tgt_solubility"],
            cot_key="cot_result",
            template_keys=["src"],
        ),
    )
    
    for task_info in [mol_edit_info, mol_opt_info, mol_und_info]:
    # for task_info in [mol_opt_info]:
        for task_name in task_info.keys():
            template_info = task_info[task_name]
            raw_info = json.load(open(task_info[task_name]['datapath'], "r"))
            result_list = list(); idx = 0
            
            for info in tqdm(raw_info, desc=task_name):
                # question generation
                if template_info['question_template'] == None:
                    question = info['Instruction'] + " Output: <answer> count-number <answer>."
                else:
                    if len(template_info['template_keys']) == 1:
                        question = template_info['question_template'].format(info[template_info['template_keys'][0]])
                    elif len(template_info['template_keys']) == 2:
                        question = template_info['question_template'].format(info[template_info['template_keys'][0]], info[template_info['template_keys'][1]])

                answer = ""; temp = list()
                
                if template_info['cot_key'] not in info.keys(): 
                    continue
                
                if type(info[template_info['cot_key']]) != dict: 
                    info[template_info['cot_key']] = tranform_str_to_json(info[template_info['cot_key']])
                    if type(info[template_info['cot_key']]) != dict: 
                        continue
                    
                for key, sentence in info[template_info['cot_key']].items():
                    temp.append((key, sentence))
                
                for i in range(len(temp)-1):
                   answer += f"Step-{i+1}, {temp[i][0]}: {temp[i][1]}" 
                   if answer[-1] != '.': answer += '. '
                   else: answer += ' '
                answer += f"Output: <answer> {temp[-1][1]} </answer>"
                
                # extra information
                extra_info = dict()
                for key in template_info['extra_info_keys']:
                    extra_info[key] = info[key]
                extra_info["index"] = idx; idx += 1
                
                result_list.append(dict(
                    question=question,
                    answer=answer,
                    extra_info=extra_info,
                    task=task_name,
                ))
            
            json.dump(result_list, open(f"chemcot_json/chemcotdataset/{task_name}.json", "w"), indent=4)


if __name__ == "__main__":
    get_json()

