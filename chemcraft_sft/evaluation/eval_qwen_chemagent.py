## 这个函数专门评测 使用tool的RL/SFT模型 在各个任务的性能
## 相较于之前的 ChemCoTBench 使用的评测, 这个评测存在两个区别点
## (1) 不同任务不是放在add, delete, sub等单独json，是放在一个大的json
## (2) 一个eval函数评测所有的mol_edit, mol_und, mol_opt的任务

import re
import json
from tqdm import tqdm

from eval_moledit import eval_moledit_from_list
from eval_molopt import eval_molopt_from_list
from eval_molund import eval_molund_from_list
from eval_rxn import MoleculeSMILESEvaluator

def get_answer(text):
    # re字符串匹配, 从后往前第一个<answer> </answer>中的内容作为答案
    match = re.search(r'<answer>\s*(.*?)\s*</answer>(?![^<>]*</answer>)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer
    else:
        return None

def read_from_json_lines(file_path):
    sample_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行尾换行符并解析
            if line.strip():  # 确保不是空行
                sample = json.loads(line)
                sample_list.append(sample)
    return sample_list

def evaluate_moledit(taskname, task_sample_list):
    invalid_number = 0; correct_num = 0
    pred_list, src_list = list(), list()
    group_a, group_b = list(), list()
    
    for sample in task_sample_list:
        pred_smiles = None
        if sample["current_history"][-1]["role"]=="assistant":
            pred_smiles = get_answer(text=sample["current_history"][-1]["content"])
            if pred_smiles != None:
                pred_list.append(pred_smiles.rstrip('.'))
                src_list.append(sample["extra_info"]["molecule"].rstrip('.'))
                if taskname == 'add': 
                    added_group = sample["extra_info"]['added_group'].rstrip('.')
                    group_a.append(added_group)
                elif taskname == 'delete':
                    removed_group = sample["extra_info"]['removed_group'].rstrip('.')
                    group_a.append(removed_group)
                elif taskname == 'sub':
                    added_group = sample["extra_info"]['added_group'].rstrip('.')
                    group_a.append(added_group)
                    removed_group = sample["extra_info"]['removed_group'].rstrip('.')
                    group_b.append(removed_group)

        # Format Error:
        if pred_smiles == None: invalid_number += 1
    
    assert len(src_list) == len(pred_list)
    assert len(src_list) == len(group_a)
    
    result = eval_moledit_from_list(src_list=src_list, pred_list=pred_list, group_a=group_a, group_b=group_b, task=taskname, total_number=len(pred_list))
    
    return result


def evaluate_molopt(taskname, task_sample_list):
    tgt_smiles_list, src_smiles_list = list(), list()
    invalid_number = 0
    
    for sample in task_sample_list:
        pred_smiles = None
        if sample["current_history"][-1]["role"]=="assistant":
            pred_smiles = get_answer(text=sample["current_history"][-1]["content"])
            if pred_smiles != None:
                tgt_smiles_list.append(pred_smiles.rstrip('.'))
                src_smiles_list.append(sample["extra_info"]["molecule"].rstrip('.'))
        
        # Format Error:
        if pred_smiles == None: invalid_number += 1
    
    assert len(tgt_smiles_list) == len(src_smiles_list)
    result_dict = eval_molopt_from_list(optimized_prop=taskname, gt_list=src_smiles_list, pred_list=tgt_smiles_list, total_number=len(task_sample_list))
    
    return result_dict

def evaluate_molund(taskname, task_sample_list, gt_key_dict):
    result_dict = dict(); invalid_number = 0
    pred_list, gt_list = list(), list()
    for sample in task_sample_list:
        pred_answer = None
        if sample["current_history"][-1]["role"]=="assistant":
            pred_answer = get_answer(text=sample["current_history"][-1]["content"])
            if pred_answer != None:
                pred_list.append(pred_answer.rstrip('.'))
                if gt_key_dict[taskname] != '':
                    gt_list.append(sample["extra_info"][gt_key_dict[taskname]])
        # Format Error:
        if pred_answer == None: invalid_number += 1
    
    result_dict = eval_molund_from_list(gt_list=gt_list, pred_list=pred_list, total_number=len(task_sample_list), task=taskname)
    return result_dict

def evaluate_rxn(taskname, task_sample_list):
    invalid_number = 0
    pred_list, gt_list = list(), list()
    for sample in task_sample_list:
        pred_answer = None
        if sample["current_history"][-1]["role"]=="assistant":
            pred_answer = get_answer(text=sample["current_history"][-1]["content"])
            if pred_answer != None:
                pred_list.append(pred_answer.rstrip('.'))
                gt_list.append(sample["extra_info"]['gt'])
        # Format Error:
        if pred_answer == None: invalid_number += 1
    
    if taskname in ['rcr', 'nepp', 'fs_major_product', 'fs_by_product']:
        evaluator = MoleculeSMILESEvaluator()
        res = evaluator.evaluate(pred_list, gt_list)
        fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
        res['fts'] = fts * res['validity']
        return res
    
    elif taskname in ['retro']:
        evaluator = MoleculeSMILESEvaluator()
        res = evaluator.evaluate(pred_list, gt_list)
        fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
        res['fts'] = fts * res['validity']
        
        pred_modified_list = [pred.split('.')[0] for pred in pred_list]
        res_v2 = evaluator.evaluate(pred_modified_list, gt_list)
        fts_v2 = (res_v2['rdk_sims'] + res_v2['maccs_sims'] + res_v2['morgan_sims']) / 3
        res_v2['fts'] = fts_v2 * res_v2['validity']
        
        return {"v1": res, "v2_reshaped": res_v2}
    
    elif taskname in ['mechsel']:
        pred_new_list = [pred.lower() for pred in pred_list]
        gt_new_list = [gt.lower() for gt in gt_list]
        accuracy = sum(1 for pred, gt in zip(pred_new_list, gt_new_list) if pred == gt) / len(gt_new_list)
        return {'MCQ Acc': accuracy}
    else:
        assert KeyError


def main_evaluate_sft(sample_path):
    # for evaluations on LLM fully-trained and test on SFT split
    # due to the format little errors, we need to extract the molecule-optimization task names from the user content in current_history
    sample_list = json.load(open(sample_path+".json", "r"))
    task_sample_dict = dict(
        # molecule-understanding
        fg_samples=list(), Murcko_scaffold=list(), mutated=list(), permutated=list(), ring_count=list(), ring_system_scaffold=list(),                                                   
        # molecule-editing
        add=list(), delete=list(), sub=list(),
        # molecule-optimization
        qed=list(), logp=list(), solubility=list(), drd=list(), gsk=list(), jnk=list()
    )
    
    for sample in tqdm(sample_list, desc="test_samples"):
        task_name = sample['extra_info']['task']
        if task_name in task_sample_dict.keys():
            task_sample_dict[task_name].append(sample)
        else:
            if "Solubility property" in sample['current_history'][1]['content']:
                task_sample_dict['solubility'].append(sample)
            elif "QED property" in sample['current_history'][1]['content']:
                task_sample_dict['qed'].append(sample)
            elif "LogP property" in sample['current_history'][1]['content']:
                task_sample_dict['logp'].append(sample)
            elif "GSK3-beta property" in sample['current_history'][1]['content']:
                task_sample_dict['gsk'].append(sample)
            elif "JNK3 property" in sample['current_history'][1]['content']:
                task_sample_dict['jnk'].append(sample)
            elif "DRD2 property" in sample['current_history'][1]['content']:
                task_sample_dict['drd'].append(sample)
            elif "count the number of the fragment in the Molecule" in sample['current_history'][1]['content']:
                task_sample_dict['ring_count'].append(sample)
            else:
                print(sample['current_history'][1]['content'])
                assert 1 == 0
    
    for key in task_sample_dict.keys():
        print(f"Task: {key}, Number: {len(task_sample_dict[key])}")
    
    result_dict = dict()
    for key in task_sample_dict.keys():
        task_sample_list = task_sample_dict[key]
        
        if key in ["add", "delete", "sub"]:
            result = evaluate_moledit(taskname=key, task_sample_list=task_sample_list)
            result_dict[key] = result
        
        elif key in ["qed", "logp", "solubility", "drd", "gsk", "jnk"]:
            result_opt = evaluate_molopt(taskname=key, task_sample_list=task_sample_list)
            result_dict[key] = result_opt
                
        elif key in ["fg_samples", "Murcko_scaffold", "mutated", "permutated", "equivalence", "ring_count", "ring_system_scaffold"]:
            gt_key_dict = dict(
                fg_samples='fg_num', fg_count="gt", Murcko_scaffold='largest_scaffold', ring_count='count', ring_system_scaffold='', equivalence='', permutated='', mutated='',
            )
            result_und = evaluate_molund(taskname=key, task_sample_list=task_sample_list, gt_key_dict=gt_key_dict)
            result_dict[key] = result_und
        
        else:
            assert "Unknown Task Name while Evaluating"
    
    print(result_dict)
    json.dump(result_dict, open(sample_path+"_eval_full_sft_new.json", "w"), indent=4)
    
def main_evaluate_chemcotbench(sample_path):
    # for evaluations on LLM fully-trained on SFT, but evaluate on ChemCoTBench
    # difference is in the `this_task`
    # sample_list = json.load(open(sample_path+".json", "r"))
    sample_list = read_from_json_lines(file_path=sample_path+".json")

    task_sample_dict = dict(
        # molecule-understanding
        fg_count=list(), Murcko_scaffold=list(), equivalence=list(), ring_count=list(), ring_system_scaffold=list(),                                                   
        # molecule-editing
        add=list(), delete=list(), sub=list(),
        # molecule-optimization
        qed=list(), logp=list(), solubility=list(), drd=list(), gsk=list(), jnk=list(),
        # reaction-related tasks
        fs_by_product=list(), fs_major_product=list(), retro=list(), mechsel=list(), nepp=list(), rcr=list(),
    )
    
    for sample in tqdm(sample_list, desc="test_samples"):
        this_task = ""
        task_name = sample['extra_info']['task']
        subtask_name = sample['extra_info']['subtask']
        if subtask_name != None:
            this_task = subtask_name
        else:
            this_task = task_name
        if this_task in task_sample_dict.keys():
            task_sample_dict[this_task].append(sample)
        else:
            print(task_name, subtask_name, this_task)
    
    for key in task_sample_dict.keys():
        print(f"Task: {key}, Number: {len(task_sample_dict[key])}")
    
    result_dict = dict()
    for key in task_sample_dict.keys():
        task_sample_list = task_sample_dict[key]
        
        if key in ["add", "delete", "sub"]:
            result = evaluate_moledit(taskname=key, task_sample_list=task_sample_list)
            result_dict[key] = result
        
        elif key in ["qed", "logp", "solubility", "drd", "gsk", "jnk"]:
            result_opt = evaluate_molopt(taskname=key, task_sample_list=task_sample_list)
            result_dict[key] = result_opt
                
        elif key in ["fg_count", "Murcko_scaffold", "mutated", "permutated", "equivalence", "ring_count", "ring_system_scaffold"]:
            gt_key_dict = dict(
                fg_count="gt", Murcko_scaffold='gt', ring_count='gt', ring_system_scaffold='gt', equivalence='gt',
            )
            result_und = evaluate_molund(taskname=key, task_sample_list=task_sample_list, gt_key_dict=gt_key_dict)
            result_dict[key] = result_und
        
        elif key in ["fs_major_product", "fs_by_product", "retro", "mechsel", "nepp", "rcr"]:
            result_rxn = evaluate_rxn(taskname=key, task_sample_list=task_sample_list)
            result_dict[key] = result_rxn
        
        else:
            assert "Unknown Task Name while Evaluating"
    
    print(result_dict)
    json.dump(result_dict, open(sample_path+"_eval.json", "w"), indent=4)
    
if __name__ == "__main__":
    # main_evaluate_sft(
    #    sample_path="../../search_saves/results/tool_sft_results/chemcot-tool-coldstart-qwen-7B-full-tool-assistant/900"
    # )
    
    # main_evaluate_chemcotbench(
    #     # sample_path="../../search_saves/results/tool_sft_results/chemcot-tool-coldstart-qwen-7B-full-tool-assistant-murcko/chemcotbench_900",
    #     # sample_path="../../search_saves/results/tool_sft_results/chemcot-tool-coldstart-7B-for-rl/chemcotbench_125",
    #     # sample_path="../../search_saves/results/tool_sft_results/chemcot-tool-coldstart-rxn-qwen-7B-coldstart-55/chemcotbench_300_full_gt",
    #     sample_path="../../search_saves/results/tool_sft_results/chemcot-tool-coldstart-rxn-qwen-7B-coldstart-55-murcko-82/model_300",
    #     # sample_path="../../search_saves/results/tool_rl_results/ChemAgent-Qwen-7B/chemcotbench_300_smiles_loss",
    #     # sample_path="../../search_saves/results/tool_rl_results/ChemAgent-Qwen-7B-with-RXN/model_300_epoch_1_7e-7",
    # )
    
    ## for ablation, evaluation batch of model results
    for i in range(240, 1860, 60):
        main_evaluate_chemcotbench(
            sample_path=f"../../search_saves/results/tool_rl_results/ChemAgent-Qwen-7B-with-RXN-epoch3/model_{i}_epoch_3_1e-6",
        )