## 将 chemcotdataset 作为train, 将 chemcotbench 作为test, 生成他们的train-test-parquet
## 1. 研究 nq_search / nq_hotpot 的数据格式, 做转换
## 2. 把 chemcotbench, chemcotdataset 转换成 jsonl的形式
## 3. datasets.load() 读取 jsonl, 然后保存成 train.parquet, test.parquet

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

import json
import pandas as pd
import argparse
import os
import random
from glob import glob

from cold_start_v2.get_tool_sft_trajectory import sft_trajectory_builder

random.seed(42)

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def check_parquet(parquet_path=None):
    train_path = parquet_path+"train_tool_full_rxn.parquet"
    test_path = parquet_path+"test_tool_full_rxn.parquet"
    
    df_train, df_test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    print(df_train.head())
    print(df_train.columns.tolist())
    print()
    
    for i in range(len(df_train)):
        # print(df_train.iloc[i]['id'], df_train.iloc[i]['system_prompt'])
        # print(df_train.iloc[i]['user_prompt'])
        # print(df_train.iloc[i]['extra_info'])
        # print(df_train.iloc[i]['answer_assistant_tool'])
        # print("*  "*30)
        if df_train.iloc[i]['extra_info']['task'] == None or df_train.iloc[i]['extra_info']['subtask'] == None:
            print(df_train.iloc[i]['extra_info'])

def check_parquet_chemcotbench(parquet_path=None):
    train_path = parquet_path+"chemcotbench.parquet"
    
    df_train = pd.read_parquet(train_path)
    print(df_train.head())
    print(df_train.columns.tolist())
    print()
    
    for i in range(len(df_train)):
        # print(df_train.iloc[i]['id'], df_train.iloc[i]['system_prompt'])
        # print(df_train.iloc[i]['user_prompt'])
        # print(df_train.iloc[i]['extra_info'])
        if df_train.iloc[i]['extra_info']['task'] == None or df_train.iloc[i]['extra_info']['subtask'] == None:
            print(df_train.iloc[i]['extra_info'])

def check_parquet_searchr1(parquet_path=None):
    train_path = parquet_path+"train_rl_baseline.parquet"
    test_path = parquet_path+"test_rl_baseline.parquet"
    
    df_train, df_test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    print(df_train.head())
    print(df_train.columns.tolist())
    # ['id', 'question', 'golden_answers', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    print()
    
    for i in range(2):
        print(df_train.iloc[i]['golden_answers'])
        print(df_train.iloc[i]['prompt'])
        print(df_train.iloc[i]['reward_model'])
        print(df_train.iloc[i]['extra_info'])
        print("*  "*30)

def get_benchmark_dataset(input_dir, task_list):
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
        if task_info[0] in ["ring_count", "fg_count"]:
            tmp_list = list()
            for sample in task_data:
                sample['gt'] = str(sample['gt'])
                tmp_list.append(sample)
            task_data = tmp_list
        all_data.extend(task_data)
    
    return Dataset.from_list(all_data)

def get_train_test_dataset(input_dir, task_list):
    all_train_data, all_test_data = list(), list(); json_files = list()
    
    for task in task_list:
        json_files.append(os.path.join(input_dir, f"{task}.json"))

    for file_path in json_files:
        subtask_name = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(subtask_name, len(data))
        
        if not isinstance(data, list): continue
        
        # subset_size = max(1, len(data) // 4)  # 保证至少有1条数据
        subset_size = max(1, len(data))
        subset = random.sample(data, subset_size)
        
        
        if any(s in ["fs_major_product", "fs_by_product", "retro", "rcr_catalyst", "rcr_solvent", "rcr_reagent", "nepp", "mech_sel"] for s in file_path):
            print(f"Different split in {file_path}, 5:5")
            # 5:5分割rxn相关的train-test
            split_idx = int(0.5 * len(subset))
        elif any(s in ["murcko_scaffold"] for s in file_path):
            print(f"Different split in {file_path}, 2:8")
            # 2:8分割 murcko-scaffold 相关的 train-test
            split_idx = int(0.2 * len(subset))
        else:
            # 8:2分割其他train-test
            split_idx = int(0.8 * len(subset))
        
        train_data = subset[:split_idx]
        test_data = subset[split_idx:]
        
        # 添加到总集
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
    
    chemcot_dataset = DatasetDict({
        "train": Dataset.from_list(all_train_data),
        "test": Dataset.from_list(all_test_data)
    })
    
    return chemcot_dataset

def generate_chemcotdataset_parquet(rootpath):
    data_source = rootpath+"datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage4"
    tgt_dir = rootpath+"datasets/chemcot/chemcot_json/chemcot-tool-coldstart"
    
    dataset = get_train_test_dataset(
        input_dir=data_source, 
        task_list=['add', 'delete', 'sub', 'functiongroup_detect', 'murcko_scaffold', 'mutated', 'permutated', 'ring_count', 'ring_system_scaffold', 'jnk', 'drd', 'gsk', 'logp', 'solubility', 'qed'],
        # task_list = ['functiongroup_detect', 'murcko_scaffold', 'mutated', 'permutated', 'ring_count', 'ring_system_scaffold'],
    )

    train_dataset = dataset["train"]; test_dataset = dataset["test"]
    print(f"train split = {len(train_dataset)}, test split = {len(test_dataset)}")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            system_prompt = example.pop("prompt")
            if system_prompt[-1] == '\n': system_prompt = system_prompt[:-1]
            user_prompt = example.pop("question")
            answer_assistant_tool = example.pop("answer_with_tool_stage4")
            meta_info = example.pop("extra_info")
            
            ## Reshape name and gt, same as ChemCoTBench
            if meta_info['task'] in ['add', 'delete', 'sub']:
                case_gt = ""; main_task = "mol_edit"
            elif meta_info['task'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']:
                case_gt = ""; main_task = "mol_opt"
            else:
                main_task = "mol_und"
                if meta_info['task'] == 'mutated': 
                    meta_info['task'] = 'equivalence'; case_gt = "No"
                elif meta_info['task'] == 'permutated': 
                    meta_info['task'] = 'equivalence'; case_gt = "Yes"
                elif meta_info['task'] == 'Murcko_scaffold':
                    case_gt = meta_info['gt']
                elif meta_info['task'] == 'ring_count':
                    case_gt = meta_info['count']
                elif meta_info['task'] == 'ring_system_scaffold':
                    case_gt = "Yes"
                elif meta_info['task'] == 'fg_samples':
                    meta_info['task'] = 'fg_count'; case_gt = meta_info['fg_num']
                else:
                    assert f"Error in {meta_info['task']}"
                
            extra_info = {
                "gt": str(case_gt),
                "task": main_task,
                "subtask": meta_info['task'],
                "molecule": "",
                "molecule2": "",
                "added_group": "",
                "removed_group": "",
                "ring": "",
                "ring_system_scaffold": "",
                "functional_group": "",
            }
            if extra_info['subtask'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']:
                extra_info["molecule"] = meta_info['src']
            elif extra_info['subtask'] in ['add', 'delete', 'sub']:
                extra_info["molecule"] = meta_info['molecule']
                if "added_group" in meta_info.keys() and meta_info["added_group"] != None: extra_info["added_group"] = meta_info['added_group']
                if "removed_group" in meta_info.keys() and meta_info["removed_group"] != None: extra_info["removed_group"] = meta_info['removed_group']
            elif extra_info['subtask'] in ['ring_count']:
                extra_info["molecule"] = meta_info['smiles']
                extra_info["ring"] = meta_info['ring']
            elif extra_info['subtask'] in ['ring_system_scaffold']:
                extra_info["molecule"] = meta_info['smiles']; extra_info["ring_system_scaffold"] = meta_info['ring_system_scaffold']
            elif extra_info['subtask'] in ['equivalence']:
                extra_info["molecule"] = meta_info['smiles']
                if 'permutated' in meta_info.keys(): extra_info["molecule2"] = meta_info['permutated']
                elif 'mutated' in meta_info.keys(): extra_info["molecule2"] = meta_info['mutated']
                else: assert FileNotFoundError
            elif extra_info['subtask'] in ['fg_count']:
                extra_info["molecule"] = meta_info['smiles']; extra_info["functional_group"] = meta_info['fg_name']
            elif extra_info['subtask'] in ['Murcko_scaffold']:
                extra_info["molecule"] = meta_info['smiles']
            else: FileExistsError
            
            data = {
                'id': idx,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "answer_assistant_tool": answer_assistant_tool,
                "data_source": data_source,
                "extra_info": extra_info,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True).shuffle(seed=42)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True).shuffle(seed=42)

    train_dataset.to_parquet(os.path.join(tgt_dir, "train_tool_full.parquet"))
    test_dataset.to_parquet(os.path.join(tgt_dir, "test_tool_full.parquet"))
    return


def generate_chemcotbench_parquet(rootpath):
    data_source = rootpath+"datasets/chemcot/chemcot_json/chemcotbench/"
    tgt_dir = rootpath+"datasets/chemcot/chemcot_json/chemcotbench/"
    
    eval_dataset = get_benchmark_dataset(
        input_dir=data_source, 
        task_list=['mol_edit', 'mol_opt', 'mol_und', 'reaction'],
        # task_list=['reaction'],
    )
    print(f"ChemCoTBench = {len(eval_dataset)}")
    
    my_data_builder = sft_trajectory_builder()
    system_prompt = my_data_builder._get_stage4_system_prompt()
    
    # add a row to each data item that represents a unique id
    def make_map_fn():
        def process_fn(example, idx):
            sample_id = example.pop("id")
            user_prompt = example.pop("query")
            meta_info = json.loads(example.pop("meta"))
            extra_info = {
                "gt": example.pop("gt"),
                "task": example.pop("task"),
                "subtask": example.pop("subtask"),
                "molecule": "",
                "molecule2": "",
                "added_group": "",
                "removed_group": "",
                "ring": "",
                "ring_system_scaffold": "",
                "functional_group": "",
            }
            if extra_info['gt'] == '':
                extra_info.update((k, meta_info[k]) for k in ["molecule", "added_group", "removed_group"] if k in meta_info.keys())
            
            if extra_info['subtask'] in ['ring_count']:
                extra_info["molecule"] = meta_info['molecule']; extra_info["ring"] = meta_info['ring']
            elif extra_info['subtask'] in ['ring_system_scaffold']:
                extra_info["molecule"] = meta_info['molecule']; extra_info["ring_system_scaffold"] = meta_info['ring_system_scaffold']
            elif extra_info['subtask'] in ['equivalence']:
                extra_info.update((k, meta_info[k]) for k in ["molecule", "molecule2"] if k in meta_info.keys())
            elif extra_info['subtask'] in ['fg_count']:
                extra_info["molecule"] = meta_info['molecule']; extra_info["functional_group"] = meta_info['functional_group']
            elif extra_info['subtask'] in ['Murcko_scaffold']:
                extra_info["molecule"] = meta_info['molecule']
            
            data = {
                'id': sample_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "data_source": data_source,
                "extra_info": extra_info,
            }
            return data

        return process_fn

    eval_dataset = eval_dataset.map(function=make_map_fn(), with_indices=True)
    eval_dataset.to_parquet(os.path.join(tgt_dir, "chemcotbench_with_rxn.parquet"))
    
    # check
    final_dataset = pd.read_parquet(os.path.join(tgt_dir, "chemcotbench_with_rxn.parquet"))
    print(len(final_dataset))

def generate_chemcotdataset_rxn_parquet(rootpath, dtype="tool"):
    # dtype for tool or non-tool; tool: sft trajectory with tool-call; non-tool: trajectory without tool-call
    if dtype == "tool":
        data_source = rootpath+"datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage4"
        tgt_dir = rootpath+"datasets/chemcot/chemcot_json/chemcot-tool-coldstart"
        file_name = "{}_tool_full_rxn55_murcko28.parquet"
    elif dtype == "non-tool":
        data_source = rootpath+"datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage4"
        tgt_dir = rootpath+"datasets/chemcot/chemcot_json/chemcot-sft"
        file_name = "{}_non_tool_sft_full.parquet"
    
    dataset = get_train_test_dataset(
        input_dir=data_source, 
        task_list=['add', 'delete', 'sub', 'functiongroup_detect', 'murcko_scaffold', 'mutated', 'permutated', 'ring_count', 'ring_system_scaffold', 'jnk', 'drd', 'gsk', 'logp', 'solubility', 'qed', 'fs_major_product', 'fs_by_product', 'mech_sel', 'nepp', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro'],
        # task_list=['fs_major_product', 'fs_by_product', 'mech_sel', 'nepp', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro'],
    )

    train_dataset = dataset["train"]; test_dataset = dataset["test"]
    print(f"train split = {len(train_dataset)}, test split = {len(test_dataset)}")

    def process_fn(example, idx, dtype):
        # dtype for tool or non-tool; tool: sft trajectory with tool-call; non-tool: trajectory without tool-call
        if dtype == "tool":
            system_prompt = example.pop("prompt")
            if system_prompt[-1] == '\n': system_prompt = system_prompt[:-1]
            user_prompt = example.pop("question")
            answer_assistant = example.pop("answer_with_tool_stage4")
        elif dtype == "non-tool":
            user_prompt = example.pop("question")
            answer_assistant = example.pop("answer")
        
        meta_info = example.pop("extra_info")
        
        ## Reshape name and gt, same as ChemCoTBench
        if meta_info['task'] in ['add', 'delete', 'sub']:
            case_gt = ""; main_task = "mol_edit"
        elif meta_info['task'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']:
            case_gt = ""; main_task = "mol_opt"
        elif meta_info['task'] in ['fs_major_product', 'fs_by_product', 'mech_sel', 'nepp', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro']:
            case_gt = meta_info['gt']; main_task = "rxn"
        else:
            main_task = "mol_und"
            if meta_info['task'] == 'mutated': 
                meta_info['task'] = 'equivalence'; case_gt = "No"
            elif meta_info['task'] == 'permutated': 
                meta_info['task'] = 'equivalence'; case_gt = "Yes"
            elif meta_info['task'] == 'Murcko_scaffold':
                case_gt = meta_info['gt']
            elif meta_info['task'] == 'ring_count':
                case_gt = meta_info['count']
            elif meta_info['task'] == 'ring_system_scaffold':
                case_gt = "Yes"
            elif meta_info['task'] == 'fg_samples':
                meta_info['task'] = 'fg_count'; case_gt = meta_info['fg_num']
            else:
                assert f"Error in {meta_info['task']}"
        try: 
            extra_info = {
                "gt": str(case_gt),
                "task": main_task,
                "subtask": meta_info['task'],
                "molecule": "",
                "molecule2": "",
                "added_group": "",
                "removed_group": "",
                "ring": "",
                "ring_system_scaffold": "",
                "functional_group": "",
            }
        except:
            print(main_task)
            print(meta_info)
            
        if extra_info['subtask'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']:
            extra_info["molecule"] = meta_info['src']
        elif extra_info['subtask'] in ['add', 'delete', 'sub']:
            extra_info["molecule"] = meta_info['molecule']
            if "added_group" in meta_info.keys() and meta_info["added_group"] != None: extra_info["added_group"] = meta_info['added_group']
            if "removed_group" in meta_info.keys() and meta_info["removed_group"] != None: extra_info["removed_group"] = meta_info['removed_group']
        elif extra_info['subtask'] in ['ring_count']:
            extra_info["molecule"] = meta_info['smiles']
            extra_info["ring"] = meta_info['ring']
        elif extra_info['subtask'] in ['ring_system_scaffold']:
            extra_info["molecule"] = meta_info['smiles']; extra_info["ring_system_scaffold"] = meta_info['ring_system_scaffold']
        elif extra_info['subtask'] in ['equivalence']:
            extra_info["molecule"] = meta_info['smiles']
            if 'permutated' in meta_info.keys(): extra_info["molecule2"] = meta_info['permutated']
            elif 'mutated' in meta_info.keys(): extra_info["molecule2"] = meta_info['mutated']
            else: assert FileNotFoundError
        elif extra_info['subtask'] in ['fg_count']:
            extra_info["molecule"] = meta_info['smiles']; extra_info["functional_group"] = meta_info['fg_name']
        elif extra_info['subtask'] in ['Murcko_scaffold']:
            extra_info["molecule"] = meta_info['smiles']
        elif extra_info['subtask'] in ['fs_major_product', 'fs_by_product', 'mech_sel', 'nepp', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro']:
            pass
        else: FileExistsError
        
        if dtype == "tool":
            data = {
                'id': idx,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "answer_assistant_tool": answer_assistant,
                "data_source": data_source,
                "extra_info": extra_info,
            }
        elif dtype == "non-tool":
            data = {
                'id': idx,
                "question": user_prompt,
                "answer": answer_assistant,
                "data_source": data_source,
                "extra_info": extra_info,
            }
        return data

    train_dataset = train_dataset.map(
        function=process_fn, 
        with_indices=True,
        fn_kwargs={"dtype": dtype}  # 这里传入你在 generate 函数开头定义的 dtype
    ).shuffle(seed=42)
    test_dataset = test_dataset.map(
        function=process_fn, 
        with_indices=True,
        fn_kwargs={"dtype": dtype}  # 这里传入你在 generate 函数开头定义的 dtype
    ).shuffle(seed=42)

    train_dataset.to_parquet(os.path.join(tgt_dir, file_name.format("train")))
    test_dataset.to_parquet(os.path.join(tgt_dir, file_name.format("test")))
    return

def check_prompt_length():
    root_path = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/"
    parquet_path_list = [
        os.path.join(root_path, "datasets/chemcot/chemcot_json/chemcot-tool-coldstart/train_tool_full_rxn.parquet"),
        # os.path.join(root_path, "datasets/chemcot/chemcot_json/chemcot-tool-coldstart/test_tool_only_rxn.parquet"),
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/verl_checkpoints/chemcot-tool-coldstart-rxn-qwen-7B-sft/global_step_150"
    )
    
    for parquet_path in parquet_path_list:
        print(parquet_path)
        parquet_info = pd.read_parquet(parquet_path).to_dict('records')
        print(len(parquet_info), parquet_info[0].keys())
        
        result_dict = dict()
        for i in range(len(parquet_info)):
            task_name = parquet_info[i]['extra_info']['subtask']
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": parquet_info[i]['system_prompt']}, 
                    {"role": "user", "content": parquet_info[i]['user_prompt']}, 
                ], 
                tokenize=True, 
                add_generation_prompt=True
            )
            # tokens = tokenizer(prompt, add_special_tokens=True)['input_ids']
            prompt_length = len(prompt)
            
            if task_name in result_dict.keys():
                result_dict[task_name].append(prompt_length)
            else:
                result_dict[task_name] = [prompt_length]
        
        ## output:
        for task in result_dict.keys():
            temp_list = [itm for itm in result_dict[task] if itm > 2096]
            print(f"task= {task}, number= {len(result_dict[task])}, mean_length= {sum(result_dict[task])/len(result_dict[task])}, >4096= {len(temp_list)}")


if __name__ == "__main__":
    ## Generate Full-SFT dataset parquet form
    # generate_chemcotdataset_parquet(rootpath="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/")
    
    ## Generate Full-SFT-RXN dataset parquet form
    generate_chemcotdataset_rxn_parquet(rootpath="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/", dtype="tool")
    
    ## Generate ChemCoTBench parquet form
    # generate_chemcotbench_parquet(rootpath="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/")
    
    ## Check Generated Parquet files
    # check_parquet(parquet_path="./chemcot_json/chemcot-tool-coldstart/")
    # check_parquet_chemcotbench(parquet_path="./chemcot_json/chemcotbench/")

    # check_prompt_length()
    
    