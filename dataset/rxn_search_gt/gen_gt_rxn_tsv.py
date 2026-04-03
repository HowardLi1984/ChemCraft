## 把ChemCotBench 以及 ChemCoTDataset的所有的RXN数据提取，并整理成csv

import os
import ast
import json
import pandas as pd

def read_demo_tsv(tgt_file_path="../rxn_search_v3/choriso_reorgnize_search.tsv"):
    df = pd.read_csv(tgt_file_path, sep='\t')
    # 查看数据
    print("数据形状:", df.shape)
    print("\n前5行数据:")
    print(df.head())

    print("\n列名:")
    print(df.columns.tolist())

    print("\n数据类型:")
    print(df.dtypes)

def remove_dict_duplicates(data_list):
    """去重dict列表，保持顺序"""
    seen = []
    result = []
    
    for d in data_list:
        # 将dict转换为可哈希的元组
        # 需要先排序键确保一致性
        d_tuple = tuple(sorted(d.items()))
        
        if d_tuple not in seen:
            seen.append(d_tuple)
            result.append(d)
    
    return result

def get_chemcotbench_rxn():
    dataset_file_path = "../chemcot/chemcot_json/chemcotbench/reaction/"
    task_list = ['fs_by_product', 'fs_major_product', 'nepp', 'rcr', 'retro']
    
    results = list()
    for taskname in task_list:
        raw_info = json.load(open(os.path.join(dataset_file_path, f"{taskname}.json"), "r"))
        if taskname in ['fs_by_product', 'fs_major_product']:
            for raw_sample in raw_info:
                meta_info = ast.literal_eval(raw_sample['meta'])
                raw_rxn_list = meta_info['rxn_smiles'].split(">")
                byproduct = "empty"
                if taskname == 'fs_by_product':
                    byproduct = raw_sample['gt']
                results.append({
                    "reactant": raw_rxn_list[0],
                    "product": raw_rxn_list[2],
                    "byproduct": byproduct,
                    "reagent": raw_rxn_list[1],
                    "solvent": "empty",
                    "catalyst": "empty",
                    "yield": 0.0,
                })
        
        if taskname in ['nepp']:
            for raw_sample in raw_info:
                meta_info = ast.literal_eval(raw_sample['meta'])
                results.append({
                    "reactant": meta_info['starting_reactants'],
                    "product": raw_sample['gt'],
                    "byproduct": "empty",
                    "reagent": meta_info['reagents'],
                    "solvent": "empty",
                    "catalyst": "empty",
                    "yield": 0.0,
                })
        
        if taskname in ['rcr']:
            for raw_sample in raw_info:
                meta_info = ast.literal_eval(raw_sample['meta'])
                rxn = raw_sample['query'].split('\n')[1]
                rxn = rxn.replace(',', '')
                temp_sample = {
                    "reactant": rxn.split('>>')[0],
                    "product": rxn.split('>>')[1],
                    "byproduct": "empty",
                    "reagent": "empty",
                    "solvent": "empty",
                    "catalyst": "empty",
                    "yield": 0.0,
                }
                temp_sample[meta_info['query_target']] = raw_sample['gt']
                results.append(temp_sample)
        
        if taskname in ['retro']:
            for raw_sample in raw_info:
                meta_info = ast.literal_eval(raw_sample['meta'])
                results.append({
                    "reactant": raw_sample['gt'],
                    "product": meta_info['products'][0],
                    "byproduct": "empty",
                    "reagent": ", ".join(meta_info['reagents']),
                    "solvent": "empty",
                    "catalyst": "empty",
                    "yield": 0.0,
                })
    
    unique_results = remove_dict_duplicates(results)
    print(f"chemcotbench, original: {len(results)}, after_duplicate: {len(unique_results)}")
    return unique_results

def get_chemcotdataset_rxn():
    ## 在chemcotdataset里面, nepp存在问题，没把rxn显式展示出来, 所以不考虑扫rxn_nepp这个部分了
    dataset_file_path = "../chemcot/chemcot_json/chemcotdataset/"
    task_list = ['rxn_fs_by_product', 'rxn_fs_major_product', 'rxn_rcr_catalyst', 'rxn_rcr_reagent', 'rxn_rcr_solvent', 'rxn_retro']
    
    results = list()
    for taskname in task_list:
        raw_info = json.load(open(os.path.join(dataset_file_path, f"{taskname}.json"), "r"))
        
        for raw_sample in raw_info:
            temp_sample = {
                "reactant": raw_sample['extra_info']['reactants'],
                "product": raw_sample['extra_info']['products'],
                "byproduct": "empty",
                "reagent": "empty",
                "solvent": "empty",
                "catalyst": "empty",
                "yield": 0.0,
            }
            for key in ['catalyst', 'reagent', 'solvent']:
                if taskname == f"rxn_rcr_{key}":
                    temp_sample[key] = raw_sample['extra_info']["gt"]
            if taskname == "rxn_fs_by_product":
                temp_sample['byproduct'] = raw_sample['extra_info']["gt"]

            results.append(temp_sample)
                
    unique_results = remove_dict_duplicates(results)
    print(f"chemcotdataset, original: {len(results)}, after_duplicate: {len(unique_results)}")
    return unique_results
    
def save_chemcot_rxn_to_tsv(savepath):
    bench_results = get_chemcotbench_rxn()
    dataset_results = get_chemcotdataset_rxn()
    
    total_results = bench_results + dataset_results
    final_results = remove_dict_duplicates(total_results)
    print(f"chemcotdataset, original: {len(total_results)}, after_duplicate: {len(final_results)}")
    
    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(final_results)
    
    # 保存为TSV
    df.to_csv(savepath, sep='\t', index=True, encoding='utf-8')


if __name__ == "__main__":
    read_demo_tsv(
        # tgt_file_path="../rxn_search_v3/choriso_reorgnize_search.tsv",
        tgt_file_path="gt_rxn_search.tsv",
    )

    # save_chemcot_rxn_to_tsv(
    #     savepath="gt_rxn_search.tsv"
    # )

    # {'reactant': 'COC(=O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1', 'product': 'C[O-].CC1(CNC(=O)c2cc(-c3ccccc3C(=O)O)ccc2Cl)CCCCCC1', 'reagent': 'CO.C1CCOC1.O', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CCOC(=O)c1nn(Cc2ccccc2)c(C)c1-c1ccc(N2CCc3cccc(C(=O)Nc4nc5ccccc5s4)c3C2)nc1C(=O)O', 'product': 'CC[O-].Cc1c(-c2ccc(N3CCc4cccc(C(=O)Nc5nc6ccccc6s5)c4C3)nc2C(=O)O)c(C(=O)O)nn1Cc1ccccc1', 'reagent': 'CO.C1COCCO1.O', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CC(C)(C)OC(=O)CCc1ccc(OC(C)(C)C)cc1', 'product': 'CC(C)(C)[O-].CC(C)(C)Oc1ccc(CCC(=O)O)cc1', 'reagent': 'CO.O', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CC(C)(O)[C@H](N)c1ccccc1.O=C(Cl)C(Cl)(Cl)Cl', 'product': 'CC(C)(O)[C@H](NC(=O)C(Cl)(Cl)Cl)c1ccccc1.c1cc[nH+]cc1', 'reagent': 'c1ccncc1', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}