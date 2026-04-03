import pandas as pd
import json

def read_from_json_lines(file_path):
    sample_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行尾换行符并解析
            if line.strip():  # 确保不是空行
                sample = json.loads(line)
                sample_list.append(sample)
    return sample_list

def read_parquet():
    file_path = '../search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/train_tool_full_rxn.parquet'
    df = pd.read_parquet(file_path)
    print(df.info())
    print(df.head())
    
    for i in range(200, 250):
        print(df["user_prompt"][i])
        print(df['answer'][i])
        print(df["extra_info"][i])
        print("*  * "*20)

def read_outputfiles():
    root_path = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/results/ablations/chemcot-fullsft-tool-rxn-qwen-7B/model_{}.json"
    
    for i in range(30, 90, 30):
        file_path = root_path.format(i)
        sample_list = read_from_json_lines(file_path)
        print(sample_list[0])
        print("*  * "*30)
    
    
read_outputfiles()
# read_parquet()