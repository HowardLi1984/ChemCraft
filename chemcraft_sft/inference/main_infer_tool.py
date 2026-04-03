## 
from tqdm import tqdm
import pandas as pd

import requests
import torch
import re

import transformers
from transformers import StoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from main_infer_moledit import update_json_file

def call_chem_api(function_name: str, query: str):
    # payload = ChemFunctionRequest(function_name=function_name, query=query)
    payload = {
        "function_name": function_name,
        "query": query
    }
    
    response = requests.post(
        "http://22.18.120.247:8000/chem_function", 
        json=payload
    )
    return response


def model_loading(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.eval() 
    return model, tokenizer


def dataset_loading(dataset_path):
    test_dataset = list()
    dataframe = pd.read_parquet(dataset_path)
    for i in range(len(dataframe)):
        test_dataset.append({
            "question": dataframe.iloc[i]['question'],
            "answer": dataframe.iloc[i]['answer'],
            "extra_info": dataframe.iloc[i]['extra_info'],
        })
    return test_dataset

curr_eos = [151645, 151643] # for Qwen2.5 series models

# Define the custom stopping criterion
class StopOnAnyToolEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 预编码常见的tool结束标记
        self.common_tool_ends = {
            "</GetRXN>": tokenizer.encode("</GetRXN>", add_special_tokens=False),
            "</MolSimilarity>": tokenizer.encode("</MolSimilarity>", add_special_tokens=False),
            "</CompareSMILES>": tokenizer.encode("</CompareSMILES>", add_special_tokens=False),
            "</SMILES2Weight>": tokenizer.encode("</SMILES2Weight>", add_special_tokens=False),
            "</CanonicalizeSMILES>": tokenizer.encode("</CanonicalizeSMILES>", add_special_tokens=False),
            "</CountMolAtoms>": tokenizer.encode("</CountMolAtoms>", add_special_tokens=False),
            "</AddFunctionalGroup>": tokenizer.encode("</AddFunctionalGroup>", add_special_tokens=False),
            "</RemoveFunctionalGroup>": tokenizer.encode("</RemoveFunctionalGroup>", add_special_tokens=False),
            "</ReplaceFunctionalGroup>": tokenizer.encode("</ReplaceFunctionalGroup>", add_special_tokens=False),
        }
        self.max_pattern_length = max(len(ids) for ids in self.common_tool_ends.values())

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < 2:
            return False
        
        # 检查预定义的tool结束标记
        for target_ids in self.common_tool_ends.values():
            target_length = len(target_ids)
            if input_ids.shape[1] >= target_length:
                if torch.equal(input_ids[0, -target_length:], 
                              torch.tensor(target_ids, device=input_ids.device)):
                    return True
        
        # 动态检测新的tool结束标记
        recent_tokens = input_ids[0, -self.max_pattern_length:]
        recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
        
        # 使用正则表达式匹配任何</tool-name>模式
        if re.search(r'</\w+>', recent_text):
            return True
            
        return False

def check_tool_call_at_end(response_text):
    """
    检查文本是否以</tool-name>结尾, 并提取tool-name和query
    如果不是以tool调用结尾, 返回None
    """
    if not response_text.strip(): return None
    
    # 从后往前查找最后一个</tool-name>模式
    # 使用正则表达式匹配以</word>结尾的情况
    end_pattern = r"</(\b(?!result\b|answer\b)\w+\b)>$"
    end_match = re.search(end_pattern, response_text)
    
    if not end_match:
        # 如果不是以</tool-name>结尾，返回None
        return None
    
    tool_name = end_match.group(1)
    
    start_pattern = rf"<{tool_name}>(.*?)</{tool_name}>"
    matches = list(re.finditer(start_pattern, response_text, re.DOTALL))
    
    if not matches: return None
    
    last_match = matches[-1]
    if last_match.end() != len(response_text): return None
    
    query = last_match.group(1).strip()
    return tool_name, query

def inference_with_tools(model, tokenizer, prompt: str):
    current_history = prompt
    
    ## init the stopping criteria from Search-R1
    stopping_criteria = transformers.StoppingCriteriaList([StopOnAnyToolEnd(tokenizer)])

    while True:
        model_input_text = tokenizer.apply_chat_template(
            current_history,
            add_generation_prompt=True, # 添加助手的起始提示
            tokenize=False,
        )
        input_ids = tokenizer(model_input_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=False,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, # 明确指定结束标记
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if tokenizer.eos_token_id ==  generated_tokens[-1]:
            ## Find [EOS] Token, finish reasoning
            if current_history[-1]['role'] == 'assistant': current_history[-1]['content'] += response_text
            break

        print("generated response: ", response_text)

        # check tool-calling tokens
        tool_call_match = check_tool_call_at_end(response_text)

        if tool_call_match: # 检测到非result, 非answer工具
            tool_name = tool_call_match.group(1)
            query = tool_call_match.group(2)
            api_result = call_chem_api(tool_name, query)
            api_text_result = api_result.json()['result']
            result_str = f"<result>{api_result}</result>."
            print(f"using: {tool_name}")
            if current_history[-1]['role'] == 'assistant': 
                current_history[-1]['content'] += f" {response_text} {result_str}"
        else:
            # 如果最后不是以</tool-name>结束, 同时最后一个token还不是终止符, 那说明response_text就是自然中止的
            if current_history[-1]['role'] == 'assistant': current_history[-1]['content'] += response_text
            else: assert "error in assistant"
    
    return current_history

def main():
    qwen_model, qwen_tokenizer = model_loading(model_path="/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-tool-coldstart-qwen-7B-dataset-full/global_step_900/")
    
    target_pattern = "<tool_call><"
    target_tokens = qwen_tokenizer.encode(target_pattern, add_special_tokens=False)
    prefix_tokens = target_tokens[:-1]  # 去掉最后一个token
    
    test_list = dataset_loading(dataset_path="/mnt/workspace/lh/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/test_tool_full.parquet")
    
    init_input = [
        {
            "role": "system",
            "content": "You are an expert chemist that can use Chemical-Tools to address the complex chemical problem. Generate the `<tool_name>query</tool_name>` to call a special tool and the result-information is in <result>result-information</result>. Here is the description of Tool Sets that you can use:\n            \n            [Tool_Name] MolSimilarity [Description] Input two molecule SMILES (separated by ';'), returns Tanimoto similarity. [Input,Output] input: SMILES1;SMILES2, output: two SMILES are identical / not similar. [Example] <MolSimilarity>CCO;CCN</MolSimilarity> <result>The Tanimoto similarity between CCO and CCN is 0.3333, indicating that the two molecules are not similar.<result>\n[Tool_Name] FunctionalGroups [Description] Get the functional groups in a molecule. Input SMILES, return list of functional groups in the molecule. [Input,Output] input: SMILES, output: functional groups in this SMILES. [Example] <FunctionalGroups>CCO</FunctionalGroups> <result>This molecule contains alcohol groups, and side-chain hydroxyls.<result>\n[Tool_Name] CompareSMILES [Description] Input two molecule SMILES (separated by ';'), returns if they are identical. To judge if two molecules are identical, you should always use this tool, instead of directly comparing the SMILES strings. [Input,Output] input: SMILES1;SMILES2, output: identical / different. [Example] <CompareSMILES>CCO;CCN</CompareSMILES> <result>different<result>\n[Tool_Name] SMILES2Weight [Description] Calculate molecular weight. Input SMILES, returns molecular weight. [Input,Output] input: SMILES, output: SMILES weight. [Example] <SMILES2Weight>CCO</SMILES2Weight> <result>46.041864812<result>\n[Tool_Name] CanonicalizeSMILES [Description] Canonicalize SMILES representation. Input SMILES, returns canonicalized SMILES. You should use this tool when asked for canonicalized SMILES. [Input,Output] input: SMILES, output: the canonicalize SMILES. [Example] <CanonicalizeSMILES>OCC</CanonicalizeSMILES> <result>CCO<result>\n[Tool_Name] CountMolAtoms [Description] Count the number of atoms in a molecule. Input SMILES, returns the types of atoms and their numbers. [Input,Output] input: SMILES, output: the atom number and atom type in this SMILES. [Example] <CountMolAtoms>CCO</CountMolAtoms> <result>There are altogether 9 atoms. The types and corresponding numbers are: {\"C\": 2, \"O\": 1, \"H\": 6}<result>\n[Tool_Name] AddFunctionalGroup [Description] Add a functional group to a molecule. [Input,Output] input: SMILES;add_group_name, output: modified SMILES [Example] <AddFunctionalGroup>CCC;carboxyl</AddFunctionalGroup> <result>C(C(=O)O)CC<result>\n[Tool_Name] RemoveFunctionalGroup [Description] Remove a specific functional group from a molecule. [Input,Output] input: SMILES;removed_group_name, output: modified SMILES [Example] <RemoveFunctionalGroup>CCO;hydroxyl</RemoveFunctionalGroup> <result>CC<result>\n[Tool_Name] ReplaceFunctionalGroup [Description] Replace a functional group in a molecule with another functional group using functional group names from constants.py. [Input,Output] input: SMILES;old_group_name;new_group_name, output: modified SMILES [Example] <ReplaceFunctionalGroup>CCO;hydroxyl;primary_amine</ReplaceFunctionalGroup> <result>CCN<result>\n[Tool_Name] QEDPropertyPred, DRD2PropertyPred, JNK3PropertyPred, LogPPropertyPred, GSKPropertyPred, SolubilityPropertyPred [Description] each tool provides a molecule property predictor for QEDS, LogP, Solubility, DRD2, JNK3, GSK. [Example] <QEDPropertyPred>O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1</QEDPropertyPred> <result>the QED of this molecule is 5.27.</result> \n[Tool_Name] GetRXN [Description] Input the reactants SMILES, search the rection database and return similar reactions. [Input/Output] input: Reactant SMILES, output: related reactions. [Example] <GetRXN>CCO.O=S1(=O)C=Cc2ccccc21.[Pd]<GetRXN> <result>reactant: CCO.O=S1(=O)C=Cc2ccccc21.[Pd], product: O=S1(=O)CCc2ccccc21, reagent: palladium on activated charcoal|ethanol, solvent: empty<result> \n\n            \n            Here is the Chemical Task and Chemical Question that you need to solve, your final answer MUST BE in <answer> SMILES/YES/NO/count-number </answer>"
        },
        {
            "role": "user",
            "content": "Please add a benzene ring to the molecule CC1CCCC1CNS(=O)(=O)c1ccnc(Cl)c1."
        },
    ]
    current_history = inference_with_tools(model=qwen_model, tokenizer=qwen_tokenizer, prompt=init_input)
    
    # for i in tqdm(range(len(test_list)), desc="SFT-Test"):
    #     test_sample = test_list[i]
    #     init_input = [
    #         {'role': 'system', 'content': test_sample['question']},
    #         {'role': 'user', 'content': test_sample['extra_info']['Instruction']},
    #         {'role': 'assistant', 'content': ''}, # 占位
    #     ]
    #     current_history = inference_with_tools(model=qwen_model, tokenizer=qwen_tokenizer, prompt=init_input)
        
    #     update_json_file(
    #         current_history, 
    #         file_name='/mnt/workspace/lh/ChemSearch/search_saves/results/tool_sft_results/chemcot-tool-coldstart-qwen-7B-dataset-full/900.json'
    #     )

def test():
    response_text = "I am a lier, I am afriad, <tool>kk</tool>"
    response_text = "I am a lier, I am afriad, <tool>kk</tool>, I am in great pain, please kill me"
    response_text = "I am a lier, I am afriad, <tool>kk</tool>, I am in great pain, please kill me. <result>kk</result>"
    print(check_tool_call_at_end(response_text=response_text))
    
    qwen_model, qwen_tokenizer = model_loading(model_path="/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-tool-coldstart-qwen-7B-dataset-full/global_step_900/")
    
    target_pattern = "<tool_call><"
    target_tokens = qwen_tokenizer.encode(target_pattern, add_special_tokens=False)
    prefix_tokens = target_tokens[:-1]  # 去掉最后一个token
    
    print(f"完整标记: {target_pattern}")
    print(f"Token IDs: {target_tokens}")
    print(f"前缀Token IDs: {prefix_tokens}")
    print(f"前缀文本: {qwen_tokenizer.decode(prefix_tokens)}")
        
    exit()

if __name__ == "__main__":
    main()
    # test()