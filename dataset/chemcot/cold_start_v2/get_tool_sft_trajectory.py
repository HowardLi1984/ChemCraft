
import re
import os
import json
import ast
import requests
from tqdm import tqdm
from openai import OpenAI

root_path = "/cto_labs/lihao/chem_reason/"

import sys
sys.path.append(os.path.join(root_path, "ChemSearch/search_saves/datasets/chemcot/cold_start_v2"))
sys.path.append(os.path.join(root_path, "ChemSearch/search_r1/search_r1/search"))
sys.path.append(os.path.join(root_path, "ChemSearch/search_r1/search_r1/llm_agent"))
from chem_agents import MolSimilarity, FuncGroups, CompareSMILES, SMILES2Weight, CanonicalizeSMILES, CountMolAtoms
from chem_agents import  AddFunctionalGroup, RemoveFunctionalGroup, ReplaceFunctionalGroup
# from chem_prop_agents import MolPropertyPred, QEDPropertyPred, DRD2PropertyPred, JNK3PropertyPred, LogPPropertyPred, GSKPropertyPred, SolubilityPropertyPred
# from rxn_index_builder import FaissRXNSearcher

from get_task_info import _get_task_info
from utils import data_clean, transform_stage3_tool_answers, reshape_question

def update_json_file(info_dict, file_name='data.json'):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或文件为空，则创建一个新的空列表
        data = []

    # 将新的info_dict添加到数据列表中
    data.append(info_dict)

    # 将更新后的数据写回文件
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def call_chem_api(function_name: str, query: str):
    # payload = ChemFunctionRequest(function_name=function_name, query=query)
    payload = {
        "function_name": function_name,
        "query": query
    }
    
    response = requests.post(
        "http://0.0.0.0:8080/chem_function", 
        json=payload
    )
    return response

# 核心: 一个trajectory_builder包括三个STAGE的采样过程
class sft_trajectory_builder:
    def __init__(self, llm_info=None) -> None:
        self.llm_info = llm_info
        self.llm_client = OpenAI(
            api_key="sk-4tYqo3Q6bY6mzdjlAUJQJ9doUL9t40AfhZFs5EdAczsSGwJl", 
            base_url="https://api.bltcy.ai/v1"
        )
        self.tool_descriptions = self._get_tool_descriptions()
        self.task_info = _get_task_info()
    
    # def _get_tool_descriptions(self) -> str:
    #     chem_funuction_list = [MolSimilarity(), FuncGroups(), CompareSMILES(), SMILES2Weight(), CanonicalizeSMILES(), CountMolAtoms(), AddFunctionalGroup(), RemoveFunctionalGroup(), ReplaceFunctionalGroup()]
    #     chem_function = ""
        
    #     for chem_func in chem_funuction_list:
    #         case_example = f"<{chem_func.name}>"+chem_func.examples[0]['input']+f"</{chem_func.name}> "+"<result>"+chem_func.examples[0]['output']+"<result>"
    #         chem_function += f"[Tool_Name] {chem_func.name} [Description] {chem_func.description} [Input,Output] {chem_func.input_output_description} [Example] {case_example}\n"
        
    #     prop_predicter = "[Tool_Name] QEDPropertyPred, DRD2PropertyPred, JNK3PropertyPred, LogPPropertyPred, GSKPropertyPred, SolubilityPropertyPred [Description] each tool provides a molecule property predictor for QEDS, LogP, Solubility, DRD2, JNK3, GSK. [Example] <QEDPropertyPred>O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1</QEDPropertyPred> <result>the QED of this molecule is 5.27.</result> \n"
        
    #     rxn_searcher = "[Tool_Name] GetRXN [Description] Input the reactants SMILES, search the rection database and return similar reactions. [Input/Output] input: Reactant SMILES, output: related reactions. [Example] <GetRXN>CCO.O=S1(=O)C=Cc2ccccc21.[Pd]<GetRXN> <result>reactant: CCO.O=S1(=O)C=Cc2ccccc21.[Pd], product: O=S1(=O)CCc2ccccc21, reagent: palladium on activated charcoal|ethanol, solvent: empty<result> \n"
        
    #     return chem_function+prop_predicter+rxn_searcher

    def _get_tool_descriptions(self) -> str:
        tools_list = []
        chem_function_list = [MolSimilarity(), FuncGroups(), SMILES2Weight(), CanonicalizeSMILES(), CountMolAtoms(), CompareSMILES(), AddFunctionalGroup(), RemoveFunctionalGroup(), ReplaceFunctionalGroup()]

        for chem_func in chem_function_list:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": chem_func.name,
                    "description": chem_func.description,
                    "arguments": chem_func.input_argument,
                    "example": chem_func.examples[0],
                }
            }
            tools_list.append(tool_dict)
            
        prop_tools = {
            "QEDPropertyPred": "A molecule property predictor for QED.",
            "DRD2PropertyPred": "A molecule property predictor for DRD2.",
            "JNK3PropertyPred": "A molecule property predictor for JNK3.",
            "LogPPropertyPred": "A molecule property predictor for LogP.",
            "GSKPropertyPred": "A molecule property predictor for GSK.",
            "SolubilityPropertyPred": "A molecule property predictor for Solubility."
        }

        PropPredict_description = {
            "type": "function",
            "function": {
                "name": "QEDPropertyPred / DRD2PropertyPred / JNK3PropertyPred / LogPPropertyPred / GSKPropertyPred / SolubilityPropertyPred",
                "description": "Input SMILES, each tool provides a molecule property predictor for QEDS, LogP, Solubility, DRD2, JNK3, GSK.",
                "arguments": {
                    "SMILES": "your_smiles"
                },
                "example": r"<tool_call>\n{'name': 'QEDPropertyPred', 'arguments': {'SMILES': 'O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1'}}\n</tool_call> <result> the QED of this molecule is 5.27. </result>"
            }
        }
        
        GetRXN_description = {
            "type": "function",
            "function": {
                "name": "GetRXN",
                "description": "Input the SMILES as reactants or products, search the reaction database and return similar reactions.",
                "arguments": {
                    "SMILES": "Your SMILES",
                    "Type": "reactants / products"
                },
                "example": r"<tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'CCO.O=S1(=O)C=Cc2ccccc21.[Pd]', 'Type': 'reactants'}}\n</tool_call> <result> product: O=S1(=O)CCc2ccccc21, reagent: palladium on activated charcoal|ethanol, solvent: empty. </result>"
            }
        }
        
        GetRXNTemplate_description = {
            "type": "function",
            "function": {
                "name": "GetRXNTemplate",
                "description": "Input the SMILES as reactants or products, search the reaction-template database and return similar reaction templates.",
                "arguments": {
                    "SMILES": "Your SMILES",
                    "Type": "reactants / products"
                },
                "example": r"<tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'CCO', 'Type': 'reactants'}}\n</tool_call> <result> the similar reaction template is [O:1][C:2]>>[O:1]=[C:2] </result>"
            }
        }
        tools_list.extend([PropPredict_description, GetRXN_description, GetRXNTemplate_description])
        
        individual_json_strings = [json.dumps(tool, indent=2, ensure_ascii=False) for tool in tools_list]
        final_content = "\n".join(individual_json_strings)
        return f"<tools>\n{final_content}\n</tools>"
        
    def _get_stage1_system_prompt(self, taskname, question) -> str:
        # Stage-1: 添加工具调用生成的<tool-name>query</tool-name>
        molsimilarity_example = "<tool_call>\n{'name': 'MolSimilarity', 'arguments': {'SMILES1':'CCO', 'SMILES2':'CCN'}\n</tool_call>"
        getrxn_example = "<tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES':'CCO.O=S1(=O)C=Cc2ccccc21.[Pd]', 'Type':'reactant'}\n</tool_call>"
        return f"""
        You are an expert chemist. Your sole task is to analyze the user-provided chemical reasoning trajectory and insert a `<tool_name>query</tool_name>` call wherever you identify a logical leap, an unverified claim, or a calculation that should be validated by a tool.

        **Rules:**
        1.  **Insert Only:** Only add the `<tool_call>\n{{'name':'ToolName', 'arguments':dict()}}\n</tool_call>`. Do not add results or any other text.
        2.  **Be Precise:** The tool name and query must be perfectly valid. e.g. {molsimilarity_example}, {getrxn_example}, etc.
        3.  **Do Not Change:** Do not alter the original text in any other way.

        Here is the description of Tool Sets:
        {self.tool_descriptions}
        
        Task Name: {taskname}, Question: {question}.
        
        Tool recommendation for this {taskname} task: {self.task_info[taskname][1]}
        
        **Example:**
        User Trajectory: the reasoning trajectory of this chemical question.
        
        Your Output: only the reasoning trajectory with <tool>query</tool> in it. 
        """
    
    def _get_stage3_system_prompt(self, question) -> str:
        return f"""
        You are a meticulous and experienced chemical expert. Your task is to perform a deep **rethink and reshape** of a given "Original-Trace." This original trace contains your preliminary thought process, including tool calls (<tool_call>...</tool_call>) and the results they returned (<result>...</result>).

        Your goal is to produce a final, coherent natural language thought trajectory that **actively integrates** and **responds** to the tool call results.

        You must adhere strictly to these core principles:
        1. **Preserve All Tool Information**: You must retain all `<tool_call>{{'name': 'ToolName', 'arguments': {{your arguments}}}}</tool_call>` and `<result>...</result>` tags and their content EXACTLY SAME as they appear in the original trace. Also, always TRUST the tool results (maybe some small mistakes).
        
        2. **Preserve Reasoning Templates**: You must retain all `Step-1, XXX:', `Step-2, YYY:' .. reasoning structures.
        
        3. **Preserve final answer**: You must keep the Output: <answer>  </answer> EXACTLY unchanged.
        
        4.  **Let the Results Guide You**: Treat the content of each `<result>` tag as feedback from a reliable source. If a result contradicts your previous assumption or reasoning, be **bold and revise your line of thought**. Your subsequent reasoning steps should be adjusted and optimized based on this new information.
        
        5.  **Actively Modify and Supplement**: Do not simply append the result. Instead, **rewrite** your reasoning around the tool's output. Explain in natural language how the feedback confirmed, corrected, or deepened your understanding. If a result provides new information, integrate it seamlessly into your narrative to make the overall trajectory more accurate and complete.
        

        The current task question is: {question}
        
        Your Output: ONLY the refined reasoning trajectory with <tool_call>\n{{'name': 'ToolName', 'arguments': {{your arguments}}}}\n</tool_call> <result></result> in it. 
        """
    
    def _get_stage4_system_prompt(self) -> str:
        ## adding system-prompt for tool-reason trajectory, for Cold-Start
        tool_call_format = "<tool_call>\n{'name': 'function-name', 'arguments': {'query1':'', 'query2':''}}\n</tool_call>"
        
        return f"""You are an expert chemist that can use Chemical-Tools to address the complex chemical problem. Generate the `{tool_call_format}` to call a special tool and the result-information is in <result> result-information </result>. Here is the description of Tool Sets that you can use:
            
            {self.tool_descriptions}
            
            Here is the Chemical Task and Chemical Question that you need to solve, your final answer MUST BE in <answer> SMILES/YES/NO/count-number </answer>
        """

    def stage1_add_tool_calling(self, task_list, save_path):
        print("\n--- Running Stage 1: Adding Tool Calls ---")
        
        for task in self.task_info.keys():
            if task not in task_list: continue
            task_samples = json.load(open(self.task_info[task][0], "r"))
            # task_samples = task_samples[:10]
            
            if os.path.exists(os.path.join(save_path, f"{task}.json")):
                done_file_info = json.load(open(os.path.join(save_path, f"{task}.json"), "r"))
                task_samples = task_samples[len(done_file_info):]
                print(f"continue generating from {len(done_file_info)}")
            
            result_list = []
            for sample in tqdm(task_samples, desc=task):
                system_prompt = self._get_stage1_system_prompt(taskname=task, question=sample['question'])
                user_prompt = f'''User Trajectory: {sample['answer']}.'''
                
                response = self.llm_client.chat.completions.create(
                    model=self.llm_info['llm'], 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    temperature=0.1 # 低温以确保遵循指令
                )
                content = response.choices[0].message.content
                
                sample['answer_with_tool'] = content
                sample['system_promt'] = system_prompt
                sample['user_prompt'] = user_prompt
                result_list.append(sample)
                update_json_file(info_dict=sample, file_name=os.path.join(save_path, f"{task}.json"))
            
            # json.dump(result_list, open(os.path.join(save_path, f"{task}.json"), "w"), indent=4)
        
        return None

    def stage2_add_tool_results(self, stage1_path, stage2_path, task_list):
        print("\n--- Running Stage 2: Calling APIs and Adding Tool Results ---")
        tool_analysis = dict() # 记录tool的错误数量
        error_response_list = [
            "Error calculating", 
            "Invalid SMILES string",
            "Invalid function name",
            "Query format error",
        ]

        # def process_and_call(match: re.Match) -> str: # 定位<tool>, 生成results
        #     nonlocal tool_analysis
        #     original_tag = match.group(0) # group(0) 是整个匹配的字符串, e.g., "<FuncGroups>CCO</FuncGroups>"
        #     tool_name = match.group(1) # group(1) 是第一个捕获组 (\w+), 即 tool_name, e.g., "FuncGroups"
        #     query = match.group(2) # group(2) 是第二个捕获组 (.*?), 即 query, e.g., "CCO"
            
        #     if tool_name == "answer":
        #         result_tag = ""
        #     else:
        #         result = call_chem_api(tool_name, query)
        #         result = result.json()
        #         result_sentence = result['result']
                
        #         # 调用了tool, 记录一个call-number
        #         tool_analysis[tool_name] = tool_analysis.get(tool_name, {})
        #         tool_analysis[tool_name]["total-call"] = tool_analysis[tool_name].get("total-call", 0) + 1
                
        #         is_error = False; my_error_template = None
        #         for error_template in error_response_list: 
        #             if error_template in result_sentence: 
        #                 is_error = True; my_error_template = error_template
        #         if is_error == True:
        #             tool_analysis[tool_name][my_error_template] = tool_analysis[tool_name].get(my_error_template, 0) + 1
                
        #         result_tag = f"<result>{result_sentence}</result>"
            
        #     return original_tag + result_tag
        
        def process_and_call(match: re.Match) -> str:
            nonlocal tool_analysis
            original_tag = match.group(0)  # 整个 <tool_call>...</tool_call>
            content_str = match.group(1)   # 标签中间的内容
            try:
                tool_data = ast.literal_eval(content_str.strip())
                tool_name = tool_data.get('name')
                arguments = tool_data.get('arguments') # 这是一个 dict
                if arguments == None:
                    raise ValueError
            except (ValueError, SyntaxError) as e:
                print(f"Parse Error in tool_call: {e}")
                return original_tag + f"<result>Format Error: Unable to parse tool arguments.</result>"

            if tool_name == "answer":
                result_tag = ""
            else:
                argument_list = [value for key, value in arguments.items()]
                argument_str = ";".join(argument_list)
                result = call_chem_api(tool_name, argument_str)
                if hasattr(result, 'json'):
                    result = result.json()
                result_sentence = result['result']

                # 4. 统计分析 (保持原有逻辑)
                tool_analysis[tool_name] = tool_analysis.get(tool_name, {})
                tool_analysis[tool_name]["total-call"] = tool_analysis[tool_name].get("total-call", 0) + 1
                
                is_error = False; my_error_template = None
                for error_template in error_response_list: 
                    if error_template in str(result_sentence): 
                        is_error = True; my_error_template = error_template
                if is_error:
                    tool_analysis[tool_name][my_error_template] = tool_analysis[tool_name].get(my_error_template, 0) + 1
                
                result_tag = f"<result>{result_sentence}</result>"
            
            return original_tag + result_tag
        
        for task in task_list:
            task_info = json.load(open(os.path.join(stage1_path, f"{task}.json"), "r"))
            # task_info = task_info[:5]
            
            print(f"**** Task = {task} ****\n"); result_list = list()
            for task_sample in tqdm(task_info, desc=f"{task}"):
                trajectory_with_tool = task_sample['answer_with_tool']
                trajectory_with_result = re.sub(r"<tool_call>(.*?)</tool_call>", process_and_call, trajectory_with_tool, flags=re.DOTALL)
                
                result_list.append(
                    dict(
                        question=task_sample['question'],
                        answer=task_sample['answer'], 
                        extra_info=task_sample['extra_info'],
                        task=task_sample['task'],
                        answer_with_tool_stage1=trajectory_with_tool,
                        answer_with_tool_stage2=trajectory_with_result,
                    )
                )
            
            for tool_name, info in tool_analysis.items(): print(tool_name, info)
            json.dump(tool_analysis, open(os.path.join(stage2_path, f"{task}_tool_analysis.json"), "w"), indent=4)
            json.dump(result_list, open(os.path.join(stage2_path, f"{task}.json"), "w"), indent=4)
            
            tool_analysis = dict()
                
        return None

    def stage3_rethink_trajectory(self, stage2_path, stage3_path, task_list):
        ## 将带有工具调用和结果的轨迹, 输入给LLM, 生成最终的自然语言思考轨迹。
        print("\n--- Running Stage 3: Reshaping Trajectory ---")
        
        for task in task_list:
            task_info = json.load(open(os.path.join(stage2_path, f"{task}.json"), "r"))
            # task_info = task_info[:5]
            
            if os.path.exists(os.path.join(stage3_path, f"{task}.json")):
                done_file_info = json.load(open(os.path.join(stage3_path, f"{task}.json"), "r"))
                task_info = task_info[len(done_file_info):]
                print(f"continue generating from {len(done_file_info)}")

            result_list = []
            print(f"**** Task = {task} ****\n")
            
            for task_sample in tqdm(task_info, desc=f"{task}"):
                question = task_sample['question']
                system_prompt = self._get_stage3_system_prompt(question)
                stage2_trajectory = task_sample["answer_with_tool_stage2"]
                user_content = f"Original-Trace: {stage2_trajectory}"
                
                response = self.llm_client.chat.completions.create(
                    model=self.llm_info['llm'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    stream=False,
                    temperature=0.5 # A slightly higher temperature for more natural language
                )
                content = response.choices[0].message.content
                
                # Append the reshaped trajectory to the result list
                task_sample['answer_with_tool_stage3'] = content
                result_list.append(task_sample)
                update_json_file(info_dict=task_sample, file_name=os.path.join(stage3_path, f"{task}.json"))
        
    def stage4_dataclean(self, stage3_path, stage4_path, task_list):
        ## Data-Cleaning: 移除<tool></tool>, <result></result>, <answer></answer>格式不闭合的case, 移除<answer>内部对不上的情况
        ## (1) 把system_prompt加在 task_sample里面, 同时system_prompt的格式改成 <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>
        ## (2) 使用<tool_call> </tool_call>，因为tool-call是special-token
        ## (3) 格式改成<tool_call>\n{"name": <function-name>, "arguments": {“query1”:"", "query2":""}}\n</tool_call>
        ## (4) 把answer改成json格式, 包括{'role': 'system'; 'role': 'user'; 'role': 'assistant'; 'role': 'tool'}
        ## (5) 把Step-1, Step-2 之类的给去掉
        ## (6) 检查extra_info里面的task是否正确
        print("\n--- Running Stage 4: Data Cleaning, Add System Prompt ---")
        for task in task_list:
            task_info = json.load(open(os.path.join(stage3_path, f"{task}.json"), "r"))
            result_list = []
            format_error_dict = dict(); format_error_dict['total'] = len(task_info)
            
            for task_sample in tqdm(task_info, desc=f"{task}"):
                ## data clean, filter the cases with error <tool_call> format & name
                clean_result = data_clean(
                    text=task_sample['answer_with_tool_stage3'], 
                    tool_valid_names=['MolSimilarity', 'SMILES2Weight', 'FunctionalGroups', 'CompareSMILES', 'CanonicalizeSMILES', 'CountMolAtoms', 'ReplaceFunctionalGroup', 'RemoveFunctionalGroup', 'AddFunctionalGroup', 'QEDPropertyPred', 'DRD2PropertyPred', 'JNK3PropertyPred', 'LogPPropertyPred', 'GSKPropertyPred', 'SolubilityPropertyPred', 'GetRXN', 'GetRXNTemplate']
                )
                if clean_result[0] == False:
                    format_error_dict[clean_result[1]] = format_error_dict.get(clean_result[1], 0) + 1
                    continue
                
                ## if the task is reaction related, reshape the question, remove the "json format" and others
                question_reshaped = reshape_question(task=task, text=task_sample['question'])
                answer_with_tool_stage4 = transform_stage3_tool_answers(raw_input_trajectory=task_sample['answer_with_tool_stage3'])
                if 'task' not in task_sample['extra_info'].keys():
                    task_sample['extra_info']['task'] = task
                
                system_prompt = self._get_stage4_system_prompt()
                tmp_sample = dict(
                    question=question_reshaped,
                    prompt=system_prompt,
                    answer=task_sample['answer'],
                    extra_info=task_sample['extra_info'],
                    # answer_with_tool_stage1=task_sample['answer_with_tool_stage1'],
                    # answer_with_tool_stage2=task_sample['answer_with_tool_stage2'],
                    # answer_with_tool_stage3=task_sample['answer_with_tool_stage3'],
                    answer_with_tool_stage4 = answer_with_tool_stage4,
                )
                result_list.append(tmp_sample)
            
            for error_name, info in format_error_dict.items(): print(error_name, info)
            print(f"Task: {task}, original data-num: {len(task_info)}, cleaned data-num: {len(result_list)}")
            json.dump(format_error_dict, open(os.path.join(stage4_path, f"{task}_format_analysis.json"), "w"), indent=4)
            json.dump(result_list, open(os.path.join(stage4_path, f"{task}.json"), "w"), indent=4)
            
        return None


if __name__ == "__main__":
    # model_list = ["qwen3_235b_a22b_instruct"]
    # llm_infos = {
    #     "qwen3_30b_a3b_instruct": dict(llm="Qwen3-30B-A3B-Instruct-2507", node="81.85", port="7042"),
    #     "qwen3_235b_a22b_instruct": dict(llm="Qwen3-235B-A22B-Instruct-2507", node="80.36", port="7042"),
    # }
    
    my_data_builder = sft_trajectory_builder(llm_info={'llm':'gemini-2.5-pro-preview-06-05'})
    
    ## generate stage-1 data
    # my_data_builder.stage1_add_tool_calling(
    #     # task_list=['add', 'delete', 'sub', 'ring_count', 'ring_system_scaffold', 'mutated', 'permutated', 'functiongroup_detect', 'murcko_scaffold', 'gsk', 'jnk', 'drd', 'logp', 'qed', 'solubility'],
    #     task_list=['fs_major_product','fs_by_product','retro'],
    #     # task_list=['rcr_catalyst','rcr_reagent','rcr_solvent','mech_sel','nepp'],
    #     save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage1/",
    # )
    
    ## generate stage-2 data
    # my_data_builder.stage2_add_tool_results(
    #     stage1_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage1/",
    #     stage2_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage2/",
    #     # task_list = ['add', 'delete', 'sub', 'ring_count', 'ring_system_scaffold', 'mutated', 'permutated', 'functiongroup_detect', 'murcko_scaffold'],
    #     # task_list = ['nepp','fs_major_product','fs_by_product','retro','rcr_catalyst','rcr_reagent','rcr_solvent','mech_sel'],
    # )
    
    ## generate stage-3 data
    # my_data_builder.stage3_rethink_trajectory(
    #     stage2_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage2/",
    #     stage3_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage3/",
    #     # task_list = ['add', 'delete', 'sub', 'ring_count', 'ring_system_scaffold', 'mutated', 'permutated', 'functiongroup_detect', 'murcko_scaffold'],
    #     # task_list= ['nepp','fs_major_product','fs_by_product','retro','rcr_catalyst','rcr_reagent','rcr_solvent','mech_sel'],
    # )
    
    ## data-clean and add system-prompt in stage-4
    my_data_builder.stage4_dataclean(
        stage3_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage3/",
        stage4_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcot-tool-coldstart/stage4/",
        # task_list= ['add', 'delete', 'sub', 'ring_count', 'ring_system_scaffold', 'mutated', 'permutated', 'functiongroup_detect', 'murcko_scaffold'],
        task_list=['nepp','fs_major_product','fs_by_product','retro','rcr_catalyst','rcr_reagent','rcr_solvent','mech_sel'],
    )
    
    # print(my_data_builder.tool_descriptions)