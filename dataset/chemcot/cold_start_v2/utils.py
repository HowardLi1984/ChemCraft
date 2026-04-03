import re
import json
import ast

# 检测一个str里面三个东西是否符合格式要求
# 1. <tool-name></tool-name>是闭合的，比如有个<CountMolAtoms>那么必须有个</CountMolAtoms>在后面，而且这<CountMolAtoms></CountMolAtoms>的中间没有别的<>
# 2. <result></result> 是闭合的，同时中间的内容没有其他的<>
# 3. <answer></answer> 出现在最后部分，同时是闭合的，同时中间的内容没有其他的<>
def data_clean(text, tool_valid_names: list):
    # 构建一个包含所有合法标签对的集合
    valid_tags = set()
    for match in re.finditer(r'<([A-Z][a-zA-Z0-9]+)>(.*?)<\/\1>', text, re.DOTALL):
        valid_tags.add(match.span())
    
    # 检查是否有未闭合或非法嵌套的标签
    tag_pattern = r'<([a-zA-Z0-9_]+)>'
    for tag_match in re.finditer(tag_pattern, text):
        tag_name = tag_match.group(1)
        tag_start = tag_match.start()
        
        # 排除 <answer> 标签
        if tag_name == 'answer':
            continue
        
        # 寻找对应的闭合标签
        end_tag_pattern = r'</' + re.escape(tag_name) + r'>'
        end_match = re.search(end_tag_pattern, text[tag_start:])
        
        if not end_match:
            return False, f"cannot find <{tag_name}> 's close tag"
        
        # 检查闭合标签之间的内容是否包含其他标签
        content = text[tag_start + len(tag_name) + 2: tag_start + end_match.start()]
        if re.search(r'<[^>]*>', content):
            return False, f"Tag <{tag_name}>...</{tag_name}> contains other tags"
        
        if tag_name != 'result':
            s_content = content.strip()
            # 1. 检查外观是否包裹在 {} 中
            if not (s_content.startswith('{') and s_content.endswith('}')):
                return False, f"Tag <{tag_name}> content must start with {{ and end with }}"
            # 2. 检查是否为合法 JSON
            try:
                # ast.literal_eval 支持单引号 {'a': 1} 和双引号 {"a": 1}
                parsed_dict = ast.literal_eval(s_content)
                if not isinstance(parsed_dict, dict):
                    return False, f"Tag <{tag_name}> content is not a dictionary"
                # find if the tag_name is in tool_valid_names, is valid tool
                if parsed_dict['name'] not in tool_valid_names:
                    return False, f"Tool <{parsed_dict['name']}>...</{parsed_dict['name']}> is not valid, does not exist"
            except Exception:
                return False, f"Tag <{tag_name}> content is not a valid python dict struct"

    # 2. 检查 <result>...</result> 标签
    # 查找所有 <result> 标签
    result_matches = re.finditer(r'<result>(.*?)</result>', text, re.DOTALL)
    
    for match in result_matches:
        content = match.group(1)
        if re.search(r'<[^>]*>', content):
            return False, f"Tag <result>...</result> contains other tags"

    # 3. 检查 <answer> 标签
    answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    
    if len(answer_matches) == 0:
        return False, "Cannot find <answer> tag"
    
    if len(answer_matches) > 1:
        return False, "Find multiple <answer> tags"
    
    answer_match = answer_matches[0]
    answer_content = answer_match.group(1)
    
    if re.search(r'<[^>]*>', answer_content):
        return False, "Tag <answer>...</answer> contains other tags"
        
    return True, "All formarts are correct"

def find_tool_calls(text):
    # pattern = r"<(\w+)>(.*?)<\/\1><result>(.*?)</result>"
    pattern = r"<(\w+)>(.*?)<\/\1>\s*<result>(.*?)</result>" # 加入\s* 
    matches_found = []
    # 使用 re.finditer 来查找所有匹配项及其位置信息
    for match in re.finditer(pattern, text, re.DOTALL):
        matches_found.append({
            'match': match.group(0),          # 完整匹配的字符串
            'tool_name': match.group(1),      # 第一个捕获组: tool-name
            'query': match.group(2),          # 第二个捕获组: query
            'result': match.group(3),         # 第三个捕获组: result标签内容
            'start': match.start(),           # 匹配的起始索引
            'end': match.end()                # 匹配的结束索引
        })
        
    return matches_found

def reform_tool_call(match_info):
    # 把格式化提取的match_info转换成<tool_call>\n{}\n</tool_call>以及<result></result>的形式
    if match_info['tool_name'] == 'tool_call':
        tool_use = f"<tool_call>{match_info['query']}</tool_call>"
        return tool_use
    
    tool_name = match_info['tool_name']
    query_list = match_info['query'].split(';')
    
    if tool_name in ["QEDPropertyPred", "DRD2PropertyPred", "JNK3PropertyPred", "LogPPropertyPred", "GSKPropertyPred", "SolubilityPropertyPred"]:
        tool_argument = {'SMILES': match_info['query']}
    elif tool_name == "GetRXN":
        tool_argument = {'reactants': match_info['query']}
    elif tool_name == 'AddFunctionalGroup' and len(query_list) >= 2:
        tool_argument = {'SMILES':query_list[0], 'new_group_name':query_list[1]}
    elif tool_name == 'RemoveFunctionalGroup' and len(query_list) >= 2:
        tool_argument = {'SMILES':query_list[0], 'old_group_name':query_list[1]}
    elif tool_name == 'ReplaceFunctionalGroup' and len(query_list) >= 3:
        tool_argument = {'SMILES':query_list[0], 'old_group_name':query_list[1], 'new_group_name':query_list[2]}
    elif tool_name in ["CompareSMILES", "MolSimilarity"] and len(query_list) >= 2:
        tool_argument = {'SMILES1':query_list[0], 'SMILES2':query_list[1]}
    elif tool_name in ['CountMolAtoms', 'CanonicalizeSMILES', 'FunctionalGroups', 'SMILES2Weight']:
        tool_argument = {'SMILES': match_info['query']}
    else:
        print("tool_name:", tool_name)
        print("tool_argu:", match_info['query'])
        tool_argument = {'SMILES': match_info['query']}
    
    tool_inside = f"'name':'{tool_name}', 'arguments':{tool_argument}"
    tool_use = f"<tool_call>\n{{{tool_inside}}}\n</tool_call>"
    return tool_use

def transform_stage3_tool_answers(raw_input_trajectory):
    # remove Step-1, Step-2, Step-3, etc ...
    # transforming tool-call to <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n
    pattern = r"Step-\d+, "
    input_trajectory = re.sub(pattern, "", raw_input_trajectory)
    
    matches = find_tool_calls(input_trajectory)
    if not matches:
        return [{'type':'assistant', 'content':input_trajectory}]
    
    result = []; last_end = 0
    for match in matches:
        if match['start'] > last_end:
            result.append({'type':'assistant', 'content':input_trajectory[last_end: match['start']].lstrip()})
        
        tool_use_str = reform_tool_call(match) 
        if 'assistant' == result[-1]['type']:
            result[-1]['content'] += tool_use_str
        elif 'tool' == result[-1]['type']:
            result.append({'type':'assistant', 'content':f"Then I use the tool: {tool_use_str}"})
        else:
            assert "tool use apply error"
        
        result.append({'type':'tool', 'content':f"<result> {match['result']} </result>"})
        last_end = match['end']
    
    if last_end < len(input_trajectory):
        result.append({'type':'assistant', 'content':input_trajectory[last_end:].lstrip()})
    
    # formatted_string = json.dumps(result, indent=2, ensure_ascii=False)
    return result

def reshape_question(task, text):
    if task in ['nepp']:
        new_text = text.replace('Just return the SMILES of prediction. Your response must contains directly parsable JSON format: \n\n{\n    \"pred_smi\": str\n}', 'Output: <answer> SMILES of the prediction </answer>.')
    elif task in ['mech_sel']:
        new_text = text.replace("in JSON format:\n{\n    \"choice\": str # (e.g. 'A'/'B')\n}", 'in format, Output: <answer> A/B/C ... </answer>.')
    elif task in ['fs_by_product']:
        new_text = text.replace("Answer:\n{\n    \"By Product\": ...\n}\n", "Output: <answer> SMILES of By Product </answer>.")
    elif task in ['fs_major_product']:
        new_text = text.replace("Answer:\n{\n    \"Major Product\": ...\n}\n", "Output: <answer> SMILES of Major Product </answer>.")
    elif task in ['rcr_catalyst', 'rcr_reagent', 'rcr_solvent']:
        rcr_type = task.split("_")[-1]
        new_text = text.split("Now, I'll secretly tell you that the Reactant Target molecule")[0]
        new_text = new_text + f"Output: <answer> SMILES of {rcr_type} </answer>."
    elif task in ['retro']:
        new_text = text.replace("Answer:\n{\n    \"Reactants\": ...\n}\n", "Output: <answer> SMILES of Reactants </answer>.")
    elif task in ['add', 'delete', 'sub', 'ring_count', 'ring_system_scaffold', 'mutated', 'permutated', 'functiongroup_detect', 'murcko_scaffold', 'gsk', 'jnk', 'drd', 'logp', 'qed', 'solubility']:
        new_text = text
    else:
        assert FileNotFoundError
    return new_text

if __name__ == "__main__":
    # text = """
    # "input_structure: CC[C@H](C)CN(C)C(=O)C1([NH3+])CCCCC1.  \ncore_ring_identification: The input molecule contains a six-membered aliphatic ring, which is represented as C1CCCCC1 in SMILES notation, indicating a cyclohexane ring. This ring is part of the core structure.  \n<CountMolAtoms>CC[C@H](C)CN(C)C(=O)C1([NH3+])CCCCC1</CountMolAtoms>  \n<result>There are altogether 20 atoms (omitting hydrogen atoms). The types and corresponding numbers are: {'C': 13, 'O': 1, 'H': 4, 'N': 2}</result>  \nThis atom count confirms the presence of multiple carbon atoms, consistent with a cyclohexane ring and several side chains. The high number of carbon atoms (13) suggests extensive substitution around the ring, supporting the presence of alkyl and functionalized side groups.  \n<FunctionalGroups>CC[C@H](C)CN(C)C(=O)C1([NH3+])CCCCC1</FunctionalGroups>  \n<result>This molecule contains 1 ketones, 1 amides, 1 primary amines, and 1 amine.</result>  \nThe presence of an amide group (C(=O)N) and a primary ammonium group ([NH3+]) indicates that the cyclohexane ring is substituted with nitrogen-containing functional groups. These are considered exocyclic modifications and must be removed when generating the Murcko scaffold.  \n\nscaffold_analysis: According to the definition, the Murcko scaffold is derived by removing all side chains, functional groups, and exocyclic bonds, retaining only the ring system and any connecting bonds. In this case, the core ring is a single cyclohexane ring (C1CCCCC1), with no additional fused rings or bridging bonds. Therefore, the expected Murcko scaffold is simply the unsubstituted cyclohexane ring.  \n\nmatching_analysis: To verify identity, the generated scaffold from the input molecule must be compared directly with the given Murcko scaffold C1CCCCC1.  \n<CompareSMILES>C1CCCCC1;C1CCCCC1</CompareSMILES>  \n<result>identical</result>  \nThe comparison confirms that the ring system extracted from the input molecule is chemically identical to the provided Murcko scaffold. Despite the presence of substituents such as the amide, ammonium, and alkyl chains in the original molecule, their removal leaves behind the pure cyclohexane ring, which matches exactly.  \n\nfeasibility_analysis: Even though the original molecule contains charged groups ([NH3+]) and polar functional groups (amide), these are not part of the core ring system and are correctly excised during Murcko scaffold generation. The tool results support that the underlying carbon ring remains intact and unmodified in terms of ring connectivity.  \n\nOutput: <answer> Yes </answer>"
    # """
    
    # output = transform_stage3_tool_answers(text)
    
    text = """
    Step-1, Parse Reactants/Reagents: The reactants are a complex organic molecule, COc1ccc(COc2ccc(Cn3c(N)nc4cc(-c5cnn(CC(=O)OC(C)(C)C)c5)cnc43)cc2OC)cc1, and sodium hydroxide, O[Na]. The organic molecule is a large, multi-ring structure containing a tert-butyl ester functional group. The reagent, sodium hydroxide, is a strong base.\n\nStep-2, Identify Functional Groups: I need to identify the reactive functional groups to predict the reaction. The primary reactant is a complex molecule, so I will use a tool to ensure all key groups are identified.\n<tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES': 'COc1ccc(COc2ccc(Cn3c(N)nc4cc(-c5cnn(CC(=O)OC(C)(C)C)c5)cnc43)cc2OC)cc1'}}\n</tool_call>\n<result>\nThis molecule contains 1 esters, 1 t-butyl, 2 methoxy, 1 primary amines, 2 benzene ring, and 1 amine.\n</result>\nThe tool confirms the presence of an ester group, specifically a tert-butyl ester, which is known to be susceptible to hydrolysis. Other functional groups identified, such as methoxy, primary amine, and aromatic rings (benzene, pyrazole, triazolopyrimidine), are generally stable under basic conditions. The reagent, O[Na], provides hydroxide ions (OH-), which are strong nucleophiles and bases, and sodium ions (Na+). The most likely site for reaction is the electrophilic carbonyl carbon of the ester.\n\nStep-3, Predict Major Product: The reaction between an ester and a strong base like sodium hydroxide is a classic saponification reaction, which results in the hydrolysis of the ester. I will check for similar known reactions to confirm this pathway.\n<tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'COc1ccc(COc2ccc(Cn3c(N)nc4cc(-c5cnn(CC(=O)OC(C)(C)C)c5)cnc43)cc2OC)cc1.O[Na]', 'Type': 'reactants'}}\n</tool_call>\n<result>\nCannot find similar reactions for your input SMILES\n</result>\nAlthough the tool did not find a direct match for this specific complex molecule, the fundamental chemical principle of saponification remains the most plausible reaction pathway. The hydroxide ion (OH-) will attack the carbonyl carbon of the tert-butyl ester. This nucleophilic attack leads to the cleavage of the acyl-oxygen bond. The initial products are a carboxylate anion and tert-butanol. The carboxylate anion will then associate with the sodium cation (Na+) from the sodium hydroxide, forming the sodium carboxylate salt as the major organic product.\n\nStep-4, Identify Minor Pathways: Given the reaction conditions implied by the reagent (aqueous NaOH), the hydrolysis of the tert-butyl ester is the most favorable and dominant reaction. The other functional groups\u2014ethers, methoxy groups, aromatic systems, and the primary amine\u2014are robust and not expected to react under these conditions. Therefore, significant side reactions leading to other byproducts are not anticipated.\n\nStep-5, Construct Byproduct Structures: The saponification reaction cleaves the ester into two parts: the carboxylate salt (the main product) and the alcohol part. In this case, the ester is a tert-butyl ester, R-COOC(CH3)3. The hydrolysis liberates the tert-butoxy group, which is protonated by water (present as the solvent for NaOH) to form tert-butanol, HOC(CH3)3. The SMILES for tert-butanol is CC(C)(C)O. The sodium ion, [Na+], from the NaOH reagent is incorporated into the main product but can also be considered a species present at the end of the reaction. Therefore, the primary byproduct generated from the organic reactant is tert-butanol.\n\nOutput:\n<answer>\nCC(C)(C)O.[Na+]\n</answer>
    """
    
    clean_result = data_clean(text)
    print(clean_result)