# Reward Function for Qwen-Chem-Agent

import re
import string
import random
import requests
import rdkit.Chem as Chem

import sys
sys.path.append("/cto_labs/lihao/chem_reason/ChemSearch/search_sft/evaluation/")
from eval_moledit import check_edit_add_valid, check_edit_del_valid, check_edit_sub_valid
from eval_rxn import MoleculeSMILESEvaluator

def _call_chemical_tool(function_name:str, query:str):
    payload = {
        "function_name": function_name,
        "query": query
    }
    response = requests.post("http://127.0.0.1:8080/chem_function", json=payload)
    return response.json()['result']

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        else: return True
    except:
        return False

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_property(str_result):
    # extract the property from "The DRD2 property of {smiles} is {prop_predict}."
    if "property" in str_result:
        # extract the property number, can be -2, 1.00, 100, etc
        pattern = r'The (\w+) property of (\S+) is ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\.'
        match = re.search(pattern, str_result)
        if match:
            info = {
                'prop_name': match.group(1),
                'smiles': match.group(2),
                'prop_value': match.group(3)
            }
            return float(info['prop_value'])
        else:
            return None
    else:
        # input str is "Error calculating DRD2: {e}"
        return None

def opt_score_fn(answer, extra_info, min_score, max_score):
    # calculate the molopt score
    taskname = extra_info['subtask']
    src_smiles = extra_info['molecule']
    if src_smiles == None: print(f"Mol-Opt, src_smiles is None, {extra_info}, {answer}")
    assert src_smiles is not None  # in mol_opt task, extra_info['molecule'] is not None
    src_smiles = src_smiles.rstrip('.')
    if not is_valid_smiles(src_smiles):
        return 1.
    elif not is_valid_smiles(answer):
        return 0.
    else: # both src_molecule and optimized_molecule are valid
        function_name_list = {
            "qed": "QEDPropertyPred", "drd": "DRD2PropertyPred", 
            "jnk": "JNK3PropertyPred", "logp": "LogPPropertyPred",
            "gsk": "GSKPropertyPred", "solubility": "SolubilityPropertyPred",
        }
        tool_name = function_name_list[taskname]
        src_property = extract_property(_call_chemical_tool(function_name=tool_name, query=src_smiles))
        tgt_property = extract_property(_call_chemical_tool(function_name=tool_name, query=answer))
        if tgt_property == None:
            return min_score
        elif src_property == None:
            print(f"Opt: {taskname}, Src_molecule optimization score fails: {src_smiles}")
            return max_score
        elif tgt_property-src_property > 0:
            return max_score
        else:
            return min_score

def edit_score_fn(answer, extra_info, min_score, max_score):
    # calculate the moledit score
    taskname = extra_info['subtask']
    src_molecule, add_group, remove_group = extra_info['molecule'], extra_info['added_group'], extra_info['removed_group']
    if src_molecule != None: src_molecule = src_molecule.rstrip('.')
    if add_group != None: add_group = add_group.rstrip('.')
    if remove_group != None: remove_group = remove_group.rstrip('.')
    score = min_score
    
    if not is_valid_smiles(src_molecule):
        print(f"Edit: {taskname}, Src_molecule fails: {src_molecule}")
        score = max_score
    elif not is_valid_smiles(answer):
        score = min_score
    else:
        if taskname in ['add']:
            if check_edit_add_valid(src=src_molecule, tgt=answer, group=add_group):
                score = max_score
        if taskname in ['delete']:
            if check_edit_del_valid(src=src_molecule, tgt=answer, group=remove_group):
                score = max_score
        if taskname in ['sub']:
            if check_edit_sub_valid(src=src_molecule, tgt=answer, remove_group=remove_group, add_group=add_group):
                score = max_score
    return score

def rxn_score_fn(answer, gt, extra_info, min_score, max_score):
    # calculate the reaction-related score
    # 'fs_major_product', 'fs_by_product', 'mech_sel', 'nepp', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro'
    taskname = extra_info['subtask']
    score = min_score
    
    if taskname in ['fs_major_product', 'fs_by_product', 'nepp', 'rcr', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro']:
        evaluator = MoleculeSMILESEvaluator()
        preds = [answer]; gts = [gt]
        res = evaluator.evaluate(preds, gts)
        fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
        score = fts
    elif taskname in ['mechsel', "mech_sel"]:
        if answer == gt:
            score = max_score

    return score

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1.0
            break
    return score

def extract_solution(solution_str):
    """
    Extract the final answer from the solution string.
    Supports both <answer> tags and JSON format with "output" field.
    Returns the last occurrence found in the string.
    """
    # Pattern 1: Extract content between <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    # Pattern 2: Extract JSON output field, {\n    "output": "CCN(Cc1ccccn1)c1ccco1"\n}
    json_pattern_1 = r'\{[^{}]*"output"\s*:\s*"([^"]*)"[^{}]*\}'
    json_matches_1 = list(re.finditer(json_pattern_1, solution_str, re.DOTALL))
    
    # Pattern 3: Extract JSON output field, {\n    'output': 'CCN(Cc1ccccn1)c1ccco1'\n}
    json_pattern_2 = r"\{[^{}]*'output'\s*:\s*'([^']*)'[^{}]*\}"
    json_matches_2 = list(re.finditer(json_pattern_2, solution_str, re.DOTALL))
    
    all_matches = []
    
    # Add answer tag matches
    for match in answer_matches:
        all_matches.append({'type': 'answer_tag', 'content': match.group(1).strip(), 'position': match.start()})
        
    # Add JSON output matches
    for match in json_matches_1:
        all_matches.append({'type': 'json_output1', 'content': match.group(1).strip(), 'position': match.start()})
    
    # Add JSON output matches
    for match in json_matches_2:
        all_matches.append({'type': 'json_output2', 'content': match.group(1).strip(), 'position': match.start()})
    
    if not all_matches: return None
    
    # Sort matches by position and get the last one
    all_matches.sort(key=lambda x: x['position'])
    last_match = all_matches[-1]
    
    return last_match['content']

def format_acc_calculation(solution_str, min_score, max_score):
    # Evaluate the format score from solution_str
    # (1) Tool_Call_Format:    <tool_call>\n{'name':'', 'arguments':{}}\n</tool_call>
    # (2) Final_Answer_Format: Output: <answer>  </answer>
    
    tool_call_blocks = re.findall(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
    total_tool_calls = len(tool_call_blocks)
    
    tool_call_pattern = r"<tool_call>\n\{('name':\s*'.*?',\s*'arguments':\s*\{.*?\})|(\"name\":\s*\".*?\",\s*\"arguments\":\s*\{.*?\})\}\n</tool_call>"
    correct_tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
    
    answer_pattern = r"Output:\s*<answer>.*?</answer>"
    answer_matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    
    if len(answer_matches) != 1: # only allow one answer
        return min_score
    elif len(correct_tool_calls) != total_tool_calls:
        return min_score
    else:
        return max_score

# def compute_chem_score(solution_str, ground_truth, format_score=0., score=1., extra_info=None):
#     """The scoring function for exact match (EM).

#     Args:
#         solution_str: the solut ion text
#         ground_truth: the ground truth
#         format_score: the score for the format
#         score: the score for the correct answer
#         extra_info: contains task, subtask, added_group, removed_groups, molecule
#     """
#     answer = extract_solution(solution_str=solution_str)
#     do_print = random.randint(1, 64) == 1
#     if do_print:
#         print(f"--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")
    
#     format_score = format_acc_calculation(solution_str=solution_str, )
    
#     if answer is None:
#         return 0.0
#     elif extra_info['subtask'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']: # mol_opt
#         assert ground_truth['target'] == ''
#         ans_score = opt_score_fn(answer, extra_info)
#         return ans_score + format_score
#     elif extra_info['subtask'] in ['add', 'delete', 'sub']: # mol_edit
#         assert ground_truth['target'] == ''
#         ans_score = edit_score_fn(answer, extra_info)
#         return ans_score + format_score
#     elif extra_info['subtask'] in ['ring_count', 'ring_system_scaffold', 'equivalence', 'fg_count', 'Murcko_scaffold']: # mol_und
#         if em_check(answer, ground_truth['target']):
#             ans_score = 1.0
#         else: ans_score = 0.0
#         return ans_score + format_score
#     else: 
#         assert NotImplementedError

## format reward v1: completely use the ToolRL reward
# def compute_chem_score(solution_str, ground_truth, format_score=0., score=1., extra_info=None):
#     """The scoring function for exact match (EM).

#     Args:
#         solution_str: the solut ion text
#         ground_truth: the ground truth
#         format_score: the score for the format
#         score: the score for the correct answer
#         extra_info: contains task, subtask, added_group, removed_groups, molecule
#     """
#     answer = extract_solution(solution_str=solution_str)
#     do_print = random.randint(1, 64) == 1
#     if do_print:
#         print(f"--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")
#         print(f"Current Step: {extra_info['current_step'].tolist()[0]}")
    
#     # Get upper_bond and lower_bond of Format Reward and Correctness Reward
#     current_step = extra_info['current_step'].tolist()[0] 
#     if current_step < 300: 
#         print(f"current step: {current_step}")
#         current_step = current_step - 300
#     format_max_score = max(1.0, 2-current_step/60)
#     format_min_score = min(-1.0, -2+current_step/60)
    
#     correct_max_score = min(3.0, 2+current_step/60)
#     correct_min_score = max(-3.0, -2-current_step/60)
    
#     # Calculate Format Reward and Correctness Reward
#     format_score = format_acc_calculation(solution_str=solution_str, min_score=format_min_score, max_score=format_max_score)
#     correct_score = correct_min_score
    
#     if answer is None:
#         return correct_min_score + format_min_score
#     elif extra_info['subtask'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']: # mol_opt
#         assert ground_truth['target'] == ''
#         correct_score = opt_score_fn(answer, extra_info, min_score=correct_min_score, max_score=correct_max_score)
#     elif extra_info['subtask'] in ['add', 'delete', 'sub']: # mol_edit
#         assert ground_truth['target'] == ''
#         correct_score = edit_score_fn(answer, extra_info, min_score=correct_min_score, max_score=correct_max_score)
#     elif extra_info['subtask'] in ['ring_count', 'ring_system_scaffold', 'equivalence', 'fg_count', 'Murcko_scaffold']: # mol_und
#         if em_check(answer, ground_truth['target']):
#             correct_score = correct_max_score
#         else: correct_score = correct_min_score
#     else: 
#         assert NotImplementedError
    
#     return correct_score + format_score

## format rewward v2: 
## total-reward: 0.0-1.0,  for the first 60 step, total-reward = 0.5*max(0, (1-step/30))*format-score + 0.5*min(2, (1+step/30))*correct-score
def compute_chem_score(solution_str, ground_truth, format_score=0., score=1., extra_info=None):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solut ion text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
        extra_info: contains task, subtask, added_group, removed_groups, molecule
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"Current Step: {extra_info['current_step'].tolist()[0]}")
    
    # # Get upper_bond and lower_bond of Format Reward and Correctness Reward
    # current_step = extra_info['current_step'].tolist()[0] 
    # if current_step > 300: print(f"current step: {current_step}")
    # alpha_format = 0.5 * max(0, (1 - current_step/30))
    # alpha_correct = 0.5 * min(2, (1 + current_step/30))
    alpha_format = 0.0; alpha_correct = 1.0

    format_min_score, format_max_score = 0.0, 1.0
    correct_min_score, correct_max_score = 0.0, 1.0
    
    # Calculate Format Reward and Correctness Reward
    format_score = format_acc_calculation(solution_str=solution_str, min_score=format_min_score, max_score=format_max_score)
    correct_score = correct_min_score
    
    if answer is None:
        return alpha_format*format_min_score + alpha_correct*correct_min_score
    elif extra_info['subtask'] in ['qed', 'logp', 'solubility', 'drd', 'gsk', 'jnk']: # mol_opt
        assert ground_truth['target'] == ''
        correct_score = opt_score_fn(answer, extra_info, min_score=correct_min_score, max_score=correct_max_score)
    elif extra_info['subtask'] in ['add', 'delete', 'sub']: # mol_edit
        assert ground_truth['target'] == ''
        correct_score = edit_score_fn(answer, extra_info, min_score=correct_min_score, max_score=correct_max_score)
    elif extra_info['subtask'] in ['ring_count', 'ring_system_scaffold', 'equivalence', 'fg_count', 'Murcko_scaffold']: # mol_und
        if em_check(answer, ground_truth['target']):
            correct_score = correct_max_score
        else: correct_score = correct_min_score
    elif extra_info['subtask'] in ['fs_major_product', 'fs_by_product', 'mechsel', 'mech_sel', 'nepp', 'rcr', 'rcr_catalyst', 'rcr_reagent', 'rcr_solvent', 'retro']:
        assert ground_truth['target'] == extra_info['gt']
        correct_score = rxn_score_fn(answer, gt=ground_truth['target'], extra_info=extra_info, min_score=correct_min_score, max_score=correct_max_score)
    else: 
        raise NotImplementedError(f"Cannot find the reward for task: {extra_info['subtask']}")
    
    return alpha_format*format_score + alpha_correct*correct_score