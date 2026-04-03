import sys, re, os, json
from rdkit import Chem

from abc import ABC, abstractmethod
from typing import List, Dict
from functools import partial
from Levenshtein import distance as lev
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import selfies as sf
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score


def exact_match(ot_smi, gt_smi):
    m_out = Chem.MolFromSmiles(ot_smi)
    m_gt = Chem.MolFromSmiles(gt_smi)

    try:
        if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
            return 1
    except:
        pass
    return 0

class Evaluator(ABC):

    @abstractmethod
    def build_evaluate_tuple(self, pred, gt):
        pass

    @abstractmethod
    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        pass

def maccs_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        MACCSkeys.GenMACCSKeys(gt_m), 
        MACCSkeys.GenMACCSKeys(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )

def morgan_similarity(ot_m, gt_m, radius=2):
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprint(gt_m, radius), 
        AllChem.GetMorganFingerprint(ot_m, radius)
    )

def rdk_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(gt_m), 
        Chem.RDKFingerprint(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )

class MoleculeSMILESEvaluator(Evaluator):
    _metric_functions = {
        "exact_match": exact_match,
        "bleu": corpus_bleu,
        "levenshtein": lev,
        "rdk_sims": rdk_similarity,
        "maccs_sims": maccs_similarity,
        "morgan_sims": morgan_similarity,
        "validity": lambda smiles: smiles is not None,
    }

    @staticmethod
    def sf_decode(selfies):
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception:
            return None

    @staticmethod
    def convert_to_canonical_smiles(smiles):
        if not smiles:
            return None
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
                return canonical_smiles
            else:
                return None
        except Exception:
            return None

    def build_evaluate_tuple(self, pred, gt, decode_selfies=False):
        if decode_selfies:
            pred = self.sf_decode(pred)
            gt = self.sf_decode(gt)
        return self.convert_to_canonical_smiles(pred), self.convert_to_canonical_smiles(gt)

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, decode_selfies: bool = False, full_results: bool = False):
            
        if metrics is None:
            metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]

        results = {metric: [] for metric in metrics}
        if "bleu" in metrics:
            results["bleu"] = [[], []]

        for pred, gt in zip(predictions, references):
            pred, gt = self.build_evaluate_tuple(pred, gt, decode_selfies=decode_selfies)

            for metric in metrics:
                if metric == "bleu" and pred and gt:
                    gt_tokens = [c for c in gt]
                    pred_tokens = [c for c in pred]
                    results[metric][0].append([gt_tokens])
                    results[metric][1].append(pred_tokens)
                elif pred is None or gt is None:
                    results[metric].append(0)
                    continue
                elif metric == "validity":
                    results[metric].append(self._metric_functions[metric](pred))
                elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                    results[metric].append(self._metric_functions[metric](Chem.MolFromSmiles(pred), Chem.MolFromSmiles(gt)))
                else:
                    results[metric].append(self._metric_functions[metric](pred, gt))

        if "bleu" in metrics:
            if results["bleu"][0] and results["bleu"][1]:
                results["bleu"] = corpus_bleu(results["bleu"][0], results["bleu"][1])
            else:
                results["bleu"] = 0

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def is_valid_smiles(smiles, strict:bool=True):
    if strict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
    else:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False
        except:
            return False
    return True

def parse_response(response: str):
    # Attempt to extract a JSON code block and parse it
    # str format: "<|think_start|>...<|think_end|><|answer_start|>...<|answer_end|>"
    # we extract the answer part
    answer_match = re.search(r'<\|answer_start\|>(.*?)<\|answer_end\|>', response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        return answer
    return ""

def evaluate_RCR(model_name:str, log_dir:str="logs/RCR"):
    """
    Evaluate the reaction condition recommendation task
    Metric: SMILES similarity

    Args:
        model_name (str): the name of the model
        log_dir (str): the directory of the logs

    Returns:
        None
    """
    # make sure the log_dir is correct
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    samples = read_json(f"{log_dir}/{model_name}.json")
    preds = []
    gts = []
    for sample in samples:
        gts.append(sample['gt'])

        pred_smiles = parse_response(sample['json_response'])
        # further parsing to remove the answer tag
        if pred_smiles.endswith("</answer>"):
            pred_smiles = pred_smiles[:-len("</answer>")]
        
        # further parsing to extract condition smiles
        if ">>" in pred_smiles:
            pred_smiles = pred_smiles.split(">>")[0]
        elif '>' in pred_smiles:
            pred_smiles = pred_smiles.split('>')[1]
        preds.append(pred_smiles)

    evaluator = MoleculeSMILESEvaluator()
    res = evaluator.evaluate(preds, gts)
    # pretty print the res
    print("exact_match: ", round(res['exact_match'], 2))
    print("bleu: ", round(res['bleu'], 2))
    print("levenshtein: ", round(res['levenshtein'], 2))
    print("rdk_sims: ", round(res['rdk_sims'], 2))
    print("maccs_sims: ", round(res['maccs_sims'], 2))
    print("morgan_sims: ", round(res['morgan_sims'], 2))
    print("validity: ", round(res['validity'], 2))
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    print("fts: ", round(fts * res['validity'], 2))
    
def evaluate_NEPP(model_name: str, log_dir: str):
    """
    Evaluate the next element-step product prediction task
    Metric: SMILES similarity

    Args:
        model_name (str): the name of the model
        log_dir (str): the directory of the logs

    Returns:
        None
    """
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    samples = read_json(f"{log_dir}/{model_name}.json")
    preds = []
    gts = []
    for sample in samples:
        gts.append(sample['gt'])
        json_response = sample['json_response']
        pred_smiles = parse_response(json_response)
        preds.append(pred_smiles)

    evaluator = MoleculeSMILESEvaluator()
    res = evaluator.evaluate(preds, gts)

    # pretty print the res
    print("exact_match: ", round(res['exact_match'], 2))
    print("bleu: ", round(res['bleu'], 2))
    print("levenshtein: ", round(res['levenshtein'], 2))
    print("rdk_sims: ", round(res['rdk_sims'], 2))
    print("maccs_sims: ", round(res['maccs_sims'], 2))
    print("morgan_sims: ", round(res['morgan_sims'], 2))
    print("validity: ", round(res['validity'], 2))
    
def evaluate_mechsel(model_name: str, logs_dir: str):
    """
    Evaluate the reaction mechanism selection prediction.

    Args:
        logs_dir (str): The directory where the logs are stored.
        model_name (str): The name of the model.

    Returns:
        None
    """
    if not os.path.exists(logs_dir):
        raise ValueError(f"logs_dir {logs_dir} is not correct")
    samples = read_json(f"{logs_dir}/{model_name}.json")

    preds = []
    gts = []
    for sample in samples:
        pred_choice = parse_response(sample['json_response'])
        if len(pred_choice) > 1:
            # if multiple chars, we take the first one
            pred_choice = pred_choice[0]
        # if pred_choice is not a valid choice, we treat it as empty
        if pred_choice.lower() not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            pred_choice = ""

        pred_choice = pred_choice.lower()
        gt = sample['gt'].lower()
        preds.append(pred_choice)
        gts.append(gt)

    accuracy = sum(1 for pred, gt in zip(preds, gts) if pred == gt) / len(gts)
    print(f"MCQ Accuracy (mean): {accuracy:.2f}")
    
def evaluate_fs_major(model_name: str, log_dir: str):
    """
    Evaluate SMILES predictions against ground truth.

    Args:
        model_name (str): The name of the model.
        log_dir (str): The directory where the logs are stored.

    Returns:
        None
    """
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    samples = read_json(f"{log_dir}/{model_name}.json")
    preds = []
    gts = []
    for sample in samples:
        ## parse the gt
        try:
            gt = sample['products']
        except:
            gt = json.loads(sample['gt']).get("Major Product")
        if isinstance(gt, list):
            gt = '.'.join(gt)
        gts.append(gt)

        ## parse the pred
        pred_smiles = parse_response(sample['json_response'])
        preds.append(pred_smiles)

    evaluator = MoleculeSMILESEvaluator()
    res = evaluator.evaluate(preds, gts)

    print("exact_match: ", round(res['exact_match'], 2))
    print("bleu: ", round(res['bleu'], 2))
    print("levenshtein: ", round(res['levenshtein'], 2))
    print("rdk_sims: ", round(res['rdk_sims'], 2))
    print("maccs_sims: ", round(res['maccs_sims'], 2))
    print("morgan_sims: ", round(res['morgan_sims'], 2))
    print("validity: ", round(res['validity'], 2))
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    print("fts: ", round(fts * res['validity'], 2))

def parse_raw_response(raw_response: str, field: str) -> str:
    """从原始响应字符串中提取指定字段的值
    
    Args:
        raw_response: 可能包含 JSON 块或键值对的原始字符串
        field: 需要提取的字段名（如 "Byproduct(s)"）
    
    Returns:
        提取到的字段值字符串，未找到则返回空字符串
    """
    
    # 改进点 1: 更健壮的 JSON 块检测（支持不同格式的代码块）
    json_block_match = re.search(
        r'```(?:json)?\s*({.*?})\s*```',  # 匹配可选的 json 标记和任意空格
        raw_response, 
        flags=re.DOTALL | re.IGNORECASE
    )
    
    if json_block_match:
        try:
            # 改进点 2: 更严格的 JSON 解析（处理嵌套结构）
            json_str = json_block_match.group(1).strip()
            data = json.loads(json_str)
            
            # 改进点 3: 处理数组型结果（如 ["HCl", "H2O"]）
            if value := data.get(field):
                if isinstance(value, list):
                    return ", ".join(map(str, value))
                return str(value)
        except (json.JSONDecodeError, AttributeError):
            pass  # 解析失败则继续尝试其他方法

    # 改进点 4: 更灵活的正则匹配（处理多种引号和转义）
    escaped_field = re.escape(field)  # 转义特殊字符如括号
    
    # 模式 1: 匹配双引号字符串（允许转义双引号）
    pattern = fr'"{escaped_field}":\s*"((?:\\"|[^"])*)"'
    if match := re.search(pattern, raw_response):
        try:
            # 使用 JSON 解析处理转义字符
            return json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return match.group(1).replace(r'\"', '"')
    
    # 模式 2: 匹配单引号字符串（允许转义单引号）
    pattern = fr'"{escaped_field}":\s*\'((?:\\\'|[^\'])*)\''
    if match := re.search(pattern, raw_response):
        return match.group(1).replace(r"\'", "'")
    
    # 改进点 5: 匹配无引号的值（如数值或布尔值）
    pattern = fr'"{escaped_field}":\s*([^,\s}}]+)'
    if match := re.search(pattern, raw_response):
        return match.group(1).strip()

    return ""  # 所有模式均未匹配

def evaluate_fs_byproduct(model_name, log_dir:str="logs/fs"):
    samples = read_json(f"{log_dir}/{model_name}.json")
    preds = []
    gts = []
    
    for sample in samples:
        ## parse the gt
        try:
            gt = sample['byproducts']
        except:
            gt = json.loads(sample['gt'])["Byproduct(s)"]
        if len(gt) == 0:  # if no byproducts, skip the sample
            continue
        if isinstance(gt, list):
            gt = '.'.join(gt)

        ## parse the pred
        if isinstance(sample['json_response'], dict):
            if "Byproduct(s)" in sample['json_response']:
                pred_smiles = sample['json_response']["Byproduct(s)"]
            else:
                pred_smiles = parse_raw_response(sample['raw_response'], "Byproduct(s)")
        else:
            try:
                pred_smiles = parse_raw_response(sample['raw_response'], "Byproduct(s)")
            except:
                pred_smiles = ""

        gts.append(gt)
        preds.append(pred_smiles)

    evaluator = MoleculeSMILESEvaluator()
    res = evaluator.evaluate(preds, gts)
    print("exact_match: ", round(res['exact_match'], 2))
    print("bleu: ", round(res['bleu'], 2))
    print("levenshtein: ", round(res['levenshtein'], 2))
    print("rdk_sims: ", round(res['rdk_sims'], 2))
    print("maccs_sims: ", round(res['maccs_sims'], 2))
    print("morgan_sims: ", round(res['morgan_sims'], 2))
    print("validity: ", round(res['validity'], 2))
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    print("fts: ", round(fts * res['validity'], 2))

def evaluate_retro(model_name: str, log_dir: str):
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    samples = read_json(f"{log_dir}/{model_name}.json")
    preds = []
    gts = []

    for sample in samples:
        ## parse the gt
        gt = sample['reactants']
        if len(gt) == 0:
            continue
        if isinstance(gt, list):
            gt = '.'.join(gt)
        gts.append(gt)

        ## parse the pred
        pred_smiles = parse_response(sample['json_response'])
        preds.append(pred_smiles)

    evaluator = MoleculeSMILESEvaluator()
    res = evaluator.evaluate(preds, gts)

    print("exact_match: ", round(res['exact_match'], 2))
    print("bleu: ", round(res['bleu'], 2))
    print("levenshtein: ", round(res['levenshtein'], 2))
    print("rdk_sims: ", round(res['rdk_sims'], 2))
    print("maccs_sims: ", round(res['maccs_sims'], 2))
    print("morgan_sims: ", round(res['morgan_sims'], 2))
    print("validity: ", round(res['validity'], 2))
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    print("fts: ", round(fts * res['validity'], 2))
