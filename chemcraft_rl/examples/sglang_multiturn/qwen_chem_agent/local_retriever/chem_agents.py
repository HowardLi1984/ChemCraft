from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict 

import os
import sys
import pickle
# add the qwen_chem_agent
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from base_agent import BaseTool
from constants import GROUP_TO_SMARTS, GROUP_TO_ADD_SMILES, find_functional_group_key_add, find_functional_group_key_replace_sub
from chem_utils import tanimoto, canonicalize_molecule_smiles, get_molecule_id, is_smiles, ChemAgentInputError

from rdkit.rdBase import DisableLog, WrapLogs
import contextlib, io

def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False, "error"
        return True, None
    except:
        return False, "error"
   
# def judge_smiles(smiles: str):
#     WrapLogs()
#     f = io.StringIO()
#     try: 
#         with contextlib.redirect_stderr(f):
#             mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return False, f"Capturing info: {f.getvalue()}"
#         else:
#             return True, None
#     except Exception as e:
#         return False, f"Python Exception: {str(e)}"

import io
import logging
import contextlib # 你的原始 import，现在不再需要
from rdkit import Chem
from rdkit.rdBase import LogToPythonLogger, LogToCppStreams, WrapLogs # 导入正确的函数

# 1. 在你的 agent 文件顶部，获取 RDKit 的 logger
# RDKit 的 C++ 日志会被转发到这个名为 'rdkit' 的 logger
rdkit_logger = logging.getLogger('rdkit')
# 设置级别，确保能捕获到 Info/Warning/Error
rdkit_logger.setLevel(logging.INFO) 


def judge_smiles(smiles: str):
    # 2. 为本次调用创建一个内存流 (StringIO)
    log_stream = io.StringIO()
    
    # 3. 创建一个专门向这个内存流写入的 Handler
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    
    mol = None
    error_message = ""
    
    try:
        # 4. 【关键】将我们的 handler 临时绑定到 RDKit logger
        rdkit_logger.addHandler(handler)
        
        # 5. 【关键】告诉 RDKit 开始使用 Python logging
        # 你原来的 WrapLogs() 也可以，但 LogToPythonLogger() 更现代
        LogToPythonLogger() 
        # WrapLogs() # 如果你坚持使用 WrapLogs() 也可以，它也能工作
        
        # 6. 清空流 (以防万一)
        log_stream.seek(0)
        log_stream.truncate(0)

        # 7. 执行 RDKit 操作
        # RDKit 错误 -> logging -> rdkit_logger -> 我们的 handler -> log_stream
        mol = Chem.MolFromSmiles(smiles)
        handler.flush(); log_stream.flush() # 刷新缓冲区, 确保多条报错都被跟踪
        
        if mol is None:
            # 8. 从我们自己的流中获取错误信息
            raw_error_message = log_stream.getvalue()
            print(raw_error_message)
            error_message = raw_error_message.replace("\n", " ").strip()
            if not error_message:
                error_message = "RDKit returned None without error message."
            return False, f"Capturing info: {error_message}"
        else:
            return True, None
            
    except Exception as e:
        return False, f"Python Exception: {str(e)}"
    
    finally:
        # 9. 【关键】必须清理，否则会内存泄漏！
        
        # 停止 RDKit 包装，恢复 C++ 默认行为
        LogToCppStreams() # <--- 这是正确的函数，用来替代 ClearLogs
        
        # 从 RDKit logger 中移除我们的 handler
        rdkit_logger.removeHandler(handler)
        
        # 关闭流
        log_stream.close()
    
class MolSimilarity(BaseTool):
    name = "MolSimilarity"
    func_name = 'cal_molecule_similarity'
    # description = (
    #     "Input two molecule SMILES (separated by ';'), returns Tanimoto similarity."
    # )
    # func_doc = ("smiles1: str, smiles2: str", "str")
    # input_output_description = "input: SMILES1;SMILES2, output: two SMILES are identical / not similar."
    # func_description = description
    # examples = [
    #     {'input': 'CCO;CCN', 'output': 'The Tanimoto similarity between CCO and CCN is 0.3333, indicating that the two molecules are not similar.'},
    #     {'input': 'CCO;CCO', 'output': 'Input Molecules Are Identical'},
    # ]
    
    description = (
        "Input molecule SMILES1 and SMILES2, returns Tanimoto similarity between two molecules."
    )
    input_argument = {'SMILES1': 'your_smiles1', 'SMILES2': 'your_smiles2'}
    examples = [
        "<tool_call>{'name': 'MolSimilarity', 'arguments': {'SMILES1':'CCO', 'SMILES2':'CCN'}</tool_call> <result> The Tanimoto similarity between CCO and CCN is 0.3333, indicating that the two molecules are not similar. </result>"
    ]

    def _run_text(self, smiles_pair: str):
        smi_list = smiles_pair.split(";")
        if len(smi_list) != 2:
            return "Query format error, please input exactly two SMILES strings separated by ';'"
        else:
            smiles1, smiles2 = smi_list
        return self._run_base(smiles1, smiles2)

    def _run_base(self, smiles1, smiles2, *args, **kwargs) -> str:
        similarity = tanimoto(smiles1, smiles2)

        if isinstance(similarity, str):
            return similarity

        sim_score = {
            0.9: "very similar",
            0.8: "similar",
            0.7: "somewhat similar",
            0.6: "not very similar",
            0: "not similar",
        }
        if similarity == 1:
            return "The input molecules are identical."
        else:
            val = sim_score[
                max(key for key in sim_score.keys() if key <= round(similarity, 1))
            ]
            message = f"The Tanimoto similarity between {smiles1} and {smiles2} is {round(similarity, 4)}, indicating that the two molecules are {val}."
        return message


class SMILES2Weight(BaseTool):
    name = "SMILES2Weight"
    func_name = 'cal_molecular_weight'
    # description = "Calculate molecular weight. Input SMILES, returns molecular weight."
    # func_doc = ("smiles: str", "str")
    # input_output_description = "input: SMILES, output: SMILES weight."
    # func_description = description
    # examples = [
    #     {'input': 'CCO', 'output': '46.041864812'},
    # ]
    
    description = (
        "Input molecule SMILES, returns the molecule weight."
    )
    input_argument = {'SMILES': 'your_smiles'}
    examples = [
        "<tool_call>{'name': 'SMILES2Weight', 'arguments': {'SMILES':'CCO'}</tool_call> <result> 46.041864812 </result>"
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg
        try: 
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ChemAgentInputError("Invalid SMILES string. Please make sure that you input a valid SMILES string, and only one at a time.")
            mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
            return str(mol_weight)
        except Exception as e:
            return f"Error calculationg molecule weight: {e}"

class FunctionalGroups(BaseTool):
    name = "FunctionalGroups"
    func_name = 'get_functional_groups'
    # description = "Get the functional groups in a molecule. Input SMILES, return list of functional groups in the molecule."
    # func_doc = ("smiles: str", "str")
    # input_output_description = "input: SMILES, output: functional groups in this SMILES."
    # func_description = description
    # examples = [
    #     {'input': 'CCO', 'output': 'This molecule contains alcohol groups, and side-chain hydroxyls.'},
    # ]
    
    description = (
        "Get the functional groups in a molecule. Input SMILES, return list of functional groups in the molecule."
    )
    input_argument = {'SMILES': 'your_smiles'}
    examples = [
        "<tool_call>{'name': 'FunctionalGroups', 'arguments': {'SMILES':'CCO'}</tool_call> <result> This molecule contains alcohol groups, and side-chain hydroxyls. </result>"
    ]

    def __init__(
        self, init=True, interface='text'
    ):
        super().__init__(init, interface)
        # 你的官能团字典
        self.dict_fgs = {
            "furan": "o1cccc1",
            "aldehydes": " [CX3H1](=O)[#6]",
            "esters": " [#6][CX3](=O)[OX2H0][#6]",
            "ketones": " [#6][CX3](=O)[#6]",
            "amides": " C(=O)-N",
            "thiol groups": " [SH]",
            "methylamide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
            "carboxylic acids": "*-C(=O)[O;D1]",
            "carbonyl methylester": "*-C(=O)[O;D2]-[C;D1;H3]",
            "terminal aldehyde": "*-C(=O)-[C;D1]",
            "amide": "*-C(=O)-[N;D1]",
            "carbonyl methyl": "*-C(=O)-[C;D1;H3]",
            "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]",
            "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]",
            "nitro": "*-[N;D3](=[O;D1])[O;D1]",
            "nitroso": "*-[N;R0]=[O;D1]",
            "oximes": "*=[N;R0]-[O;D1]",
            "Imines": "*-[N;R0]=[C;D1;H2]",
            "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]",
            "hydrazines": "*-[N;D2]=[N;D1]",
            "diazo": "*-[N;D2]#[N;D1]",
            "cyano": "*-[C;D2]#[N;D1]",
            "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
            "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
            "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]",
            "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
            "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]",
            "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]",
            "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]",
            "methyl thio": "*-[S;D2]-[C;D1;H3]",
            "methylthio": "*-[S;D2]-[C;D1;H3]", # 甲硫基
            "thiols": "*-[S;D1]",
            "thio carbonyls": "*=[S;D1]",
            "halogens": "*-[#9,#17,#35,#53]",
            "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
            "tri fluoromethyl": "*-[C;D4](F)(F)F",
            "acetylenes": "*-[C;D2]#[C;D1;H]",
            "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1",
            "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]",
            "methoxy": "*-[O;D2]-[C;D1;H3]",
            "side-chain hydroxyls": "*-[O;D1]",
            "ketones": "*=[O;D1]",
            "primary amines": "*-[N;D1]",
            "nitriles": "*#[N;D1]",
            # 新增的官能团模板
            "sulfoxide": "[S;X3](=[O])", # 亚砜
            "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])", # 酸酐
            "borane": "[BX3]", # 硼烷
            "thiol": "[SH]", # 硫醇
            "carboxyl": "C(=O)[OH]", # 羧基
            "disulfide": "[S;D2]-[S;D2]", # 二硫键
            "benzene ring": "c1ccccc1", # 苯环
            "amine": "[N;X3]", # 胺基
            "halo": "[F,Cl,Br,I]", # 卤素
            "thioether": "[S;D2]", # 硫醚
            "sulfide": "[S;D2]", # 硫化物
            "nitro": "[N+](=O)[O-]", # 硝基
            "sulfone": "[S;D4](=O)(=O)", # 砜
            "aldehyde": "[CX3H1](=O)[#6]", # 醛基
            "nitrile": "C#N", # 腈基
        }

    def _is_fg_in_mol(self, mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        if fgmol is None: return 0
        mol_obj = Chem.MolFromSmiles(mol.strip())
        if mol_obj is None: return 0
        matches = Chem.Mol.GetSubstructMatches(mol_obj, fgmol, uniquify=True)
        return len(matches)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        """
        Input a molecule SMILES or name.
        Returns a list of functional groups identified by their common name (in natural language).
        """
        smiles_valid, error_msg = judge_smiles(smiles)
        print(smiles, error_msg)
        if not smiles_valid:
            return error_msg
        
        try:
            fgs_in_molec = list()
            for name, fg in self.dict_fgs.items():
                fg_count = self._is_fg_in_mol(smiles, fg)
                if fg_count > 0:
                    fgs_in_molec.append((name, fg_count))
            
            # --- 核心修改部分 ---
            # 找到包含酮基的更复杂官能团
            complex_fg_names_with_ketone = [
                "esters", "aldehydes", "amides", "carboxylic acids", 
                "carbonyl methylester", "terminal aldehyde", "amide", 
                "carbonyl methyl"
            ]
            
            # 检查这些复杂的官能团是否在识别结果中
            has_complex_fg = any(name in [fg_info[0] for fg_info in fgs_in_molec] for name in complex_fg_names_with_ketone)
            
            # 如果存在包含酮基的复杂官能团，则从结果中移除“酮基”
            if has_complex_fg:
                fgs_in_molec = [fg_info for fg_info in fgs_in_molec if fg_info[0] != "ketones"]
            
            # --- 修改结束 ---
            
            if len(fgs_in_molec) >= 1:
                if len(fgs_in_molec) == 1:
                    return f"This molecule contains {fgs_in_molec[0][1]} {fgs_in_molec[0][0]}."
                else:
                    fg_str = ""
                    for fg_info in fgs_in_molec[:-1]:
                        fg_str += f" {fg_info[1]} {fg_info[0]},"
                    return f"This molecule contains{fg_str} and {fgs_in_molec[-1][1]} {fgs_in_molec[-1][0]}."
            else:
                return f"This molecule does not contain common functional groups."
        except Exception as e:
            return f"Error calculating functional group: {e}"

class CompareSMILES(BaseTool):
    name = "CompareSMILES"
    func_name = 'check_molecule_identical'
    # description = "Input two molecule SMILES (separated by ';'), returns if they are identical. To judge if two molecules are identical, you should always use this tool, instead of directly comparing the SMILES strings."
    # func_doc = ("smiles1: str, smiles2: str", "str")
    # input_output_description = "input: SMILES1;SMILES2, output: identical / different."
    # func_description = description
    # examples = [
    #     {'input': 'CCO;CCN', 'output': 'different'},
    #     {'input': 'OCC;CCO', 'output': 'identical'},
    # ]
    
    description = (
        "Input molecule SMILES1 and SMILES2, returns if they are identical. To judge if two molecules are identical, you should always use this tool, instead of directly comparing the SMILES strings."
    )
    input_argument = {'SMILES1': 'your_smiles1', 'SMILES2': 'your_smiles2'}
    examples = [
        "<tool_call>{'name': 'CompareSMILES', 'arguments': {'SMILES1':'CCO', 'SMILES2':'CCN'}</tool_call> <result> different </result>"
    ]

    def _run_text(self, smiles_pair: str):
        smi_list = smiles_pair.split(';')
        if len(smi_list) != 2:
            return "Query format error, please input two smiles strings separated by ';'"
        else:
            smiles1, smiles2 = smi_list
        return self._run_base(smiles1, smiles2)

    def _run_base(self, smiles1, smiles2, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(smiles1)
        if not smiles_valid:
            return error_msg
        smiles_valid, error_msg = judge_smiles(smiles2)
        if not smiles_valid:
            return error_msg
        
        try:
            smiles1 = canonicalize_molecule_smiles(smiles1)
            smiles2 = canonicalize_molecule_smiles(smiles2)
            if smiles1 is None and smiles2 is None:
                return "Invalid SMILES string, both."
            elif smiles1 is None:
                return 'Invalid SMILES string, first.'
            elif smiles2 is None:
                return 'Invalid SMILES string, second.'
            else:
                id1 = get_molecule_id(smiles1, remove_duplicate=False)
                id2 = get_molecule_id(smiles2, remove_duplicate=False)
                if id1 == id2:
                    return 'identical'
                else:
                    return 'different'
        except Exception as e:
            return f"Error calculating canonicalize and identical: {e}"


class CanonicalizeSMILES(BaseTool):
    name = "CanonicalizeSMILES"
    func_name = 'canonicalize_smiles'
    # description = "Canonicalize SMILES representation. Input SMILES, returns canonicalized SMILES. You should use this tool when asked for canonicalized SMILES."
    # func_doc = ("smiles: str", "str")
    # func_description = description
    # input_output_description = "input: SMILES, output: the canonicalize SMILES."
    # examples = [
    #     {'input': 'OCC', 'output': 'CCO'},
    # ]
    
    description = (
        "Canonicalize SMILES representation. Input SMILES, returns canonicalized SMILES. You should use this tool when asked for canonicalized SMILES."
    )
    input_argument = {'SMILES': 'your_smiles'}
    examples = [
        "<tool_call>{'name': 'CanonicalizeSMILES', 'arguments': {'SMILES':'OCC'}</tool_call> <result> CCO </result>"
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg
        
        try:
            smiles = canonicalize_molecule_smiles(smiles)
            if smiles is None:
                return "Invalid SMILES string, canonicalize."
            
            return smiles
        except Exception as e:
            return f"Error calculating canonicalizeSMILES: {e}"
    

class CountMolAtoms(BaseTool):
    name = "CountMolAtoms"
    func_name = 'count_molecule_atoms'
    # description = "Count the number of atoms in a molecule. Input SMILES, returns the types of atoms and their numbers."
    # func_doc = ("smiles: str", "str")
    # func_description = description
    # input_output_description = "input: SMILES, output: the atom number and atom type in this SMILES."
    # examples = [
    #     {'input': 'CCO', 'output': 'There are altogether 9 atoms. The types and corresponding numbers are: {"C": 2, "O": 1, "H": 6}'},
    # ]
    
    description = (
        "Count the number of atoms in a molecule. Input SMILES, returns the types of atoms and their numbers."
    )
    input_argument = {'SMILES': 'your_smiles'}
    examples = [
        "<tool_call>{'name': 'CountMolAtoms', 'arguments': {'SMILES':'CCO'}</tool_call> <result> There are altogether 9 atoms. The types and corresponding numbers are: {'C': 2, 'O': 1, 'H': 6}'} </result>"
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol_with_h = Chem.AddHs(mol)
            num_atoms = mol_with_h.GetNumAtoms()
            
            # 统计每种原子类型的数量
            atom_type_counts = {}
            for atom in mol_with_h.GetAtoms():
                symbol = atom.GetSymbol()
                atom_type_counts[symbol] = atom_type_counts.get(symbol, 0) + 1
            
            sorted_atom_counts = dict(sorted(atom_type_counts.items()))    
            text = "There are altogether %d atoms. The types and corresponding numbers are: %s" % (num_atoms, str(sorted_atom_counts))
            return text
        
        except Exception as e:
            return f"Error calculating CountMolAtoms: {e}"

class ReplaceFunctionalGroup(BaseTool):
    name = "ReplaceFunctionalGroup"
    func_name = 'replace_functional_group'
    # description = "Replace a functional group in a molecule with another functional group using functional group names from constants.py."
    # func_doc = ("base_smiles: str, old_group_name: str, new_group_name: str", "str")
    # func_description = description
    # input_output_description = "input: SMILES;old_group_name;new_group_name, output: modified SMILES"
    # examples = [
    #     {'input': 'CCO;hydroxyl;primary_amine', 'output': 'CCN'},
    #     {'input': 'CC=O aldehyde carboxyl', 'output': 'CC(=O)O'},
    # ]
    
    description = "Input: SMILES, old_group_name, new_group_name; Output: SMILES_new. Replace a functional group in a molecule with another functional group using functional group names."
    input_argument = {'SMILES': 'your_smiles', 'old_group_name': 'functional_group', 'new_group_name': 'functional_group'}
    examples = [
        "<tool_call>{'name': 'ReplaceFunctionalGroup', 'arguments': {'SMILES':'CCO', 'old_group_name':'hydroxyl', 'new_group_name':'primary_amine'}</tool_call> <result> CCN </result>"
    ]
    
    def _run_text(self, smiles_group_pair: str):
        smi_group_list = smiles_group_pair.split(';')
        if len(smi_group_list) != 3:
            return "Query format error, please input SMILES;old_group_name;new_group_name separated by ';'"
        else:
            smiles, old_group_name, new_group_name = smi_group_list
        return self._run_base(base_smiles=smiles, old_group_name=old_group_name, new_group_name=new_group_name)

    def _run_base(self, base_smiles: str, old_group_name: str, new_group_name: str, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(base_smiles)
        if not smiles_valid:
            return error_msg
        mol = Chem.MolFromSmiles(base_smiles)

        # 查找旧官能团的SMARTS模式
        old_group_name_standerd = find_functional_group_key_replace_sub(old_group_name)
        if old_group_name not in GROUP_TO_SMARTS and old_group_name_standerd is None:
            available_groups = ", ".join(list(GROUP_TO_SMARTS.keys())[:10]) + "..."
            return f"Error: Unknown old group name '{old_group_name}'. Available groups: {available_groups}"
        
        old_group_smarts = GROUP_TO_SMARTS[old_group_name_standerd]
        pattern = Chem.MolFromSmarts(old_group_smarts)
        if not pattern:
            return f"Error: Invalid SMARTS pattern for '{old_group_name}': {old_group_smarts}"

        # 查找新官能团的SMILES - 优先从ADD_SMILES查找，去掉*
        new_group_standerd = find_functional_group_key_add(function_group=new_group_name)
        if new_group_name in GROUP_TO_ADD_SMILES:
            new_group_smiles = GROUP_TO_ADD_SMILES[new_group_name].replace('*', '')
        elif new_group_standerd != None:
            new_group_smiles = GROUP_TO_ADD_SMILES[new_group_standerd].replace('*', '')
        else:
            # 一些常见的简单替换
            simple_groups = {
                'hydrogen': '[H]',
                'methyl': 'C',
                'ethyl': 'CC', 
                'hydroxyl': 'O',
                'primary_amine': 'N',
                'chloro': 'Cl',
                'bromo': 'Br',
                'fluoro': 'F',
                'iodo': 'I'
            }
            if new_group_name in simple_groups:
                new_group_smiles = simple_groups[new_group_name]
            else:
                available_groups = ", ".join(list(GROUP_TO_ADD_SMILES.keys())[:10]) + "..."
                return f"Error: Unknown new group name '{new_group_name}'. Available groups: {available_groups}"

        replacement = Chem.MolFromSmiles(new_group_smiles)
        if not replacement:
            return f"Error: Invalid SMILES for '{new_group_name}': {new_group_smiles}"

        # 执行替换
        new_mols = AllChem.ReplaceSubstructs(mol, pattern, replacement, replaceAll=True)

        if not new_mols:
            return f"Warning: No '{old_group_name}' functional group found in molecule, returning original: {base_smiles}"

        # ReplaceSubstructs 返回一个元组，我们取第一个结果
        result_mol = new_mols[0]
        try:
            Chem.SanitizeMol(result_mol)
            return f"The molecule after functional group replacement: {Chem.MolToSmiles(result_mol)}"
        except:
            return "Error: Failed to sanitize resulting molecule"


class RemoveFunctionalGroup(BaseTool):
    name = "RemoveFunctionalGroup"
    func_name = 'remove_functional_group'
    # description = "Remove a specific functional group from a molecule."
    # func_doc = ("base_smiles: str, group_name: str", "str")
    # func_description = description
    # input_output_description = "input: SMILES;removed_group_name, output: modified SMILES"
    # examples = [
    #     {'input': 'CCO;hydroxyl', 'output': 'CC'},
    #     {'input': 'CCCl;halo', 'output': 'CC'},
    # ]
    
    description = "Input: SMILES, old_group_name; Output: SMILES_new. Remove a specific functional group from a molecule."
    input_argument = {'SMILES': 'your_smiles', 'old_group_name': 'functional_group'}
    examples = [
        "<tool_call>{'name': 'RemoveFunctionalGroup', 'arguments': {'SMILES':'CCO', 'old_group_name':'hydroxyl'}</tool_call> <result> CC </result>"
    ]
    
    def _run_text(self, smiles_group_pair: str):
        smi_group_list = smiles_group_pair.split(';')
        if len(smi_group_list) != 2:
            return "Query format error, please input SMILES;removed_group_name separated by ';'"
        else:
            smiles, removed_group_name = smi_group_list
        return self._run_base(base_smiles=smiles, group_name=removed_group_name)

    def _run_base(self, base_smiles: str, group_name: str, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(base_smiles)
        if not smiles_valid:
            return error_msg
        mol = Chem.MolFromSmiles(base_smiles)

        stander_fg_name = find_functional_group_key_replace_sub(function_group=group_name)
        if stander_fg_name == None:
            available_groups = ", ".join(list(GROUP_TO_SMARTS.keys())[:10]) + "..."
            return f"Error: Unknown group name '{group_name}'. Available groups: {available_groups}"
        
        group_smarts = GROUP_TO_SMARTS[stander_fg_name]
        pattern = Chem.MolFromSmarts(group_smarts)
        if not pattern:
            return f"Error: Invalid SMARTS pattern for '{group_name}': {group_smarts}"

        try:
            # result_mol = AllChem.DeleteSubstructs(mol, pattern, onlyFrags=False)
            
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return f"Warning: No '{group_name}' functional group found in molecule, returning original: {base_smiles}"
            first_match_atoms = matches[0]
            rw_mol = Chem.RWMol(mol)
            
            for atom_idx in sorted(list(first_match_atoms), reverse=True):
                rw_mol.RemoveAtom(atom_idx)

            result_mol = rw_mol.GetMol()
            Chem.SanitizeMol(result_mol)
            result_smiles = Chem.MolToSmiles(result_mol)
            if result_smiles == base_smiles:
                return f"Warning: No '{group_name}' functional group found in molecule, returning original: {base_smiles}"
            return f"The molecule after removing functional group: {result_smiles}"
        except Exception as e:
            return f"Error: Failed to remove functional group or sanitize resulting molecule, catching: {e}"


class AddFunctionalGroup(BaseTool):
    name = "AddFunctionalGroup"
    func_name = 'add_functional_group'
    description = "Add a functional group to a molecule."
    # func_doc = ("base_smiles: str, group_name: str", "str")
    # func_description = description
    # input_output_description = "input: SMILES;add_group_name, output: modified SMILES"
    # examples = [
    #     {'input': 'CCC;carboxyl', 'output': 'C(C(=O)O)CC'},
    #     {'input': 'CCC;hydroxyl', 'output': 'CCO'},
    # ]
    
    description = "Input: SMILES, new_group_name; Output: SMILES_new. Add a functional group to a molecule."
    input_argument = {'SMILES': 'your_smiles', 'new_group_name': 'functional_group'}
    examples = [
        "<tool_call>{'name': 'AddFunctionalGroup', 'arguments': {'SMILES':'CCC', 'new_group_name':'carboxyl'}</tool_call> <result> C(C(=O)O)CC </result>"
    ]
    
    def _run_text(self, smiles_group_pair: str):
        smi_group_list = smiles_group_pair.split(';')
        if len(smi_group_list) != 2:
            return "Query format error, please input SMILES;add_group_name separated by ';'"
        else:
            smiles, add_group_name = smi_group_list
        return self._run_base(base_smiles=smiles, group_name=add_group_name)

    def _run_base(self, base_smiles: str, group_name: str, atom_index_to_attach=0, *args, **kwargs) -> str:
        smiles_valid, error_msg = judge_smiles(base_smiles)
        if not smiles_valid:
            return error_msg
        mol = Chem.MolFromSmiles(base_smiles)

        if atom_index_to_attach >= mol.GetNumAtoms():
            return f"Error: Atom index {atom_index_to_attach} out of range (molecule has {mol.GetNumAtoms()} atoms)"

        stander_fg_name = find_functional_group_key_add(function_group=group_name)
        if stander_fg_name == None:
            available_groups = ", ".join(list(GROUP_TO_ADD_SMILES.keys())[:10]) + "..."
            return f"Error: Unknown group name '{group_name}'. Available groups: {available_groups}"
        
        group_to_add_smiles = GROUP_TO_ADD_SMILES[stander_fg_name]

        # 检查目标原子是否有可用的氢原子
        target_atom = mol.GetAtomWithIdx(atom_index_to_attach)
        if target_atom.GetTotalNumHs() == 0:
            return f"Error: Atom at index {atom_index_to_attach} has no hydrogen atoms to replace"

        # 使用EditableMol来直接编辑分子结构
        editable_mol = Chem.EditableMol(mol)
        
        # 解析要添加的官能团，去掉连接点标记(*)
        group_smiles_clean = group_to_add_smiles.replace('*', '')
        group_mol = Chem.MolFromSmiles(group_smiles_clean)
        if not group_mol:
            return f"Error: Invalid SMILES for functional group '{group_name}': {group_smiles_clean}"

        # 获取原始分子的原子数，用于后续的原子索引计算
        original_atom_count = mol.GetNumAtoms()
        
        # 添加官能团的所有原子到分子中
        atom_index_map = {}
        for atom in group_mol.GetAtoms():
            new_atom_idx = editable_mol.AddAtom(atom)
            atom_index_map[atom.GetIdx()] = new_atom_idx
            
        # 添加官能团内部的键
        for bond in group_mol.GetBonds():
            begin_idx = atom_index_map[bond.GetBeginAtomIdx()]
            end_idx = atom_index_map[bond.GetEndAtomIdx()]
            editable_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
            
        # 连接官能团的第一个原子到目标原子
        if len(atom_index_map) > 0:
            first_group_atom_idx = atom_index_map[0]  # 官能团的第一个原子
            editable_mol.AddBond(atom_index_to_attach, first_group_atom_idx, Chem.BondType.SINGLE)
        
        try:
            result_mol = editable_mol.GetMol()
            Chem.SanitizeMol(result_mol)
            result_smiles = Chem.MolToSmiles(result_mol)
            return f"The molecule after adding functional group: {result_smiles}"
        except Exception as e:
            return f"Error: Failed to create or sanitize resulting molecule: {str(e)}"

class GetRXNTemplate(BaseTool):
    name = "GetRXNTemplate"
    func_name = 'get_reaction_templates'
    description = "Input the SMILES as reactants or products, search the reaction-template database and return similar reaction templates."
    input_argument = {'SMILES': 'Your_SMILES', 'Type': 'reactants / products'}
    examples = [
        "<tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'CCO', 'Type': 'reactants'}}\n</tool_call> <result> the similar reaction template is [O:1][C:2]>>[O:1]=[C:2] </result>"
    ]
    
    def _run_text(self, smiles_type_pair: str):
        smi_type_list = smiles_type_pair.split(';')
        if len(smi_type_list) != 2:
            return "Query format error, please input SMILES;type separated by ';' type must be 'reactants' or 'products'"
        else:
            smiles, query_type = smi_type_list
        return self._run_base(base_smiles=smiles, query_type=query_type)
    
    # def _run_base(self, base_smiles: str, query_type: str, *args, **kwargs) -> str:
    #     smiles_valid, error_msg = judge_smiles(base_smiles)
    #     if not smiles_valid:
    #         return error_msg
    #     if query_type not in ['reactants', 'products']:
    #         return "Query format error, the type must be 'reactants' or 'products'"
    #     mol = Chem.MolFromSmiles(base_smiles)
        
    #     # Start Retrieving from RXN Templates
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     with open(os.path.join(current_dir, "templates.pkl"), 'rb') as f:  
    #         rxn_templates = pickle.load(f)
        
    #     templates_lst = []
    #     for template in rxn_templates:
    #         reactant_mol_tpl = Chem.MolFromSmarts(template.split('>>')[0])
    #         product_mol_tpl = Chem.MolFromSmarts(template.split('>>')[1])
    #         templates_lst.append({'reactants':reactant_mol_tpl, 'products':product_mol_tpl, 'template':template})
        
    #     matched_templates = []
    #     for template in templates_lst:
    #         try:
    #             if mol.HasSubstructMatch(template[query_type]):
    #                 matched_templates.append(template['template'])
    #         except: pass

    #     if len(matched_templates) == 0:
    #         return f"Cannot find suitable reaction templates for {base_smiles} as {query_type}"
    #     else:
    #         total_templates = ", ".join(matched_templates)
    #         return f"For {base_smiles} as {query_type}, we find {total_templates} as potential reaction templates"
    
    def _run_base(self, base_smiles: str, query_type: str, *args, **kwargs) -> str:
        ## 更新, 面对SMILES1;SMILES2这种输入, 也能多SMILES匹配模板
        # 1. 验证 SMILES 有效性
        smiles_valid, error_msg = judge_smiles(base_smiles)
        if not smiles_valid:
            return error_msg
        if query_type not in ['reactants', 'products']:
            return "Query format error, the type must be 'reactants' or 'products'"

        # 2. 处理多分子输入：将 base_smiles 拆分为 Mol 列表
        # 即使只有一个分子，也统一处理为列表 [mol]
        input_mols = [Chem.MolFromSmiles(s.strip()) for s in base_smiles.split(';')]
        if None in input_mols:
            return "Error: One or more SMILES strings could not be parsed."

        # 3. 加载模板库（建议在实际应用中将加载移出循环以提高效率）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "templates.pkl"), 'rb') as f:  
            rxn_templates = pickle.load(f)
        
        matched_templates = []

        # 4. 遍历模板进行匹配
        for template_str in rxn_templates:
            try:
                # 提取对应类型的 SMARTS 部分
                side_smarts = template_str.split('>>')[0] if query_type == 'reactants' else template_str.split('>>')[1]
                
                # 将模板按 '.' 拆分为多个组分
                # 示例："(A).(B)" -> ["(A)", "(B)"]
                template_components = [Chem.MolFromSmarts(s.strip('()')) for s in side_smarts.split('.')]
                
                if not template_components:
                    continue

                # 核心逻辑：判断模板的每一个组分是否都能在输入的 mols 中找到匹配
                all_component_matched = True
                for t_comp in template_components:
                    # 只要模板中有一个组分在所有输入分子中都找不着，该模板就失效
                    if not any(m.HasSubstructMatch(t_comp) for m in input_mols):
                        all_component_matched = False
                        break
                
                if all_component_matched:
                    matched_templates.append(template_str)

            except Exception:
                continue

        # 5. 返回结果
        if len(matched_templates) == 0:
            return f"Cannot find suitable reaction templates for {base_smiles} as {query_type}"
        else:
            total_templates = ", ".join(matched_templates)
            return f"For {base_smiles} as {query_type}, we find {total_templates} as potential reaction templates"
            

if __name__ == "__main__":
    # my_CountMolAtoms = CountMolAtoms()
    # print(my_CountMolAtoms._run_base(smiles="CCOC(=O)C(C(=O)OCC)C(CC(=O)c1ccc(N(C)C)cc1)Cc1ccc([N+](=O)[O-])cc1"))
    
    # my_FunctionalGroup = FuncGroups()
    # print(my_FunctionalGroup._run_base(smiles="CCOC(=O)C(C(=O)OCC)C(CC(=O)c1ccc(N(C)C)cc1)Cc1ccc([N+](=O)[O-])cc1"))
    
    # Initialize tools
    replace_tool = ReplaceFunctionalGroup()
    remove_tool = RemoveFunctionalGroup()
    add_tool = AddFunctionalGroup()
    rxn_template_tool = GetRXNTemplate()

    # # Replace hydroxyl with amine in ethanol
    # result = replace_tool._run_base('CCO', 'hydroxyl', 'primary_amine')
    # print(result)  # Output: CCN

    # # Remove hydroxyl from ethanol
    # result = remove_tool._run_base('CCO', 'hydroxyl') 
    # print(result)  # Output: CC

    # # Add carboxyl group to propane at position 0
    # result = add_tool._run_base('CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl', 'carboxyl.')
    # print(result)  # Output: C(C(=O)O)CC
    
    # result = remove_tool._run_base('CCC(NN)c1cc(C)c(F)cc1F', 'amine group')
    # print(result)
    
    # result = replace_tool._run_base("Cc1cc(C=C2C(=O)NC(=O)N(c3cccc(Cl)c3)C2=O)c(C)n1-c1ccccc1C(=O)[O-]", old_group_name="halo group", new_group_name="carboxyl group")
    # print(result)
    
    reactants = ["CCO", "CCC=O", "CCN"]
    for smiles in reactants:
        result = rxn_template_tool._run_text(smiles_type_pair=f"{smiles};reactants")
        print(result)
        print("#   "*20)