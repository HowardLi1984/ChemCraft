import json

def load_config():
    from omegaconf import OmegaConf
    searchr1_config_file = "/cto_labs/lihao/chem_reason/ChemSearch/verl/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"
    
    searchr1_tools_config = OmegaConf.load(searchr1_config_file)
    print(searchr1_tools_config)
    
    chemagent_config_file = "/cto_labs/lihao/chem_reason/ChemSearch/verl/examples/sglang_multiturn/config/tool_config/chem_tool_config.yaml"
    
    chemagent_tools_config = OmegaConf.load(chemagent_config_file)
    print(chemagent_tools_config)

def parser_tools():
    import ast
    import re
    content = 'I am Lihao, <tool_call>\n{"name": "search", "arguments": {"query_list": ["ZIP code 38834 North Mississippi"]}}\n</tool_call>'
    # action = json.loads(content)
    # print(action)
    
    bot_token = "<tool_call>\n"
    eot_token = "\n</tool_call>"
    idx = content.find(eot_token)
    normal_text = content[:idx].strip()+eot_token if idx != -1 else content
    
    print(normal_text)
    
    match_result = r'{"name":"FunctionalGroups", "arguments":{"SMILES": "CCc1ccc(/C=C2\NC(=O)NC2=O)cc1"}}'
    match_result = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match_result)
    match_result = match_result.replace("'", '"') # change ' to "
    match_result = match_result.replace('}}}', '}}')
    parsed_call = ast.literal_eval(match_result.strip())
    print(parsed_call)
    
    action = json.loads(match_result)
    print(action)
    
    kk = json.dumps(action)
    print(kk)

def working_function(num):
    return num*num

def multiprocess_test():
    import multiprocessing as mp
    
    with mp.Pool(processes=4) as pool:
        results = pool.map(working_function, range(10))
        print(results)

def load_yaml():
    from omegaconf import OmegaConf
    tools_config_file = "/cto_labs/lihao/chem_reason/ChemSearch/verl/examples/sglang_multiturn/config/tool_config/chem_tool_config_v3.yaml"
    tools_config = OmegaConf.load(tools_config_file)
    print(tools_config)

class MolSimilarity():
    def __init__(self) -> None:
        self.name = self.__class__.__name__
    
    def print(self):
        print(self.name)

from rdkit.rdBase import DisableLog, WrapLogs
import contextlib, io
        
def judge_smiles(smiles: str):
    WrapLogs()
    f = io.StringIO()
    try: 
        with contextlib.redirect_stderr(f):
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Capturing info: {f.getvalue()}"
        else:
            return True, None
    except Exception as e:
        return False, f"Python Exception: {str(e)}"
       
def is_smiles():
    import logging
    from io import StringIO
    import rdkit.Chem as Chem
    from rdkit import RDLogger
    from rdkit import rdBase
    
    import sys
    from contextlib import redirect_stderr, redirect_stdout # 关键导入：用于捕获 C 级别的 stderr

    smiles1 = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    is_valid, smiles_fail_reminder = judge_smiles(smiles=smiles1)
    print(is_valid, smiles_fail_reminder)
    
    smiles1 = "CN1C=NC2=C1C(=O)N(C(=O)N2CC"
    is_valid, smiles_fail_reminder = judge_smiles(smiles=smiles1)
    print(is_valid, smiles_fail_reminder)


from examples.sglang_multiturn.qwen_chem_agent.local_retriever.base_agent import BaseTool
import rdkit.Chem as Chem
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

def test_functional_group():
    smiles1 = "CN1C=NC2=C1C(=O)N(C(=O)N2CC"
    
    my_func = FunctionalGroups()
    my_func._run_base(smiles=smiles1)

import re
def format_acc_calculation(solution_str):
    # Evaluate the format score from solution_str
    # (1) Tool_Call_Format:    <tool_call>\n{'name':'', 'arguments':{}}\n</tool_call>
    # (2) Final_Answer_Format: Output: <answer>  </answer>
    format_score = 1.0
    
    tool_call_blocks = re.findall(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
    total_tool_calls = len(tool_call_blocks)
    
    tool_call_pattern = r"<tool_call>\n\{('name':\s*'.*?',\s*'arguments':\s*\{.*?\})|(\"name\":\s*\".*?\",\s*\"arguments\":\s*\{.*?\})\}\n</tool_call>"
    correct_tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
    format_score -= 0.4*(total_tool_calls - len(correct_tool_calls))
    print(tool_call_blocks, len(tool_call_blocks))
    print(correct_tool_calls, len(correct_tool_calls))
    answer_pattern = r"Output:\s*<answer>.*?</answer>"
    answer_matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    
    if len(answer_matches) != 1: # only allow one answer
        format_score -= 0.8
    
    if format_score < 0.0: 
        return 0.0
    else:
        return format_score

def test_evaluate_format():
    """测试函数"""
    test_cases = [
        """'input_structure: OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1. <tool_call>\n{\'name\':\'FunctionalGroups\', \'arguments\':{\'SMILES\': \'OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1\'}}\n</tool_call>\n<tool_call>\n{"name": "FunctionalGroups", "arguments": {"SMILES": "OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "This molecule contains 1 side-chain hydroxyls, 2 benzene ring, 1 amine, 1 thioether, and 1 sulfide."}\n</tool_response>\nassistant\nThe tool confirms the presence of key functional groups: two benzene rings, a thioether (S in CCSCC1), a sulfur atom in the thioether backbone, an aliphatic amine (CNC...), and a hydroxyl group attached to the aromatic ring. This aligns with structural features visible in the SMILES string.\n\n<tool_call>\n{\'name\':\'CountMolAtoms\', \'arguments\':{\'SMILES\': \'OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1\'}}\n</tool_call>\n<tool_call>\n{"name": "CountMolAtoms", "arguments": {"SMILES": "OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "There are altogether 49 atoms. The types and corresponding numbers are: {\'C\': 20, \'H\': 25, \'N\': 1, \'O\': 2, \'S\': 1}"}\n</tool_response>\nassistant\nThe atom count supports the molecular formula C₂₀H₂₅NO₂S. However, the result can be slightly interpreted: one oxygen is in the -OH group on the benzene ring, another in the thioether (SO), and one nitrogen is in the aliphatic secondary amine (CNC). This composition is consistent with the functional groups identified.\n\n<tool_call>\n{\'name\':\'RemoveFunctionalGroup\', \'arguments\':{\'SMILES\': \'OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1\', \'old_group_name\': \'hydroxyl\'}}\n</tool_call>\n<tool_call>\n{"name": "RemoveFunctionalGroup", "arguments": {"SMILES": "OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1", "old_group_name": "hydroxyl"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "The molecule after removing functional group: c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1"}\n</tool_response>\nassistant\nRemoving the hydroxyl group (-OH) yields **c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1**, indicating that the OH was on the aromatic ring (c1cccc(OC...)), simplifying the structure by deleting this labile side chain. The remaining scaffold still retains the benzene ring and the linked heterocyclic rings.\n\n<tool_call>\n{\'name\':\'RemoveFunctionalGroup\', \'arguments\':{\'SMILES\': \'c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1\', \'old_group_name\': \'secondary_amine\'}}\n</tool_call>\n<tool_call>\n{"name": "RemoveFunctionalGroup", "arguments": {"SMILES": "c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1", "old_group_name": "secondary_amine"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "The molecule after removing functional group: C1CCSCC1.CCc1cccc(OCc2ccccc2)c1"}\n</tool_response>\nassistant\nRemoving the secondary amine (N in CNC) leaves two separate substructures: **C1CCSCC1** (a thioether ring) and **.Cc1cccc(OCc2ccccc2)c1** (an aromatic ring with alkoxy substituent). However, this suggests the break between rings, which contradicts the definition of the Murcko scaffold.\n\n<tool_call>\n{\'name\':\'RemoveFunctionalGroup\', \'arguments\':{\'SMILES\': \'c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1\', \'old_group_name\': \'thioether\'}}\n</tool_call>\n<tool_call>\n{"name": "RemoveFunctionalGroup", "arguments": {"SMILES": "c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1", "old_group_name": "thioether"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "The molecule after removing functional group: CC(C)NCCc1cccc(OCc2ccccc2)c1"}\n</tool_response>\nassistant\nRemoving the thioether group (CSCC1) from the thienopyrrole-like core (CNC1CCSCC1) simplifies it to **CC(C)NCC**, but the context suggests keeping ring systems and linking bonds. This result appears incomplete or interpreted incorrectly by the tool.\n\n<tool_call>\n{\'name\':\'RemoveFunctionalGroup\', \'arguments\':{\'SMILES\': \'c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1\', \'old_group_name\': \'benzene_ring\'}}\n</tool_call>\n<tool_call>\n{"name": "RemoveFunctionalGroup", "arguments": {"SMILES": "c1ccc(COc2cccc(CCNC3CCSCC3)c2)cc1", "old_group_name": "benzene_ring"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "The molecule after removing functional group: COc1cccc(CCNC2CCSCC2)c1"}\n</tool_response>\nassistant\nRemoving one benzene ring (the one with the hydroxyl) leaves **COc1cccc(CCNC2CCSCC2)c1**: this suggests that the remaining scaffold consists of the other benzene ring (c1cccc(OC...) — the one looped back with the thieno-dioxole) and the pyrrole-containing ring formed by CCNC3CCSCC3 (a benzofuran-like ring with a pyrrole-oxazole fusion). The carbonyl (CO) now appears as a bridge.\n\n<tool_call>\n{\'name\':\'MurckoScaffold\', \'arguments\':{\'SMILES\': \'OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1\'}}\n</tool_call>\n<tool_call>\n{"name": "FailFunctionName", "arguments": {"SMILES": "OC(CNC1CCSCC1)c1cccc(OCc2ccccc2)c1"}}\n</tool_call>\nuser\n<tool_response>\n{"result": "Tool Name Error. DO NOT have this tool. Try again."}\n</tool_response>\nassistant\nThe tool \'MurckoScaffold\' does not exist in the available library, so we must infer the scaffold manually rather than relying on automated tool output. Based on the functional group analysis and scaffold removal steps, the intact ring systems and their interconnections must be preserved. The molecule contains:\n- A benzene ring (with OH and alkyl chain)\n- A thieno-dioxole ring (with CNC linkage)\n- A benzene ring (with O substituent and pyrrole ring)\n\nWhen removing side chains and functional groups, we are left with:\n- The core benzene ring (c1cccc(OC...))\n- The thieno-dioxole ring (CNC1CCSCC1)\n- The bridging structure formed by the carbonyl and the C atom between the two rings\n\nThe correct Murcko scaffold should therefore consist of the two benzene rings (linked via the bridge) and the heterocyclic rings themselves. The resulting scaffold is the connectivity between the two aromatic systems via the connecting bond, forming a fused bicyclic system.\n\nHowever, in SMILES representation, the scaffold is often simplified to ring closures and the bridge atoms. By piecing together structural analysis and tool-derived results, the expected scaffold SMILES is likely **c1ccccc1-C1CCCCC1**, but with appropriate ring closures and the bridge.\n\nThus, the correct scaffold is the connectivity between the two aromatic systems via the connecting bond: **c1ccccc1-C2CCCCC2**.\n\nOutput: <answer> c1ccccc1-C2CCCCC2 </answer>'
        """,
        
        # 多个tool_call，其中一个格式错误
        """Some text here.
<tool_call>
{'name':'Tool1','arguments':{'param':'value1'}}
</tool_call>
<tool_call>{'name':'Tool2','arguments':{'param':'value2'}}</tool_call>
Output: <answer> result </answer>""",
        
        # 缺少工具调用
        """Some text here.
Output: <answer> result </answer>""",
        
        # 工具调用格式完全错误
        """Some text here.
<tool_call>invalid format</tool_call>
Output: <answer> result </answer>""",
        
        # 缺少最终结果
        """Some text here.
<tool_call>
{'name':'test','arguments':{}}
</tool_call>"""
    ]
    
    for i, text in enumerate(test_cases):
        print(i, format_acc_calculation(solution_str=text))

import numpy as np
def scale_calculate():
    current_batch_step = np.array([301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301, 301,
       301, 301, 301, 301])
    step = current_batch_step.tolist()[0] - 300
    
    correct_max_possible = 3.0; correct_min_possible = -3.0
    format_max_possible = 1.0; format_min_possible = -1.0
    
    # format upper and lower scaling:
    format_max_reward = 2 - (2 - format_max_possible) * step / 150
    format_min_reward = -2 + (2 + format_min_possible) * step / 150
    if format_max_reward < 1.0:
        format_max_reward = 1.0
    if format_min_reward > -1.0:
        format_min_reward = -1.0
    

if __name__ == "__main__":
    # load_config()
    # parser_tools()

    # load_yaml()
    # my_class = MolSimilarity(); my_class.print()
    
    # is_smiles()
    # test_functional_group()
    
    test_evaluate_format()
    
    # scale_calculate()
    
    
