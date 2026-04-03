
import tdc
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

import sys
sys.path.append("/cto_labs/lihao/chem_reason/ChemSearch/search_r1/search_r1/llm_agent")
from chem_utils.error import *
from base_agent import BaseTool
from chem_utils import is_smiles

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
        
        
class QEDPropertyPred(BaseTool):
    name = "QEDPropertyPred"
    func_name = "qed_molecule_property_predictor"
    description = "Predict the QED property of a molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'smiles', 'output': "The QED property of SMILES is xxx."}
    ]
    def __init__(self, init=True, interface='text') -> None:
        super().__init__(init, interface)
        self.oracle = tdc.Oracle(name="qed")
    
    def _run_text(self, smiles: str) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg

        try:
            prop_predict = self.oracle(smiles)
            prop_predict = round(prop_predict, 4)
            return f"The QED property of {smiles} is {prop_predict}."
        except Exception as e:
            return f"Error calculating qed: {e}"
    
    def _run_base(self) -> str:
        return "departure"

class DRD2PropertyPred(BaseTool):
    name = "DRD2PropertyPred"
    func_name = "drd2_molecule_property_predictor"
    description = "Predict the DRD2 property of a molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'smiles', 'output': "The DRD2 property of SMILES is xxx."}
    ]
    def __init__(self, init=True, interface='text') -> None:
        super().__init__(init, interface)
        self.oracle = tdc.Oracle(name="drd2")
    
    def _run_text(self, smiles: str) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg

        try:
            prop_predict = self.oracle(smiles)
            prop_predict = round(prop_predict, 4)
            return f"The DRD2 property of {smiles} is {prop_predict}."
        except Exception as e:
            return f"Error calculating DRD2: {e}"
    
    def _run_base(self) -> str:
        return "departure"
    

class JNK3PropertyPred(BaseTool):
    name = "JNK3PropertyPred"
    func_name = "jnk3_molecule_property_predictor"
    description = "Predict the JNK3 property of a molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'smiles', 'output': "The JNK3 property of SMILES is xxx."}
    ]
    def __init__(self, init=True, interface='text') -> None:
        super().__init__(init, interface)
        self.oracle = tdc.Oracle(name="jnk3")
    
    def _run_text(self, smiles: str) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg
    
        try:
            prop_predict = self.oracle(smiles)
            prop_predict = round(prop_predict, 4)
            return f"The JNK3 property of {smiles} is {prop_predict}."
        except Exception as e:
            return f"Error calculating JNK3: {e}"
    
    def _run_base(self) -> str:
        return "departure"
    

class LogPPropertyPred(BaseTool):
    name = "LogPPropertyPred"
    func_name = "LogP_molecule_property_predictor"
    description = "Predict the LogP property of a molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'smiles', 'output': "The LogP property of SMILES is xxx."}
    ]
    def __init__(self, init=True, interface='text') -> None:
        super().__init__(init, interface)
        self.oracle = tdc.Oracle(name="logp")
    
    def _run_text(self, smiles: str) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg

        try:
            prop_predict = self.oracle(smiles)
            prop_predict = round(prop_predict, 4)
            return f"The LogP property of {smiles} is {prop_predict}."
        except Exception as e:
            return f"Error calculating LogP: {e}"
    
    def _run_base(self) -> str:
        return "departure"


class GSKPropertyPred(BaseTool):
    name = "GSKPropertyPred"
    func_name = "gsk3beta_molecule_property_predictor"
    description = "Predict the gsk3-beta property of a molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'smiles', 'output': "The gsk3-beta property of SMILES is xxx."}
    ]
    def __init__(self, init=True, interface='text') -> None:
        super().__init__(init, interface)
        self.oracle = tdc.Oracle(name="gsk3b")
    
    def _run_text(self, smiles: str) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg

        try:
            prop_predict = self.oracle(smiles)
            prop_predict = round(prop_predict, 4)
            return f"The gsk3-beta property of {smiles} is {prop_predict}."
        except Exception as e:
            return f"Error calculating GSK-3-beta: {e}"
    
    def _run_base(self) -> str:
        return "departure"


class SolubilityPropertyPred(BaseTool):
    name = "SolubilityPropertyPred"
    func_name = "solubility_molecule_property_predictor"
    description = "Predict the solubility property of a molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'smiles', 'output': "The solubility property of SMILES is xxx."}
    ]
    def __init__(self, init=True, interface='text') -> None:
        super().__init__(init, interface)
    
    def _run_text(self, smiles: str) -> str:
        smiles_valid, error_msg = judge_smiles(smiles)
        if not smiles_valid:
            return error_msg

        try:
            mol = Chem.MolFromSmiles(smiles)
            # Calculate relevant descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_bond_donors = Descriptors.NumHDonors(mol)
            h_bond_acceptors = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            # Simple linear model (based on published QSPR models)
            logS = 0.16 - 0.63*logp - 0.0062*mw + 0.066*h_bond_donors - 0.074*h_bond_acceptors
            
            prop_predict = round(logS, 4)
            return f"The solubility property of {smiles} is {prop_predict}."
        except Exception as e:
            return f"Error calculating solubility: {e}"
    
    def _run_base(self) -> str:
        return "departure"