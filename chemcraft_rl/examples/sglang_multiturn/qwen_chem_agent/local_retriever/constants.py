GROUP_TO_SMARTS = {
    # 芳香环类
    "benzene_ring": "c1ccccc1",                          # 苯环
    "pyridine": "n1ccccc1",                              # 吡啶环
    "pyrrole": "[nH]1cccc1",                             # 吡咯环
    "furan": "o1cccc1",                                  # 呋喃环
    # 饱和杂环类
    "piperidine": "N1CCCCC1",                           # 哌啶环
    
    # 含氧官能团
    "hydroxyl": "[OX2H]",                                # 羟基(包括酚羟基和醇羟基)
    "aldehyde": "[CX3H1](=O)",                           # 醛基
    "ketone": "[#6][CX3](=O)[#6]",                       # 酮基
    "carboxyl": "[CX3](=O)[OX2H1]",                      # 羧基
    "ester": "[#6][CX3](=O)[OX2H0][#6]",                 # 酯基(排除酸酐)
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",      # 酸酐
    "ether": "[OD2]([#6])[#6]",                          # 醚键
    "acyl_chloride": "[CX3](=[OX1])[Cl]",                # 酰氯
    "hemiacetal": "[CX4]([OX2H])([OX2])[#6]",            # 半缩醛
    "acetal": "[CX4]([OX2])([OX2])([#6])[#6]",           # 缩醛
    
    # 含氮官能团
    "primary_amine": "[NX3;H2;!$(NC=O)]",                # 伯胺(排除酰胺)
    "secondary_amine": "[NX3;H1;!$(NC=O)]",              # 仲胺(排除酰胺)
    "tertiary_amine": "[NX3;H0;!$(NC=O)]",               # 叔胺(排除酰胺)
    "amide": "[NX3][CX3](=[OX1])[#6]",                   # 酰胺
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",  # 硝基
    "imine": "[NX2]=[CX3]",                              # 亚胺/希夫碱
    "oxime": "[NX2]=[CX3][#6]",                          # 肟基
    "hydrazone": "[NX2]=[CX3]",                          # 腙基
    "isocyanate": "[NX2]=[CX2]=[OX1]",                   # 异氰酸酯
    
    # 卤素
    "halo": "[F,Cl,Br,I]",                               # 卤素
    
    # 含硫官能团
    "thiol": "[#16X2H]",                                 # 巯基
    "thioether": "[SX2]([CX4])[CX4]",                    # 硫醚
    "disulfide": "[#16X2H0][#16X2H0]",                   # 二硫键
    "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]", # 亚砜
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # 砜
    
    # 含磷官能团
    "phosphine": "[PX3]",                                # 膦基
    "phosphate": "[PX4](=[OX1])([OX2])([OX2])[OX2]",     # 磷酸基团
    "phosphonate": "[PX4](=[OX1])([OX2])([#6])[OX2]",    # 膦酸酯
    
    # 其他
    "nitrile": "[NX1]#[CX2]",                            # 氰基
    "borane": "[BX3]",                                   # 硼烷基
}

# Mapping of functional group names to SMILES for adding (with * as connection point)
# This allows AddFunctionalGroup to use the same group names as constants
GROUP_TO_ADD_SMILES = {
    # 芳香环类 (simplified versions for addition)
    "benzene_ring": "*c1ccccc1",                         # 苯环
    "pyridine": "*c1ccncc1",                             # 吡啶环
    "pyrrole": "*c1cc[nH]c1",                            # 吡咯环
    "furan": "*c1ccoc1",                                 # 呋喃环
    # 饱和杂环类
    "piperidine": "*C1CCNCC1",                           # 哌啶环
    
    # 含氧官能团
    "hydroxyl": "*O",                                    # 羟基
    "aldehyde": "*C=O",                                  # 醛基  
    "ketone": "*C(=O)C",                                 # 酮基（甲基酮）
    "carboxyl": "*C(=O)O",                               # 羧基
    "ester": "*C(=O)OC",                                 # 酯基（甲酯）
    "anhydride": "*C(=O)OC(=O)C",                        # 酸酐
    "ether": "*OC",                                      # 醚键（甲醚）
    "acyl_chloride": "*C(=O)Cl",                         # 酰氯
    "hemiacetal": "*C(O)OC",                             # 半缩醛
    "acetal": "*C(OC)OC",                                # 缩醛
    
    # 含氮官能团
    "primary_amine": "*N",                               # 伯胺
    "secondary_amine": "*NC",                            # 仲胺（N-甲基）
    "tertiary_amine": "*N(C)C",                          # 叔胺（N,N-二甲基）
    "amide": "*C(=O)N",                                  # 酰胺
    "nitro": "*[N+](=O)[O-]",                            # 硝基
    "imine": "*C=N",                                     # 亚胺
    "oxime": "*C=NO",                                    # 肟基
    "hydrazone": "*C=NN",                                # 腙基
    "isocyanate": "*N=C=O",                              # 异氰酸酯
    
    # 卤素
    "halo": "*Cl",                                       # 卤素（氯）
    "chloro": "*Cl",                                     # 氯
    "bromo": "*Br",                                      # 溴
    "fluoro": "*F",                                      # 氟
    "iodo": "*I",                                        # 碘
    
    # 含硫官能团
    "thiol": "*S",                                       # 巯基
    "thioether": "*SC",                                  # 硫醚（甲硫醚）
    "disulfide": "*SSC",                                 # 二硫键
    "sulfoxide": "*S(=O)C",                              # 亚砜
    "sulfone": "*S(=O)(=O)C",                            # 砜
    
    # 含磷官能团
    "phosphine": "*P(C)C",                               # 膦基
    "phosphate": "*P(=O)(O)(O)O",                        # 磷酸基
    "phosphonate": "*P(=O)(OC)OC",                       # 膦酸酯
    
    # 其他
    "nitrile": "*C#N",                                   # 氰基
    "borane": "*B(C)C",                                  # 硼烷基
    
    # 常用烷基
    "methyl": "*C",                                      # 甲基
    "ethyl": "*CC",                                      # 乙基
    "propyl": "*CCC",                                    # 丙基
    "isopropyl": "*C(C)C",                               # 异丙基
    "butyl": "*CCCC",                                    # 丁基
    "tert_butyl": "*C(C)(C)C",                           # 叔丁基
}

def validate_group_name(group_name: str, operation: str = "any") -> bool:
    """
    验证官能团名称是否存在于定义的字典中
    
    Args:
        group_name (str): 要验证的官能团名称
        operation (str): 操作类型 "remove"/"replace"/"add"/"any"
        
    Returns:
        bool: 如果官能团名称有效返回True，否则返回False
    """
    if operation in ["remove", "replace", "any"]:
        if group_name in GROUP_TO_SMARTS:
            return True
    
    if operation in ["add", "any"]:
        if group_name in GROUP_TO_ADD_SMILES:
            return True
            
    return False

def get_available_groups(operation: str = "any") -> list:
    """
    获取可用的官能团名称列表
    
    Args:
        operation (str): 操作类型 "remove"/"replace"/"add"/"any"
        
    Returns:
        list: 可用的官能团名称列表
    """
    available_groups = set()
    
    if operation in ["remove", "replace", "any"]:
        available_groups.update(GROUP_TO_SMARTS.keys())
    
    if operation in ["add", "any"]:
        available_groups.update(GROUP_TO_ADD_SMILES.keys())
        
    return sorted(list(available_groups))

def get_group_smarts(group_name: str) -> str:
    """
    根据官能团名称获取对应的SMARTS模式
    
    Args:
        group_name (str): 官能团名称
        
    Returns:
        str: SMARTS模式, 如果不存在返回None
    """
    return GROUP_TO_SMARTS.get(group_name)

def get_group_add_smiles(group_name: str) -> str:
    """
    根据官能团名称获取用于添加的SMILES字符串
    
    Args:
        group_name (str): 官能团名称
        
    Returns:
        str: 添加用的SMILES字符串，如果不存在返回None
    """
    return GROUP_TO_ADD_SMILES.get(group_name)

def find_functional_group_key_add(function_group: str) -> str | None:
    normalized_input = function_group.lower().strip()

    normalized_input = normalized_input.replace(".", "")
    normalized_input = normalized_input.replace(" group", "")
    normalized_input = normalized_input.replace(" ring", "")
    # normalized_input = normalized_input.replace("s", "")  # 处理复数形式

    if normalized_input in GROUP_TO_ADD_SMILES.keys():
        return normalized_input

    for key in GROUP_TO_ADD_SMILES.keys():
        if normalized_input in key:
            return key

    return None

def find_functional_group_key_replace_sub(function_group: str) -> str | None:
    normalized_input = function_group.lower().strip()

    normalized_input = normalized_input.replace(".", "")
    normalized_input = normalized_input.replace(" group", "")
    normalized_input = normalized_input.replace(" ring", "")
    # normalized_input = normalized_input.replace("s", "")  # 处理复数形式

    if normalized_input in GROUP_TO_SMARTS.keys():
        return normalized_input

    for key in GROUP_TO_SMARTS.keys():
        if normalized_input in key:
            return key

    return None