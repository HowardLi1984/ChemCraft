def _get_task_info()->dict:
    return {
        # Molecule Editing Tasks
        'add': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/add.json',
            "Tool Recommend: After 'Step-1, molecule_analysis:', highly recommend call <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>. You can call <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> and <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> multiple times to analyse molecule. DO NOT use <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants / products'}}\n</tool_call>, DO NOT use <tool_call>\n{'name': 'AddFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'new_group_name':'Your group'}}\n</tool_call>. DO NOT call any tool after <answer></answer>"
        ],
        'delete': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/delete.json',
            "Tool Recommend: After 'Step-1, molecule_analysis:', highly recommend call <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>. You can call <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> and <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> multiple times to analyse molecule. DO NOT use <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants / products'}}\n</tool_call>, DO NOT use <tool_call>\n{'name': 'RemoveFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'old_group_name':'Your group'}}\n</tool_call>. DO NOT call any tool after <answer></answer>"
        ],
        'sub': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/sub.json',
            "Tool Recommend: After 'Step-1, molecule_analysis:', highly recommend call <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>. You can call <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> and <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> multiple times to analyse molecule. DO NOT use <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants / products'}}\n</tool_call>, DO NOT use <tool_call>\n{'name': 'ReplaceFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'old_group_name':'Your group', 'new_group_name':'Your group'}}\n</tool_call>. DO NOT call any tool after <answer></answer>"
        ],

        # Molecule Understanding Tasks
        'ring_count': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/ring_count.json',
            "use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. DO NOT use <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants / products'}}\n</tool_call>. DO NOT use <tool_call>\n{'name': 'CanonicalizeSMILES', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>."
        ],
        'ring_system_scaffold': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/ring_system_scaffold.json',
            "use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. DO NOT use <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants / products'}}\n</tool_call>."
        ],
        'mutated': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/mutated.json',
            "use <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> to check for differences between molecules. use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule"
        ],
        'permutated': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/permutated.json',
            "use <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> to verify if molecules are identical. use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule"
        ],
        'functiongroup_detect': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/fg_detect.json',
            "use <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> to detect all functional groups in the molecule."
        ],
        'murcko_scaffold': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/murcko_scaffold.json',
            "use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. DO NOT use <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants / products'}}\n</tool_call>. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct."
        ],

        # Molecule Optimization Tasks
        'gsk': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/gsk.json',
            "use <tool_call>\n{'name': 'GSKPropertyPred', 'arguments': {'SMILES': 'Your SMILES'}}\n</tool_call> to predict the GSK property. <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. use <tool_call>\n{'name': 'RemoveFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'old_group_name':'Your group'}}\n</tool_call> and <tool_call>\n{'name': 'AddFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'new_group_name':'Your group'}}\n</tool_call> to edit the molecule. Suggest to do <GSKPropertyPred> before <answer></answer>. DO NOT call any tool after <answer></answer>"
        ],
        'jnk': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/jnk.json',
            "use <tool_call>\n{'name': 'JNK3PropertyPred', 'arguments': {'SMILES': 'Your SMILES'}}\n</tool_call> to predict the JNK3 property. <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. Suggest to do <JNK3PropertyPred> before <answer></answer>. DO NOT call any tool after <answer></answer>"
        ],
        'drd': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/drd.json',
            "use <tool_call>\n{'name': 'DRD2PropertyPred', 'arguments': {'SMILES': 'Your SMILES'}}\n</tool_call> to predict the DRD2 property. <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. Suggest to do <DRD2PropertyPred> before <answer></answer>. DO NOT call any tool after <answer></answer>"
        ],
        'logp': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/logp.json',
            "use <tool_call>\n{'name': 'LogPPropertyPred', 'arguments': {'SMILES': 'Your SMILES'}}\n</tool_call>to predict the LogP value. <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. use <tool_call>\n{'name': 'RemoveFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'old_group_name':'Your group'}}\n</tool_call> and <tool_call>\n{'name': 'AddFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'new_group_name':'Your group'}}\n</tool_call> to edit the molecule. Suggest to do <LogPPropertyPred> before <answer></answer>. DO NOT call any tool after <answer></answer>"
        ],
        'qed': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/qed.json',
            "use <tool_call>\n{'name': 'QEDPropertyPred', 'arguments': {'SMILES': 'Your SMILES'}}\n</tool_call> to predict the QED score. <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. use <tool_call>\n{'name': 'RemoveFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'old_group_name':'Your group'}}\n</tool_call> and <tool_call>\n{'name': 'AddFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'new_group_name':'Your group'}}\n</tool_call> to edit the molecule. Suggest to do <QEDPropertyPred> before <answer></answer>. DO NOT call any tool after <answer></answer>"
        ],
        'solubility': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/solubility.json',
            "use <tool_call>\n{'name': 'SolubilityPropertyPred', 'arguments': {'SMILES': 'Your SMILES'}}\n</tool_call> to predict solubility. <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. use <tool_call>\n{'name': 'RemoveFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'old_group_name':'Your group'}}\n</tool_call> and <tool_call>\n{'name': 'AddFunctionalGroup', 'arguments': {'SMILES':'Your SMILES', 'new_group_name':'Your group'}}\n</tool_call> to edit the molecule. Suggest to do <SolubilityPropertyPred> before <answer></answer>. DO NOT call any tool after <answer></answer>"
        ],
        
        # rxn
        'fs_major_product': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_fs_major_product.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get correlating reation templates. use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct. DO NOT call any tool after <answer></answer>"
        ],
        'fs_by_product': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_fs_by_product.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get correlating reation templates. use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct. DO NOT call any tool after <answer></answer>"
        ],
        'retro': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_retro.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. Use <tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reation templates. DO NOT call any tool after <answer></answer>"
        ],
        'rcr_catalyst': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_rcr_catalyst.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse the molecules from reactants and products. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct. DO NOT call any tool after <answer></answer>"
        ],
        'rcr_reagent': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_rcr_reagent.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse the molecules from reactants and products. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct. DO NOT call any tool after <answer></answer>"
        ],
        'rcr_solvent': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_rcr_solvent.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse the molecules from reactants and products. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct. DO NOT call any tool after <answer></answer>"
        ],
        'mech_sel': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_mech_sel.json',
            "You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get correlating reation templates. use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. <tool_call>\n{'name': 'CompareSMILES', 'arguments': {'SMILES1':'Your SMILES', 'SMILES2':'Your SMILES'}}\n</tool_call> MUST BE format correct. DO NOT call any tool after <answer></answer>"
        ],
        'nepp': [
            '/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/chemcot/chemcot_json/chemcotdataset/rxn_nepp.json',
            "Use <tool_call>\n{'name': 'CountMolAtoms', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call>, <tool_call>\n{'name': 'FunctionalGroups', 'arguments': {'SMILES':'Your SMILES'}}\n</tool_call> analyse molecule. You MUST Use at least ONE time <tool_call>\n{'name': 'GetRXN', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reaction. Use <tool_call>\n{'name': 'GetRXNTemplate', 'arguments': {'SMILES': 'Your SMILES', 'Type': 'reactants/products'}}\n</tool_call> to get the correlating reation templates. DO NOT call any tool after <answer></answer>"
        ],
    }