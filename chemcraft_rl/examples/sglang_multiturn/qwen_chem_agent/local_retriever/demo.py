from tqdm import tqdm
from rdkit import Chem
import pickle

def matching(reactants, templates):
    """
    :param reactants: list [reactant_smiles1, reactant_smiles2, ...]
    :param templates: list [template_smarts1, template_smarts2, ...]
    :return:
    """
    reactants_none_tpl = []
    reactants_with_tpl = []
    templates_lst = []
    for template in templates:
        reactant_mol_tpl = Chem.MolFromSmarts(template.split('>>')[0])
        product_mol_tpl = Chem.MolFromSmarts(template.split('>>')[1])
        templates_lst.append([reactant_mol_tpl, product_mol_tpl, template])

    for i in tqdm(range(len(reactants))):
        matched_templates = []
        reactant_mol = Chem.MolFromSmiles(reactants[i])
        for template in templates_lst:
            try:
                if reactant_mol.HasSubstructMatch(template[0]):
                    matched_templates.append(template[2])
            except:
                pass

        if len(matched_templates) == 0:
            reactants_none_tpl.append(reactants[i])
        else:
            reactants_with_tpl.append([reactants[i]] + [matched_templates])

    print(len(reactants_none_tpl), len(reactants_with_tpl))
    print(reactants_with_tpl)
    return reactants_none_tpl, reactants_with_tpl


if __name__ == "__main__":
    reactants = [
        "CCO",
        "CCC=O",
        "CCN",
        # "invalid_smiles"
    ]
    templates = [
        "[O:1][C:2]>>[O:1]=[C:2]",
        "[C:1]=[O:2]>>[C:1][O:2H]",
        "[N:1][C:2]>>[N:1]=[C:2]"
    ]

    # res = matching(reactants, templates)
    with open("templates.pkl", 'rb') as f:  
        rxn_templates = pickle.load(f)
    print(type(rxn_templates), len(rxn_templates))
    print(rxn_templates[:10])
