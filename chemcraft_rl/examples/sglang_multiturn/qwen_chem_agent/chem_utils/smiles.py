import re

import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import WrapLogs
import contextlib, io

# def is_smiles(text):
#     try:
#         m = Chem.MolFromSmiles(text, sanitize=False)
#         if m is None:
#             return False
#         return True
#     except:
#         return False

def is_smiles(smiles: str):
    WrapLogs()
    f = io.StringIO()
    try: 
        with contextlib.redirect_stderr(f):
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES string. {f.getvalue()}"
        else:
            return True, None
    except Exception as e:
        return False, f"Python Exception: {str(e)}"

def split_smiles(text):
    return text.split(".")


def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    smiles_valid = True
    while not smiles_valid:
        current_valid, _ = is_smiles(ss[-1])
        smiles_valid = current_valid
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]


def tanimoto(s1, s2):
    """Calculate the Tanimoto similarity of two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except (TypeError, ValueError, AttributeError):
        return "Invalid SMILES string."
