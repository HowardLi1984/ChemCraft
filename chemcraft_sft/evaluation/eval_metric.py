## For Single-Objective Molecule Optimization Benchmark

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import DataStructs
from rdkit.Chem import Descriptors
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MurckoScaffoldSmiles
from rdkit.Chem import Crippen, Lipinski
from collections import namedtuple

import tdc

# def calculate_solubility(smiles):
#     ## Calculate aqueous solubility(logS) using RDKit descriptors and a simple linear model.
#     try:
#         # Convert SMILES to RDKit molecule
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             raise ValueError("Invalid SMILES string")
        
#         # Calculate relevant descriptors
#         mw = Descriptors.MolWt(mol)
#         logp = Descriptors.MolLogP(mol)
#         h_bond_donors = Descriptors.NumHDonors(mol)
#         h_bond_acceptors = Descriptors.NumHAcceptors(mol)
#         rotatable_bonds = Descriptors.NumRotatableBonds(mol)
#         # Simple linear model (based on published QSPR models)
#         logS = 0.16 - 0.63*logp - 0.0062*mw + 0.066*h_bond_donors - 0.074*h_bond_acceptors
#         return logS
    
#     except Exception as e:
#         print(f"Error calculating solubility: {e}")
#         return 0.0

class ESOLCalculator:
    # from https://github.com/PatWalters/solubility/blob/master/esol.py
    # a better ESOL calculator for solubility prediction
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol_orig(self, mol):
        """
        Original parameters from the Delaney paper, just here for comparison
        :param mol: input molecule
        :return: predicted solubility
        """
        # just here as a reference don't use this!
        intercept = 0.16
        coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol

    def calc_esol(self, smiles):
        """
        Use This Function !
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        mol = Chem.MolFromSmiles(smiles)
        try:
            intercept = 0.26121066137801696
            coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
            desc = self.calc_esol_descriptors(mol)
            esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
                + coef["ap"] * desc.ap
            return esol
        
        except Exception as e:
            print(f"Error calculating solubility: {e}")
            return None

def compute_statistics(numbers, prop, skew=False):
    if numbers == []:
        return {
            "mean": 0,
            "variance": 0,
            "min": 0,
            "max": 0,
            "success_rate": 0,  # success opt that increase the property
            "best_rate": 0,  # rate of best property mol-opt
        }

    easy_thres, hard_thres = 0.5, 0.3
    threshold_dict = {
        "gsk3b": hard_thres,
        "qed": hard_thres,
        "drd2": hard_thres,
        "jnk3": hard_thres,
        "logp": easy_thres,
        "solubility": easy_thres,
    }

    n = len(numbers)
    # if skew is True, use median and IQR
    if skew:
        lower = np.percentile(numbers, 5)
        upper = np.percentile(numbers, 95)
        winsorized = np.clip(numbers, lower, upper)
        mean = np.mean(winsorized)
        variance = np.var(winsorized)
        min_val = np.min(winsorized)
        max_val = np.max(winsorized)
    else:
        mean = sum(numbers) / n
        # Calculate variance (using population variance: 1/N * sum((x_i - mean)^2))
        variance = sum((x - mean) ** 2 for x in numbers) / n
        min_val = min(numbers)
        max_val = max(numbers)

    success_rate = sum(1 for itm in numbers if itm > 0) / len(numbers)
    best_rate = sum(1 for itm in numbers if itm >= threshold_dict[prop]) / len(numbers)

    return {
        "mean": mean,
        "variance": variance,
        "min": min_val,
        "max": max_val,
        "success_rate": success_rate,  # success opt that increase the property
        "best_rate": best_rate,  # rate of best property mol-opt
    }


class mol_opt_evaluater():
    def __init__(self, prop=None, ) -> None:
        self.prop = prop
        if prop in ['gsk3b', 'qed', 'drd2', 'jnk3', 'logp']:
            self.property_oracle = tdc.Oracle(name=prop)
        elif prop == 'solubility':
            esolcalculator = ESOLCalculator()
            self.property_oracle = esolcalculator.calc_esol
    
    def property_improvement(self, src_mol_list, tgt_mol_list, total_num):
        ## evaluate the property improvement after the mol-opt
        ## First, Check the validation of SMILES, remove the invalid SMILES
        ## Second, Calculate the property of SMILES
        ## Finally, Statistic the improvement score with valid cases and invalid numbers.
        assert len(src_mol_list) == len(tgt_mol_list)
        src_mol_check_valid = [is_valid_smiles(smiles) for smiles in src_mol_list]
        tgt_mol_check_valid = [is_valid_smiles(smiles) for smiles in tgt_mol_list]
        src_mol_score, tgt_mol_score = list(), list()
        for i in range(len(src_mol_check_valid)):
            if src_mol_check_valid[i] and tgt_mol_check_valid[i]:
                src_mol_score.append(self.property_oracle(src_mol_list[i]))
                tgt_mol_score.append(self.property_oracle(tgt_mol_list[i]))
        
        # continue filtering, remove the `None` score from src_mol_score & tgt_mol_score
        prop_improve_list = list()
        for i in range(len(src_mol_score)):
            if src_mol_score[i] != None and tgt_mol_score[i] != None:
                prop_improve_list.append(tgt_mol_score[i]-src_mol_score[i])
        valid_score = len(prop_improve_list)
        prop_improve_list = prop_improve_list + [0.0]*(total_num - len(prop_improve_list))
        
        statistic = compute_statistics(prop_improve_list, self.prop, skew=True)
        statistic['valid_smiles_rate'] = len(src_mol_score) / total_num
        statistic['valid_score_rate'] = valid_score / total_num
        statistic['valid_smiles_extract_rate'] = len(src_mol_list) / total_num
            
        return statistic

    def property_list_calculation(self, src_mol_list, tgt_mol_list, total_num):
        ## only for data analysis
        assert len(src_mol_list) == len(tgt_mol_list)
        src_mol_check_valid = [is_valid_smiles(smiles) for smiles in src_mol_list]
        tgt_mol_check_valid = [is_valid_smiles(smiles) for smiles in tgt_mol_list]
        src_mol_score, tgt_mol_score = list(), list()
        src_valid_mol_list, tgt_valid_mol_list = list(), list()
        for i in range(len(src_mol_check_valid)):
            if src_mol_check_valid[i] and tgt_mol_check_valid[i]:
                src_mol_score.append(self.property_oracle(src_mol_list[i]))
                tgt_mol_score.append(self.property_oracle(tgt_mol_list[i]))
                src_valid_mol_list.append(src_mol_list[i])
                tgt_valid_mol_list.append(tgt_mol_list[i])
        
        # continue filtering, remove the `None` score from src_mol_score & tgt_mol_score
        prop_improve_list = list()
        for i in range(len(src_mol_score)):
            if src_mol_score[i] != None and tgt_mol_score[i] != None:
                tmp = dict(
                    src_mol=src_valid_mol_list[i],
                    tgt_mol=tgt_valid_mol_list[i],
                    src_score=src_mol_score[i],
                    tgt_score=tgt_mol_score[i],
                    delta=tgt_mol_score[i]-src_mol_score[i],
                )
                prop_improve_list.append(tmp)
        
        prop_improve_list = prop_improve_list + [dict(src_mol=None, tgt_mol=None, src_score=None, tgt_score=None, delta=0.0)]*(total_num - len(prop_improve_list))
        
        return prop_improve_list
    
    def scaffold_consistency(self, src_mol_list, tgt_mol_list):
        ## evaluate the scaffold consistency before&after mol-opt, consistency includes: same or contain
        assert len(src_mol_list) == len(tgt_mol_list)
        
        count_same = 0
        scaffold_score = list()
        
        for i in range(len(tgt_mol_list)):
            src_smiles, tgt_smiles = src_mol_list[i], tgt_mol_list[i]
            try:
                src_mol, tgt_mol = Chem.MolFromSmiles(src_smiles), Chem.MolFromSmiles(tgt_smiles)
            except:
                continue
            
            if src_mol == None or tgt_mol == None:
                scaffold_score.append(0.0); continue
            
            opt_smiles = [src_smiles, tgt_smiles]
            if '' in opt_smiles:
                scaffold_score.append(0.0); continue
            murcko_scaffold_list = [MurckoScaffoldSmiles(smiles) for smiles in opt_smiles]
            
            if len(set(murcko_scaffold_list)) == 1:
                scaffold_score.append(1.0)
                count_same += 1
            else:
                ## Morgan Fingerprint for scaffold similarity
                murcko_scaffold_mol_list = [Chem.MolFromSmiles(murcko_scaffold_list[0]), Chem.MolFromSmiles(murcko_scaffold_list[1])]
                mcs = rdFMCS.FindMCS(murcko_scaffold_mol_list)
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString) if mcs.numAtoms > 0 else None
                
                if mcs_mol:
                    # 计算基于指纹的Tanimoto相似度
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(murcko_scaffold_mol_list[0], 2, nBits=1024)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(murcko_scaffold_mol_list[1], 2, nBits=1024)
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                else:
                    similarity = 0.0
                
                scaffold_score.append(similarity)  
        
        if len(tgt_mol_list) == 0:
            return 0.0, 0.0
        
        return count_same, sum(scaffold_score)
    

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        else: return True
    except:
        return False