from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from typing import Literal
from chem_agents import ReplaceFunctionalGroup, RemoveFunctionalGroup, AddFunctionalGroup
from chem_agents import MolSimilarity, FunctionalGroups, CompareSMILES, SMILES2Weight, CanonicalizeSMILES, CountMolAtoms, GetRXNTemplate
from chem_prop_agents import QEDPropertyPred, DRD2PropertyPred, JNK3PropertyPred, LogPPropertyPred, GSKPropertyPred, SolubilityPropertyPred
from rxn_index_builder_v3 import FaissSearcherWithoutPCA

app = FastAPI()

FunctionName = Literal["MolSimilarity", "FuncGroups", "CompareSMILES"]

class ChemFunctionRequest(BaseModel):
    function_name: str
    query: str

class ChemFunctionResponse(BaseModel):
    result: str
    status: str = "success"

# calculation function
my_MolSimilarity = MolSimilarity()
my_FuncGroups = FunctionalGroups()
my_CompareSMILES = CompareSMILES()
my_SMILES2Weight = SMILES2Weight()
my_CanonicalizeSMILES = CanonicalizeSMILES()
my_CountMolAtoms = CountMolAtoms()

my_ReplaceFunctionalGroup = ReplaceFunctionalGroup()
my_RemoveFunctionalGroup = RemoveFunctionalGroup()
my_AddFunctionalGroup = AddFunctionalGroup()

# property function
my_QEDPropertyPred = QEDPropertyPred()
my_DRD2PropertyPred = DRD2PropertyPred()
my_JNK3PropertyPred = JNK3PropertyPred()
my_LogPPropertyPred = LogPPropertyPred()
my_GSKPropertyPred = GSKPropertyPred()
my_SolubilityPropertyPred = SolubilityPropertyPred()

# rxn searcher
rxn_reactant_searcher = FaissSearcherWithoutPCA.load_index(
    # save_dir="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/",
    # tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/choriso_reorgnize_search.tsv",
    save_dir="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt",
    tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt/gt_rxn_search.tsv",
    query_type='reactants',
)
rxn_product_searcher = FaissSearcherWithoutPCA.load_index(
    # save_dir="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/",
    # tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/choriso_reorgnize_search.tsv",
    save_dir="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt",
    tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt/gt_rxn_search.tsv",
    query_type='products',
)
my_GetRXNTemplate = GetRXNTemplate()

def rxn_result_response(match_rxn_list):
    ## transform the match_rxn_list to the response sentence
    response = ""
    if len(match_rxn_list) > 0:
        for rxn in match_rxn_list:
            response += f"Reaction: reactants: {rxn['reactant']}, major product: {rxn['product']} with by product: {rxn['byproduct']}, reagent: {rxn['reagent']}, solvent: {rxn['solvent']}, catalyst: {rxn['catalyst']}\n"
    else:
        response = "Cannot find similar reactions for your input SMILES"
    return response.strip()

@app.post("/chem_function")
async def call_chem_function(request: ChemFunctionRequest) -> ChemFunctionResponse:
    
    if request.function_name == "MolSimilarity":
        result = my_MolSimilarity._run_text(request.query)
    elif request.function_name == "FunctionalGroups":
        result = my_FuncGroups._run_base(smiles=request.query)
    elif request.function_name == "CompareSMILES":
        result = my_CompareSMILES._run_text(request.query)
    elif request.function_name == "SMILES2Weight":
        result = my_SMILES2Weight._run_text(request.query)
    elif request.function_name == "CanonicalizeSMILES":
        result = my_CanonicalizeSMILES._run_text(request.query)
    elif request.function_name == "CountMolAtoms":
        result = my_CountMolAtoms._run_text(request.query)
        
    elif request.function_name == "AddFunctionalGroup":
        result = my_AddFunctionalGroup._run_text(request.query)
    elif request.function_name == "RemoveFunctionalGroup":
        result = my_RemoveFunctionalGroup._run_text(request.query)
    elif request.function_name == "ReplaceFunctionalGroup":
        result = my_ReplaceFunctionalGroup._run_text(request.query)
    
    elif request.function_name == "QEDPropertyPred":
        result = my_QEDPropertyPred._run_text(request.query)
    elif request.function_name == "DRD2PropertyPred":
        result = my_DRD2PropertyPred._run_text(request.query)
    elif request.function_name == "JNK3PropertyPred":
        result = my_JNK3PropertyPred._run_text(request.query)
    elif request.function_name == "LogPPropertyPred":
        result = my_LogPPropertyPred._run_text(request.query)
    elif request.function_name == "GSKPropertyPred":
        result = my_GSKPropertyPred._run_text(request.query)
    elif request.function_name == "SolubilityPropertyPred":
        result = my_SolubilityPropertyPred._run_text(request.query)
    
    elif request.function_name == "GetRXN":
        smi_type_list = request.query.split(';')
        if len(smi_type_list) != 2:
            result = "Query format error, please input SMILES;type separated by ';' type must be 'reactants' or 'products'"
        else:
            smiles, query_type = smi_type_list
            if query_type not in ['reactants', 'products']: 
                result = "GetRXN Type error, type must be 'reactants' or 'products'"
            elif query_type == 'reactants':
                match_rxn_list = rxn_reactant_searcher.search(smiles, top_k=3, min_similarity=0.8)
                result = rxn_result_response(match_rxn_list=match_rxn_list)
            elif query_type == 'products':
                match_rxn_list = rxn_product_searcher.search(smiles, top_k=3, min_similarity=0.8)
                result = rxn_result_response(match_rxn_list=match_rxn_list)
    elif request.function_name == "GetRXNTemplate":
        result = my_GetRXNTemplate._run_text(request.query)
    else:
        return ChemFunctionResponse(
            result="Invalid function name", 
            status="error"
        )
    
    return ChemFunctionResponse(result=str(result))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)