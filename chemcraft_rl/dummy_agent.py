
import requests

def call_chem_api(function_name: str, query: str):
    # payload = ChemFunctionRequest(function_name=function_name, query=query)
    payload = {
        "function_name": function_name,
        "query": query
    }
    
    response = requests.post(
        "http://127.0.0.1:8080/chem_function", 
        json=payload
    )
    return response

if __name__ == "__main__":
    ## test GetRXN and GetRXNTemplate
    result = call_chem_api("GetRXN", "COC(=O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1;reactants")
    print(result.json())
    
    result = call_chem_api("GetRXNTemplate", "O=S1(=O)C=Cc2ccccc21.[Na+].[OH-].[Zn];reactants")
    print(result.json())
    
    # result = call_chem_api("MolSimilarity", "CCO;CC=O")
    # print("MolSimilarity:", result, type(result), result.json())
    
    # result = call_chem_api("FunctionalGroups", "CSc1nc(N)cc(-c2c(-c3ccc(F)cc3)ncn2CCCN2CCOCC2)n1")
    result = call_chem_api("FunctionalGroups", "CN1C=NC2=C1C(=O)N(C(=O)N2CC")
    # result = call_chem_api("FunctionalGroups", "C(C)(C)(C)(C)C")
    print("FunctionalGroups", result, type(result), result.json())
    
    # result = call_chem_api("CompareSMILES", "CCO;CC=O")
    # print("CompareSMILES", result, type(result), result.json())
    
    # result = call_chem_api("SMILES2Weight", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    # print("SMILES2Weight", result, type(result), result.json())
    
    # result = call_chem_api("CanonicalizeSMILES", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    # print("CanonicalizeSMILES", result, type(result), result.json())
    
    # result = call_chem_api("CountMolAtoms", "Fc1c(C[NH2+][C@@H]2[C@@H](C)CN(CC3CC3)CC2)cccc1")
    # print("CountMolAtoms", result, type(result), result.json())
    
    # result = call_chem_api("AddFunctionalGroup", "CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl; carboxyl.")
    # print("AddFunctionalGroup", result, type(result), result.json())
    
    # result = call_chem_api("RemoveFunctionalGroup", "CCC(NN)c1cc(C)c(F)cc1F;amine group")
    # print("AddFunctionalGroup", result, type(result), result.json())
    
    # result = call_chem_api("ReplaceFunctionalGroup", "Cc1cc(C=C2C(=O)NC(=O)N(c3cccc(Cl)c3)C2=O)c(C)n1-c1ccccc1C(=O)[O-];anti-halo;carboxyl group")
    # print("ReplaceFunctionalGroup", result, type(result), result.json())
    
    # result_list = [
    #     ("QEDPropertyPred", "O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1"), 
    #     ("DRD2PropertyPred", "O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1"), 
    #     ("JNK3PropertyPred", "O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1"), 
    #     ("LogPPropertyPred", "O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1"), 
    #     ("GSKPropertyPred", "O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1"), 
    #     ("SolubilityPropertyPred", "O=C(CCN1CCN(CCOC(c2ccccc2)c2ccccc2)CC1)c1ccco1"), 
    # ]
    
    # for info in result_list:
    #     function_name, smiles = info
    #     result = call_chem_api(function_name, smiles)
    #     print(function_name, result, result.json())
    
    