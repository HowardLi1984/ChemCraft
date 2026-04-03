import pandas as pd
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle

import pandas as pd

def read_tsv_first_10_lines(file_path):
    # 读取 reaction.tsv 文件的前10行内容
    df = pd.read_csv(file_path, sep='\t', nrows=10)
    
    print(f"成功读取文件: {file_path}")
    print(f"数据形状: {df.shape}")
    print("\n前10行数据内容:")
    print("=" * 50)
    print(df)
    
    return df

class FaissRXNSearcher:
    def __init__(self, tsv_path, ngram_range=(2, 3), query_type='reactant'):
        """
        :param tsv_path: TSV文件路径
        :param ngram_range: N-gram范围，例如(2,3)表示提取2-gram和3-gram
        """
        self.query_type = query_type
        # 1. 加载数据
        self.df = pd.read_csv(tsv_path, sep="\t")
        
        # 3. 提取所有 query-key=reactant/product 并清洗
        self.query = None
        if query_type == 'reactant':
            self.query = self.df["reactant"].fillna("").astype(str).tolist()
        elif query_type == 'product':
            self.query = self.df["product"].fillna("").astype(str).tolist()
        
        # 4. 构建N-gram TF-IDF向量
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,  # 捕获化学式的局部模式
            min_df=5,                # 忽略罕见ngram
            dtype=np.float32
        )
        X = self.vectorizer.fit_transform(self.query)
        
        # 5. 转换为FAISS支持的格式
        self.embeddings = X.toarray().astype("float32")
        faiss.normalize_L2(self.embeddings)  # 归一化以便使用余弦相似度
        
        # 6. 构建FAISS索引
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # 内积=余弦相似度
        self.index.add(self.embeddings)
    
    def save_index(self, save_dir="faiss_index"):
        """保存索引和元数据"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, f"{save_dir}/{self.query_type}.index")
        
        # 保存其他必要数据
        with open(f"{save_dir}/metadata_{self.query_type}.pkl", "wb") as f:
            pickle.dump({
                "reactants_or_products": self.query,
                "vectorizer": self.vectorizer,
                "df_columns": self.df.columns.tolist()
            }, f)
    
    @classmethod
    def load_index(cls, save_dir, tsv_path, query_type):
        """加载预建索引"""
        if query_type not in ['reactants', 'products']:
            assert 1 == 0
        print(f"Start loading reactant.index and metadata_{query_type}.pkl")
        index = faiss.read_index(f"{save_dir}/{query_type}.index")
        
        # 加载元数据
        with open(f"{save_dir}/metadata_{query_type}.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # 重建对象
        searcher = cls.__new__(cls)
        searcher.query_type = query_type
        searcher.index = index
        if query_type == 'reactants':
            searcher.query = metadata[query_type]
        elif query_type == 'products':
            searcher.query = metadata['reactants_or_products']
        searcher.vectorizer = metadata["vectorizer"]
        # searcher.df = pd.DataFrame(columns=metadata["df_columns"])
        searcher.df = pd.read_csv(tsv_path, sep="\t")
        return searcher
    
    def search(self, query, top_k=5, min_similarity=0.7):
        """
        模糊搜索reactant
        :param query: 查询字符串（如 "CCO")
        :param top_k: 返回前K个结果
        :param min_similarity: 最低余弦相似度阈值（0-1）
        :return: 匹配的DataFrame（包含similarity列）
        """
        # 1. 将查询文本转换为向量
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        faiss.normalize_L2(query_vec)
        
        # 2. FAISS搜索
        distances, indices = self.index.search(query_vec, top_k)
        # print("rxn: ", distances, indices)
        
        # 3. 处理结果
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist < min_similarity:
                continue
            match = {
                "similarity": dist,
                **self.df.iloc[idx].to_dict()
            }
            matches.append(match)
        
        return matches if matches else []

def build_index(rxn_tsv_path, index_save_path, query_type='reactant'):
    # 首次运行：构建索引（约5-10分钟，200万数据）
    searcher = FaissRXNSearcher(tsv_path=rxn_tsv_path, query_type=query_type)
    searcher.save_index(save_dir=index_save_path)
    
def search_demo():
    from time import time
    start_time = time()
    index_save_dir = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search"
    searcher = FaissRXNSearcher.load_index(
        save_dir=index_save_dir,
        tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/choriso_reorgnize_search.tsv",
        query_type='reactants'
    )
    
    load_time = time()
    print(f"load-time: {load_time-start_time:.6f}")
    
    # 实时查询（约10-50ms）
    query_list = [
        "O=S1(=O)C=Cc2ccccc21.[Na+].[OH-].[Zn]", 
        'CCO.O=S1(=O)C=Cc2ccccc21.[Pd]',
        'CO.O=C1CCCN1C1CCN(Cc2ccccc2)CC1.O=C[O-].[NH4+].[Pd]',
        'CC(=O)O[C@@H](C)C#CC(=O)O.CC(=O)[O-].CC(=O)[O-].[Pb+2].[Pd].CCCCOCCOCCOCc1cc2c(cc1CCC)OCO2.CCO.[H][H].c1ccc2ncccc2c1',
    ]
    for query in query_list:
        start_time = time()
        results = searcher.search(query, min_similarity=0.6)
        end_time = time()
        print(f"retrieve-time: {end_time-start_time:.6f}")
        
        if results is not None:
            print(f"Top {len(results)} matches for '{query}':")
            print(results)
        else:
            print(f"No matches found for '{query}'")
    
    # for products
    start_time = time()
    index_save_dir = "/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search"
    searcher = FaissRXNSearcher.load_index(
        save_dir=index_save_dir,
        tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/choriso_reorgnize_search.tsv",
        query_type='products'
    )
    
    load_time = time()
    print(f"load-time: {load_time-start_time:.6f}")
    
    # 实时查询（约10-50ms）
    query_list = [
        'O=S1(=O)CCc2ccccc21', 
        'O=C1CCCN1C1CCNCC1',
        'CC(C)(C)OC(=O)N1CC2CNCC(C2)C1',
        'CC(=O)O[C@@H](C)/C=C\\C(=O)O',
    ]
    for query in query_list:
        start_time = time()
        results = searcher.search(query, min_similarity=0.6)
        end_time = time()
        print(f"retrieve-time: {end_time-start_time:.6f}")
        
        if results is not None:
            print(f"Top {len(results)} matches for '{query}':")
            print(results)
        else:
            print(f"No matches found for '{query}'")



if __name__ == "__main__":
    print("--- Building Optimized Index ---")
    
    ## First, build the product queries
    build_index(
        rxn_tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/choriso_reorgnize_search.tsv",
        index_save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/",
        query_type='product'
    )   
    
    ## Second, build the reactant queries
    # build_index(
    #     rxn_tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/choriso_reorgnize_search.tsv",
    #     index_save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search/",
    #     query_type='reactant'
    # )  
    
    search_demo() 
    
    # {'reactant': 'O=S1(=O)C=Cc2ccccc21.[Na+].[OH-].[Zn]', 'product': 'O=S1(=O)CCc2ccccc21', 'reagent': 'sodium hydroxide|zinc', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CCO.O=S1(=O)C=Cc2ccccc21.[Pd]', 'product': 'O=S1(=O)CCc2ccccc21', 'reagent': 'palladium on activated charcoal|ethanol', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CO.O=C1CCCN1C1CCN(Cc2ccccc2)CC1.O=C[O-].[NH4+].[Pd]', 'product': 'O=C1CCCN1C1CCNCC1', 'reagent': 'palladium 10 on activated carbon|ammonium formate', 'solvent': 'methanol', 'catalyst': 'empty', 'yield': 100.0}
    # {'reactant': 'CC(C)(C)OC(=O)N1CC2CC(CN(Cc3ccccc3)C2)C1.CCO.[H][H].[Pd]', 'product': 'CC(C)(C)OC(=O)N1CC2CNCC(C2)C1', 'reagent': 'hydrogen', 'solvent': 'ethanol', 'catalyst': 'palladium on activated charcoal', 'yield': 51.0}
    # {'reactant': 'CCNC1(C(N)=O)CCN(Cc2ccccc2)CC1.CO.O.[H][H].[OH-].[OH-].[Pd+2]', 'product': 'CCNC1(C(N)=O)CCNCC1', 'reagent': 'palladium hydroxide on carbon|hydrogen', 'solvent': 'methanol|water', 'catalyst': 'empty', 'yield': 100.0}
    # {'reactant': 'CC(=O)O[C@@H](C)C#CC(=O)O.CC(=O)[O-].CC(=O)[O-].[Pb+2].[Pd].CCO.[H][H].c1ccc2ncccc2c1', 'product': 'CC(=O)O[C@@H](C)/C=C\\C(=O)O', 'reagent': 'quinoline|hydrogen', 'solvent': 'ethanol', 'catalyst': "lindlar's catalyst", 'yield': 78.0}
    # {'reactant': 'CC(=O)O[C@@H](C)C#CC(=O)O.CC(=O)[O-].CC(=O)[O-].[Pb+2].[Pd].CCCCOCCOCCOCc1cc2c(cc1CCC)OCO2.CCO.[H][H].c1ccc2ncccc2c1', 'product': 'CC(=O)O[C@@H](C)/C=C\\C(=O)O', 'reagent': 'quinoline|hydrogen', 'solvent': 'ethanol', 'catalyst': "lindlar's catalyst|alpha-[2-(2-butoxyethoxy)ethoxy]-4,5-methylenedioxy-2-propyltoluene", 'yield': 80.0}
    # {'reactant': 'CC(=O)O[C@@H](C)C#CC(=O)O.CC(=O)[O-].CC(=O)[O-].[Pb+2].[Pd]', 'product': 'CC(=O)O[C@@H](C)/C=C\\C(=O)O', 'reagent': "lindlar's catalyst", 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CC(=O)O[Pd]c1c(CP(c2ccccc2)c2ccccc2)c(C)cc(C)c1CP(c1ccccc1)c1ccccc1.CCCCO.COc1ccc(C=CC(=O)c2ccccc2)cc1.O=C([O-])[O-].[K+].[K+]', 'product': 'COc1ccc(CCC(=O)c2ccccc2)cc1', 'reagent': 'acetoxy(2,6-bis((diphenylphosphino)methyl)-3,5-dimethylphenyl)palladium(ii)|potassium carbonate|butan-1-ol', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 94.0}