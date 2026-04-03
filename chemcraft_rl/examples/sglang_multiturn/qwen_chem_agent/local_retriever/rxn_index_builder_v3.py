## 用 IndexIVFFlat 聚类能获得更快的速度, rxn_index_builder.py 5s/case, rxn_index_builder_v3.py 0.02s/case

import pandas as pd
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
from time import time

class FaissSearcherWithoutPCA:
    def __init__(self, tsv_path, query_type, ngram_range=(2, 3)):
        """
        初始化加载数据，为构建索引做准备。
        :param tsv_path: TSV文件路径
        :param ngram_range: N-gram范围
        """
        print("1. Loading and preprocessing data...")
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.query_type = query_type
        self.query = None
        if query_type == 'reactants':
            self.query = self.df["reactant"].fillna("").astype(str).tolist()
        elif query_type == 'products':
            self.query = self.df["product"].fillna("").astype(str).tolist()
        # self.reactants = self.df["reactant"].fillna("").astype(str).tolist()
        
        # 定义向量化器，使用max_features进行维度控制
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=5,
            max_features=20000,  # 依然建议保留，以防维度爆炸
            dtype=np.float32
        )

        # 初始化索引
        self.index = None
        self.is_on_gpu = False
        self.gpu_res = None

    def build_index(self, d=512, nlist=1024, m=None, train_on_gpu=True):
        """
        构建优化的FAISS索引 (IndexIVFPQ) - 通过向量填充确保维度可整除。
        :param nlist: IVF的聚类中心数。
        :param m: Product Quantization的子向量数。必须是GPU支持的值 (e.g., 16, 32, 64)
        """
        print("2. Vectorizing reactants with TF-IDF...")
        X = self.vectorizer.fit_transform(self.query)
        embeddings = X.toarray().astype("float32")
        
        d_original = embeddings.shape[1]
        print(f"   Original vector dimension: {d_original}")

        # **【核心修复】: 根据固定的、GPU友好的m值，来计算新的向量维度**
        # 我们不再使用 m_div，而是直接传入一个确定的 m
        if m is not None and d_original % m != 0:
            d_new = (d_original + m - 1) // m * m
            print(f"   Padding dimension from {d_original} to {d_new} to be a multiple of m={m}.")
            
            self.padded_dim = d_new
            padded_embeddings = np.zeros((embeddings.shape[0], d_new), dtype='float32')
            padded_embeddings[:, :d_original] = embeddings
            embeddings = padded_embeddings
        else:
            d_new = d_original
            self.padded_dim = d_new

        faiss.normalize_L2(embeddings)

        # 使用固定的m值创建索引
        # self.index = faiss.IndexIVFPQ(quantizer, d_new, nlist, m, 8)
        print(f"4. Building FAISS IndexIVFFlat (d={d_new}, nlist={nlist})...")
        quantizer = faiss.IndexFlatIP(d_new) 
        self.index = faiss.IndexIVFFlat(quantizer, d_new, nlist, faiss.METRIC_INNER_PRODUCT)
        
        if train_on_gpu:
            print("5. Training the index in GPU:0, fast")
            
            co = faiss.GpuClonerOptions()
            # 启用半精度(float16)查询表，将共享内存需求减半
            co.useFloat16 = True
            
            # 将CPU索引临时克隆到GPU上进行训练
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index, co)
            gpu_index.train(embeddings)
            # 训练完成后，将训练好的索引结构移回CPU
            print("   Training on GPU complete. Moving index back to CPU.")
            self.index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            print("5. Training the index in CPU, slow")
            self.index.train(embeddings)
        
        print("6. Adding embeddings to the index...")
        self.index.add(embeddings)
        print(f"   Index built successfully. Total entries: {self.index.ntotal}")

    def to_gpu(self, gpu_id=0):
        """将索引移至GPU"""
        if not faiss.get_num_gpus():
            print("No GPU found. Staying on CPU.")
            return
        
        print(f"Moving index to GPU {gpu_id}...")
        self.gpu_res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.index = faiss.index_cpu_to_gpu(self.gpu_res, gpu_id, self.index, co)
        self.is_on_gpu = True
        print("Index successfully moved to GPU.")

    def search(self, query, top_k=5, min_similarity=0.7, nprobe=10):
        if self.index is None:
            raise RuntimeError("Index is not built or loaded...")

        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        
        # **【核心修复】: 对查询向量进行同样的填充**
        if hasattr(self, 'padded_dim') and query_vec.shape[1] != self.padded_dim:
            d_original = query_vec.shape[1]
            # 检查原始维度是否大于填充后的维度，这不应该发生
            if d_original > self.padded_dim:
                raise ValueError(f"Query vector dimension ({d_original}) is greater than the index's padded dimension ({self.padded_dim}).")
            padded_query_vec = np.zeros((1, self.padded_dim), dtype='float32')
            padded_query_vec[:, :d_original] = query_vec
            query_vec = padded_query_vec

        faiss.normalize_L2(query_vec)
        
        self.index.nprobe = nprobe
        distances, indices = self.index.search(query_vec, top_k)

        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist < min_similarity:
                continue
            match = { "similarity": dist, **self.df.iloc[idx].to_dict() }
            matches.append(match)
        
        return matches

    def save_index(self, save_dir="faiss_index_no_pca"):
        """保存索引和元数据"""
        Path(save_dir).mkdir(exist_ok=True)
        
        index_to_save = self.index
        if self.is_on_gpu:
            print("Moving index back to CPU for saving...")
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(index_to_save, f"{save_dir}/{self.query_type}.index")
        
        with open(f"{save_dir}/metadata_{self.query_type}.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "df_columns": self.df.columns.tolist(),
                "padded_dim": self.padded_dim 
            }, f)
        print(f"Index and metadata (no PCA) saved to {save_dir}")

    @classmethod
    def load_index(cls, save_dir, tsv_path, query_type):
        """加载预建的优化索引"""
        if query_type not in ['reactants', 'products']:
            assert 1 == 0
        print(f"Start loading {query_type} index and metadata (no PCA)...")
        index = faiss.read_index(f"{save_dir}/{query_type}.index")
        
        with open(f"{save_dir}/metadata_{query_type}.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        searcher = cls.__new__(cls)
        searcher.query_type = query_type
        searcher.index = index
        searcher.vectorizer = metadata["vectorizer"]
        
        # 使用 .get() 提供向后兼容性，如果旧的 pkl 文件没有这个键，默认为0
        searcher.padded_dim = metadata.get("padded_dim", 0)
        if searcher.padded_dim == 0:
            # 如果没有保存padded_dim, 就假定它等于索引的维度
            searcher.padded_dim = index.d
            print(f"Warning: 'padded_dim' not found in metadata.pkl. Assuming it's the index dimension: {index.d}")
        
        searcher.df = pd.read_csv(tsv_path, sep="\t")
        searcher.is_on_gpu = False
        searcher.gpu_res = None
        
        print("Index loaded successfully.")
        return searcher

def build_index_no_pca(rxn_tsv_path, index_save_path, query_type):
    # 1. 初始化
    searcher = FaissSearcherWithoutPCA(tsv_path=rxn_tsv_path, query_type=query_type)
    
    # 2. 构建索引 (无PCA)
    searcher.build_index(nlist=2048, m=64)
    
    # 3. 保存
    searcher.save_index(save_dir=index_save_path)

def search_demo_optimized(index_save_dir, tsv_path):
    from time import time
    start_time = time()
    # Start Reactants Retrieval
    searcher = FaissSearcherWithoutPCA.load_index(
        save_dir=index_save_dir,
        tsv_path=tsv_path,
        query_type="reactants"
    )

    # **最终加速：将索引移至GPU**
    # searcher.to_gpu()
    end_load_time = time()
    print(f"Loading Time cost: {end_load_time-start_time:.2f}")
    
    query_list = [
        "COc1ccc(COc2ccc(Cn3c(N)nc4cc(-c5cnn(CC(=O)OC(C)(C)C)c5)cnc43)cc2OC)cc1.O[Na]", 
        'CCOc1cc(C(C)(C)C)ncc1C1=NC(C)(c2ccc(Cl)cc2)C(C)(c2ccc(Cl)cc2)N1C(=O)N1CCN(CC(=O)OC(C)(C)C)CC1.[Li]O',
        'COCCn1ccc2c(Nc3ncc(C4CC4)cc3C(=O)OC(C)(C)C)cccc21.O[Na]',
        'CC(=O)c1nn(CC(=O)OC(C)(C)C)c2c3c(c(-c4cnc(C)nc4)cc12)CCCO3.[Li]O',
    ]
    
    for query in query_list:
        start_query_time = time()
        # 搜索时调整 nprobe, nprobe越大越准但越慢
        results = searcher.search(query, min_similarity=0.1, top_k=3, nprobe=20)
        
        if results:
            print(f"\nTop {len(results)} matches for '{query}':")
            print(results)
            print("*  "*30)
        else:
            print(f"No matches found for '{query}'")
        
        end_query_time = time()
        print(f"Searching Time cost: {end_query_time-start_query_time:.2f}")
        
    # # Start Product Retrieval
    # searcher = FaissSearcherWithoutPCA.load_index(
    #     save_dir=index_save_dir,
    #     tsv_path=tsv_path,
    #     query_type="products"
    # )
    
    # query_list = [
    #     'O=S1(=O)CCc2ccccc21', 
    #     'O=C1CCCN1C1CCNCC1',
    #     'CC(C)(C)OC(=O)N1CC2CNCC(C2)C1',
    #     'CC(=O)O[C@@H](C)/C=C\\C(=O)O',
    # ]
    
    # for query in query_list:
    #     start_query_time = time()
    #     # 搜索时调整 nprobe, nprobe越大越准但越慢
    #     results = searcher.search(query, min_similarity=0.2, top_k=3, nprobe=20)
        
    #     if results:
    #         print(f"\nTop {len(results)} matches for '{query}':")
    #         print(results)
    #         print("*  "*30)
    #     else:
    #         print(f"No matches found for '{query}'")
        
    #     end_query_time = time()
    #     print(f"Searching Time cost: {end_query_time-start_query_time:.2f}")

if __name__ == "__main__":
    # print("--- Building Reactant Optimized Index ---")
    # build_index_no_pca(
    #     # rxn_tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_v3/choriso_reorgnize_search.tsv",
    #     # index_save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_v3",
        
    #     rxn_tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt/gt_rxn_search.tsv",
    #     index_save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt",
        
    #     query_type='reactants',
    # )
    
    # print("--- Building Product Optimized Index ---")
    # build_index_no_pca(
    #     # rxn_tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_v3/choriso_reorgnize_search.tsv",
    #     # index_save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_v3",
        
    #     rxn_tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt/gt_rxn_search.tsv",
    #     index_save_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt",
        
    #     query_type='products',
    # )
    
    print("\n--- Running Optimized Search Demo ---")
    search_demo_optimized(
        tsv_path="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt/gt_rxn_search.tsv",
        index_save_dir="/cto_labs/lihao/chem_reason/ChemSearch/search_saves/datasets/rxn_search_gt",
    )
    
    
    # {'reactant': 'COc1ccc(COc2ccc(Cn3c(N)nc4cc(-c5cnn(CC(=O)OC(C)(C)C)c5)cnc43)cc2OC)cc1.O[Na]', 'product': 'COc1ccc(COc2ccc(Cn3c(N)nc4cc(-c5cnn(CC(=O)O)c5)cnc43)cc2OC)cc1', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CCOc1cc(C(C)(C)C)ncc1C1=NC(C)(c2ccc(Cl)cc2)C(C)(c2ccc(Cl)cc2)N1C(=O)N1CCN(CC(=O)OC(C)(C)C)CC1.[Li]O', 'product': 'CCOc1cc(C(C)(C)C)ncc1C1=NC(C)(c2ccc(Cl)cc2)C(C)(c2ccc(Cl)cc2)N1C(=O)N1CCN(CC(=O)O)CC1', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'COCCn1ccc2c(Nc3ncc(C4CC4)cc3C(=O)OC(C)(C)C)cccc21.O[Na]', 'product': 'COCCn1ccc2c(Nc3ncc(C4CC4)cc3C(=O)O)cccc21', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CC(=O)c1nn(CC(=O)OC(C)(C)C)c2c3c(c(-c4cnc(C)nc4)cc12)CCCO3.[Li]O', 'product': 'CC(=O)c1nn(CC(=O)O)c2c3c(c(-c4cnc(C)nc4)cc12)CCCO3', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CC(OCC(=O)OC(C)(C)C)c1sccc1Br.O[Na]', 'product': 'CC(OCC(=O)O)c1sccc1Br', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'Cc1ncc(-c2ccc3c(c2)c(C2OCCO2)nn3CC(=O)OC(C)(C)C)cn1.[Li]O', 'product': 'Cc1ncc(-c2ccc3c(c2)c(C2OCCO2)nn3CC(=O)O)cn1', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'O.Cc1c(Sc2ccc(Cl)cc2)c2c(-c3ccccc3)cccc2n1CC(=O)OC(C)(C)C', 'product': 'Cc1c(Sc2ccc(Cl)cc2)c2c(-c3ccccc3)cccc2n1CC(=O)O', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'CC(C)(C)OC(=O)Cn1cc(C2NS(=O)(=O)c3ccccc32)c2cc(Cl)ccc21.O[Na]', 'product': 'O=C(O)Cn1cc(C2NS(=O)(=O)c3ccccc32)c2cc(Cl)ccc21', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'O.CCC1CC1(C(=O)OC(C)(C)C)C(=O)OC(C)(C)C', 'product': 'CCC1CC1(C(=O)O)C(=O)OC(C)(C)C', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    # {'reactant': 'O.COc1ccc(S(=O)(=O)N(Cc2ccccc2N)C(CO)C(=O)OC(C)(C)C)cc1', 'product': 'COc1ccc(S(=O)(=O)N(Cc2ccccc2N)C(CO)C(=O)O)cc1', 'reagent': 'empty', 'solvent': 'empty', 'catalyst': 'empty', 'yield': 0.0}
    
    