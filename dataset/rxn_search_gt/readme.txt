rxn_search_gt: 为了验证qwen-chemnagent在当前的推理流程下，极限能到什么程度

1. 把ChemCotBench 以及 ChemCoTDataset的所有的RXN数据提取，并整理成csv, 然后用verl里面的rxn_index_builder_v3.py来提取.index, .pkl格式

2. 把GetRXNTemplate的反应模板也收缩一下

3. 评测推理性能