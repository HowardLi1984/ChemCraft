
## Introduction for this DIR

- cold_start_v1: 最早的tool引入尝试, 工具只考虑<rdkit>, <search>这两个, 构建冷起数据的时候没考虑result对后续token的影响

- cold_start_v2: 结合了MiroFlow之类的经验后, 首先Tool-Platform更全面(结合ChemToolAgent的经验), 其次使用二阶段来构建冷启数据：
    - stage-1 QWen把化学推理轨迹的不确定的地方加入<tool>以及中间的query
    - stage-2 使用<tool-api>加入<result>
    - stage-3: 在化学推理轨迹+<tool>+<result>的基础上, 把他们整体重新输入给LLM, 重新根据LLM的思路让他think一边，确保轨迹和即将训练的小模型轨迹距离不大