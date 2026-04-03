

Environments:

## 安装 llm_sft

conda create -n llm_sft python==3.10
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 接下来安装 flash-attn, 会存在很多问题
pip install flash-attn==2.6.3 --no-build-isolation

    1. 直接安装flash-attn会报错：ImportError: /mnt/workspace/lh/envs/miniconda3/envs/llm_sft/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs

    2. 解决方案：MAX_JOBS=8 pip install flash-attn==2.8.0.post2 --no-build-isolation  装一个新的

(新的安装方案)：pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    这个是从github的编译库中下载好的

如果import flash-attn, 报错 import flash_attn_2_cuda as flash_attn_gpu; ImportError: libcudart.so.11.0: cannot open shared object file: No such file or directory
这说明缺少cuda-11.0对应的链接库，但是你的环境是cuda-12.0，那说明你的flash-attn下错版本了，下载到cuda-11.0对应的flash-attn版本了


# 接下来，执行 pip install -e . -c constraints.txt   防止torch之类的关键版本被改变

pip install peft==0.17.1 (记得这个peft别弄到0.18.0上)
pip install vllm==0.6.3


## 安装 llm_finetune
(这里我把llm_finetune和retriver两个环境, 也就是search-r1和retriever两个功能集成在一起了)

直接执行 pip install -e .
再补充 MAX_JOBS=8 pip install flash-attn==2.6.1 --no-build-isolation
最后安装faiss   pip install faiss-gpu-cu12 --no-build-isolation



# For Latest Environment Install

## 安装 SFT Training 环境

1. conda create -n llm_sft python==3.10
2. conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
3. [Install Flash-Attn]:  pip install https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
4. pip install peft==0.17.1
5. pip install vllm==0.6.3
6. pip install torch_memory_saver -i https://pypi.tuna.tsinghua.edu.cn/simple # the new version of sglang requires torch_memory_saver package


## 安装 Verl-Latest for RL Training

1. conda create -n verl-latest python==3.10.18
2. pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 (if to slow, use: pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple )
3. Enter the verl/ folder, pip install -e . -c constraints.txt, to install verl
4. pip3 install -e .[sglang] -c constraints.txt
5. pip install transformers==4.56.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
5. pip install https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
6.  pip install rdkit nltk rouge-score python-Levenshtein selfies -c constraints.txt


## Installing Retriever Environment
1. conda create -n retriever python==3.10.18
2. pip install rdkit rdchiral -i https://pypi.tuna.tsinghua.edu.cn/simple
3. pip install pytdc -i https://pypi.tuna.tsinghua.edu.cn/simple
4. pip install fastapi pydantic uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
5. pip install faiss-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple


# Some Common Issue:

1. pip install torch, then, import torch, error: ImportError: /hdd/conda/envs/lh_sft/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent

   Solve: pip install mkl==2024.0

2. install flash-attn, many issue
pip install flash-attn==2.6.3 --no-build-isolation

    1. 直接安装flash-attn会报错：ImportError: /mnt/workspace/lh/envs/miniconda3/envs/llm_sft/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs

    2. 解决方案：MAX_JOBS=8 pip install flash-attn==2.8.0.post2 --no-build-isolation  装一个新的

(新的安装方案)：pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    这个是从github的编译库中下载好的

如果import flash-attn, 报错 import flash_attn_2_cuda as flash_attn_gpu; ImportError: libcudart.so.11.0: cannot open shared object file: No such file or directory
这说明缺少cuda-11.0对应的链接库，但是你的环境是cuda-12.0，那说明你的flash-attn下错版本了，下载到cuda-11.0对应的flash-attn版本了