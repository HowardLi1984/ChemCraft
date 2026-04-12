# ChemCraft ⚗️

**Official Implementation of "Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-huggingface-blue)](https://huggingface.co/OzymandisLi/ChemCraft-7B)
[![ArXiv](https://img.shields.io/badge/Arxiv-2601.17687-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2601.17687)
[![Dataset](https://img.shields.io/badge/Dataset-huggingface-blue)](https://huggingface.co/datasets/OzymandisLi/ChemCraft-Agent-Trajectory/)

## 📢 News & Updates

- **[2026-01-26] Paper Available:** Due to the extended moderation process on arXiv, we are temporarily hosting the **pre-print draft** directly in this repository. You can find the PDF in the `assets/` folder or download it [here](./assets/paper_draft.pdf). We will update this section with the official arXiv link as soon as it becomes available.
- **[2026-04-12] Code & Weights & Dataset Available** We have finalized the code / weight / dataset cleanup and documentation. The full source code and model weights have been open-sourced!

## 🚀 Getting Started

### 1. Model Weights Preparation
The pre-trained weights for **ChemCraft-7B** are hosted on Hugging Face.

| Model | Link | Size |
| :--- | :--- | :--- |
| **ChemCraft-7B** | [🤗 Hugging Face](https://huggingface.co/OzymandisLi/ChemCraft-7B) | ~155 GB |

**Installation:**
Create a `checkpoint/` directory in the root of this repository and download the weights into it:
```bash
mkdir -p checkpoint/
# Using huggingface-cli to download
huggingface-cli download OzymandisLi/ChemCraft-7B --local-dir checkpoint/
```

### 2. Dataset & Agent Trajectories
While this repository contains basic data samples in `dataset/`, the complete **Chemical Agent Sandbox** data (including massive intermediate trajectories and construction scripts) is hosted separately on Hugging Face Datasets.

| Dataset | Link | Size |
| :--- | :--- | :--- |
| **ChemCraft-Agent-Trajectory** | [📊 HF Dataset](https://huggingface.co/datasets/OzymandisLi/ChemCraft-Agent-Trajectory) | ~1.5 GB |

> [!IMPORTANT]
> To fully utilize ChemCraft's capabilities, you must replace the local `dataset/` folder with the complete content from Hugging Face.

### 3. Training & Execution Scripts

We provide a complete pipeline to reproduce **ChemCraft-7B**, covering trajectory generation, environment setup, and the two-stage training process (SFT & RL).

**[1] Chemical Agent Trajectory Construction:** Generate the raw trajectories for agentic reasoning and convert them into the final training format.
```bash
# Step A: Generate tool-use SFT trajectories
cd dataset/chemcot/colde_start_v2 && python get_tool_sft_trajectory.py

# Step B: Process data into parquet format
cd ../ && python get_parquet.py
```

**[2] Launch Chemical Agent Sandbox:** Start the sandbox environment to support agent-based retrieval and molecular interaction.
```bash
cd chemcraft_rl/ && bash retriever_chemagent.sh
```

**[3] Cold-start SFT (Supervised Fine-tuning):** Train the initial policy model by fine-tuning Qwen-7B-Instruct into ChemCraft-7B-coldstart.
```bash
cd chemcraft_sft/ && bash train_coldstart_sft_7B.sh
```

**[4] SMILES-GRPO Reinforcement Learning:** Advance the model through Group Relative Policy Optimization (GRPO) to obtain the final ChemCraft-7B.
```bash
cd chemcraft_rl/ && bash train_qwenagent_withrxn.sh
```

## 🚀 Environment Setup

This project requires three distinct environments for SFT, RL training, and Retrieval tasks. We recommend using `conda` for environment management.

---

### 1. SFT Training Environment
Focused on Supervised Fine-Tuning with high-performance inference support.

```bash
# Create environment
conda create -n llm_sft python=3.10 -y
conda activate llm_sft

# Install Core Dependencies (PyTorch & CUDA 12.4)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install Flash-Attention (Pre-built Wheel)
pip install [https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)

# Install Extension Libraries
pip install peft==0.17.1 vllm==0.6.3
pip install torch_memory_saver -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple) # Required for latest sglang
```

### 2. Verl-Latest for RL Training
The core environment for Reinforcement Learning (RL) using the verl framework.

```bash
# Create environment
conda create -n verl-latest python=3.10.18 -y
conda activate verl-latest

# Install PyTorch 2.8.0
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# Install Verl from source
cd verl/
pip install -e . -c constraints.txt
pip install -e .[sglang] -c constraints.txt

# Install Science & NLP dependencies
pip install transformers==4.56.1 rdkit nltk rouge-score python-Levenshtein selfies -c constraints.txt

# Install specific Flash-Attention version
pip install [https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
```

### 3. Retriever Environment
Designed for chemical structure retrieval and vector database management.

```bash
conda create -n retriever python=3.10.18 -y
conda activate retriever

# Chemistry and Bio-Library
pip install rdkit rdchiral pytdc -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# API and Vector DB
pip install fastapi pydantic uvicorn faiss-gpu -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```

## 🛠 Troubleshooting

#### 1. PyTorch / MKL Symbol Errors
**Error:** `ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`  
**Description:** This typically occurs when there is a version mismatch in the Math Kernel Library (MKL) used by PyTorch, often seen in specific conda environments.  
**Solution:** Manually install a compatible MKL version to resolve the symbol link:
```bash
pip install mkl==2024.0
```

#### 2. Flash-Attention Installation & Import Issuess
Flash-Attention installation can fail due to CUDA/PyTorch version mismatches or build environment issues.

##### Case A: Undefined Symbol during Import
**Error:** `ImportError: ...flash_attn_2_cuda...so: undefined symbol: _ZN3c105Error...`  
**Description:** This indicates the binary was built against a different version of PyTorch than the one currently installed. 
**Solution:** Avoid building from source without isolation. Use the following command to build with a specific job limit, or use a pre-built wheel:
```bash
# Option 1: Reinstall with no-build-isolation
MAX_JOBS=8 pip install flash-attn==2.8.0.post2 --no-build-isolation

# Option 2: Use the stable pre-built wheel (Recommended for CUDA 12.4 + Torch 2.4)
pip install [https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
```

##### Case B: Shared Object File Missing (libcudart.so.11.0)
**Error:** `ImportError: libcudart.so.11.0: cannot open shared object file: No such file or directory`  
**Description:** Your system/environment uses CUDA 12.x, but the installed Flash-Attention version was compiled for CUDA 11.x. 
**Solution:** Ensure you download and install the wheel specifically marked with +cu12.

## 🔗 Citations and Related Projects
If you find our ChemCraft-7B helpful, please cite it as follows:
```bibtex
@article{li2026agentic,
  title={Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis},
  author={Li, Hao and Cao, He and Peng, Shenyao and Liu, Zijing and Feng, Bin and Wang, Yu and Yan, Zhiyuan and Tian, Yonghong and Li, Yu and Yuan, Li},
  journal={arXiv preprint arXiv:2601.17687},
  year={2026}
}
```

We highly recommend checking out our related benchmark work: **ChemCotBench-v1**
* **Paper:** [NeurIPS-2025 - ChemCotBench](https://arxiv.org/pdf/2505.21318?)
* **Dataset:** [Hugging Face - ChemCotDataset](https://huggingface.co/datasets/OpenMol/ChemCoTDataseth)
* **Benchmark:** [Hugging Face - ChemCotBench](https://huggingface.co/datasets/OpenMol/ChemCoTBench)
```bibtex
@article{li2025beyond,
  title={Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations},
  author={Li, Hao and Cao, He and Feng, Bin and Shao, Yanjun and Tang, Xiangru and Yan, Zhiyuan and Yuan, Li and Tian, Yonghong and Li, Yu},
  journal={arXiv preprint arXiv:2505.21318},
  year={2025}
}
```
