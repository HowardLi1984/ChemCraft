# ChemCraft ⚗️

**Official Implementation of "Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-Coming%20Soon-blue)]()
[![ArXiv](https://img.shields.io/badge/Arxiv-2601.17687-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2601.17687)

## 📢 News & Updates

- **[2026-01-26] Paper Availability:** Due to the extended moderation process on arXiv, we are temporarily hosting the **pre-print draft** directly in this repository. You can find the PDF in the `assets/` folder or download it [here](./assets/paper_draft.pdf). We will update this section with the official arXiv link as soon as it becomes available.
- **[Coming Soon] Code & Weights:** We are currently finalizing the code cleanup and documentation. The full source code and model weights are scheduled to be open-sourced **within two weeks**. Please star 🌟 the repo to stay tuned!

## 📄 Abstract

Language models are revolutionizing the biochemistry domain, assisting scientists in drug design and chemical synthesis with high efficiency. Yet current approaches struggle between small language models prone to hallucination and limited knowledge retention, and large cloud-based language models plagued by privacy risks and high inference costs. To bridge this gap, we introduce ChemCraft, a novel framework leveraging agentic reinforcement learning to decouple chemical reasoning from knowledge storage. Instead of forcing the model to memorize vast chemical data, our approach empowers the language model to interact with a sandbox for precise information retrieval. This externalization of knowledge allows a locally deployable small model to achieve superior performance with minimal inference costs. To enable small language models for agent-calling ability, we build an agentic trajectory construction pipeline and a comprehensive chemical-agent sandbox. Based on sandbox interactions, we constructed ChemToolDataset, the first large-scale chemical tool trajectory dataset. Simultaneously, we propose SMILES-GRPO to build a dense chemical reward function, promoting the model's ability to call chemical agents. Evaluations across diverse aspects of drug design show that ChemCraft outperforms current cloud-based LLMs in molecular structure analysis, molecular optimization, and synthesis pathway prediction, demonstrating that scientific reasoning is not solely an emergent ability of model scale, but a learnable policy of tool orchestration. This work establishes a cost-effective and privacy-preserving paradigm for AI-aided chemistry, opening new avenues for accelerating molecular discovery with locally deployable agents.

## 🔗 Related Projects

We highly recommend checking out our related benchmark work:

**ChemCotBench**
* **Paper:** [NeurIPS-2025 - ChemCotBench](https://arxiv.org/pdf/2505.21318?)
* **Dataset:** [Hugging Face - ChemCotDataset](https://huggingface.co/datasets/OpenMol/ChemCoTDataseth)
* **Benchmark:** [Hugging Face - ChemCotBench](https://huggingface.co/datasets/OpenMol/ChemCoTBench)

## 🗓️ Roadmap

- [x] Release pre-print paper draft.
- [ ] Release inference code (Expected within 2 weeks).
- [ ] Release model checkpoints (Expected within 2 weeks).
- [ ] Release training scripts.
- [ ] Update official arXiv link.
