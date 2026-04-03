import torch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

def qwen_inference(
    model_path: str,
    system_content: str,
    user_content: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.8
):
    
    # 1. 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    # 2. 设置生成配置
    model.generation_config = GenerationConfig.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p
    
    # 3. 构建对话格式
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # 4. 生成响应
    response = model.chat(
        tokenizer,
        messages,
        generation_config=model.generation_config
    )
    
    return response

# 使用示例
if __name__ == "__main__":
    # 替换为你的模型路径
    model_path = "/mnt/workspace/lh/ChemSearch/search_saves/verl_checkpoints/chemcot-sft-nips-qwen-2.5-7B-instruct/global_step_90"
    
    system_prompt = "你是一个有帮助的AI助手。"
    user_input = "请解释一下量子计算的基本原理。"
    
    output = qwen_inference(
        model_path=model_path,
        system_content=system_prompt,
        user_content=user_input
    )
    
    print("系统提示:", system_prompt)
    print("用户输入:", user_input)
    print("模型输出:", output)