import torch

# 假设 IGNORE_INDEX 和 compute_position_id_with_mask 已经定义
IGNORE_INDEX = -100

def __getitem__(self, item):
    tokenizer = self.tokenizer

    prompt = self.prompts[item]
    response = self.responses[item]

    # apply chat template
    prompt_chat = [{"role": "user", "content": prompt}]

    # string
    prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
    response_chat_str = response + tokenizer.eos_token

    # tokenize
    prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_ids_output["input_ids"][0]
    prompt_attention_mask = prompt_ids_output["attention_mask"][0]

    response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
    response_ids = response_ids_output["input_ids"][0]
    response_attention_mask = response_ids_output["attention_mask"][0]

    prompt_length = prompt_ids.shape[0]
    response_length = response_ids.shape[0]

    # 核心修改点：loss_mask 的生成和对 prompt_ids 的修改
    # 这里我们只对 prompt_ids 进行操作，不对 response_ids 做任何处理
    prompt_labels = prompt_ids.clone()
    
    # 获取 <result> 和 </result> 的 token IDs
    start_marker_str = "<result>"
    end_marker_str = "</result>"
    start_marker = tokenizer.encode(start_marker_str, add_special_tokens=False)
    end_marker = tokenizer.encode(end_marker_str, add_special_tokens=False)
    
    inside_result_tag = False
    i = 0
    while i < len(prompt_labels):
        # 检查是否是 <result> 的开始
        if not inside_result_tag and i + len(start_marker) <= len(prompt_labels) and all(prompt_labels[i + j] == start_marker[j] for j in range(len(start_marker))):
            inside_result_tag = True
            # 将 <result> 标签本身也忽略
            for j in range(len(start_marker)):
                if i + j < len(prompt_labels):
                    prompt_labels[i + j] = IGNORE_INDEX
            i += len(start_marker)
            continue
        
        # 检查是否是 </result> 的结束
        if inside_result_tag and i + len(end_marker) <= len(prompt_labels) and all(prompt_labels[i + j] == end_marker[j] for j in range(len(end_marker))):
            inside_result_tag = False
            # 将 </result> 标签本身也忽略
            for j in range(len(end_marker)):
                if i + j < len(prompt_labels):
                    prompt_labels[i + j] = IGNORE_INDEX
            i += len(end_marker)
            continue

        # 如果在 <result>...</result> 内部，则设置 prompt_labels 为 IGNORE_INDEX
        if inside_result_tag:
            if i < len(prompt_labels):
                prompt_labels[i] = IGNORE_INDEX
        
        i += 1

    # 在 Verl 框架中，我们不需要显式修改 attention_mask
    # Verl 的训练通常依赖于 loss_mask 来计算损失，而 attention_mask 保持不变，用于表示序列的有效部分。
    # 理论上，attention_mask 和 loss_mask 是两个不同的概念：
    # attention_mask: 告诉模型哪些 token 是真实的输入（1），哪些是填充（0）。
    # loss_mask: 告诉损失函数哪些 token 需要计算损失（1），哪些需要忽略（0）。
    
    # 结合 prompt 和 response 的 token IDs
    input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

    # 结合 prompt 和 response 的 loss_mask
    # 这里的 loss_mask 应该与你的 Verl 框架中的逻辑一致
    # 原始的 prompt loss_mask 通常全为 0
    # 我们这里使用 `prompt_labels` 来判断
    # 也就是说，只有 `prompt_labels` 不等于 `IGNORE_INDEX` 的部分才参与损失计算
    prompt_loss_mask = (prompt_labels != IGNORE_INDEX).to(torch.long)
    response_loss_mask = torch.ones_like(response_ids)
    loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

    # Padding and truncation remains the same
    sequence_length = input_ids.shape[0]
    if sequence_length < self.max_length:
        padded_input_ids = (
            torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
            * self.tokenizer.pad_token_id
        )
        padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
        padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)
        
        input_ids = torch.cat((input_ids, padded_input_ids))
        attention_mask = torch.cat((attention_mask, padded_attention_mask))
        loss_mask = torch.cat((loss_mask, padded_loss_mask))
    elif sequence_length > self.max_length:
        if self.truncation == "left":
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]
            loss_mask = loss_mask[-self.max_length :]
        elif self.truncation == "right":
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
        elif self.truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {self.truncation}")

    position_ids = compute_position_id_with_mask(attention_mask)

    # 在 Verl 中，通常需要提供 labels 和 loss_mask
    # labels 是输入序列的下一个词，loss_mask 决定哪些位置计算损失
    # 这里的 labels 应该就是 input_ids 的右移版本
    labels = input_ids.clone()
    labels[:-1] = input_ids[1:]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
    }