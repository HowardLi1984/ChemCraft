# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import Union

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

IGNORE_INDEX=-100

class SFTToolDataset(Dataset):
    """
    This is an in-memory SFTToolDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, ListConfig], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()        
        # import ipdb; ipdb.set_trace()
        
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

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

        # Ignoring <result>Result</result>
        response_labels = response_ids.clone()
        response_labels = self.ignore_tool_result(response_labels, tokenizer)
        
        # 结合 prompt 和 response 的 token IDs
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # 结合 prompt 和 response 的 loss_mask
        # Verl 框架通常对 prompt 不计算损失
        prompt_loss_mask = torch.zeros_like(prompt_ids)
        # response 部分的 loss_mask 由上面修改过的 response_labels 决定
        response_loss_mask = (response_labels != IGNORE_INDEX).to(torch.long)
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

        # labels 依然是 input_ids 的右移版本
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
    
    def ignore_tool_result(self, response_labels, tokenizer):
        # 获取 <result> 和 </result> 的 token IDs
        start_marker_str = "<result>"
        end_marker_str = "</result>"
        start_marker = tokenizer.encode(start_marker_str, add_special_tokens=False)
        end_marker = tokenizer.encode(end_marker_str, add_special_tokens=False)
        
        inside_result_tag = False
        i = 0
        while i < len(response_labels):
            # 检查是否是 <result> 的开始
            if (not inside_result_tag and
                i + len(start_marker) <= len(response_labels) and
                all(response_labels[i + j] == start_marker[j] for j in range(len(start_marker)))):
                inside_result_tag = True
                # 将 <result> 标签本身也忽略
                for j in range(len(start_marker)):
                    if i + j < len(response_labels):
                        response_labels[i + j] = IGNORE_INDEX
                i += len(start_marker)
                continue
            
            # 检查是否是 </result> 的结束
            if (inside_result_tag and
                i + len(end_marker) <= len(response_labels) and
                all(response_labels[i + j] == end_marker[j] for j in range(len(end_marker)))):
                inside_result_tag = False
                # 将 </result> 标签本身也忽略
                for j in range(len(end_marker)):
                    if i + j < len(response_labels):
                        response_labels[i + j] = IGNORE_INDEX
                i += len(end_marker)
                continue

            # 如果在 <result>...</result> 内部，则设置 response_labels 为 IGNORE_INDEX
            if inside_result_tag:
                if i < len(response_labels):
                    response_labels[i] = IGNORE_INDEX
            
            i += 1
        
        return response_labels
