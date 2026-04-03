
# Debugs

- 按照 verl-document 安装完之后报错
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found (required by /home/lihao/miniconda3/envs/verl-latest/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)

下降版本到  flash-attn=2.6.3

- Import flash_attn_2_cuda as flash_attn_gpu 报错 ImportError: libc10.so: cannot open shared object file: No such file or directory
这个东西可以通过, 先 import torch, 再 import flash_attn_2_cuda as flash_attn_gpu 来解决报错

- The server socket on [team-slb-api]:30194 has timed out
这个报错很奇怪，不过可以通过:
1. ray stop
2. ray start --head
多重启几次，解决


改进：
qwen2.5 <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n
1. 需要把数据调成这种格式，然后重新训一边sft OK(2025-10-20)

## 加入RXN之后的调整
1. data.max_prompt_length=6144 从4096扩展为6144, 以适配更长的rxn输入
2. 
3. actor_rollout_ref.rollout.n 从5->2查看是否能够防止timeout

