
export RAY_DEBUG_POST_MORTEM=1 # for debug
export MY_HOST_IP=127.0.0.1 # to fix the "The server socket on [team-slb-api]:30502 has timed out" bug
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export n_gpus_per_node=8

CHECKPOINT_CONTENTS=['model','hf_model','optimizer','extra'] # save on huggingface format

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_MODEL="$PROJECT_DIR/checkpoints/coldstart_models/chemcot-tool-coldstart-rxn-qwen-7B-coldstart-55-murcko-82/global_step_250"
WAND_PROJECT='ChemAgent-Qwen-7B-with-RXN'
EXPERIMENT_NAME="chemagent-epoch-3-lr-1e-6"

set -x
ulimit -n 65535

CONFIG_PATH="$PROJECT_DIR/chemcraft_rl/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/chem_tool_config_v3.yaml"

TRAIN_DATA="$PROJECT_DIR/dataset/chemcot/chemcot_json/chemcot-tool-coldstart/train_tool_full_rxn.parquet"
VAL_DATA="$PROJECT_DIR/dataset/chemcot/chemcot_json/chemcotbench/chemcotbench_with_rxn.parquet"

# 支持每个sample的工具调度数量 - actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10
# data.max_response_length 在default情况下会付给 actor_rollout_ref.rollout.response_length, 表示每次rollout的最大长度

## default parameter
# export data_train_batch_size=512
# export actor_ppo_mini_batch_size=256
# export actor_ppo_micro_batch_size=64
# export val_batch_size=256

# smaller
export data_train_batch_size=32
export actor_ppo_mini_batch_size=16
export actor_ppo_micro_batch_size=4
export val_batch_size=128

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$data_train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=6144 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_CONTENTS} \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=30 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=3 $@ \
    trainer.val_before_train=False \
    trainer.ray_wait_register_center_timeout=1000

