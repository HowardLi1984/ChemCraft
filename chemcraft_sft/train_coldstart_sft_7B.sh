export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='../dataset/chemcot/chemcot_json/chemcot-tool-coldstart'
WAND_PROJECT='Qwen-ChemCraft-SFT'

# export BASE_MODEL='../search_saves/checkpoints/qwen/Qwen2.5-32B-Instruct'
# export EXPERIMENT_NAME=chemcot-tool-coldstart-qwen-2.5-32B-instruct

## qwen-2.5-7B coldstart by lihao
export BASE_MODEL='../checkpoints/base_models/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=chemcot-coldstart-tool-qwen-7B-coldstart-55-murcko-82

export SAVE_PATH=../checkpoints/coldstart_models/$EXPERIMENT_NAME

export VLLM_ATTENTION_BACKEND=XFORMERS

nproc_per_gpu=1
nnodes=1
ngpu_per_node=8

torchrun --standalone --nnodes=${nnodes} --nproc_per_node=$ngpu_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.use_shm=True \
    data.train_files=$DATA_DIR/test_tool_full_rxn55_murcko28.parquet \
    data.val_files=$DATA_DIR/test_tool_full_rxn55_murcko28.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.prompt_dict_keys=null \
    data.response_dict_keys=null \
    optim.lr=2e-5 \
    data.train_batch_size=48 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 \
    data.truncation=right \
    ulysses_sequence_parallel_size=1 \
    model.use_shm=True \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    model.target_modules=all-linear \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=700 \
    trainer.save_freq=50 \
    trainer.total_epochs=4 \
    trainer.default_hdfs_dir=null $@ \
    2>&1 | tee ../search_saves/logs/$EXPERIMENT_NAME.log