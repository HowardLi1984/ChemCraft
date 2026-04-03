export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='../search_saves/datasets/chemcot/chemcot_json/chemcot-sft'
WAND_PROJECT='Search-R1'

## qwen-2.5-7B sft by lihao
# export BASE_MODEL='/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=chemcot-sft-nips-qwen-2.5-7B-instruct

export BASE_MODEL='/mnt/workspace/lh/ChemSearch/search_saves/checkpoints/qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=chemcot-sft-nips-qwen-2.5-3B-instruct

export SAVE_PATH=../search_saves/verl_checkpoints/$EXPERIMENT_NAME

export VLLM_ATTENTION_BACKEND=XFORMERS

nproc_per_gpu=1
nnodes=1
ngpu_per_node=8
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))
mini_batch_size=$(( total_procs / 4 ))

torchrun --standalone --nnodes=${nnodes} --nproc_per_node=$ngpu_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.use_shm=True \
    data.train_files=$DATA_DIR/train_sft.parquet \
    data.val_files=$DATA_DIR/test_sft.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.prompt_dict_keys=null \
    data.response_dict_keys=null \
    optim.lr=2e-5 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
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
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console'] \
    trainer.total_training_steps=200 \
    trainer.save_freq=10 \
    trainer.total_epochs=4 \
    trainer.default_hdfs_dir=null $@ \
    2>&1 | tee ../search_saves/logs/$EXPERIMENT_NAME.log