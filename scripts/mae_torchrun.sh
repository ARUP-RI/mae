#!/usr/bin/bash

# MASTER_ADDR=$1
# MASTER_PORT=$2
# REPO_ROOT=$3
# GPUS_PER_NODE=$4
GPUS_PER_NODE=4
# RESULTS_ROOT=$5
# RUN_NAME=$6
# MODEL_CONFIG=$(realpath "$7")

PYTHON_ENV=/home/$USER/miniforge3/envs/fb_mae/

let "NUM_WORKERS = 4 * $GPUS_PER_NODE"

DATA_ROOT=/mnt/ri_share/Data/vision_datasets/nvidia_hemepath_pretrain/


#RUNCMD="$REPO_ROOT/scripts/listgpus.py"
RUNCMD="$HOME/git/mae/main_pretrain.py \
    --batch_size 160 \
    --ckpt_freq 50 \
    --model mae_vit_large_patch16 \
    --blr 3e-4 \
    --warmup_epochs 2 \
    --data_path $DATA_ROOT \
    --path_csv $DATA_ROOT/afewcells.csv \
    --num_workers=$NUM_WORKERS"

$PYTHON_ENV/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 $RUNCMD

    # --resume /home/31716/git/mae/mae_finetuned_vit_large.pth \