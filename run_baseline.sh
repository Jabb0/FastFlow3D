#!/bin/bash
set -e

if [[ "$VIRTUAL_ENV" == "" ]]
then
        echo "not inside virtual env!"
        exit 1
fi

echo "Make sure to log to the correct wandb project!"

# Run a training run
# Batch size is PER GPU
# Num workers is likely PER GPU and each gpu has another process as well
python train.py \
../data/preprocessed/full \
FlowNet3D_batchSize_16_lr_0.001_BN_full \
--architecture FlowNet \
--n_points 2048 \
--accelerator ddp \
--sync_batchnorm True \
--batch_size 16 \
--gpus 4 \
--num_workers 4 \
--wandb_enable True \
--wandb_project fastflow3d \
--learning_rate 0.001 \
--disable_ddp_unused_check True
