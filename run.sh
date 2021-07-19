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
../data/preprocessed/8_2 \
FastFlowNet_batchSize_16_lr_0.0001_BN_8_2 \
--accelerator ddp \
--sync_batchnorm True \
--batch_size 4 \
--gpus 4 \
--num_workers 4 \
--wandb_enable True \
--wandb_project fastflow3d \
--learning_rate 0.0001 \
--disable_ddp_unused_check True