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
../data/flyingthings/flyingthings3d_preprocessed \
FlowNet3D_batchSize_16_lr_0.001_BN_flying \
--architecture FlowNet \
--dataset flying_things \
--n_points 16000 \
--accelerator ddp \
--sync_batchnorm True \
--batch_size 4 \
--gpus 4 \
--num_workers 4 \
--wandb_enable True \
--wandb_project fastflow3d \
--learning_rate 0.001 \
--disable_ddp_unused_check True