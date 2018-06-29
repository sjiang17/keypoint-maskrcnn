#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_CPU_WORKER_NTHREADS=24
export PYTHONUNBUFFERED=1
#export MXNET_ENGINE_TYPE=NaiveEngine
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

MODEL=alternate_maskrcnn_mask_head_gaussian_init_check2
TRAIN_DIR=model/res50-fpn/cityscape/${MODEL}/
DATASET=Cityscape
SET=train
TEST_SET=val
mkdir -p ${TRAIN_DIR}

echo "current workspace: $(pwd)"

GPU=0,1,2,3,4,5,6,7
GPU=0
PRETRAIN=resnet-50

# Train
#python train_alternate_mask_fpn_stage1.py \
#    --network resnet_fpn \
#    --dataset ${DATASET} \
#    --image_set ${SET} \
#    --root_path ${TRAIN_DIR} \
#    --pretrained model/${PRETRAIN} \
#    --prefix ${TRAIN_DIR} \
#    --pretrained_epoch 0 \
#    --gpu ${GPU} && \
python train_alternate_mask_fpn_stage2.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/${PRETRAIN} \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu ${GPU} && \
python train_alternate_mask_fpn_stage3.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/${PRETRAIN} \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu ${GPU} && \
python train_alternate_mask_fpn_stage4.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/${PRETRAIN} \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu ${GPU} && \
bash scripts/eval.sh -m ${MODEL} -p final
