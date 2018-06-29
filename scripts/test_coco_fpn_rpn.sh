#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=4
#export MXNET_ENGINE_TYPE=NaiveEngine
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

TRAIN_DIR=model/res50-fpn/coco/alternate_detection/
DATASET=coco
SET=minival2014
mkdir -p ${TRAIN_DIR}
GPU=0,1,2,3,4,5,6,7
PRETRAIN=resnet-50

echo "current workspace: $(pwd)"

# Train
python test_fpn_rpn.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/${PRETRAIN} \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu ${GPU}
