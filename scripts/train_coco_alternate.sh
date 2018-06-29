#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONUNBUFFERED=1
#export MXNET_ENGINE_TYPE=NaiveEngine
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

TRAIN_DIR=model/res50-fpn/coco/alternate_detection/
DATASET=coco
SET=train2017
#TEST_SET=minival2014
mkdir -p ${TRAIN_DIR}

# Train
echo "current workspace: $(pwd)"

GPU=0
#GPU=0

# Train
python train_alternate_mask_fpn_stage1.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/resnet-50 \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu $GPU && \
python train_alternate_mask_fpn_stage2.py \
   --network resnet_fpn \
   --dataset ${DATASET} \
   --image_set ${SET} \
   --root_path ${TRAIN_DIR} \
   --pretrained model/resnet-50 \
   --prefix ${TRAIN_DIR} \
   --pretrained_epoch 0 \
   --gpu $GPU && \
python train_alternate_mask_fpn_stage3.py \
   --network resnet_fpn \
   --dataset ${DATASET} \
   --image_set ${SET} \
   --root_path ${TRAIN_DIR} \
   --pretrained model/resnet-50 \
   --prefix ${TRAIN_DIR} \
   --pretrained_epoch 0 \
   --gpu $GPU && \
python train_alternate_mask_fpn_stage4.py \
   --network resnet_fpn \
   --dataset ${DATASET} \
   --image_set ${SET} \
   --root_path ${TRAIN_DIR} \
   --pretrained model/resnet-50 \
   --prefix ${TRAIN_DIR} \
   --pretrained_epoch 0 \
   --gpu $GPU
