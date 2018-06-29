#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_CPU_WORKER_NTHREADS=24
export PYTHONUNBUFFERED=1
#export MXNET_ENGINE_TYPE=NaiveEngine
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

MODEL_NAME=alternate_panseg_fpn_consist
TRAIN_DIR=model/res50-fpn/cityscape/${MODEL_NAME}/
DATASET=Cityscape
SET=train
TEST_SET=val
#mkdir -p ${TRAIN_DIR}
#cp -r model/res50-fpn/cityscape/alternate_panseg_baseline/cache ${TRAIN_DIR}

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
python train_alternate_mask_fcn_stage2.py \
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
python train_alternate_mask_fcn_stage4.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/${PRETRAIN} \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu ${GPU}

if [[ -f ${TRAIN_DIR}/final-0000.params ]]; then
    bash scripts/eval_all.sh -m ${MODEL_NAME} -p final
fi
