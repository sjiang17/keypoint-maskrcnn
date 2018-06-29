#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export LD_LIBRARY_PATH=../cuda_cudnn7/lib64

python train_alternate_mask_fpn_stage4.py \
--network resnet_fpn \
--dataset Cityscape \
--image_set train \
--root_path model/res50-fpn/cityscape/alternate/ \
--pretrained model/resnet-50 \
--prefix model/res50-fpn/cityscape/alternate/ \
--pretrained_epoch 0 \
--gpu 0,1,2,3,4,5,6,7