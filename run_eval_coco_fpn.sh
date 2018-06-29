#@IgnoreInspection BashAddShebang
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONUNBUFFERED=1
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

TRAIN_DIR=model/res50-fpn/coco/alternate_detection/
DATASET=coco
#SET=train2014+valminusminival2014
TEST_SET=train2014
mkdir -p ${TRAIN_DIR}
#GPU=0,1,2,3,4,5,6,7
GPU=0

mkdir -p eval_ins
rm -rf eval_ins/*
find ./ -name "*pyc" | xargs rm

# Test
python eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}final \
    --epoch 0 \
    --gpu $GPU
