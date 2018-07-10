export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONUNBUFFERED=1

MODEL_PATH=model/res50-fpn/coco/alternate_minitrain/
RESULT_PATH=data/coco/results/alternate_minitrain_final/

PREFIX=${MODEL_PATH}final
DATASET=coco
SET=train
TEST_SET=val2017

mkdir -p ${RESULT_PATH}

python demo_mask.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${PREFIX} \
    --result_path ${RESULT_PATH} \
    --has_rpn \
    --epoch 0 \
    --gpu 0
