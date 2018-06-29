export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONUNBUFFERED=1

MODEL_PATH=model/res50-fpn/cityscape/alternate_maskfcn_gap_g4_renorm/
RESULT_PATH=data/cityscape/results/maskfcn_gap/

PREFIX=${MODEL_PATH}final
DATASET=Cityscape
SET=train
TEST_SET=val

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
