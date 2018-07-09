#@IgnoreInspection BashAddShebang
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONUNBUFFERED=1
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

#while getopts ":m:p:" opt; do
#  case $opt in
#    m) MODEL="$OPTARG"
#    ;;
#    p) PREFIX="$OPTARG"
#    ;;
#    \?) echo "Invalid option -$OPTARG" >&2
#    ;;
#  esac
#done
#
#if [ -z ${MODEL} ] || [ -z ${PREFIX} ]; then
#  echo "usage: $0 -m[MODEL] -p[PREFIX]"
#  exit -1
#fi
#
#echo "MODEL=${MODEL}"
#echo "PREFIX=${PREFIX}"

GPU=3

TRAIN_DIR=model/res50-fpn/coco/alternate_detection/
DATASET=coco
TEST_SET=val2017

mkdir -p eval_ins
rm -rf eval_ins/*

# Test
python eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}final \
    --epoch 0 \
    --gpu $GPU
