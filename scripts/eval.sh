#@IgnoreInspection BashAddShebang
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export PYTHONUNBUFFERED=1
#export MXNET_PROFILER_MODE=1
#export MXNET_PROFILER_AUTOSTART=1

while getopts ":m:p:" opt; do
  case $opt in
    m) MODEL="$OPTARG"
    ;;
    p) PREFIX="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ -z ${MODEL} ] || [ -z ${PREFIX} ]; then
  echo "usage: $0 -m[MODEL] -p[PREFIX]"
  exit -1
fi

echo "MODEL=${MODEL}"
echo "PREFIX=${PREFIX}"

TRAIN_DIR=model/res50-fpn/cityscape/${MODEL}/
DATASET=Cityscape
SET=train
TEST_SET=val

mkdir -p eval_ins/
rm -rf eval_ins/*

# Test
python eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}${PREFIX} \
    --epoch 0 \
    --gpu 0

export CITYSCAPES_RESULTS=eval_ins/
python data/cityscape/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py
