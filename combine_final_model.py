from rcnn.utils.combine_model import combine_model

# model_path1 = '/mnt/truenas/scratch/siyu/deep-detection/model/train_front_4layers_grad50_phase2/tucson_forward'
model_path1 = '/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/pretrained/resnet_slim_tucson_forward_v12_2018_11_15'
model_path2 = '/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/pretrained/resnet-18_2block_cut' 
rpn_epoch = 50
rcnn_epoch = 0

combine_model(model_path1 + '', rpn_epoch, model_path2 + '', rcnn_epoch, model_path1 + '_final', 0)
