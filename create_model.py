from __future__ import print_function
from rcnn.utils.load_model import load_checkpoint
from rcnn.utils.save_model import save_checkpoint

def print_model(prefix, epoch):
    args1, auxs1 = load_checkpoint(prefix, epoch)
    arg_names = sorted(args1.keys())
    aux_names = sorted(auxs1.keys())
    print('\nargs:')
    print(len(arg_names))
    for arg in arg_names:
        print(arg, args1[arg].shape)

    print('\naux:')
    print(len(aux_names))
    for aux in aux_names:
        print(aux, auxs1[aux].shape)

def delete_model(prefix, epoch):
    args1, auxs1 = load_checkpoint(prefix, epoch)
    arg_names = sorted(args1.keys())
    aux_names = sorted(auxs1.keys())
    # args = dict()
    print('\nargs:')
    print(len(arg_names))
    for arg in arg_names:
        if arg.startswith('conv') or arg.startswith('cls') or arg.startswith('bbox') or arg.startswith('fc') or arg.startswith('rpn'):
            args1.pop(arg)
        elif arg.startswith('stage') or arg.startswith('bn'):
            new_name = 'tl_' + arg
            args1[new_name] = args1[arg]
            args1.pop(arg)
            print(arg)

    print('\naux:')
    print(len(aux_names))
    for aux in aux_names:
        if aux.startswith('stage') or aux.startswith('bn'):
            new_name = 'tl_' + aux
            auxs1[new_name] = auxs1[aux]
            auxs1.pop(aux)
            print(aux)
    prefix_out = '/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/pretrained/resnet-18_2block_cut'
    save_checkpoint(prefix_out, 0, args1, auxs1)

def change_model(prefix, epoch):
    args1, auxs1 = load_checkpoint(prefix, epoch)
    arg_names = sorted(args1.keys())
    aux_names = sorted(auxs1.keys())

    new_args, new_auxs = {}, {}
    # args = dict()
    print('\nargs:')
    print(len(arg_names))
    for arg in arg_names:
        if arg.startswith('stage1'):
            args1.pop(arg)
        print(arg)

    print('\naux:')
    print(len(aux_names))
    for aux in aux_names:
        if aux.startswith('stage1'):
            auxs1.pop(aux)
        print(aux)

    for arg in sorted(args1.keys()):
        if arg.startswith('stage2'):
            new_arg = arg.replace('stage2', 'stage1')
        elif arg.startswith('stage3'):
            new_arg = arg.replace('stage3', 'stage2')
        else:
            new_arg = arg
        new_args[new_arg] = args1[arg]

    for aux in sorted(auxs1.keys()):
        if aux.startswith('stage2'):
            new_aux = aux.replace('stage2', 'stage1')
        elif aux.startswith('stage3'):
            new_aux = aux.replace('stage3', 'stage2')
        else:
            new_aux = aux
        new_auxs[new_aux] = auxs1[aux]
            
    prefix_out = '/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/pretrained/vgg-16_resnet-18_3block_cut'
    save_checkpoint(prefix_out, 0, new_args, new_auxs)

def convert_open_to_mask(prefix1, epoch1, prefix2, epoch2, prefix_out):
    args1, auxs1 = load_checkpoint(prefix1, epoch1)
    args2, auxs2 = load_checkpoint(prefix2, epoch2)
    arg_names1 = sorted(args1.keys())
    aux_names1 = sorted(auxs1.keys())

    arg_names2 = sorted(args2.keys())
    aux_names2 = sorted(auxs2.keys())

    new_args = dict()
    print('\narg')
    for arg in arg_names2:
        if arg.startswith('P') and 'aggregate' in arg:
            temp = arg.replace('aggregate', 'conv')
            new_args[temp] = args2[arg]
        else:
            assert arg in arg_names1, "{} not in mask model".format(arg)
            new_args[arg] = args2[arg]

    print('\naux')
    new_auxs = dict()
    for aux in aux_names2:
        assert aux in aux_names1, "{} not in mask model".format(aux)
        new_auxs[aux] = auxs2[aux]

    save_checkpoint(prefix_out, 0, new_args, new_auxs)


if __name__ == "__main__":
    # model_path2 = '/mnt/truenas/scratch/siyu/mx-maskrcnn/model/'
    # model_path1 = '/mnt/truenas/scratch/siyu/maskrcnn/model/res50-fpn/coco/alternate_detection/'
    # model_path_out = '/mnt/truenas/scratch/siyu/mx-maskrcnn/model/converted'
    # rpn_epoch = 6
    # rcnn_epoch = 9
    # prefix = 'final'
    # epoch = 0
    # convert_open_to_mask(model_path1 + prefix, epoch, model_path2+prefix, epoch, model_path_out+prefix)
    # model_path = '/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/taillight_tucson_detnet_pair_3block/taillight_tucson_detnet'
    model_path = '/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/pretrained/vgg-16_resnet-18_2block_cut'
    # check_model(model_path, 0)
    # print_model(model_path, 0)
    delete_model(model_path, 0)
    # print_model('/mnt/truenas/scratch/siyu/deep-vehicle-intention/inte/models/pretrained/tucson_forward_v12_2018_11_14_res50_final', 0)
    # change_model(model_path, 0)
