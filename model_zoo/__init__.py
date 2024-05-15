from .badnet import BadNet
from . import resnet

def get_model(args, num_classes):
    ### Create Model and Training
    if args.model_name == 'badnet':
        return BadNet(input_channels=3, output_num=num_classes).to(args.device)
    elif 'resnet' in args.model_name:
        return resnet.build_model(args.model_name, num_classes).to(args.device)
    else:
        raise ValueError
