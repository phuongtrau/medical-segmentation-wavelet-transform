import os
import sys
import time
import yaml

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import argparse

import numpy as np
import torch.optim as optim
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MonusegData
from model.model import UNet, SpectralDiscriminator, Wavelet_Segmentation
from model.losses import VGGLoss, DiceBCELoss, DiceLoss, edge_aware_loss

from tqdm import tqdm
import random
import clip


from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from utils import calculate_iou

def parse_args_and_config():

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    ### training ### 

    parser.add_argument(
        "--gpu", type=int, default='0', help="GPU"
    )

    parser.add_argument('--ckpt', default='netG_100.pth')
    ### config for the model ###
    parser.add_argument(
        "--config", type=str, default= "monuseg.yml", help="Path to the config file"
    )
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config

def dict2namespace(config):
    
    namespace = argparse.Namespace()
    
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)

    return namespace
    
def main():

    args, config = parse_args_and_config()
    print("=========================================================")
    print(config)
    print("=========================================================")
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0')

    test_data = MonusegData(path= config.data.data_path, mode='test')

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True,
                          num_workers=4, pin_memory=False, drop_last=True)

    net_G = UNet(n_channels=3, n_classes=1).to(device)
    
    ### logging experiment ###
    ### experiment name ###
    parent_dir = "/mnt/hdd_1A/ds_project/result" 
    exp_path = os.path.join(parent_dir, config.model.name)
    if not os.path.exists(os.path.join(exp_path,'test')):
        os.makedirs(os.path.join(exp_path,'test'))
    ### load model ###
    if os.path.exists(os.path.join(exp_path, args.ckpt)):
        
        checkpoint_file = os.path.join(exp_path, args.ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # load G
        net_G.load_state_dict(checkpoint)
        
    else:
        raise ValueError (f'input pretrained model please !!!')

    for i,inputs in  enumerate(test_loader):
        net_G.eval()

        input_image = inputs['image'].cuda()
        target = inputs['target'].cuda()
        predict = net_G(input_image)   

        predict = torch.round(predict)
        ### sample image ### 
        
        predicts = torch.cat([predict]*3,1)
        targets = torch.cat([target]*3,1)
        input_image_01 = ((input_image + 1) * 0.5).clamp(0,1)
        
        ### sample image ### 
        combine = torch.cat([input_image_01, predicts, targets], 3)

        torchvision.utils.save_image(
                    predict, 
                    os.path.join(exp_path, 'test/pred_{}.png'.format(i+1))
                    )
        torchvision.utils.save_image(
                    target, 
                    os.path.join(exp_path, 'test/gt_{}.png'.format(i+1))
                    )
        
        torchvision.utils.save_image(
                    combine, 
                    os.path.join(exp_path, 'test/combine_{}.png'.format(i+1))
                    )
    
if __name__ == "__main__":
    main()