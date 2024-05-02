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
from model.model import UNet, SpectralDiscriminator, Wavelet_Segmentation, Wavelet_Segmentation_GN
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
        "--gpu", type=int, default='6', help="GPU"
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
    
    net_G = Wavelet_Segmentation(config).to(device)
    
    ### CLIP Condition model ### 
    clip_model, _ = clip.load(name = config.clip.clip_model, device=device, jit=False)
    clip_model.eval()

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

    ### Wavelet Pooling ### 
    dwt = DWTForward(J=1, mode='zero', wave='haar').float().cuda()
    iwt = DWTInverse(mode='zero', wave='haar').float().cuda()

        
    ### evaluate ###
    for i,inputs in  enumerate(test_loader):
        net_G.eval()

        input_image = inputs['image'].cuda()
        target = inputs['target'].cuda()

        input_clip = inputs['image_clip'].cuda()

        ### wavelet input ### 
        xll, xh = dwt(input_image)  # [b, 3, h, w], [b, 3, 3, h, w]
        xlh, xhl, xhh = torch.unbind(xh[0], dim=2)

        input_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [b, 12, h, w]
        input_data = input_data  /2.0 

        assert -1 <= input_data.min() < 0
        assert 0 < input_data.max() <= 1

        # text = clip.tokenize(prompt).to(device)

        latent_z = torch.randn(1, config.wavelet_model.nz, device=device)
        clip_feature = clip_model.encode_image(input_clip) #+ clip_model.encode_text(text)

        predict = net_G(x = input_data, z = latent_z, clip_feature = clip_feature)   

        predict = predict * 2

        predict_sample = iwt((predict[:, :1], [torch.stack(
            (predict[:, 1:2], predict[:, 2:3], predict[:, 3:4]), dim=2)]))
        
        predict_sample = torch.sigmoid(predict_sample)
        
        predict_sample = torch.round(predict_sample)
        
        torchvision.utils.save_image(
                    predict, 
                    os.path.join(exp_path, 'test/pred_{}.png'.format(i+1))
                    )
        torchvision.utils.save_image(
                    target, 
                    os.path.join(exp_path, 'test/gt_{}.png'.format(i+1))
                    )

if __name__ == "__main__":
    main()