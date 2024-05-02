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
        "--gpu", type=int, default='6', help="GPU"
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )

    parser.add_argument(
        "--num_epoch", type=int, default=500, help="Number of epoch for training"
    )

    parser.add_argument(
        "--shuffle", action='store_true', help='shuffle input data'
    )

    parser.add_argument(
        "--lr_g", type=float, default=0.00005, help="Learning rate with the generator"
    )

    parser.add_argument(
        "--lr_d", type=float, default=0.00005, help="Learning rate with the net_D"
    )

    ### displaying ###
    parser.add_argument(
        "--display_count", type=int, default= 10
    )

    parser.add_argument(
        "--save_count", type=int, default=1
    )

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=5,
                        help='save content for resuming every x epochs')
    
    parser.add_argument('--resume', action='store_true', default=False)
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

def get_optimizer(config, parameters, lr):
    
    if config.optim.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))
    
def main():

    args, config = parse_args_and_config()
    print("=========================================================")
    print(config)
    print("=========================================================")
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0')

    train_data = MonusegData(path= config.data.data_path)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)

    test_data = MonusegData(path= config.data.data_path, mode='test')

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True,
                          num_workers=4, pin_memory=False, drop_last=True)
    
    start_epoch = 0
    net_G = UNet(n_channels=3, n_classes=1).to(device)
    # net_G = Wavelet_Segmentation(config).to(device)

    net_D = SpectralDiscriminator(input_nc=1).to(device)

    optim_G = get_optimizer(config, 
                            filter(lambda p: p.requires_grad, net_G.parameters()), 
                            args.lr_g)
    optim_D = get_optimizer(config, 
                            filter(lambda p: p.requires_grad, net_D.parameters()), 
                            args.lr_d)
    
    # schedu_G = lr_scheduler.CosineAnnealingLR(
    #             optim_G, args.num_epoch, eta_min=1e-5)
    
    # schedu_D = lr_scheduler.CosineAnnealingLR(
    #         optim_D, args.num_epoch, eta_min=1e-5)
    schedu_G = lr_scheduler.MultiStepLR(optim_G, milestones=[1000], gamma=0.5)
    
    schedu_D = lr_scheduler.MultiStepLR(optim_D, milestones=[1000], gamma=0.5)
    
    ### logging experiment ###
    ### experiment name ###
    parent_dir = "/mnt/hdd_1A/ds_project/result" 
    exp_path = os.path.join(parent_dir, config.model.name)
    
    # if rank == 0:
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    ### Logging ### 
    board = SummaryWriter(logdir=os.path.join(exp_path, 'tensorboard'))

    ### load model ###
    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G
        net_G.load_state_dict(checkpoint['netG_dict'])
        optim_G.load_state_dict(checkpoint['optimizerG'])
        schedu_G.load_state_dict(checkpoint['schedulerG'])
        # load D
        net_D.load_state_dict(checkpoint['netD_dict'])
        optim_D.load_state_dict(checkpoint['optimizerD'])
        schedu_D.load_state_dict(checkpoint['schedulerD'])

        step = checkpoint['step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        step, epoch, init_epoch = 0, 0, 0

    ### Define Loss Function ### 
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionDice = DiceLoss()
    # criterionEdge = edge_aware_loss()

    for epoch in range(start_epoch, args.num_epoch + 1):

        for inputs in  tqdm(train_loader):
            net_G.train()
            step +=1

            input_image = inputs['image'].cuda()
            target = inputs['target'].cuda()

            predict = net_G(input_image)   

            predicts = torch.cat([predict]*3,1)
            targets = torch.cat([target]*3,1)
            
            ### loss function ### 
            loss_l1 = criterionL1(predict, target)
            loss_vgg = criterionVGG(predicts, targets)
            loss_dice = criterionDice(predict, target)
            # loss_edge = edge_aware_loss(target, predict)

            loss_all = loss_dice + loss_vgg + loss_l1 #+ loss_edge
            
            ### loss Adversarial ### 

            for p in net_D.parameters():
                p.requires_grad = True
            
            for p in net_G.parameters():
                p.requires_grad = False

            net_D.zero_grad()
            
            D_in_fake = predict.detach().cuda()
            
            noise = torch.randn_like(D_in_fake).cuda() * random.uniform(-0.05 , 0.05)

            D_in_fake = D_in_fake + noise

            D_in_real = target + noise
            
            ### Relativistic average LSGAN ###
            loss_gan_D = (torch.mean((net_D(D_in_real) - torch.mean(net_D(D_in_fake)) - torch.ones_like(net_D(D_in_real))) ** 2) \
                    + torch.mean((net_D(D_in_fake) - torch.mean(net_D(D_in_real)) + torch.ones_like(net_D(D_in_real))) ** 2))/2 * 0.1

            loss_gan_D.backward()
            optim_D.step()
            schedu_D.step()

            for p in net_D.parameters():
                p.requires_grad = False
            
            for p in net_G.parameters():
                p.requires_grad = True
            optim_G.zero_grad()

            D_in_fake_G = predict.cuda() + noise

            ### Relativistic average LSGAN ###
            loss_gan_G = (torch.mean((net_D(D_in_real) - torch.mean(net_D(D_in_fake_G)) + torch.ones_like(net_D(D_in_real))) ** 2) \
                          + torch.mean((net_D(D_in_fake_G) - torch.mean(net_D(D_in_real)) - torch.ones_like(net_D(D_in_real))) ** 2))/2 * 0.1


            loss_all = loss_all + loss_gan_G
            
            loss_all.backward()
            optim_G.step()
            schedu_G.step()

            if (step+1) % args.display_count == 0:
                
                board.add_scalar('lr_G', schedu_G.get_last_lr(), step+1)
                board.add_scalar('lr_D', schedu_D.get_last_lr(), step+1)

                board.add_scalar('loss_all', loss_all.item(), step+1)
                board.add_scalar('loss_l1', loss_l1.item(), step+1)
                board.add_scalar('loss_vgg', loss_vgg.item(), step+1)
                board.add_scalar('loss_dice', loss_dice.item(), step+1)
                # board.add_scalar('loss_edge', loss_edge.item(), step+1)

                board.add_scalar('loss_gan_D', loss_gan_D.item(), step+1)
                board.add_scalar('loss_gan_G', loss_gan_G.item(), step+1)
                
                
                input_image_01 = ((input_image + 1) * 0.5).clamp(0,1)
                
                ### sample image ### 
                combine = torch.cat([input_image_01, predicts, targets], 3)
                
                torchvision.utils.save_image(
                            combine, 
                            os.path.join(exp_path, 'sample_epoch_{}.png'.format(epoch+1))
                            )
            
        if args.save_content:
            if (epoch + 1) % args.save_content_every == 0:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'step': step, 'args': args,
                            'netG_dict': net_G.state_dict(), 'optimizerG': optim_G.state_dict(),
                            'schedulerG': schedu_G.state_dict(), 
                            'netD_dict': net_D.state_dict(),
                            'optimizerD': optim_D.state_dict(), 'schedulerD': schedu_D.state_dict(), 
                        }

                torch.save(content, os.path.join(exp_path, 'content.pth'))
        
        if (epoch+1) % args.save_count == 0:
    
            torch.save(net_G.state_dict(), os.path.join(
                exp_path, 'netG_{}.pth'.format(epoch+1)))

        ### evaluate ###
        miou = 0 
        for inputs in  tqdm(test_loader):
            net_G.eval()

            input_image = inputs['image'].cuda()
            target = inputs['target'].cuda()
            predict = net_G(input_image)   

            predicts = torch.cat([predict]*3,1)
            targets = torch.cat([target]*3,1)
            predict = torch.round(predict)
            miou += calculate_iou(mask1 = predict, mask2 = target)

            input_image_01 = ((input_image + 1) * 0.5).clamp(0,1)
            ### sample image ### 
            combine = torch.cat([input_image_01, predicts, targets], 3)
            
            torchvision.utils.save_image(
                        combine, 
                        os.path.join(exp_path, 'sample_test_epoch_{}.png'.format(epoch+1))
                        )
        with open(os.path.join(exp_path,'log.txt'), 'a') as f:
            f.write("Epoch {}: MIou = {} \n".format(epoch+1, miou/len(test_loader)))

if __name__ == "__main__":
    main()