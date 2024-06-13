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
from model.model import UNet, SpectralDiscriminator, Wavelet_Segmentation, Wavelet_Segmentation_GN, Wavelet_Segmentation_Cross_Attn
from model.losses import VGGLoss, DiceBCELoss, DiceLoss, edge_aware_loss

from tqdm import tqdm
import random
import clip


from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from utils import calculate_iou, rand_brightness, rand_contrast, rand_cutout, rand_saturation, rand_translation

def diffaug(x):
    ### color ### 
    x = rand_brightness(x)
    x = rand_saturation(x)
    # x = rand_contrast(x)
    ### translation ### 
    x = rand_translation(x)
    ### cutout ### 
    x = rand_cutout(x)
    return x.contiguous()

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
    net_G = Wavelet_Segmentation(config).to(device)
    optim_G = get_optimizer(config, 
                            filter(lambda p: p.requires_grad, net_G.parameters()), 
                            args.lr_g)
    schedu_G = lr_scheduler.MultiStepLR(optim_G, milestones=[1000], gamma=0.5)

    if config.wavelet_model.use_gan:
        net_D = SpectralDiscriminator(input_nc=1).to(device)
        optim_D = get_optimizer(config, 
                            filter(lambda p: p.requires_grad, net_D.parameters()), 
                            args.lr_d)
        schedu_D = lr_scheduler.MultiStepLR(optim_D, milestones=[1000], gamma=0.5)
    
    if config.wavelet_model.use_clip:
        ### CLIP Condition model ### 
        clip_model, _ = clip.load(name = config.clip.clip_model, device=device, jit=False)
        clip_model.eval()
    

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
        if config.wavelet_model.use_gan:
            # load D
            net_D.load_state_dict(checkpoint['netD_dict'])
            optim_D.load_state_dict(checkpoint['optimizerD'])
            schedu_D.load_state_dict(checkpoint['schedulerD'])

        step = checkpoint['step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        step, epoch, init_epoch = 0, 0, 0

    ### Wavelet Pooling ### 
    dwt = DWTForward(J=1, mode='zero', wave='haar').float().cuda()
    iwt = DWTInverse(mode='zero', wave='haar').float().cuda()

    ### Define Loss Function ### 
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionDice = DiceLoss()
    # criterionBCE = nn.BCELoss()
    # criterionCE = nn.CrossEntropyLoss()
    # criterionEdge = edge_aware_loss()

    for epoch in range(start_epoch, args.num_epoch + 1):

        for inputs in  tqdm(train_loader):
            net_G.train()
            step +=1

            input_image = inputs['image'].cuda()
            target = inputs['target'].cuda()
            input_clip = inputs['image_clip'].cuda()
            # prompt = inputs['prompt']

            ### wavelet input ### 
            xll, xh = dwt(input_image)  # [b, 3, h, w], [b, 3, 3, h, w]
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)

            input_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [b, 12, h, w]
            input_data = input_data  /2.0 

            assert -1 <= input_data.min() < 0
            assert 0 < input_data.max() <= 1

            latent_z = torch.randn(args.batch_size, config.wavelet_model.nz, device=device)

            if config.wavelet_model.use_clip:
                clip_feature = clip_model.encode_image(input_clip) #+ clip_model.encode_text(text)
            else:
                clip_feature = None

            predict = net_G(x = input_data, z = latent_z, clip_feature = clip_feature)   

            predict = predict * 2

            predict_sample = iwt((predict[:, :1], [torch.stack(
                (predict[:, 1:2], predict[:, 2:3], predict[:, 3:4]), dim=2)]))

            predict_sample = torch.sigmoid(predict_sample)

            predicts = torch.cat([predict_sample]*3, 1)
            targets = torch.cat([target]*3, 1)
            net_G.zero_grad()
            ### loss function ### 
            loss_l1 = criterionL1(predict_sample, target)
            loss_vgg = criterionVGG(predicts, targets)
            loss_dice = criterionDice(predict_sample, target)
            loss_edge = edge_aware_loss(target, predict_sample) 

            loss_all =  loss_dice + loss_l1 + loss_vgg + loss_edge

            if config.wavelet_model.use_gan:
                ### loss Adversarial ### 

                for p in net_D.parameters():
                    p.requires_grad = True
                
                for p in net_G.parameters():
                    p.requires_grad = False

                net_D.zero_grad()
                
                D_in_fake = predict_sample.detach().cuda()
                
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

                D_in_fake_G = predict_sample.cuda() + noise

                ### Relativistic average LSGAN ###
                loss_gan_G = (torch.mean((net_D(D_in_real) - torch.mean(net_D(D_in_fake_G)) + torch.ones_like(net_D(D_in_real))) ** 2) \
                            + torch.mean((net_D(D_in_fake_G) - torch.mean(net_D(D_in_real)) - torch.ones_like(net_D(D_in_real))) ** 2))/2 * 0.1


                loss_all = loss_all + loss_gan_G

            loss_all.backward()
            optim_G.step()
            schedu_G.step()

            if (step+1) % args.display_count == 0:
                
                board.add_scalar('lr_G', schedu_G.get_last_lr(), step+1)

                board.add_scalar('loss_all', loss_all.item(), step+1)
                board.add_scalar('loss_l1', loss_l1.item(), step+1)
                board.add_scalar('loss_vgg', loss_vgg.item(), step+1)
                board.add_scalar('loss_dice', loss_dice.item(), step+1)
                board.add_scalar('loss_edge', loss_edge.item(), step+1)

                if config.wavelet_model.use_gan:
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
                if config.wavelet_model.use_gan:
                    content = {'epoch': epoch + 1, 'step': step, 'args': args,
                                'netG_dict': net_G.state_dict(), 'optimizerG': optim_G.state_dict(),
                                'schedulerG': schedu_G.state_dict(), 
                                'netD_dict': net_D.state_dict(),
                                'optimizerD': optim_D.state_dict(), 'schedulerD': schedu_D.state_dict(), 
                            }
                else:
                    content = {'epoch': epoch + 1, 'step': step, 'args': args,
                                'netG_dict': net_G.state_dict(), 'optimizerG': optim_G.state_dict(),
                                'schedulerG': schedu_G.state_dict(),  
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
            # prompt = inputs['prompt']

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
            if config.wavelet_model.use_clip:
                clip_feature = clip_model.encode_image(input_clip) #+ clip_model.encode_text(text)
            else:
                clip_feature = None
                
            predict = net_G(x = input_data, z = latent_z, clip_feature = clip_feature)   

            predict = predict * 2

            predict_sample = iwt((predict[:, :1], [torch.stack(
                (predict[:, 1:2], predict[:, 2:3], predict[:, 3:4]), dim=2)]))
            
            predict_sample = torch.sigmoid(predict_sample)
            
            predict_sample = torch.round(predict_sample)

            miou += calculate_iou(mask1 = predict_sample, mask2 = target)

            predicts = torch.cat([predict_sample]*3,1)
            targets = torch.cat([target]*3,1)
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