import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import cv2 

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class MonusegData(data.Dataset):
    def __init__(self, path, mode = 'train'):
        self.path = path
        
        folder_name = 'monuseg_patches_{}'.format(mode)
        if mode == 'train':
            self.ls_images = [os.path.join(path, folder_name, 'images/train', e) for e in \
                        os.listdir(os.path.join(path, folder_name, 'images/train')) if e.endswith('.png')] 
        elif mode =='test' :
            self.ls_images = [os.path.join(path, folder_name, 'images/val', e) for e in \
                        os.listdir(os.path.join(path, folder_name, 'images/val')) if e.endswith('.png')]
        else:
            raise ValueError(f'{mode} mode does not support, only train and test mode.')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1, 1 #
        ])

        self.transform_clip = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.width, self.height = 256, 256

    def __getitem__(self, index):
        im_name = self.ls_images[index]
        im = Image.open(im_name)
        im_clip = self.transform_clip(im)
        im = im.resize((self.width, self.height)) ### input size of the model ### 
        im = self.transform(im)

        tar_name = im_name.replace('images','masks')
        tar = Image.open(tar_name).convert("L")
        tar = tar.resize((self.width, self.height)) ### input size of the model ### 

        tar_array = np.array(tar)
        tar_array = (tar_array >= 128).astype(np.float32)
        tar = torch.from_numpy(tar_array)  # [0,1]
        tar.unsqueeze_(0)

        result = {
            
            'image':   im,     # input
            'image_clip':   im_clip,     # input clip
            'target':  tar,    # ground truth
            'prompt': "Nuclei segmentation"
        }

        return result
    
    def __len__(self):
        return len(self.ls_images)
    
if __name__ == '__main__':
    main_path = '/mnt/hdd_1A/ds_project/dataset/monuseg_patches'
    # MonusegData(main_path)