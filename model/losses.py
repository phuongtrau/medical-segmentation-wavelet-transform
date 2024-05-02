### Perceptual loss model ### 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class Vgg19(nn.Module): #### For perceptual loss calculation ###
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def edge_aware_loss(y_true, y_pred, edge_weight=2.0):
    """
    Edge-aware loss function for segmentation tasks.
    Args:
    - y_true: Ground truth masks (torch tensors).
    - y_pred: Predicted masks (torch tensors).
    - edge_weight: Weight multiplier for edge regions.
    Returns:
    - Loss value.
    """
    # Calculate binary cross-entropy
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    # Detect edges in the ground truth mask using Sobel filter
    sobel_x = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    sobel_y = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    # Define Sobel operators
    sobel_x.weight = nn.Parameter(torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).cuda().float(), requires_grad=False)
    sobel_y.weight = nn.Parameter(torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).cuda().float(), requires_grad=False)
    # Calculate gradients
    grad_x = sobel_x(y_true).squeeze(1)
    grad_y = sobel_y(y_true).squeeze(1)
    edge_mask = torch.sqrt(grad_x**2 + grad_y**2)
    # Threshold for edges and convert to binary mask
    edge_mask = (edge_mask > 0.1).float()
    # Weights for edge regions
    weights = 1 + edge_mask * (edge_weight - 1)
    # Apply weights to BCE loss
    weighted_bce = bce * weights
    # Return the mean of the weighted loss
    return weighted_bce.mean()