import os
import math
import numpy as np
import cv2
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import lpips
from pytorch_fid import fid_score
from torchvision.transforms import functional as trans_fn
from PIL import Image

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
def calculate_lpips(img1, img2, net="vgg", device="cuda"):
    '''calculate LPIPS
    '''
    if net == "alex":
        lpips_alex = lpips.LPIPS(net="alex")  # best forward scores
    elif net == "vgg":
        lpips_vgg = lpips.LPIPS(net="vgg") 
    else:
        raise ValueError('Invalid network name.')

    img1 = torch.from_numpy(img1).to(device)/255
    img2 = torch.from_numpy(img2).to(device)/255
    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if len(img1.shape) == 2:
        h, w = img1.shape
        img1 = img1.reshape(-1, 1, h, w).repeat(1, 3, 1, 1)
        img2 = img2.reshape(-1, 1, h, w).repeat(1, 3, 1, 1)
        if net == "alex":
            lpips_alex = lpips_alex.to(device)
            return lpips_alex(img1, img2).item()
        elif net == "vgg":
            lpips_vgg = lpips_vgg.to(device)
            return lpips_vgg(img1, img2).item()
    elif len(img1.shape) == 3:
        if img1.shape[2] == 3:
            _, h, w = img1.shape
            img1 = img1.unsqueeze(0).permute([0, 3, 1, 2])
            img2 = img2.unsqueeze(0).permute([0, 3, 1, 2])
            if net == "alex":
                lpips_alex = lpips_alex.to(device)
                return lpips_alex(img1, img2).item()
            elif net == "vgg":
                lpips_vgg = lpips_vgg.to(device)
                return lpips_vgg(img1, img2).item()
        elif img1.shape[2] == 1:
            _, h, w = img1.shape
            img1 = img1.unsqueeze(0).permute([0, 3, 1, 2]).repeat(1, 3, 1, 1)
            img2 = img2.unsqueeze(0).permute([0, 3, 1, 2]).repeat(1, 3, 1, 1)
            if net == "alex":
                lpips_alex = lpips_alex.to(device)
                return lpips_alex(img1, img2).item()
            elif net == "vgg":
                lpips_vgg = lpips_vgg.to(device)
                return lpips_vgg(img1, img2).item()
    else:
        raise ValueError('Wrong input image dimensions.')
    
    
def calculate_fid(hr_path, sr_path):
    return fid_score.calculate_fid_given_paths([hr_path, sr_path], batch_size=1,device='cuda', dims=2048, num_workers=4)


def resize_and_convert(img, size, resample=Image.BICUBIC):
    if(img.size[0] != size[0] and img.size[1] != size[1]):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img

def calculate_consistency(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = Image.fromarray(np.uint8(img1)).convert('RGB')
    img2 = Image.fromarray(np.uint8(img2)).convert('RGB')
    img1 = resize_and_convert(img1, (img2.size[1], img2.size[0]))
    img1 = trans_fn.to_tensor(img1)
    img2 = trans_fn.to_tensor(img2)
    mse = torch.mean(((img1 - img2))**2)
    return mse.item() * 1e5