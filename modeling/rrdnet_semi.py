import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import cv2
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
iterations = 10

# parameters settings
illu_factor = 1
reflect_factor = 1
noise_factor = 5000
reffac = 1
gamma = 0.4

# Gaussian Kernel Initialization
g_kernel_size = 5
g_padding = 2
sigma = 3
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma)
gaussian_kernel = np.multiply(kx,np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(device)

class RRDNet(nn.Module):
    def __init__(self):
        super(RRDNet, self).__init__()

        self.illumination_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
        )

        self.reflectance_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

        self.noise_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, input):
        illumination = torch.sigmoid(self.illumination_net(input))
        reflectance = torch.sigmoid(self.reflectance_net(input))
        noise = torch.tanh(self.noise_net(input))

        return illumination, reflectance, noise

def reconstruction_loss(image, illumination, reflectance, noise):
    reconstructed_image = illumination*reflectance+noise
    return torch.norm(image-reconstructed_image, 1)


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w

def normalize01(img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)

def gaussianblur3(input):
    slice1 = F.conv2d(input[:,0,:,:].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
    slice2 = F.conv2d(input[:,1,:,:].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
    slice3 = F.conv2d(input[:,2,:,:].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
    x = torch.cat([slice1,slice2, slice3], dim=1)
    return x

def illumination_smooth_loss(image, illumination):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    max_rgb, _ = torch.max(image, 1)
    max_rgb = max_rgb.unsqueeze(1)
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    max_rgb.detach()
    return loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1)


def reflectance_smooth_loss(image, illumination, reflectance):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_reflect_h, gradient_reflect_w = gradient(reflectance)
    weight = 1/(illumination*gradient_gray_h*gradient_gray_w+0.0001)
    weight = normalize01(weight)
    weight.detach()
    loss_h = weight * gradient_reflect_h
    loss_w = weight * gradient_reflect_w
    refrence_reflect = image/illumination
    refrence_reflect.detach()
    return loss_h.sum() + loss_w.sum() + reffac*torch.norm(refrence_reflect - reflectance, 1)


def loss_noise(illumination, noise):
    weight_illu = illumination
    weight_illu.detach()
    loss = weight_illu*noise
    return torch.norm(loss, 2)

def retinex(net, img):
    img_tensor = transforms.ToTensor()(img)  # [c, h, w]
    img_tensor = img_tensor.to(device)
    img_tensor = img_tensor.unsqueeze(0)     # [1, c, h, w]

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # iterations
    for i in range(iterations+1):
        # forward
        illumination, reflectance, noise = net(img_tensor)  # [1, c, h, w]
        # loss computing
        loss_recons = reconstruction_loss(img_tensor, illumination, reflectance, noise)
        loss_illu = illumination_smooth_loss(img_tensor, illumination)
        loss_reflect = reflectance_smooth_loss(img_tensor, illumination, reflectance)
        loss_noise = loss_noise(img_tensor, illumination, reflectance, noise)

        loss = loss_recons + illu_factor*loss_illu + reflect_factor*loss_reflect + noise_factor*loss_noise

        # backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # adjustment
    adjust_illu = torch.pow(illumination, gamma)
    res_image = adjust_illu*((img_tensor-noise)/illumination)
    res_image = torch.clamp(res_image, min=0, max=1)

    if device != 'cpu':
        res_image = res_image.cpu()
        illumination = illumination.cpu()
        adjust_illu = adjust_illu.cpu()
        reflectance = reflectance.cpu()
        noise = noise.cpu()

    res_img = transforms.ToPILImage()(res_image.squeeze(0))
    res_img = np.array(res_img)
    # res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    # illum_img = transforms.ToPILImage()(illumination.squeeze(0))
    # adjust_illu_img = transforms.ToPILImage()(adjust_illu.squeeze(0))
    # reflect_img = transforms.ToPILImage()(reflectance.squeeze(0))
    # noise_img = transforms.ToPILImage()(normalize01(noise.squeeze(0)))
    # return res_img, illum_img, adjust_illu_img, reflect_img, noise_img
    return res_img
