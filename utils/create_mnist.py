import pandas as pd
import os
from PIL import Image
import utils.config as config
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np


def create_masks(img_size, raw_image, left_color, right_color):
    img_bg_mask = raw_image != 255
    left_bg_mask = torch.zeros(1, img_size, img_size)
    left_bg_mask[:, :, : img_size // 2] = 1
    left_bg_mask *= img_bg_mask
    right_bg_mask = torch.zeros(1, img_size, img_size)
    right_bg_mask[:, :, img_size // 2 :] = 1
    right_bg_mask *= img_bg_mask

    image = (
        raw_image
        + left_bg_mask * 255 * left_color
        + right_bg_mask * 255 * right_color
    )
    image = image.int()
    image = torch.clamp(image, 0, 255)
    return image


def balance_dataset(data, l1, r1, l2, r2, img_shape):
    data_final = torch.zeros(data.shape[0], 3, img_shape, img_shape)
    data_final = data_final.int()
    colors = torch.zeros(data.shape[0], 2)
    data_range = [int(0.25 * data.shape[0]), int(0.5 * data.shape[0]), int(0.75 * data.shape[0]), data.shape[0]]
    for i in range(data.shape[0]):
        if i<=data_range[0]:
            colors[i, 0] = 0
            colors[i, 1] = 0
            data_final[i] = create_masks(img_shape, data[i], l1, r1) 
        elif i <= data_range[1]:
            colors[i, 0] = 0
            colors[i, 1] = 1
            data_final[i] = create_masks(img_shape, data[i], l1, r2)
        elif i <= data_range[2]:
            colors[i, 0] = 1
            colors[i, 1] = 0
            data_final[i] = create_masks(img_shape, data[i], l2, r1) 
        else:
            colors[i, 0] = 1
            colors[i, 1] = 1
            data_final[i] = create_masks(img_shape, data[i], l2, r2)
    return data_final, colors


def iterate_for_masking(data, data_range, l1, r1, l2, r2, img_shape):
    data_final = torch.zeros(data.shape[0], 3, img_shape, img_shape)
    data_final = data_final.int()
    choices = [0 if i % 2 == 1 else 1 for i in range(data_final.shape[0])]
    colors = torch.zeros(data.shape[0], 2)
    for i in range(data.shape[0]):
        if i <= (int(0.94 * data.shape[0])):
            colors[i, 0] = 1
            colors[i, 1] = 1
            data_final[i] = create_masks(img_shape, data[i], l1, r1)
        elif i <= (int(0.95 * data.shape[0])):
            colors[i, 0] = 0
            colors[i, 1] = 1
            data_final[i] = create_masks(img_shape, data[i], l2, r1)
        elif i <= (int(0.99 * data.shape[0])):
            colors[i, 0] = 1
            colors[i, 1] = 0
            data_final[i] = create_masks(img_shape, data[i], l1, r2)
        else:
            colors[i, 0] = 0
            colors[i, 1] = 0
            data_final[i] = create_masks(img_shape, data[i], l2, r2)
    # for i in range(data.shape[0]):
    #     choice = choices[i]
    #     if i<=data_range[0]:
    #         if choice == 0:
    #             colors[i, 0] = 1
    #             colors[i, 1] = 1
    #             data_final[i] = create_masks(img_shape, data[i], l1, r1) 
    #         else:
    #             colors[i, 0] = 1
    #             colors[i, 1] = 0
    #             data_final[i] = create_masks(img_shape, data[i], l1, r2) 
    #     elif i>data_range[1]:
    #         if choice == 0:
    #             colors[i, 0] = 1
    #             colors[i, 1] = 0
    #             data_final[i] = create_masks(img_shape, data[i], l1, r2) 
    #         else:
    #             colors[i, 0] = 0
    #             colors[i, 1] = 1
    #             data_final[i] = create_masks(img_shape, data[i], l2, r1)
    #         # colors[i, 0] = 1
    #         # colors[i, 1] = 1
    #         # data_final[i] = create_masks(img_shape, data[i], l1, r1) 
    #     else:
    #         if choice == 0:
    #             colors[i, 0] = 1
    #             colors[i, 1] = 1
    #             data_final[i] = create_masks(img_shape, data[i], l1, r1) 
    #         else:
    #             colors[i, 0] = 0
    #             colors[i, 1] = 1
    #             data_final[i] = create_masks(img_shape, data[i], l2, r1)
    return data_final, colors


def generate_masked_data(img_shape, data, targets, split=0):
    colors = torch.load(config.color_path)
    left_color_1 = colors[config.left1]
    right_color_1 = colors[config.right1]
    left_color_2 = colors[config.left2]
    right_color_2 = colors[config.right2]

    data_1 = data[targets == config.class1]
    data_2 = data[targets == config.class2]

    if split in [0, 1]:
        randperm_1 = torch.randperm(data_1.shape[0])
        data_1 = data_1[randperm_1]

        
        randperm_2 = torch.randperm(data_2.shape[0])
        data_2 = data_2[randperm_2]

        if split == 0:
            data_1 = data_1[:int(config.split*data_1.shape[0])]
            data_2 = data_2[:int(config.split*data_2.shape[0])]
            range_1 = [int(0.4 * data_1.shape[0]), int(0.8 * data_1.shape[0])]
            range_2 = [int(0.4 * data_2.shape[0]), int(0.8 * data_2.shape[0])]
            data_1_final, colors_1 = iterate_for_masking(data_1, range_1, left_color_1, right_color_1, left_color_2, right_color_2, img_shape)
            data_2_final, colors_2 = iterate_for_masking(data_2, range_2, left_color_2, right_color_2, left_color_1, right_color_1, img_shape)
            colors_1 = 1 - colors_1
            colors_1 = colors_1.int()
            colors_2 = colors_2.int()
            data_final = torch.cat((data_1_final, data_2_final), dim=0)
            colors = torch.cat((colors_1, colors_2), dim=0)
            print(colors_1[:, 0].sum()/colors_1.shape[0], colors_1[:, 1].sum()/colors_1.shape[0])
            print(colors_2[:, 0].sum()/colors_2.shape[0], colors_2[:, 1].sum()/colors_2.shape[0])
            targets = torch.zeros(data_final.shape[0])
            targets[:data_1.shape[0]] = 0
            targets[data_1.shape[0]:] = 1

        elif split == 1:
            data_1 = data_1[int(config.split*data_1.shape[0]):]
            data_2 = data_2[int(config.split*data_2.shape[0]):]
            data_1_final, colors_1 = balance_dataset(data_1, left_color_1, right_color_1, left_color_2, right_color_2, img_shape)
            data_2_final, colors_2 = balance_dataset(data_2, left_color_1, right_color_1, left_color_2, right_color_2, img_shape)
            colors_1 = colors_1.int()
            colors_2 = colors_2.int()
            data_final = torch.cat((data_1_final, data_2_final), dim=0)
            colors = torch.cat((colors_1, colors_2), dim=0)
            targets = torch.zeros(data_final.shape[0])
            targets[:data_1.shape[0]] = 0
            targets[data_1.shape[0]:] = 1
    else:
        data_1_final, colors_1 = balance_dataset(data_1, left_color_1, right_color_1, left_color_2, right_color_2, img_shape)
        data_2_final, colors_2 = balance_dataset(data_2, left_color_1, right_color_1, left_color_2, right_color_2, img_shape)
        colors_1 = colors_1.int()
        colors_2 = colors_2.int()
        data_final = torch.cat((data_1_final, data_2_final), dim=0)
        colors = torch.cat((colors_1, colors_2), dim=0)
        targets = torch.zeros(data_final.shape[0])
        targets[:data_1.shape[0]] = 0
        targets[data_1.shape[0]:] = 1
    # create_masks(img_shape, data_1[0], left_color, right_color)
    
    return data_final, targets, colors
    


def create_colored_MNIST(split):
    if split in [0, 1]:
        dataset = MNIST(config.MNIST_dir, train=True, download=True)
    else:
        dataset = MNIST(config.MNIST_dir, train=False, download=True)
    targets = dataset.targets
    data = dataset.data
    data = data[(targets == config.class1) | (targets == config.class2),:,:]
    targets = targets[(targets == config.class1) | (targets == config.class2)]
    img_shape = data.shape[1]
    return generate_masked_data(img_shape, data, targets, split)
    

if __name__ == '__main__':
    create_colored_MNIST(split=0)