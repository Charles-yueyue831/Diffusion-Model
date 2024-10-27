# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 22:05
# @Author  : 楚楚
# @File    : ConditionTrain.py
# @Software: PyCharm

import os

"""
Dict 类型是一种类型注解，用于表示字典类型
"""
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
import torch.nn.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .ConditionDiffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from .ConditionModel import UNet
from .ConditionScheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    """
    :param modelConfig:
        modelConfig = {
        "state": "train", # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "condition_ddpm.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
        }
    """
    device = torch.device(modelConfig["device"])

    # dataset
    """
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))对图像数据进行标准化的一种操作
    """
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    """
    pin_memory=True：表示在数据加载过程中将数据固定在内存中，从而加速数据传输到GPU的速度
    """
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    unet = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        """
        strict=True：加载的参数必须严格匹配模型中定义的参数名称和形状
        strict=False：加载的参数必须尽可能匹配模型中定义的参数名称和形状
        """
        unet.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")

    """
    权值衰减（weight decay）是一种正则化技术，用于防止神经网络过拟合。
    在优化算法中，权值衰减通过在损失函数中添加一个惩罚项，来限制模型参数的大小，从而降低模型的复杂度，提高泛化能力
    """
    optimizer = optim.AdamW(
        unet.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)

    """
    CosineAnnealingLR()：余弦退火学习率调度器，这个调度器根据余弦函数的形状动态地调整学习率，以平滑地降低学习率
        optimizer=optimizer：指定了要进行学习率调整的优化器对象
        T_max=modelConfig["epoch"]：指定了一个周期的迭代次数，即余弦函数的周期。在每个周期内，学习率将从初始值线性降低到最小值
        eta_min=0：指定了学习率的最小值。在余弦退火过程中，学习率将从初始值降低到这个最小值
        last_epoch=-1：指定了上一个周期的迭代次数。默认值为 -1，表示从头开始训练。在每个周期结束时，调度器会自动更新 last_epoch，因此不需要手动更新
    """
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)

    """
    GradualWarmupScheduler()：在开始训练的时候逐渐增加学习率，以帮助模型更快地收敛到合适的参数值
        optimizer=optimizer：指定了要进行学习率调整的优化器对象
        multiplier=modelConfig["multiplier"]：指定了学习率增加的倍数。在 Gradual Warmup 阶段，学习率将从初始值按照这个倍数进行逐渐增加
        warm_epoch=modelConfig["epoch"] // 10：指定了 Gradual Warmup 阶段的持续周期数。在这个阶段内，学习率会逐渐增加，直到达到设定的初始学习率
        after_scheduler=cosineScheduler：指定了 Gradual Warmup 阶段结束后要使用的学习率调度器
    """
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    # 增加噪声
    trainer = GaussianDiffusionTrainer(
        unet, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    best_loss = 1e9

    # start training
    for e in range(modelConfig["epoch"]):
        epoch_loss = 0.

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                batch_size = images.shape[0]

                optimizer.zero_grad()
                x_0 = images.to(device)

                labels = labels.to(device) + 1

                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)

                loss = trainer(x_0, labels).sum() / batch_size ** 2
                epoch_loss += loss.item()

                loss.backward()

                utils.clip_grad_norm_(unet.parameters(), modelConfig["grad_clip"])

                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        warmUpScheduler.step()

        if epoch_loss < best_loss:
            torch.save(unet.state_dict(), os.path.join(
                modelConfig["save_dir"], 'condition_ddpm.pt'))

            best_loss = epoch_loss

def eval(modelConfig:Dict):
    """
    :param modelConfig:
        modelConfig = {
        "state": "train", # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "condition_ddpm.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
        }
    """
    device = torch.device(modelConfig["device"])

    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)

        labelList = []

        # 标签
        k = 0

        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1

        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)

        unet = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)

        unet.load_state_dict(ckpt)
        print("model load weight done.")

        unet.eval()
        sampler = GaussianDiffusionSampler(
            unet, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)

        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)

        """
        torch.clamp(noisyImage * 0.5 + 0.5, 0, 1) 的作用是将张量 noisyImage 先进行线性变换，然后将其元素限制在区间 [0, 1] 内
            noisyImage * 0.5 + 0.5：将 noisyImage 中的每个元素乘以 0.5，然后加上 0.5，相当于对像素值进行了一个缩放和平移的操作，将像素值的范围从 [-1, 1] 映射到 [0, 1]
            torch.clamp(..., 0, 1)：对上一步得到的张量进行截断操作，将小于 0 的元素置为 0，将大于 1 的元素置为 1，而在区间内的元素保持不变
        """
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)

        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])

        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])