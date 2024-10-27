import os

"""
Dict 类型是一种类型注解，用于表示字典类型
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .Model import UNet
from .Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from .Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    """
    :param modelConfig:
        modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ddpm.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    """
    device = torch.device(modelConfig["device"])

    """
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))对图像数据进行标准化的一种操作
    transforms.RandomHorizontalFlip() 是 PyTorch 中的一个数据增强操作，用于对图像进行随机水平翻转
    """
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    """
    pin_memory=True：表示在数据加载过程中将数据固定在内存中，从而加速数据传输到GPU的速度
    """
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    unet = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                attn=modelConfig["attn"],
                num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        unet.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))

    """
    权值衰减（weight decay）是一种正则化技术，用于防止神经网络过拟合。
    在优化算法中，权值衰减通过在损失函数中添加一个惩罚项，来限制模型参数的大小，从而降低模型的复杂度，提高泛化能力
    """
    optimizer = torch.optim.AdamW(
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
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)

    # 增加噪声
    trainer = GaussianDiffusionTrainer(
        unet, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    best_loss = 1e9

    for e in range(modelConfig["epoch"]):
        epoch_loss = 0.

        """
        dynamic_ncols=True 参数表示进度条的宽度会动态调整，以适应终端窗口的宽度
        """
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataloader:
            for images, labels in tqdmDataloader:
                # train
                optimizer.zero_grad()

                # 初始图像
                x_0 = images.to(device)

                loss = trainer(x_0).sum() / 1000
                loss.backward()

                epoch_loss += loss.item()

                """
                torch.nn.utils.clip_grad_norm_()用于梯度裁剪，其作用是限制梯度的大小，防止梯度爆炸的问题
                    函数 torch.nn.utils.clip_grad_norm_ 会计算所有参数的梯度的范数，并根据指定的阈值对梯度进行裁剪。
                    具体来说，它会计算所有参数的梯度的 L2 范数，然后将这个范数限制在给定的阈值之内。
                    如果梯度的 L2 范数超过了阈值，则会对所有参数的梯度进行按比例缩放，使得其范数等于阈值
                """
                nn.utils.clip_grad_norm_(
                    unet.parameters(), modelConfig["grad_clip"])

                optimizer.step()

                """
                tqdm.set_postfix()：更新进度条的附加信息
                optimizer.state_dict()['param_groups'][0]["lr"]：获取了优化器当前默认参数组的学习率
                    optimizer.state_dict() 返回一个包含优化器当前状态信息的字典
                    param_groups 键对应的值是一个列表，每个元素代表一个参数组，而列表中的第一个元素 param_groups[0] 就是默认参数组
                    在默认参数组中，"lr" 键对应的值就是学习率
                """
                tqdmDataloader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        warmUpScheduler.step()

        if epoch_loss < best_loss:
            torch.save(unet.state_dict(), os.path.join(modelConfig["save_weight_dir"], f'ddpm.pt'))

            best_loss = epoch_loss

def eval(modelConfig: Dict):
    """
    :param modelConfig:
        modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ddpm.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    """
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])

        unet = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                    attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)

        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)

        unet.load_state_dict(ckpt)

        print("model load weight done.")

        unet.eval()

        sampler = GaussianDiffusionSampler(
            unet, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        # Sampled from standard normal distribution
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32],
                                 device=device)  # noisyImage.shape = [8, 3, 32, 32]

        """
        torch.clamp(noisyImage * 0.5 + 0.5, 0, 1) 的作用是将张量 noisyImage 先进行线性变换，然后将其元素限制在区间 [0, 1] 内
            noisyImage * 0.5 + 0.5：将 noisyImage 中的每个元素乘以 0.5，然后加上 0.5，相当于对像素值进行了一个缩放和平移的操作，将像素值的范围从 [-1, 1] 映射到 [0, 1]
            torch.clamp(..., 0, 1)：对上一步得到的张量进行截断操作，将小于 0 的元素置为 0，将大于 1 的元素置为 1，而在区间内的元素保持不变
        """
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)

        """
        nrow 参数表示每行显示的图像数量
        """
        save_image(saveNoisy, os.path.join(modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]),
                   nrow=modelConfig["nrow"])

        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"], modelConfig["sampledImgName"]),
                   nrow=modelConfig["nrow"])