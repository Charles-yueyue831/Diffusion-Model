# -*- coding: utf-8 -*-
# @Time    : 2024/2/5 19:03
# @Author  : 楚楚
# @File    : main.py
# @Software: PyCharm

import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from UNet import Unet
from dataset import LiverDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
标准化操作：零均值单位方差—>减去均值并除以标准差
    零均值：消除数据中的偏移
    单位方差：将数据的尺度缩放到统一的水平
"""
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataloader, num_epoch=20):
    for epoch in tqdm(range(num_epoch)):
        print(f"Epoch {epoch}/{num_epoch}")
        print(f"{'-' * 10}")

        size = len(dataloader.dataset)

        epoch_loss = 0
        step = 0

        minimum_loss = 0

        for x, y in dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outpus = model(inputs)

            # 计算误差
            loss = criterion(outpus, labels)

            # 误差的反向传播
            loss.backward()

            # 更新梯度
            optimizer.step()

            epoch_loss += loss.item()

            print(f"{step}/{(size - 1) // dataloader.batch_size + 1}, train_loss: {loss.item():0.3f}")

        print(f"epoch {epoch} loss: {epoch_loss / step:0.3f}")

        if epoch==0:
            minimum_loss=epoch_loss/step

        if epoch_loss / step <= minimum_loss:
            torch.save(model.state_dict(), f"./unet.pth")

            minimum_loss = epoch_loss / step

    return model


# 训练模型
def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size

    """
    BCEWithLogitsLoss()：二分类交叉损失函数
        将 Sigmoid 激活函数和二分类交叉熵损失结合在一起。通常，在模型的最后一层不使用激活函数，而是直接输出未经激活的 logits
    """
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_dataset = LiverDataset("./data/train", transform=x_transforms, target_transform=y_transforms)

    """
    num_workers定义了加载数据的子进程的数量
    当num_workers大于0时，pytorch会使用多个子进程来加载数据，而不是在主进程中进行
    可以在数据加载的同时进行模型的训练，从而提高整体的训练效率
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_model(model, criterion=criterion, optimizer=optimizer, dataloader=train_dataloader)


# 显示模型的输出结果
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    val_dataset = LiverDataset("./data/val", transform=x_transforms, target_transform=y_transforms)

    val_dataloader = DataLoader(val_dataset, batch_size=1)

    model.eval()

    """
    plt.ion()：打开交互模式
        在绘图窗口中实时更新图像，不会阻塞程序的进行
    """
    plt.ion()

    with torch.no_grad():
        for x, _ in val_dataloader:
            y = model(x).sigmoid()
            """
            torch.squeeze()：去除张量中维度为1的维度
            """
            image_y = torch.squeeze(y).numpy()

            plt.imshow(image_y)
            plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    """
    --表示该参数是可选参数
    """
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str)

    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
