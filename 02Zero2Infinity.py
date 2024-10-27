# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : 02Zero2Infinity.py
# @Software : PyCharm

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from diffusers import DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = torchvision.datasets.MNIST(root="./mnist", train=True, download=True,
                                     transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
# make_grid()：将多张图片合成一张图片
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

"""
添加噪声：Corrupt the input `x` by mixing it with noise according to `amount`
"""


def corrupt(x, amount):
    """
    :param x:[batch_size, channel, width, height]
    :param amount:[batch_size]
    :return:
    """
    noise = torch.rand_like(x)  # [batch_size, channel, width, height]
    amount = amount.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
    return x * (1 - amount) + noise * amount


# Plotting the input data
"""
plt.subplots(2, 1, figsize=(12, 5)) 表示创建一个 2 行 1 列的子图网格，也就是说，图形将被分为两行，每行包含一个子图。figsize=(12, 5) 定义了整个图形的大小，宽度为 12 个单位，高度为 5 个单位。
fig, axs 这一部分是用来接收 plt.subplots() 函数的返回值的，其中 fig 是整个图形对象，axs 是一个包含两个 Axes 对象的数组。Axes 对象是 Matplotlib 中用于绘制图形的基本单元。
"""
fig, axs = plt.subplots(2, 1, figsize=(12, 5))

axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

# Adding noise
"""
从0到1等分为x.shape[0]份
"""
amount = torch.linspace(0, 1, x.shape[0])
noised_x = corrupt(x, amount)

# Plottinf the noised version
axs[1].set_title('Corrupted data (-- amount increases -->)')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')

plt.show()

'''
UNet由一个“压缩路径和”扩展路径“组成
    压缩路径使通过该路径的数据被压缩
    扩展路径会将数据扩展回原始维度
    模型中的残差连接允许信息和梯度在不同层级之间流动
'''


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, (3, 3), padding=1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, inputs):
        return self.conv(inputs)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)  # [8, 32, 28, 28]
        self.pool1 = nn.MaxPool2d(2)  # [8, 32. 14, 14]
        self.conv2 = DoubleConv(32, 64)  # [8, 64, 14, 14]
        self.pool2 = nn.MaxPool2d(2)  # [8, 64, 7, 7]

        """
        转置卷积计算公式：(height-1)*stride-2*padding+size
        """
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # [8, 64, 14, 14]
        self.conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # [8, 32, 28, 28]
        self.conv4 = DoubleConv(64, 32)

        self.conv5 = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)  # [8, 64, 14, 14]
        p2 = self.pool2(c2)

        up_3 = self.up3(p2)  # [8, 64, 14, 14]
        merge3 = torch.cat([up_3, c2], dim=1)  # [8, 128, 14, 14]
        c3 = self.conv3(merge3)  # [8, 64, 14, 14]

        up_4 = self.up4(c3)  # [8, 32, 28, 28]
        merge4 = torch.cat([up_4, c1], dim=1)  # [8, 64, 28, 28]
        c4 = self.conv4(merge4)  # [8, 32, 28, 28]

        c5 = self.conv5(c4)

        return c5


net = Unet(1, 1)  # 228897个参数
net.to(device)

"""
给定一个损坏的输入noise_x，模型应该输出它对原本x的最佳猜测，通过均方差误差将预测与真实值进行比较

1、获取一批数据
2、添加随机噪声
3、将数据输入模型
4、将模型预测与干净图进行比较，以计算loss
5、更新模型的参数
"""
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_epochs = 20

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

losses = list()

best_loss = 0

for epoch in range(n_epochs):
    for x, y in train_dataloader:
        # Get some data and prepare the corrupted version
        x = x.to(device)  # Data on the GPU
        noised_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
        noisy_x = corrupt(x, noised_amount)  # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x)

        # Calculate the loss
        loss = criterion(pred, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

    if epoch==0:
        best_loss=avg_loss
        torch.save(net.state_dict(),"./model/unet.pth")

    if epoch>0 and avg_loss<best_loss:
        torch.save(net.state_dict(),"./model/unet.pth")

plt.plot(losses)
plt.ylim(0, 0.1)  # 用于设置图形y轴范围的函数
plt.show()