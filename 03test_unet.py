# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : 03test_unet.py
# @Software : PyCharm

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist=torchvision.datasets.MNIST(root="./mnist",download=False,train=True,transform=torchvision.transforms.ToTensor())
train_dataloader=DataLoader(dataset=mnist,batch_size=8,shuffle=True)

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

net=Unet(1,1)

net.load_state_dict(torch.load("./model/unet.pth",map_location="cpu"))
net.eval()

def corrupt(x, amount):
    """
    :param x:[batch_size, channel, width, height]
    :param amount:[batch_size]
    :return:
    """
    noise = torch.rand_like(x)  # [batch_size, channel, width, height]
    amount = amount.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
    return x * (1 - amount) + noise * amount

# Fetch some data
x,y=next(iter(train_dataloader))
x=x[:8]

amount=torch.linspace(0,1,x.shape[0])
noised_x=corrupt(x,amount)

with torch.no_grad():
    preds = net(noised_x).detach().cpu()

fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Corrupted data')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')

plt.show()

n_steps=5
x=torch.randn(8,1,28,28)
step_history=[x]
pred_output_history=list()

with torch.no_grad():
    for i in range(n_steps):
        pred=net(x)

        pred_output_history.append(pred)

        # How much we move towards the prediction
        mix_factor=1/(n_steps-i)
        x=x*(1-mix_factor)+pred*mix_factor

        step_history.append(x)

fig, axs = plt.subplots(n_steps, 2, figsize=(8, 4), sharex=True)
axs[0,0].set_title('x (model input)')
axs[0,1].set_title('model prediction')

for i in range(n_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')

plt.show()