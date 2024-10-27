# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : 01Unet2Model.py
# @Software : PyCharm

from diffusers import UNet2DModel
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

net = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet models
    block_out_channels=(32, 64, 64),  # 每个UNet模块的输出通道， (32,64,64) 表示第一个 UNet 模型块输出通道数为 32，第二个和第三个 UNet 模型块的输出通道数都为 64
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D"
    ),

    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    )
)

print(net)

"""
SiLU是Sigmoid和ReLU的改进版。SiLU具备无上界有下界、平滑、非单调的特性
SiLU（Sigmoid Linear Unit）激活函数也被称为 Swish 激活函数
    f(x) = x * sigmoid(x)
    
ReLU(): max(0,x)
优点：
    解决梯度消失问题：ReLU 函数在正值区域上是线性的，不会出现梯度消失问题
    稀疏激活性：ReLU 函数在负值区域上输出为零，因此可以带来稀疏激活性，有助于减少模型过拟合
缺点：
    神经元死亡问题：在训练过程中，某些神经元可能会永远输出零，导致相应的权重不会更新，这被称为神经元死亡问题
"""

mnist = torchvision.datasets.MNIST(root="./mnist", download=True, train=True,
                                   transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset=mnist, batch_size=8, shuffle=True)

def corrupt(x,amount):
    noise=torch.rand_like(x)
    amount=amount.view(-1,1,1,1)
    return noise*amount+x*(11-amount)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

net.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

losses = list()
best_loss = 0

n_epochs=3

for epoch in tqdm(range(n_epochs)):
    for x,y in train_dataloader:
        x=x.to(device)
        noise_amount=torch.randn(x.shape[0]).to(device)
        noise_x=corrupt(x,noise_amount)

        pred=net(noise_x,0).sample # Using timestep 0 always

        loss=criterion(pred,x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

    if epoch==0:
        best_loss=avg_loss

    if avg_loss<=best_loss:
        best_loss=avg_loss
        torch.save(net,"./model/unet2dmodel.pth")

fig,axs=plt.subplots(1,2,figsize=(12,5))
axs[0].plot(losses)
axs[0].set_ylim(0, 0.1)
axs[0].set_title('Loss over time')

n_steps=40
x=torch.randn(64,1,28,28).to(device)

with torch.no_grad():
    for i in range(n_steps):
        pred=net(x,0).sample

        mix_factor=1/(n_steps-i)

        x=x*(1-mix_factor)+pred*mix_factor

axs[1].imshow(torchvision.utils.make_grid(x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Generated Samples')