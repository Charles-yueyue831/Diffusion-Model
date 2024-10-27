import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from scipy.stats import norm

latent_dim = 32
inter_dim = 128
mid_dim = (256, 2, 2)
mid_num = 1

# mid_num = 256*2*2
for i in mid_dim:
    mid_num *= i

class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        super(ConvVAE, self).__init__()

        """
		Batch Normalization 层的作用是在网络的训练过程中对每个小批量的输入进行标准化处理，使得每个特征通道的均值接近于0，方差接近于1
		nn.LeakyReLU(0.2) 表示一个带有 Leaky ReLU 激活函数的层，其中参数 0.2 是指定的负斜率（slope），即当输入值 x 小于0时，Leaky ReLU 函数的输出值是输入值乘以该负斜率。当输入值 x 大于或等于0时，Leaky ReLU 函数的输出值与 ReLU 函数相同
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2),
        )

        self.fc1 = nn.Linear(mid_num, inter_dim)
        self.fc2 = nn.Linear(inter_dim, latent * 2)

        self.fcr2 = nn.Linear(latent, inter_dim)
        self.fcr1 = nn.Linear(inter_dim, mid_num)

        """
        转置卷积用于上采样，从而将特征图的尺寸放大，转置卷积层在生成模型（如生成对抗网络GAN）中常用于从潜在空间生成图像，其中用于将低维潜在向量映射到高维图像空间
        转置卷积的计算公式：(Height-1)*Stride-2*padding+Size

        nn.ConvTranspose2d(256, 128, 4, 2)：
        	输入通道, 输出通道, 卷积核, 步长
        """
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(16, 3, 4, 2),
            nn.Sigmoid()
        )

    # 重参数化技巧
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.encoder(x)
        x = self.fc1(x.view(batch_size, -1))
        h = self.fc2(x)  # 低维潜在向量

        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterise(mu, logvar)  # 重参数化技巧

        decode = self.fcr2(z)
        decode = self.fcr1(decode)
        recon_x = self.decoder(decode.view(batch, *mid_dim))

        return recon_x, mu, logvar

vae = ConvVAE()
vae.load_state_dict(torch.load('./pokemon.pth'))

n = 10
image_size = 40

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

vae.eval()

selected = 21
coll = [(selected, i) for i in range(latent_dim) if i != selected]

"""
unsqueeze(0)：与 squeeze(0) 相反，该函数的作用是在指定位置（这里是索引为0的位置）上增加一个大小为1的维度
squeeze(0)：该函数的作用是挤压（squeeze）张量的维度，即去除指定维度上的大小为1的维度
"""
for idx, (p, q) in enumerate(coll):
    figure = np.zeros((3, image_size * n, image_size * n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            t = [random.random() for i in range(latent_dim)]
            t[p], t[q] = xi, yi
            z_sampled = torch.FloatTensor(t).unsqueeze(0)
            with torch.no_grad():
                decode = vae.fcr1(vae.fcr2(z_sampled))
                decode = decode.view(1, *mid_dim)
                decode = vae.decoder(decode)
                decode = decode.squeeze(0)

                figure[:,
                    i * image_size: (i + 1) * image_size,
                    j * image_size: (j + 1) * image_size
                ] = decode

    plt.title("X: {}, Y: {}".format(p, q))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(figure.transpose(1, 2, 0))
    plt.show()