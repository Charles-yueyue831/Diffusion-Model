import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm

latent_dim = 2  # 隐变量维度: latent_dim = 2, 方便后续可视化
input_dim = 28 * 28  # 输入层维度: input_dim = 784
inter_dim = 256  # 过渡层维度: inter_dim = 256


class VAE(nn.Module):
    def __init__(self, input_dim=input_dim, inter_dim=inter_dim, latent_dim=latent_dim):
        super(VAE, self).__init__()

        """
        Encoder末尾千万别像网上某些例子在再接一个ReLU
        在优化过程中, 我们的隐变量Z是要逐渐趋向于\mathcal{N}(0,1)的，如果非要加个ReLU的话, 本身假设的隐变量维度就很小, 小于0的隐变量直接就没了… 
        Decoder在解码时直接就会因为信息不足而崩掉
        """
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        """
        :param mu: 正态分布的均值
        :param logvar: 正态分布的方差
        :return:
        """
        epsilon = torch.rand_like(mu)  # 重参数化技巧 Z = \mu + \epsilon \times \sigma
        return mu + epsilon * torch.exp(logvar / 2)  # 将对数方差转为标准差

    def forward(self, x):
        x_size = x.size()
        x = x.view(x_size[0], -1)

        h = self.encoder(x)

        """
        torch.chunk():
            将张量沿着指定的维度进行切片，将张量分割成指定数量的块，并返回这些块的列表
        """
        mu, logvar = h.chunk(2, dim=1)  # 在变分自动编码器（VAE）中，通常会将潜在变量的均值和对数方差分开处理

        z = self.reparameterise(mu, logvar)  # 重参数化

        reconstruct_x = self.decoder(z).view(x_size)

        return reconstruct_x, mu, logvar

vae = VAE()
vae.load_state_dict(torch.load('./best_model_mnist.pth'))

n = 20
digit_size = 28

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
print(f"grid x: {grid_x}\tgrid y: {grid_y}")


vae.eval()
figure = np.zeros((digit_size * n, digit_size * n))
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        t = [xi, yi]
        z_sampled = torch.FloatTensor(t)
        with torch.no_grad():
            decode = vae.decoder(z_sampled)
            digit = decode.view((digit_size, digit_size))
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size
            ] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.xticks([])
plt.yticks([])
plt.axis('off');
plt.show()