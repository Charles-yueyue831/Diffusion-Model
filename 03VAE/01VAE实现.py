# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : 01VAE实现.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm

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


"""
VAE的损失由重构损失和KL损失组成
    KL(\mathcal{N}(μ,σ^2)‖\mathcal{N}(0,1)) = \frac{1}{2}(-log\sigma^2 + \mu^2 + \sigma^2 - 1)
    
VAE的目标是最小化Z和\mathcal{N}(0,1))之间的KL散度
因为MNIST是黑白二值图像, 所以的Decoder就可以用Sigmoid后的值当做灰度, 重构损失可以直接用BCE

MSE计算公式：MSE = (1/n) * Σ(actual – prediction)^2
binary_cross_entropy计算公式：\text{Binary Cross-Entropy Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)] 
"""
kl_criterion = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

"""
参数 size_average=False 表示不对每个样本的损失值进行平均，而是直接对所有样本的损失值进行求和
"""
reconstruct_criterion = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False)

epochs = 100
batch_size = 128

"""
TypeError: 'ToTensor' object is not iterable
    It's because transforms.Compose() needs to be a list
"""
transform = transforms.Compose([transforms.ToTensor()])

train_data = MNIST(root="./mnist", download=False, train=True, transform=transform)
test_data = MNIST(root="./mnist", download=True, train=False, transform=transform)

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(input_dim, inter_dim, latent_dim)
vae.to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

best_loss = 1e9
best_epoch = 0

valid_losses = []
train_losses = []

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch}")
    vae.train()

    train_loss = 0
    train_num = len(train_dataloader.dataset)

    for idx, (x, _) in enumerate(train_dataloader):
        batch = x.size(0)
        x = x.to(device)

        reconstruct_x, mu, logvar = vae(x)

        # 重构损失
        reconstruct_loss = reconstruct_criterion(reconstruct_x, x)
        # KL损失
        kl_loss = kl_criterion(mu, logvar)

        loss = reconstruct_loss + kl_loss

        train_loss += loss.item()

        loss = loss / batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(
                f"Training loss {loss: .3f} \t Recon {reconstruct_loss / batch: .3f} \t KL {kl_loss / batch: .3f} in Step {idx}")

    train_losses.append(train_loss / train_num)

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.
    valid_num = len(test_dataloader.dataset)

    vae.eval()
    with torch.no_grad():
        for idx, (x, _) in enumerate(test_dataloader):
            x = x.to(device)
            recon_x, mu, logvar = vae(x)

            recon = reconstruct_criterion(recon_x, x)
            kl = kl_criterion(mu, logvar)

            loss = recon + kl
            valid_loss += loss.item()
            valid_kl += kl.item()
            valid_recon += recon.item()
        valid_losses.append(valid_loss / valid_num)

        print(f"Valid loss {valid_loss / valid_num: .3f} \t Recon {valid_recon / valid_num: .3f} \t KL {valid_kl / valid_num: .3f} in epoch {epoch}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(vae.state_dict(), 'best_model_mnist')
            print("Model saved")

plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Valid')
plt.legend()
plt.title('Learning Curve')

plt.show()