"""
Pokemon原始数据大小为(3, 40, 40)
"""

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

from tqdm import tqdm


class Pokemon(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Pokemon, self).__init__()

        self.root = root
        self.image_path = [os.path.join(root, x) for x in os.listdir(root)]
        random.shuffle(self.image_path)

        if transform is not None:
            self.transform = transform

        if train:
            self.images = self.image_path[: int(.8 * len(self.image_path))]
        else:
            self.images = self.image_path[int(.8 * len(self.image_path)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        数据集中实际上存储的是图像文件的路径, 在需要使用的时候再读出来, 我们将这一Pipeline集成在transform中
        """
        return self.transform(self.images[item])


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


# KL损失
kl_criterion = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
# 重构损失 MSE
recon_criterion = lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)

epochs = 2000
batch_size = 16

best_loss = 1e9
best_epoch = 0

valid_losses = []
train_losses = []

"""
transform将图像从路径中读取出来, 并通过transforms.ToTensor转换为0, 1之间的RGB值
"""
transform = transforms.Compose([
    lambda x: Image.open(x).convert("RGB"),
    transforms.ToTensor()
])

pokemon_train = Pokemon('./pokemon/', train=True, transform=transform)
pokemon_valid = Pokemon('./pokemon/', train=False, transform=transform)

train_dataloader = DataLoader(pokemon_train, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(pokemon_valid, batch_size=batch_size, shuffle=False,drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = ConvVAE()
vae.to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch}")
    vae.train()

    train_loss = 0.
    train_num = len(train_dataloader.dataset)

    for idx, x in enumerate(train_dataloader):
        batch = x.size(0)
        x = x.to(device)

        recon_x, mu, logvar = vae(x)

        recon = recon_criterion(recon_x, x)
        kl = kl_criterion(mu, logvar)

        loss = recon + kl
        train_loss += loss.item()
        loss = loss / batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} in Step {idx}")

    train_losses.append(train_loss / train_num)

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.

    valid_num = len(test_dataloader.dataset)
    vae.eval()

    with torch.no_grad():
        for idx, x in enumerate(test_dataloader):
            x = x.to(device)
            recon_x, mu, logvar = vae(x)

            recon = recon_criterion(recon_x, x)
            kl = kl_criterion(mu, logvar)

            loss = recon + kl
            valid_loss += loss.item()
            valid_kl += kl.item()
            valid_recon += recon.item()

        valid_losses.append(valid_loss / valid_num)

        print(
            f"Valid loss {valid_loss / valid_num: .3f} \t Recon {valid_recon / valid_num: .3f} \t KL {valid_kl / valid_num: .3f} in epoch {epoch}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

            torch.save(vae.state_dict(), 'pokemon.pth')
            print("Model saved")

plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Valid')
plt.legend()
plt.title('Pokemon Learning Curve')
plt.show()
