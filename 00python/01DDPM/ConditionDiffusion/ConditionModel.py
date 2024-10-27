# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 16:13
# @Author  : 楚楚
# @File    : ConditionModel.py
# @Software: PyCharm

import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 正余弦对时间步进行编码
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, tdim):
        """
        :param T: 500
        :param d_model: 特征数量 128
        :param tdim: 特征数量*4 512
        """
        assert d_model % 2 == 0
        super(TimeEmbedding, self).__init__()

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)  # emb.shape = [64]
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()  # pos.shape = [500]
        emb = pos[:, None] * emb[None, :]  # emb.shape = [500, 64]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # emb.shape = [500, 64, 2]
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)  # emb.shape = [500, 128]

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, tdim),
            Swish(),
            nn.Linear(tdim, tdim)
        )

        self.initialize()

    # 初始化参数
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                """
                Xavier初始化的公式：
                    \mathbf{std}=\sqrt{\frac2{n_\mathrm{in}+n_\mathrm{out}}}
                    然后权重参数按照均值为 0、标准差为 std 的正态分布或均匀分布进行随机初始化
                """
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """
        :param t: [8]
        :return:
        """
        emb = self.time_embedding(t)  # emb.shape = [8, 512]

        return emb


# 基于标签的条件编码
class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, tdim):
        """
        :param num_labels: 标签 10
        :param d_model: 特征数量 128
        :param tdim: 特征数量*4 512
        """
        assert d_model % 2 == 0
        super().__init__()

        """
        nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0)
            num_labels：要学习的标签或类别的数量，通常会有一个额外的标记用于填充或者表示未知类别
            padding_idx=0：这个参数指定了用于填充的特殊标记的索引，通常选择索引为0的标记来作为填充标记
        """
        self.condition_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, tdim),
            Swish(),
            nn.Linear(tdim, tdim),
        )

        self.initialize()

    # 初始化参数
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                """
                Xavier初始化的公式：
                    \mathbf{std}=\sqrt{\frac2{n_\mathrm{in}+n_\mathrm{out}}}
                    然后权重参数按照均值为 0、标准差为 std 的正态分布或均匀分布进行随机初始化
                """
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """
        :param t: [8]
        :return:
        """
        emb = self.condition_embedding(t)  # emb.shape = [8, 512]
        return emb


# 下采样模块
class DownSample(nn.Module):
    def __init__(self, in_ch):
        """
        :param in_ch: 输入通道
        """
        super(DownSample, self).__init__()

        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.c1.weight)
        init.zeros_(self.c1.bias)

        init.xavier_uniform_(self.c2.weight)
        init.zeros_(self.c2.bias)

    def forward(self, x, temb, cemb):
        """
        :param x: 图像 x.shape = [8, out_ch, width, height]
        :param temb:
        :param cemb:
        :return:
        """

        # self.c1(x).shape = [8, out_ch, (width-3+2)//2+1, (height-3+2)//2+1]
        # self.c2(x).shape = [8, out_ch, (width-5+4)//2+1, (height-5+4)//2+1]
        x = self.c1(x) + self.c2(x)  # x.shape = [8, out_ch, (width-5+4)//2+1, (height-5+4)//2+1]
        return x


# 上采样
class UpSample(nn.Module):
    def __init__(self, in_ch):
        """
        :param in_ch: 输入通道
        """
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

        # width = (width - 1) * stride - 2 * padding + kernel_size + output_padding
        # output_padding = stride - 1
        # padding = (kernel_size - 1) / 2
        # nn.ConvTranspose2d中output_padding参数理解：https://blog.csdn.net/zzy_pphz/article/details/108291187
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.c.weight)
        init.zeros_(self.c.bias)

        init.xavier_uniform_(self.t.weight)
        init.zeros_(self.t.bias)

    def forward(self, x, temb, cemb):
        """
        :param x: 图像 x.shape = [8, out_ch, width, height]
        :param temb:
        :param cemb:
        :return:
        """
        _, _, H, W = x.shape

        # x.shape = [8, out_ch, (width-1)*2-4+5+1, (height-1)*2-4+5+1]
        x = self.t(x)

        # x.shape = [8, out_ch, (width-1)*2-4+5+1, (height-1)*2-4+5+1]
        x = self.c(x)
        return x  # x.shape = [8, out_ch, width*2, height*2]


# 注意力块
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        """
        :param in_ch: 输入通道
        """
        super(AttnBlock, self).__init__()

        # 将通道进行分组，分组进行归一化，减少组内数据的偏移，将组内数据特征缩放到统一的尺寸，同时减少内部协变量转移的影响
        self.group_norm = nn.GroupNorm(32, in_ch)

        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)

        """
        gain：调整初始化的增益，它会与初始化的标准差相乘
        """
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        """
        :param x.shape = [8, out_ch, 32, 32]
        :return:
        """
        B, C, H, W = x.shape

        h = self.group_norm(x)
        q = self.proj_q(h)  # q.shape = [8, out_ch, 32, 32]
        k = self.proj_k(h)  # k.shape = [8, out_ch, 32, 32]
        v = self.proj_v(h)  # v.shape = [8, out_ch, 32, 32]

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # q.shape = [8, 32*32, out_ch]
        k = k.view(B, C, H * W)  # k.shape = [8, out_ch, 32*32]
        w = torch.bmm(q, k) * (int(C) ** (-0.5))  # w.shape = [8, 32*32, 32*32]
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)  # w.shape = [8, 32*32, 32*32]

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)  # v.shape = [8, 32*32, out_ch]
        h = torch.bmm(w, v)  # # h.shape = [8, 32*32, out_ch]
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # h.shape = [8, out_ch, 32, 32]
        h = self.proj(h)  # h.shape = [8, out_ch, 32, 32]

        return x + h  # [8, out_ch, 32, 32]


# 残差连接块
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        """
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param tdim: ch * 4
        :param dropout: 随机选择要丢弃的单元，不参与前向传播和反向传播的过程，防止过拟合
        :param attn: 是否使用注意力机制
        """
        super(ResBlock, self).__init__()

        """
        数据分布：输入数据在网络各层之间传递时的统计特性，包括均值、方差
            在深度神经网络中，每一层的输入数据都是上一层的输出数据，因此随着网络的前向传播，数据会逐渐传递至网络的不同层
            如果输入数据的分布发生较大的变化，可能会导致某些层的激活值过大或过小，从而使得梯度消失或爆炸
        内部协变量转移（Internal Covariate Shift）是指在深度神经网络的训练过程中，由于网络参数的更新导致网络层之间输入数据的分布发生变化，从而影响网络的训练效果的现象
        """

        """
        Batch Normalization：
            1、对批次大小敏感：Batch Normalization 在计算均值和方差时依赖于整个批次的数据
            2、对于小批量大小的数据，Batch Normalization 的效果可能不稳定
        Group Normalization适用于小批量大小和卷积层
        Batch Normalization 适用于较大的批次大小和全连接层
        """
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),  # 将通道进行分组，每组32个
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        )

        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch)
        )

        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            # nn.Identity()是一个简单的恒等映射层，这意味着它接受输入，并且直接将输入作为输出返回，不进行任何变换
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb, labels):
        """
        :param x: 图像 x.shape = [8, 128, 32, 32]
        :param temb: 正余弦对图像进行编码 temb.shape = [8, 512]
        :param labels: 标签的条件编码 labels.shape = [8, 512]
        :return:
        """
        h = self.block1(x)  # h.shape = [8, out_ch, 32, 32]

        # time_proj(time_embedding).shape = [8, out_ch, 1, 1]
        h += self.time_proj(temb)[:, :, None, None]  # h.shape = [8, out_ch, 32, 32]

        # cond_proj(labels).shape = [8, out_ch, 32, 32]
        h += self.cond_proj(labels)[:, :, None, None]  # h.shape = [8, out_ch, 32, 32]

        h = self.block2(h)  # h.shape = [8, out_ch, 32, 32]

        h = h + self.shortcut(x)  # h.shape = [8, out_ch, 32, 32]
        h = self.attn(h)  # h.shape = [8, out_ch, 32, 32]
        return h


class UNet(nn.Module):
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        """
        :param T: 时间步 500
        :param num_labels: 标签 10
        :param ch: 特征数量 128
        :param ch_mult: [1, 2, 2, 2]
        :param num_res_blocks: 残差块数量 2
        :param dropout: 要丢弃的神经元的个数 0.15
        """
        super(UNet, self).__init__()

        tdim = ch * 4  # 512

        # 对时间步进行正余弦编码
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # 基于类别的条件编码
        self.condition_embedding = ConditionalEmbedding(num_labels, ch, tdim)

        # Padding added to all four sides of the input
        # 输入预处理，单个卷积Conv
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # 下采样block：收缩阶段
        # 两个残差连接ResBlock，一个下采样DownSample
        self.downblocks = nn.ModuleList()

        # record output channel when dowmsample for upsample
        chs = [ch]

        now_ch = ch

        # 第一次：两个残差连接ResBlock
        # 第一次：一个下采样DownSample
        # 第二次：两个残差连接ResBlock
        # 第二次：一个下采样DownSample
        # 第三次：两个残差连接ResBlock
        # 第三次：一个下采样DownSample
        # 第四次：两个残差连接ResBlock
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # 中间阶段
        # 两个残差连接ResBlock
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        # 上采样block：扩张阶段
        # 三个残差连接ResBlock和一个上采样UpSample
        self.upblocks = nn.ModuleList()

        # 第一次：三个残差连接ResBlock
        # 第一次：一个上采样UpSample
        # 第二次：三个残差连接ResBlock
        # 第二次：一个上采样UpSample
        # 第三次：三个残差连接ResBlock
        # 第三次：一个上采样UpSample
        # 第四次：三个残差连接ResBlock
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # 输出后处理
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

        self.initialize()

    # 初始化参数
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, labels):
        """
        :param x: 输入的图像 [8, 3, 32, 32]
        :param t: 时间步 [8]
        :param labels: 标签 [8]
        """
        # Timestep Embedding
        temb = self.time_embedding(t)  # temb.shape = [8, 512]
        cemb = self.condition_embedding(labels)  # cemb.shape = [8, 512]

        # Downsampling
        h = self.head(x)  # h.shape = [8, 128, 32, 32]
        hs = [h]

        for layer in self.downblocks:
            # 第一次：两个残差连接ResBlock：[8, out_ch, 32, 32]
            # 第一次：一个下采样DownSample：[8, out_ch, 16, 16]
            # 第二次：两个残差连接ResBlock：[8, out_ch, 16, 16]
            # 第二次：一个下采样DownSample：[8, out_ch, 8, 8]
            # 第三次：两个残差连接ResBlock：[8, out_ch, 8, 8]
            # 第三次：一个下采样DownSample：[8, out_ch, 4, 4]
            # 第四次：两个残差连接ResBlock：[8, out_ch, 4, 4]
            h = layer(h, temb, cemb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            # 两个残差连接ResBlock：[8, out_ch, 4, 4]
            h = layer(h, temb, cemb)

        # Upsampling
        for layer in self.upblocks:
            # 第一次：三个残差连接ResBlock：[8, out_ch, 4, 4]
            # 第一次：一个上采样UpSample：[8, out_ch, 8, 8]
            # 第二次：三个残差连接ResBlock：[8, out_ch, 8, 8]
            # 第二次：一个上采样UpSample：[8, out_ch, 16, 16]
            # 第三次：三个残差连接ResBlock：[8, out_ch, 16, 16]
            # 第三次：一个上采样UpSample：[8, out_ch, 16, 16]
            # 第四次：三个残差连接ResBlock：[8, out_ch, 32, 32]
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)  # h.shape = [8, out_ch_1+out_ch_2, 32, 32]
            h = layer(h, temb, cemb)

        h = self.tail(h)  # h.shape = [8, 3, 32, 32]

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    unet = UNet(T=10000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.1)

    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])

    y = unet(x, t, labels)

    print(y.shape)