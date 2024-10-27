import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


# 对时间步进行编码
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        """
        :param T: 时间步长T
        :param d_model: 通道
        :param dim: 通道*4
        """
        assert d_model % 2 == 0

        super(TimeEmbedding, self).__init__()

        time_embedding = torch.arange(0, d_model, step=2) / d_model * math.log(10000) # time_embedding.shape = [d_model//2]
        time_embedding = torch.exp(-time_embedding) # time_embedding.shape = [d_model//2]
        pos = torch.arange(T).float() # pos.shape = [T]

        """
        None表示在切片的位置增加一个新的维度
        """
        time_embedding = pos[:, None] * time_embedding[None, :]  # time_embedding.shape = [T, d_model // 2]
        assert list(time_embedding.shape) == [T, d_model // 2]

        # 正余弦对时间步编码
        time_embedding = torch.stack([torch.sin(time_embedding), torch.cos(time_embedding)],
                                     dim=-1)  # time_embedding.shape = [T, d_model // 2, 2]
        assert list(time_embedding.shape) == [T, d_model // 2, 2]

        time_embedding = time_embedding.view(T, d_model)  # time_embedding.shape = [T, d_model]

        """
        nn.Embedding.from_pretrained(emb) 用于创建一个 Embedding 层，并且使用预训练的嵌入权重来初始化该层
        """
        self.time_emb = nn.Sequential(
            nn.Embedding.from_pretrained(time_embedding),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim)
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
        :param t: 时间步t
        """
        emb = self.time_emb(t) # emb.shape = [T, dim]
        return emb


# 下采样
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super(DownSample, self).__init__()

        self.down = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.down.weight)
        init.zeros_(self.down.bias)

    def forward(self, x, time_embedding):
        """
        :param x: x.shape = [8, out_ch, width, height]
        """
        x = self.down(x) # x.shape = [8, out_ch, (width-3+2)//2 + 1, (height-3+2)//2 + 1]
        return x


# 上采样
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super(UpSample, self).__init__()

        self.up = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up.weight)
        init.zeros_(self.up.bias)


    def forward(self, x, time_embedding):
        """
        :param x: x.shape = [8, out_ch, width, height]
        """
        _, _, H, W = x.shape

        """
        F.interpolate(x, scale_factor=2, mode='nearest') 对输入张量 x 进行大小调整，使得输出张量的大小是原始大小的两倍
        插值的模式为最近邻插值，即对输出张量的每个像素，从原始输入张量中找到最近的像素值进行插值
        """
        x = F.interpolate(
            x, scale_factor=2, mode='nearest') # [8, out_ch, width*2, height*2]
        x = self.up(x) # [8, out_ch, (width*2-3+2)/1+1, (height*2-3+2)/1+1]
        return x # [8, out_ch, width*2, height*2]


# 注意力机制
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttnBlock, self).__init__()

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
        :param x: x.shape = [8, out_ch, 32, 32]
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

        return x + h # [8, out_ch, 32, 32]


# 残差块
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
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

    def forward(self, x, time_embedding):
        """
        :param x: x.shape = [8, 128, 32, 32]
        :param time_embedding: 正余弦对图像进行编码 time_embedding.shape = [8, 512]
        """
        h = self.block1(x) # h.shape = [8, out_ch, 32, 32]

        # time_proj(time_embedding).shape = [8, out_ch, 1, 1]
        h += self.time_proj(time_embedding)[:, :, None, None] # h.shape = [8, out_ch, 32, 32]
        h = self.block2(h) # h.shape = [8, out_ch, 32, 32]

        h = h + self.shortcut(x)  # h.shape = [8, out_ch, 32, 32]
        h = self.attn(h)  # h.shape = [8, out_ch, 32, 32]
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        """
        :param T: 时间步长T
        :param ch: 通道
        :param ch_mult: [1,2,3,4]
        :param attn: [2]
        :param num_res_blocks: 残差块的个数
        :param dropout: 随机选择要丢弃的单元，不参与前向传播和反向传播的过程，防止过拟合
        """
        super(UNet, self).__init__()

        """
        all()：迭代检查可迭代对象中的每个元素，如果所有元素都为真，则返回 True，否则返回 False
        """
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        # timestep特征维度是通道数的4倍，为什么？
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # Padding added to all four sides of the input
        # 输入预处理，单个卷积Conv
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # 下采样block：收缩阶段
        # 两个残差连接ResBlock，一个下采样DownSample
        self.downblocks = nn.ModuleList()

        # record output channel when dowmsample for upsample
        chs = [ch]

        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
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
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
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

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        """
        :param x: 输入的图像 [8, 3, 32, 32]
        :param t: 时间步 [8]
        """

        # Timestep embedding
        temb = self.time_embedding(t) # temb.shape = [8, 512]

        # Downsampling
        h = self.head(x) # h.shape = [8, 128, 32, 32]
        hs = [h]

        for layer in self.downblocks:
            # 第一次：两个残差连接ResBlock：[8, out_ch, 32, 32]
            # 第一次：一个下采样DownSample：[8, out_ch, 16, 16]
            # 第二次：两个残差连接ResBlock：[8, out_ch, 16, 16]
            # 第二次：一个下采样DownSample：[8, out_ch, 8, 8]
            # 第三次：两个残差连接ResBlock：[8, out_ch, 8, 8]
            # 第三次：一个下采样DownSample：[8, out_ch, 4, 4]
            # 第四次：两个残差连接ResBlock：[8, out_ch, 4, 4]
            h = layer(h, temb) 
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            # 两个残差连接ResBlock：[8, out_ch, 4, 4]
            h = layer(h, temb)

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
                h = torch.cat([h, hs.pop()], dim=1) # h.shape = [8, out_ch_1+out_ch_2, 32, 32]
            h = layer(h, temb)

        h = self.tail(h) # h.shape = [8, 3, 32, 32]

        assert len(hs) == 0
        return h


if __name__ == "__main__":
    batch_size = 8
    unet = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)

    x = torch.randn(batch_size, 3, 32, 32) # x.shape = [batch_size, channel, width, height]
    t = torch.randint(1000, (batch_size,)) # t.shape = [batch_size]

    y = unet(x, t)

    print(y.shape)


"""
UNet(
  (time_embedding): TimeEmbedding(
    (time_emb): Sequential(
      (0): Embedding(1000, 128)
      (1): Linear(in_features=128, out_features=512, bias=True)
      (2): Swish()
      (3): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (head): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (downblocks): ModuleList(
    (0): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=128, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
    (1): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=128, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
    (2): DownSample(
      (down): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (3): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): AttnBlock(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (proj_q): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_k): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_v): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (4): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): AttnBlock(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (proj_q): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_k): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_v): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (5): DownSample(
      (down): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (6): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
    (7): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
    (8): DownSample(
      (down): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (9): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
    (10): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
  )
  (middleblocks): ModuleList(
    (0): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): AttnBlock(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (proj_q): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_k): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_v): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (1): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Identity()
      (attn): Identity()
    )
  )
  (upblocks): ModuleList(
    (0): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (1): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (2): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (3): UpSample(
      (up): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (4): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (5): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (6): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (7): UpSample(
      (up): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (8): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): AttnBlock(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (proj_q): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_k): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_v): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (9): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 512, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): AttnBlock(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (proj_q): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_k): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_v): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (10): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 384, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
      (attn): AttnBlock(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (proj_q): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_k): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj_v): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (proj): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (11): UpSample(
      (up): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (12): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 384, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=128, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (13): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=128, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
    (14): ResBlock(
      (block1): Sequential(
        (0): GroupNorm(32, 256, eps=1e-05, affine=True)
        (1): Swish()
        (2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (time_proj): Sequential(
        (0): Swish()
        (1): Linear(in_features=512, out_features=128, bias=True)
      )
      (block2): Sequential(
        (0): GroupNorm(32, 128, eps=1e-05, affine=True)
        (1): Swish()
        (2): Dropout(p=0.5, inplace=False)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (attn): Identity()
    )
  )
  (tail): Sequential(
    (0): GroupNorm(32, 128, eps=1e-05, affine=True)
    (1): Swish()
    (2): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
"""