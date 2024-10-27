# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 18:40
# @Author  : 楚楚
# @File    : ConditionDiffusion.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    :param v: 系数alphas_bar v.shape = [1000]
    :param t: 时间 t.shape = [8]
    :param x_shape: 噪声
    """
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """

    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)  # out.shape = [8]

    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))  # [8, 1, 1, 1]


# 正向扩散过程
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """
        :param model: UNet模型
        :param beta_1: 1e-4
        :param beta_T: 0.02
        :param T: 1000
        """
        super(GaussianDiffusionTrainer, self).__init__()

        self.model = model

        self.T = T

        """
        register_buffer()：注册一个缓冲区（buffer），命名为 betas
            缓冲区是模型中的一种特殊张量，它与模型参数一起存储在模型的状态字典中，但不会被视为模型的可训练参数
        betas：\begin{aligned}x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon\quad\ldots(3)\\\text{;where}\ \epsilon\sim\mathcal{N}(0,I)\end{aligned}
        """
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())

        alphas = 1. - self.betas
        """
        torch.cumprod()：计算张量在指定维度的累积乘积
            alphas_bar[0] = alphas[0]
            alphas_bar[1] = alphas[0] * alphas[1]
            alphas_bar[2] = alphas[0] * alphas[1] * alphas[2]
            ...
            alphas_bar[N-1] = alphas[0] * alphas[1] * ... * alphas[N-1]
        """
        alphas_bar = torch.cumprod(alphas, dim=0)  # alphas_bar.shape = [1000]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # sqrt_alphas_bar: \sqrt{\bar{\alpha}_t}
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))

        # sqrt_one_minus_alphas_bar: \sqrt{1-\bar{\alpha}_t}
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        :param x_0: 原始图像 x_0.shape = [8, 3, 32, 32]
        :param labels: 标签 labels.shape = [10]
        :return:
        """
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=[x_0.shape[0], ], device=x_0.device)  # t.shape = [1000]

        noise = torch.randn_like(x_0)  # noise.shape = [8, 3, 32, 32]

        """
        x_t:=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon\end{aligned}
        """
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

        """
        在 DDPM（Diffusion Probabilistic Models）中，对时间步进行编码的作用在于提供了一个时间上的动态特征，使得模型能够更好地适应不同时间步的数据分布变化
        """
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


# 逆向扩散
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        """
        :param model: UNet模型
        :param beta_1: 1e-4
        :param beta_T: 0.02
        :param T: 1000
        :param w: w is the key to control the guidence
        """
        super(GaussianDiffusionSampler, self).__init__()

        self.T = T

        self.model = model

        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())  # betas.shape = [1000]
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)  # alphas_bar.shape = [1000]

        """
         F.pad(alphas_bar, [1, 0], value=1)
             [1, 0]：表示在tensor的最前面填充一个值，值为1；在tensor的最后面填充零个值
        """
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # self.posterior_var = \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}(1-\bar{\alpha}_{t-1})+\beta_{t}}
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # 预测与x_{t-1}相关的均值
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        :param x_t: 时间步t的图像 x_t.shape = [8, 3, 32, 32]
        :param t: 时间步t
        :param eps: 正向扩散过程输出的图像 eps.shape = [8, 3, 32, 32]
        """

        assert x_t.shape == eps.shape

        # \frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))
        return (
                extract(self.coeff1, t, x_t.shape) * x_t -
                extract(self.coeff2, t, x_t.shape) * eps
        )

    # 与x_{t-1}相关的均值和方差
    def p_mean_variance(self, x_t, t, labels):
        """
        :param x_t: 时间步t的图像 x_t.shape = [8, 3, 32, 32]
        :param t: 时间步t
        :param labels: 标签 labels.shape = [10]
        """

        # below: only log_variance is used in the KL computations
        # var = \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}(1-\bar{\alpha}_{t-1})+\beta_{t}}
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])  # var.shape = [1000]
        var = extract(var, t, x_t.shape)  # var.shape = [8, 1, 1, 1]

        # 预测噪声
        eps = self.model(x_t, t, labels)  # eps.shape = [8, 3, 32, 32]
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))  # nonEps.shape = [8, 3, 32, 32]

        eps = (1. + self.w) * eps - self.w * nonEps

        # xt_prev_mean = \frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        :param x_T: 时间步的图像 x_T.shape = [8, 3, 32, 32]
        :param labels: 标签 labels.shape = [10]
        """
        """
        Algorithm 2.
        """

        x_t = x_T  # x_t.shape = [8, 3, 32, 32]

        for time_step in reversed(range(self.T)):
            print(time_step)

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step  # t.shape = [8]

            # mean = \frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))
            # var = \frac{\beta_{t}(1-\bar{\alpha}_{t-1})}{\alpha_{t}(1-\bar{\alpha}_{t-1})+\beta_{t}}
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)

            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)

            else:
                noise = 0

            # 重参数化技巧
            # x_{t-1}=\frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))+\sigma_t\epsilon~\epsilon\sim\mathcal{N}(0,I)
            x_t = mean + torch.sqrt(var) * noise

            """
            torch.isnan(x_t)：生成一个与张量 x_t 相同大小的布尔类型张量，其中每个元素的值表示对应位置上 x_t 中的元素是否为 NaN。
                如果 x_t 中的元素是 NaN，则对应位置上的布尔值为 True，否则为 False
            .int()：将布尔类型的张量转换为整型张量，其中 True 被转换为 1，False 被转换为 0
            .sum()：对整型张量进行求和操作，统计其中非零元素的数量，即统计了 x_t 中 NaN 值的数量
            """
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0=x_t

        """
        torch.clip(x_0, -1, 1)：将张量 x_0 中的元素限制在区间 [-1, 1] 内，即将小于 -1 的元素置为 -1，将大于 1 的元素置为 1
        """
        return torch.clip(x_0, -1, 1)