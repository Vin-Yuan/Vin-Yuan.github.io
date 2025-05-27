---
title: Normlization
mathjax: true
date: 2025-05-19 12:05:44
categories:
tags: deeplearning LLM
---

# Layer Normalization


验证LayerNorm的，通过使用torch.mean和torch.var复现的时候发现不一致
LayerNorm默认使用的是bias的整体方差, divided by N
torch.var默认使用的是无bias的样本方差, devided by N-1


对于每一个样本的特征向量 $x \in \mathbb{R}^d$ ，LayerNorm 执行以下操作：

$$
\operatorname{LayerNorm}(x)=\gamma \cdot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

- $\mu, \sigma^2$ ：当前样本的均值和方差（仅用于归一化）
- $\gamma$ ：可学习的缩放参数（scale，类似于权重）
- $\beta$ ：可学习的偏移参数（bias，偏置）
  
[![LayerNomr](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0257ddec-b348-41d7-905d-5bc2b54fd557_1280x720.png)](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

<!-- more -->

对于这两个偏移参数如何更新，探究了一下底层实现
```python
import torch
import torch.nn as nn
import torch.optim as optim

layer_norm = nn.LayerNorm(2)
x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # 目标全为 0

optimizer = optim.SGD(layer_norm.parameters(), lr=0.1)

for i in range(3):
    optimizer.zero_grad()
    out = layer_norm(x)
    loss = ((out - target)**2).mean()
    loss.backward()
    optimizer.step()
    
    print(f"Step {i}, beta: {layer_norm.bias.data}")

```

$\gamma$ 和 $\beta$ 的维度是 (batch_size, seq_len)
$\sigma$ 以及 $\mu$ 的维度是 (batch_size, seq_len, feature_dim), 在完成标准化后，这两个会以向量的形式
例如:
```python
x = [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]

LayerNorm 会对每一行做标准化：
=> 每行变成均值=0，方差=1 的向量

gamma, beta = [γ1, γ2, γ3, γ4], [β1, β2, β3, β4]

最后输出 = normalized * gamma + beta
```
底层在相乘的时候，一般会向量化
$ \hat{x} * {\gamma}^{T} + \beta$
其中 $\hat{x} ,\gamma, \beta \in \mathbb{R}^d$

$\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2}}$
$ y = \hat{x} \cdot \gamma + \beta $
对于$\beta$, 反向传播的梯度： $\frac{\partial L}{\partial \beta}=\frac{\partial L}{\partial y}$
对于$\gamma$, 反向传播梯度： $\frac{\partial L}{\partial \gamma}=\frac{\partial L}{\partial y} \cdot \hat{x}$

那么，为什么 γ 和 β 要共享？
### 1. 归一化后丢失了尺度和偏移信息，需要 γ、β 来恢复表达能力
归一化本质上是把数据变成了零均值、单位方差。

这样虽然有助于稳定训练，但会损失一些特征的表达能力（例如，“这个神经元原本输出很大是有意义的”）。

所以通过引入 γ 和 β 这两个 可学习参数，网络可以在训练中学习“是否需要放大某些维度”或“整体平移”，以恢复这种表达能力。

🎯 关键点：γ 和 β 是 模型的一部分，并不是用来适配每个样本，而是学习一种在所有样本上都有效的变换方式，这符合深度学习模型“共享参数”的理念。
### 2. 不对每个样本单独学习 γ 和 β 是为了避免过拟合 + 保持参数效率
如果 γ 和 β 对每个样本都有独立的一套，那意味着参数量将随着 batch size 成倍增长。

这会：

大幅增加计算和内存负担

破坏模型的泛化能力（相当于为每个样本定制归一化，可能会过拟合）

所以：共享 γ 和 β 是一种在保持模型表达能力和计算效率之间的权衡。

### 3. 与 BatchNorm 的区别也体现出这种设计哲学
BatchNorm 使用的是 batch 内的统计量（跨样本统计），适用于图像等同分布样本。

LayerNorm 使用的是 样本内的统计量，避免依赖 batch 大小（适合 Transformer 这种序列建模）。

但无论哪种归一化，γ 和 β 始终是共享的参数，因为它们是模型本身的一部分，不依赖于输入样本。

🧪 举个比喻：
你有一个归一化后的图像数据集，每张图都被标准化成亮度为 0，标准差为 1。但你知道有些图像本该亮一些、有些本该暗一些。于是你训练一个“亮度增益”和“亮度偏移”参数，用来统一地调整所有图像。你不会为每张图学一个增益，而是找出一组对所有图都适用的参数。

✅ 总结：
问题	解释
为什么 γ 和 β 要 batch 共享？	因为它们是模型的一部分，用于恢复表达能力，不是输入的一部分；共享可以减少参数量、避免过拟合
为什么不对每个样本独立学习 γ 和 β？	这样会大大增加参数、容易过拟合，并且不符合深度学习“参数共享”的核心设计哲学
γ 和 β 的作用是什么？	恢复归一化过程丢失的尺度和偏移信息，使模型保留学习能力


# Batch Normliazation

Batch Normalization（批量归一化）中的一个重要概念：运行统计量（running statistics） 的更新和使用。在训练过程中，Batch Normalization 会计算每个批次的均值和方差，并用这些统计量来归一化当前批次的数据。然而，这些批次内的统计量并不直接用于最终的归一化，而是用来更新运行统计量，这些运行统计量会在推理（inference）阶段使用。

```python
import torch
import torch.nn as nn

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        if self.training:
            # 计算当前批次的均值和方差
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            # 更新运行统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            # 归一化当前批次的数据
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # 使用运行统计量归一化数据
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        # 应用缩放和偏移
        return self.gamma * x_norm + self.beta
```