---
title: Normlization
mathjax: true
date: 2025-05-19 12:05:44
categories:
tags: deeplearning LLM
---

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
  
[![LayerNomr](https://docs.pytorch.org/docs/stable/_images/layer_norm.jpg)](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

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