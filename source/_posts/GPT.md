---
title: GPT
mathjax: true
date: 2025-05-21 12:02:14
categories: LLM
tags: LLM
---

## cross_entropy的应用
```python
class BigramLanguageModel(nn.Module):
……
    def forward(self, idx, targets=None):
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# output
torch.Size([32, 65])
tensor(4.8786, grad_fn=<NllLossBackward0>)
```
```python
loss = F.cross_entropy(logits, targets)
```
其实计算的是
$$\text{CrossEntropy}(p,y) = -log(p_y)$$

其中：

$p$ 是 softmax 后的概率分布    
$y$ 是 ground-truth label (目标token)    
$p_y$ 是对应ground-truth 类别的概率    

对于这个Loss如果我们想估计一下是什么水平，那就对比随机猜的情况,，这个可以作为baseline  

$$ \text{loss} = -\text{log}(\frac{1}{65}) = \text{log}(65) \approx 4.17 $$

<!-- more -->

如果模型刚初始化，或者毫无学习能力，其交叉熵损失函数就会非常接近这个值

why loss=4.87 > 4.17 ?
这是因为：模型还没开始训练（刚初始化），预测可能不均匀分布，而是更糟糕的“错误分布”，这会导致预测目标token概率 $p_y \lt fra$


## multitorch.multinomial
这个函数用来采样，会根据传入的数据所提供的概率来采样
```python
import torch

# 一维概率分布
probs = torch.tensor([0.2, 0.5, 0.3])
# 采样 3 个样本，不重复采样
samples = torch.multinomial(probs, num_samples=3, replacement=False)
print("Samples:", samples)

# 二维概率分布
probs = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
# 采样 2 个样本，允许重复采样
samples = torch.multinomial(probs, num_samples=2, replacement=True)
print("Samples:", samples)
```

## temperature
$$\text{adjusted logits} = \frac{logits}{T} $$ 
temperature通过在softmax之前修改logits 来平滑或尖锐logits分布  
T=1.5：调整后的 logits 更小，概率分布更加平滑，所有类别的概率更加接近。  
T=0.5：调整后的 logits 更大，概率分布更加尖锐，最高概率的类别更加突出。
```python
import torch
import torch.nn.functional as F

# 定义原始 logits
logits = torch.tensor([[0.1, 0.7, 0.2]])

# 定义温度参数
temperature_high = 1.5
temperature_low = 0.5

# 调整 logits
adjusted_logits_high = logits / temperature_high
adjusted_logits_low = logits / temperature_low

# 转换为概率分布
probs_high = F.softmax(adjusted_logits_high, dim=-1)
probs_low = F.softmax(adjusted_logits_low, dim=-1)

print("Original logits:", logits)
print("Adjusted logits (T=1.5):", adjusted_logits_high)
print("Adjusted logits (T=0.5):", adjusted_logits_low)
print("Probs (T=1.5):", probs_high)
print("Probs (T=0.5):", probs_low)
```

## weighted sum
```python
B, T, C = 2, 4, 2
x = torch.arange(0, B*T*C, 1, dtype=torch.float32).reshape(B, T, C)
wei = torch.tril(torch.ones(T,T))
wei = wei/wei.sum(1, keepdim=True)
res = wei @ x
print("wei = ")
print(wei)
print("x = ")
print(x)
print("res = ")
print(res)
```
能理解到把x的每一行看做一个2维向量， W每一行对4个2维向量加权求和，如果从矩阵乘法角度去推导，用分块矩阵方式如何具象化这一过程？
关键的地方是只将w分块化，看做行向量的stack, 而x不做分块考虑，这样就直观了
每一行 output 是对所有 xᵢ（2D 向量）加权求和，权重是 W 的一行，这就是 attention 机制里 “加权求和” 的本质。
```python
W =
⎡ w₀ᵀ ⎤
⎢ w₁ᵀ ⎥
⎢ w₂ᵀ ⎥
⎣ w₃ᵀ ⎦   ∈ ℝ^{4×4}

W @ x =
⎡ w₀ᵀ @ x ⎤
⎢ w₁ᵀ @ x ⎥
⎢ w₂ᵀ @ x ⎥
⎣ w₃ᵀ @ x ⎦
```