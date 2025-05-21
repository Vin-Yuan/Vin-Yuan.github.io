---
title: Pytorch常用例子
mathjax: true
date: 2025-02-19 18:34:58
categories:
tags: pytorch
---

## 1. 构造数据

```python
import torch

# 1. 全 0 / 全 1 / 常数
torch.zeros(3, 4)           # shape=(3, 4)
torch.ones(2, 3)
torch.full((2, 2), 7.0)
x = torch.randn(10,8,4)
y = torch.ones_like(x)   # 复制shape

# 2. 随机数据
torch.randn(5, 10)          # 标准正态分布 N(0,1)
torch.rand(3, 3)            # 均匀分布 [0, 1)
torch.randint(0, 10, (2, 4)) # 整数随机数

# 3. 类似 numpy 的方式
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)     # [0., 0.25, 0.5, 0.75, 1.]

# 4. 从 numpy 转换
import numpy as np
torch.from_numpy(np.array([[1, 2], [3, 4]]))
```

## 2. 常用输入数据

```python
# 假数据：10 张 RGB 图片（3 通道，32x32），每张图一个标签
x = torch.randn(10, 3, 32, 32)     # 图像张量
y = torch.randint(0, 5, (10,))     # 标签：5 类分类任务
```

## 3. 向量化操作

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)

# 1. 矩阵乘法
out = a @ b                    # shape=(3,5)
out = torch.matmul(a, b)

# 2. 广播加法
x = torch.randn(10, 5)      # 广播机制，自动扩展
bias = torch.randn(5)
x = x + bias                  
x = torch.arange(0, 10, 1).reshape(2, 5) 
bias = torch.tensor(1.0)    # 广播机制，scalar 自动扩展
y = x + bias


# 3. 拼接
torch.cat([a, a], dim=0)      # 拼接行
torch.cat([a, a], dim=1)      # 拼接列

# 4. reshape & transpose
x = torch.randn(2, 3, 4)
x.view(6, 4)                  # 改形状
x.permute(1, 0, 2)            # 交换维度

# 5. 选择/掩码
x = torch.randn(10)
mask = x > 0
x[mask]                       # 选出正数元素

```