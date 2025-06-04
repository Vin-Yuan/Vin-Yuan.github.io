---
title: residual
mathjax: true
date: 2025-06-04 08:55:01
categories: LLM, deeplearning
tags: LLM, deeplearning
---

在两层神经网络中，**反向传播（Backpropagation）** 的目的是计算损失函数关于每一层权重（weights）的梯度，用于更新权重。下面我会一步步说明梯度是如何与权重相关联的。

---

## 假设一个简单的两层神经网络结构：

### 网络结构：

输入 → 第一层（Linear + Activation） → 第二层（Linear） → 输出（Loss）

* 输入：$\mathbf{x} \in \mathbb{R}^d$
* 第一层权重：$\mathbf{W}_1 \in \mathbb{R}^{h \times d}$，偏置：$\mathbf{b}_1 \in \mathbb{R}^h$
* 激活函数：ReLU 或 Sigmoid（记为 $f$）
* 第二层权重：$\mathbf{W}_2 \in \mathbb{R}^{o \times h}$，偏置：$\mathbf{b}_2 \in \mathbb{R}^o$
* 输出层无激活（或直接用 MSE）

---

## 前向传播公式：

1. 第一层输出（隐层激活）：
   $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$
   $\mathbf{a}_1 = f(\mathbf{z}_1)$

2. 第二层输出（模型最终输出）：
   $\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$

3. 损失函数（以 MSE 为例，标签为 $\mathbf{y}$）：
   $L = \frac{1}{2} \|\mathbf{z}_2 - \mathbf{y}\|^2$

---

## 反向传播计算梯度：

目标是求 $\frac{\partial L}{\partial \mathbf{W}_2}$ 和 $\frac{\partial L}{\partial \mathbf{W}_1}$。

### 第一步：从输出层反向传播

**输出层梯度（loss 对 $\mathbf{z}_2$ 的导数）**
$\delta_2 = \frac{\partial L}{\partial \mathbf{z}_2} = \mathbf{z}_2 - \mathbf{y}$

**对第二层权重的梯度**
$\frac{\partial L}{\partial \mathbf{W}_2} = \delta_2 \cdot \mathbf{a}_1^T$

> 💡 这个公式说明：第二层权重的梯度是误差 $\delta_2$ 与前一层激活 $\mathbf{a}_1$ 的外积。通过式子可以理解到，误差被乘以了一个权重，那些**对误差贡献比较大的激活单元**所对应的**weight**会被更大程度的减小

---

### 第二步：继续向前传播误差到第一层

**传播误差到第一层**
$\delta_1 = \left(\mathbf{W}_2^T \delta_2\right) \odot f'(\mathbf{z}_1)$

> * $\odot$ 表示按元素相乘（Hadamard product）
> * $f'(\mathbf{z}_1)$ 是激活函数的导数（例如 ReLU 的导数是 0 或 1）

**对第一层权重的梯度**
$\frac{\partial L}{\partial \mathbf{W}_1} = \delta_1 \cdot \mathbf{x}^T$

---

## 总结：梯度与权重的关系

| 权重                | 对应梯度表达式                                                                         | 梯度依赖于                                        |
| ----------------- | ------------------------------------------------------------------------------- | -------------------------------------------- |
| $\mathbf{W}_2$ | $\frac{\partial L}{\partial \mathbf{W}_2} = \delta_2 \cdot \mathbf{a}_1^T$ | 当前层误差 $\delta_2$ 和前一层输出 $\mathbf{a}_1$ |
| $\mathbf{W}_1$ | $\frac{\partial L}{\partial \mathbf{W}_1} = \delta_1 \cdot \mathbf{x}^T$    | 当前层误差 $\delta_1$ 和输入 $\mathbf{x}$       |

因此，**每层权重的梯度取决于该层输出误差与前一层的输出（或输入）的乘积**。这个过程是反向传播的核心：误差从输出层一步步反传，每层的权重根据其对最终误差的贡献进行调整。

---


当然，我们来逐句解释这段话的核心意思，尤其是关于“**aren’t very meaningful**” 的部分。这段话是为了说明**为什么早期的深层神经网络在训练时表现很差**，并为后来使用 skip connection（残差连接）做铺垫。

---


https://www.youtube.com/watch?v=Q1JCrG1bJ-A
## 📌 逐段解释：

> **when we initialize a neural network all of the layers have randomly chose weights**

神经网络初始化时，权重是随机的 —— 每一层的权重矩阵都是随机生成的。

---

> **so when we pass an input through each layer of the network the activations are being multiplied by a random weight matrix**

前向传播时，输入数据经过每一层都会被随机的权重矩阵变换，换句话说：输入信号在每一层都被“打乱”一次。

---

> **if we do this many many times then by the time the data gets to the output layer almost none of what we’re producing is actual signal related to the input**

如果层数很深（比如几十层），输入数据就会被这些随机变换一遍又一遍，到了最后一层时，输出结果几乎不再携带原始输入的“有效信息” —— 变成了一堆“随机激活”（noise）。

这就像把一个信号通过几十个带随机旋转的变压器处理之后，最后剩下的根本不知道是什么。

---

> **we've essentially scrambled that input into random noise**

我们可以说：**原始输入被“扰乱”成了随机噪声**。这种情况下，模型的输出和输入几乎没有实际联系。

---

> **then when we perform an update we’ll compute the loss on that output and propagate it back through the network**

训练时，我们计算 loss，并进行反向传播（backpropagation）以更新权重。

---

> **but that loss may not be very informative at the layers late in the network because the inputs were mostly random noise**

反向传播的梯度来源于输出误差，但**如果输出本身已经是“随机激活”**，那这个误差根本不能反映输入的真实特征，所以它对靠后的几层的更新意义也不大。

---

> **when we pass gradients back through the network each layer multiplies the deltas by its weight matrix**

在反向传播过程中，每一层的梯度也会被它的权重矩阵（也就是在前向传播时使用的那个）反向变换。因为这些矩阵是随机的，所以：

---

> **by the time we get back to the early layers that had inputs with some meaningful connection to the data the gradients have also been scrambled**

梯度回传到早期层时，已经**被多层的随机变换“扰乱”**，**梯度本身也变得和数据无关了**。

---

> **so the updates we do to the later layers aren’t very meaningful because their inputs have been scrambled and don’t mean very much**

后面的层（靠近输出）的输入是随机的，因此即使我们对它们做了梯度更新，也是在优化一些“无意义的东西”——所以这个更新 **“不是很有意义”**（not very meaningful）。

---

> **and the updates we do to the early layers aren’t very meaningful because the gradients have been scrambled**

早期层的输入其实还比较接近原始输入，但 **梯度本身已经被“污染”了**，所以即使这些层“想学”，也无法从错误中获得有效信息。

---

> **this is why we tend to see not much improvement for an awful lot of training time as gradient descent essentially wanders around randomly**

这就解释了**为什么深层网络在训练初期进展缓慢**，甚至基本不学习 —— 因为梯度下降没有清晰的方向，就像“在黑夜里瞎走”。

---

## 🧠 总结核心点：“not very meaningful”的含义

> * 对于**后面的层**，它们的输入已经像噪声，所以即使 loss 传回来，**更新的目标本身就没什么意义**。
> * 对于**前面的层**，它们的输入还比较“干净”，但传回来的梯度已经乱掉了，**更新的方向也没有意义**。
> * 所以，这些更新不是完全错误，而是**无效、低效、方向性差**，因此叫 **"not very meaningful"**。

---

需要我再结合梯度消失/梯度爆炸来进一步解释吗？或者用可视化图示、代码模拟也可以帮你理解这个“扰乱”和“梯度混乱”的过程。


