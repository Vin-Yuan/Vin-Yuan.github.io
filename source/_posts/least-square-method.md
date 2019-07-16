---
title: 最小二乘法
mathjax: true
date: 2019-05-14 16:00:42
categories: math
tags: machine_learning, math
---

在实验曲线拟合数据的时候突然有个想法：是否所有连续的函数都可以通过多项式拟和？

对于这个问题，需要先了解最小二乘法法的原理：

最小二乘法的由来 [^1 ]：

法国数学家，阿德里安-馬里·勒讓德（1752－1833）提出让总的误差的平方最小的就是真值，这是基于如果误差是随机的，应该围绕真值上下波动。通过他的假设，我们将其应用到一般回归问题上就是如下形式：[^2 ]
$$
J(\theta)=\frac{1}{2} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}=\frac{1}{2} t r\left[(X \theta-Y)^{T}(X \theta-Y)\right] \tag{1}
$$
令误差最小的参数就是我们想要的参数。但这样的假设如何证明？ [^1 ]

<!-- more -->

高斯通过概率的角度补充了这个假设：所有偏离真实值的误差都是符合高斯分布的。需要拟合的数据都是我们观测到的，那么它出现的概率就应该是最大的（极大似然的角度），具体阅读参考 [^1 ]

最小二乘法就是对上面的式子求解，通过矩阵方式得到解析解，或者说正规方程的解（**Normal Equation**)，其结果正是 Ng Andrew的《机器学习》教程中的正规方程。
$$
\begin{equation}
\begin{aligned} \frac{\partial J(\theta)}{\partial \theta} &=\frac{1}{2} \cdot \frac{\partial \operatorname{tr}\left(\theta^{T} X^{T} X \theta-\theta^{T} X^{T} Y-Y^{T} X \theta+Y^{T} Y\right)}{\partial \theta} \\ &=\frac{1}{2} \cdot\left[\frac{\partial \operatorname{tr}\left(\theta I \theta^{T} X^{T} X\right)}{\partial \theta}-\frac{\partial \operatorname{tr}\left(\theta^{T} X^{T} Y\right)}{\partial \theta}-\frac{\partial \operatorname{tr}\left(\theta Y^{T} X\right)}{\partial \theta}\right] \\ &=\frac{1}{2} \cdot\left[X^{T} X \theta I+\left(X^{T} X\right)^{T} \theta I^{T}-X^{T} Y-\left(Y^{T} X\right)^{T}\right] \\ &=X^{T} X \theta-X^{T} Y \end{aligned}
\end{equation} \tag{2}
$$

令上式为0，得到解析解，Normal Equation，

$$
\theta= {\left( {X^TX} \right)^{ - 1}}{X^T}Y \tag{3}
$$

(1)最小二乘法和梯度下降法在线性回归问题中的目标函数是一样的(或者说本质相同)，都是通过最小化均方误差来构建拟合曲线。

(2)二者的不同点可见下图(正规方程就是最小二乘法)：[^3 ]

| 梯度下降                         | 正规方程                                                     |
| -------------------------------- | ------------------------------------------------------------ |
| 需要学习率$\alpha$               | 不需要                                                       |
| 多次迭代                         | 一次计算                                                     |
| 当特征数量$n$ 很大时也能很好适用 | 需要计算$(X^TX)^{-1}$，如果特征数量$n$ 非常大，运算代价比较大，因为矩阵求逆的时间复杂度为$O(n^3)$，通常来说当n小于10000时还是可以接受的 |
| 适用于大部分模型                 | 只适用先行模型                                               |

> 需要注意的一点是最小二乘法只适用于**线性模型**(这里一般指**线性回归**)；而梯度下降适用性极强，一般而言，**只要是凸函数**，都可以通过梯度下降法得到全局最优值(对于非凸函数，能够得到局部最优解)。

最小二乘法由于是最小化均方差，所以它考虑了每个样本的贡献，也就是每个样本具有相同的权重；由于它采用距离作为度量，使得他对噪声比较敏感(**最小二乘法假设噪声服从高斯分布**)，即使得他它对异常点比较敏感。因此，人们提出了加权最小二乘法，

相当于给每个样本设置了一个权重，以此来反应样本的重要程度或者对解的影响程度。

上面所说的只适用先行模型其实是一个广义的含义：

consider a model:

$y_i = b_0+b_1 x^{n_1}_i + \cdots+ b_px^{n_p}_i + \epsilon_i.$

This can bex rewritten as:
$$
y = 
X b + \epsilon;\\
X= \begin{pmatrix}
  1 & x_{1}^{n_1} & \cdots & x_{1}^{n_p} \\
  1 & x_{2}^{n_1} & \cdots & x_{2}^{n_p} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  1 & x_{n}^{n_1} & \cdots & x_{n}^{n_p} \\
 \end{pmatrix}.
$$
这也是一种线性模型：polynomial regression is considered a special case of multiple linear regression [^4 ]

最小二乘法分为两类：

## Linear least squares
线性模型是指model通过参数的先行组合构成的
$$
\begin{equation}
f(x, \beta)=\sum_{i=1}^{m} \beta_{j} \phi_{j}(x) \tag{4}
\end{equation}
$$
其中 $\phi_j $ 是 $x$ 的函数，这也是《统计学习方法》第一章拟合非线性曲线用到的：$h(x;w) = w_2x^3 + w_1x^2+w_0x^0$形式，通过$x$的不同组合提升feature的维度，进而构成先行模型。如果令$\phi _j(x) = x_j$，通过最小二乘法的Normal Equation可以得到(3)的close-form（close-form指可以通过有限的数字组合表示的解[^5]）。

其实**多元高次**组合的多项式依旧是线性组合的特殊形式的：

| $w_0$ | $w_1$ | $w_2$ | $w_3$| $w_4$ |
| ----- | ----- | ----- | ---- | ---- |
| $x_0$ | $x_1$ | $x_2$ |$x_3$ | $x_4$|
| $x_0$ | $x_1$ | $x_0 x_1$ |$x_0^2$ | $x_1^2$|

高次多项式拟合曲面参照[^6]

以表格数据直观展现参数$W$在模型变复杂（阶数越来越大）时的变化，在没有正则项的时候scale会越来越大[^7]

### Polynomial Regression

当数据并不符合线性规律而是更复杂的时候，将每一维特征的幂次添加为**新的特征**，再对**所有的特征**进行线性回归分析。这种方法就是 **多项式回归**。

当存在多维特征时，多项式回归能够发现特征之间的**相互关系(例如$x_1x_2x_3^3$）**，这是因为在添加新特征的时候，添加的是所有特征的排列组合[^8]。

多项式回归问题需要考虑**特征维度爆炸**的问题，维度为n，幂数为d的的新特征数共有$\frac{(n+d)!}{d!n!}$个。

## Non-Linear least squares

非线性是指与线性相反，不是通过线性组合构成的，例如：$m(x,\theta_i) = \theta_1 + \theta_2x^{\theta_3}$，这种由于构成复杂，无法通过Normal equation得到close-form解，所以只有通过迭代方式求解。



[^1 ]: <https://www.matongxue.com/madocs/818.htm>
[^2 ]:<https://blog.csdn.net/u011893609/article/details/80016915>
[^3 ]: <https://www.cnblogs.com/wangkundentisy/p/7505487.html>
[^4 ]: <https://stats.stackexchange.com/questions/92065/why-is-polynomial-regression-considered-a-special-case-of-multiple-linear-regres> 
[^5]: <https://en.wikipedia.org/wiki/Closed-form_expression>
[^6]: <https://www.cnblogs.com/zzy0471/p/polynomial_regression.html>
[^7]: <https://www.jianshu.com/p/eac4c7928b56>
[^8]: <https://blog.csdn.net/tsinghuahui/article/details/80229299>
