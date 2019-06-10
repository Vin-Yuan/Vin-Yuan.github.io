---
title: EM_algorithm
mathjax: true
date: 2019-06-10 15:54:04
categories:
tags: em
---

## 1. Maximum-likelihood

最大似然方法：
$$
p(X | \theta) =	\prod_{i=1}^{N}p(x_i|\theta) = L(\theta|X)
$$
其中，样本: $X = {x_1,...,x_N}$。

左边的$p(X|\theta)$是由参数$\theta$支配的密度函数（density function)

右边的$L(\theta|X)$是参数 $\theta$ 的likelihood（在给定数据$X$的情况下)。

从公式中可以看出，在给定$\theta$ (假设参数）的情况下，对已观测的实验结果用参数形式描述其概率，在做这一步的时候用到了一个假设，即样本之间的出现是相互独立无关分布的的（i.i.d)。鉴于其已经出现在现实世界，我们有理由相信（无论是大数定律还是什么的）这种可能性是最大的，所以，如何让这种观测结果出现的可能性最大变成主要目标，这样遍得到我们的最大似然方法。