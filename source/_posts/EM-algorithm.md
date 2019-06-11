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

右边的$L(\theta|X)$是关于参数 $\theta$ 的likelihood（在给定数据$X$的情况下)。

从公式中可以看出，在给定 $\theta$ (假设参数）的情况下，**对已观测的实验结果用参数形式描述其概率**，在做这一步的时候用到了一个假设，即样本之间的出现是相互独立无关的（**i.i.d**)。鉴于其已经出现在现实世界中，我们有理由相信（无论是大数定律还是什么的）这种可能性是最大的，所以，如何让这种观测结果出现的可能性最大变成主要目标，这就是我们的使用的max likelihood 的本质。

## 2. Gussian Mixture

参照讲解[^1]，里面有一高斯混合的例子。（有几个峰值并不代表有几个高斯模型）

## 3. EM algorithm

首先看常用的一个图（来自于Chuong B Do & Serafim Batzoglou, What is the expectation maximization algorithm?)，硬币实验的一个例子。

![](http://ww1.sinaimg.cn/mw690/6bf0a364ly1g3x1kv0o7fj20j10dt79h.jpg)

$E$ (Evidence）: 我们已经观察到的结果

$A$ : 选择A硬币 

$\bar{A}$ ：选择B硬币

### E-Step

在E- step: 我们使用Bayes公式获取latent varible（隐变量）的估计值：
$$
P(A|E) = \frac{P(E,A)}{P(E)} = \frac{P(E|A)*P(A)}{P(E,A) + P(E,\bar{A})} = \frac{P(E|A) * P(A)}{P(E|A)*P(A) + P(E|\bar{A})*P(\bar{A})}
$$
$P(E|A)$ 是什么？

在选择A硬币的情况下，出现E这个evidence的概率，即用A模型生成E这种观测结果，什么样子的呢：

对于第二行，9次正面，1次反面的实验结果：$P(E|A) = (\hat{\theta}_A)^9 * (1-\hat{\theta}_A)^1 $，**正如上面所说，用假设的参数模型去描述现实中发生的结果**。

$P(A)$和$P(\bar{A})$ 这里假设相等，为 0.5，选A选B是随机。

以此采用这种方式获取5次实验选择硬币A，硬币B的概率（在试验结果下的**条件概率**）

由于选择哪一枚硬币是一个隐变量，所以我们可以将每次实验观测到的结果看作是两个模型的混合。以第二行为例子，9次正面，1次反面，9次正面是如何形成的？是两个硬币“混合”形成的，A硬币contribute了0.8，B硬币贡献了0.2，计算后就是A贡献了7.2次，B贡献了1.8次。

最后，统计A总共贡献多少次Head/Tail, B贡献多少次Head/Tail。

### M-Step

在max likelihood阶段，按照上文所说，最大似然就是让概率模型偏向于最能呈现实验现象的方法。对最大似然求解后便会得到公式 $\hat{\theta} = \frac{H}{H+T} $，计算就不细说了，图中有说明。

[^1]: 清华大学 公开课《数据挖掘：理论与算法》