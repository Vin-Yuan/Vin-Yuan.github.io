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

更具贝叶斯公式会有如下：

$$
P(\theta|X) \propto P(X|\theta) \cdot P(\theta)
$$

$P(\theta|x)$ : 	**posterior probabiity** 后验概率
$P(X|\theta)$ :	**likelihood** 似然
$P(\theta)$ : 		**prior** 先验概率

## 2. Gussian Mixture

参照讲解[^1]，里面有一高斯混合的例子。（有几个峰值并不代表有几个高斯模型，如下图）

高斯混合模型的讲解[^3 ]

### single Gussian Model

对于单个高斯模型:
$$
\mathop{\arg\min}_\theta L(\theta|X) = \mathop{\arg\min}_{\theta}[\sum_{i=1}^Nlog(N(x_i|\mu,\sigma))]
$$
参数是 $\theta = \{\mu, \sigma\}$, 我们对$log$似然函数(log likelihood)求极值后便可得到最大似然估计：

$\mu_{MLE}  = \frac{\partial L(\mu, \sigma|X)}{\partial \mu} 0$	

### Gussian Mixture Model

对于多个高斯的混合模型：
$$
\begin{equation}
\begin{gathered}
P(X|\theta) = \sum_{k=1}^{K}\alpha_{k}N(X|\mu_k, \sigma_k) \ ,\\
\sum_{k=1}^K \alpha_k = 1
\end{gathered}
\end{equation}
$$
需要指出，为什么使用$\alpha_k$这种形式来构建混合模型，而非使用$\frac{1}{k}$着各种形式？

![](http://ww1.sinaimg.cn/mw690/6bf0a364ly1g3zigk9996j20am070q3a.jpg)

参照上图，加上数据由两个高斯模型混合而成，这是后使用$\frac{1}{k}$ 直接的效果是均分了每一个高斯的贡献度，这是不合理的，从图中看出对于3.5处的点，很明显第二个模型贡献的概率大一些。所以，基于此，我们采用了$\alpha_k$ 这种结构，同时约束$\sum_{k=1}^K \alpha_k = 1$

对于gussian 混合模型，我们可以看到，对于含有隐变量的问题来讲，$\alpha_k$便是隐变量，这样一来，参数就是如下：

$\theta  = \{\mu_1\cdot\cdot\cdot\mu_k,  \sigma_1\cdot\cdot\cdot\sigma_k, \alpha_1\cdot\cdot\cdot\alpha_{k-1}\}$ ，这里考虑一下为什么$\alpha$是只到$\alpha_{k-1}$，即$\alpha$ 自由度是k-1？
$$
\theta_{MLE} =\mathop{\arg\min}_{\theta}L(\theta|X) = \mathop{\arg\min}_{\theta} \sum_{i=1}^{N}log[\sum_{j=1}^K \alpha_jN(X|\mu_j, \sigma_j)]
$$
这是后我们发现如果用极大似然求解会非常麻烦，不能得到close-form的解析解，因为log likelihood中出现了高斯模型的加和$log(A+B+C)$。

基于此，我们采用了迭代求解的方式，即我们提到的EM算法。


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
[^3 ]: 徐亦达机器学习：Expectation Maximization EM算法 【2015年版-全集】