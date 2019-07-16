---
title: EM_algorithm
mathjax: true
date: 2019-06-10 15:54:04
categories:
tags: em
---

## 1. MLE (Maximum-likelihood-estimation)

最大似然方法：
$$
\begin{equation}
p(X | \theta) =\prod_{i=1}^{N}p(x_i|\theta) = L(\theta|X)
\end{equation}
$$
其中，样本: $X = {x_1,...,x_N}$。

左边的$p(X|\theta)$是由参数$\theta$支配的密度函数（density function)，**注意这是条件概率**


右边的$L(\theta|X)$是关于参数 $\theta$ 的likelihood（在给定数据$X$的情况下)，**注意这是函数**


从公式中可以看出，在给定 $\theta$ (假设参数）的情况下，**对已观测的实验结果用参数形式描述其概率**，在做这一步的时候用到了一个假设，即样本之间的出现是相互独立无关的（**i.i.d**)。鉴于其已经出现在现实世界中，我们有理由相信（无论是大数定律还是什么的）这种可能性是最大的，所以，如何让这种观测结果出 大变成主要目标，这就是我们的使用的max likelihood 的本质。

最大化（1）便得到最大似然的式子
$$
L(\theta) = \mathop{\arg\max}_{\theta} L(\theta|X)
$$

这里通常会用log函数替换，因为可以使得函数

## 2. MAP (Maximum A posterior estimation)

根据**Bayes**公式，我们可以得到如下结论
$$
P(\theta|X) \propto P(X|\theta) \cdot P(\theta)
$$
上面的表达式分别是：

$P(\theta|X)$:	**posterior** 后验概率

$P(X|\theta)$:	**likelihood** 似然

$P(\theta)$:		 **prior** 先验概率

即相比较MLE，**在似然后面乘上prior**，然后求最大，便是MAP[^2 ]

根据贝叶斯公式会有如下：
$$
\begin{equation}
P(\theta|X) \propto P(X|\theta) \cdot P(\theta)
\end{equation}
$$

$P(\theta|x)$ : 	**posterior probabiity** 后验概率
$P(X|\theta)$ :	**likelihood** 似然
$P(\theta)$ : 		**prior** 先验概率

<!-- more-->

## 2 Gussian Mixture

参照讲解[^1]，里面有一高斯混合的例子。（有几个峰值并不代表有几个高斯模型，如下图）

高斯混合模型的讲解[^3 ]

### 2.1. Single Gussian Model

对于单个高斯模型:

$$
\begin{equation}
\arg\min_\theta L(\theta|X) = \arg\min_{\theta}[\sum_{i=1}^Nlog(N(x_i|\mu,\sigma))]
\end{equation}
$$


参数是 $\theta = \{\mu, \sigma\}$, 我们对$log$似然函数(log likelihood)求极值后便可得到最大似然估计：

$\mu_{MLE}  = \frac{\partial L(\mu, \sigma|X)}{\partial \mu} 0$	

### 2.2. Gussian Mixture Model

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

$\theta  = \{\mu_1\cdot\cdot\cdot\mu_k,  \sigma_1\cdot\cdot\cdot\sigma_k, \alpha_1\cdot\cdot\cdot\alpha_{k-1}\}$ ，这里考虑一下为什么$\alpha$是只到$\alpha_{k-1}$，即$\alpha$ 自由度是k-1？[^2]

$$
\begin{equation}
\theta_{MLE} =\arg\min_{\theta}L(\theta|X) = \arg\min_{\theta} \sum_{i=1}^{N}log[\sum_{j=1}^K \alpha_jN(X|\mu_j, \sigma_j)]
\end{equation}
$$

这是后我们发现如果用极大似然求解会非常麻烦，不能得到close-form的解析解，因为log likelihood中出现了高斯模型的加和$log(A+B+C)$。

基于此，我们采用了迭代求解的方式，即我们提到的EM算法。

### 2.3 Gussian 混合模型求解

参照板书[^3 ]，采用迭代的方式，就要构建上一次和这次迭代的“关系”:
$$
\theta^{(i+1)} = \arg \min_{\theta} \int \log P(X,Z|\theta)\cdot P(Z|X,\theta^{(i)})
$$
对于这个式子，我们引入了隐变量$Z$，引入隐变量的原则有一条：**对其边缘概率（margin）积分后不影响原概率**，其起到隐藏、辅助的功能：

$$
P(x_i) = \int_{z_i}P_{\theta}(x_i|z_i)\cdot P_{\theta}(z_i)d{z_i}
$$

放到高斯混合模型的问题上(上面公式$P_{\theta}..$代表受参数$\theta$ 支配）：
$$
= \sum_{z_i}^k \alpha_{z_i}N(x_i|\mu_{z_i},\sigma_{z_i})
$$
发现正好是混合模型的样子，所以这个隐变量$Z $（即$\alpha$ ）是可行的。

对于普通的问题，我们有最大似然方式求解，现在要换成EM算法，我们要寻求等效。我们的目的是求：
$$
\begin{gather}

\hat{\theta}_{MLE}= \arg\min_{\theta}\log P(X|\theta) \\

\log P(X|\theta) =\log P(X,Z|\theta) - log P(Z|X,\theta)
\end{gather}
$$

原本使用最大似然 （9）即可求解，但由于无法求解，所以我们寻求与其相等的等式（10）来寻求突破，公式（10）来自于Bayes公式。

对（10）两遍求期望，求期望的时候我们要考虑**基于哪个分布**，在这里我们使用$P(Z|X,\theta)$，为什么呢？这是因为左边基于此概率求期望不改变原表达式，因为其不含$Z$，直接积分为1。
$$
E_z(\log P(X|\theta)) = \log P(X|\theta)\cdot\int_zP(z|X,\theta)dz = \log P(X|\theta)\cdot1
$$
同时，对右边求期望：
$$
\begin{aligned}
 &= \int_z\log P(X,z|\theta)\cdot P(z|X,\theta^{(i)})dz-\int_z\log P(z|X,\theta)\cdot P(z|X,\theta))dz \\
 &= Q(\theta, \theta^{(i)}) - H(\theta,\theta^{(i)})
 \end{aligned}
$$


现在我们将其分解为两部分：$Q$和$H$，总体目的是求得使似然函数最大的$\theta$ ，如果我们能证明在迭代中：$Q \uparrow、 H \downarrow$, 那就完美的解决了问题， **并且，得到这样的证明后，我们可以只最大化$Q$, 而不去理会$H$。**由于我们的算法本质是最大化Q函数，所以只需证明H随着变小即可。

### 2.4 终极目的证明H(i+1) < H(i)



## 3. EM algorithm

首先看常用的一个图（来自于Chuong B Do & Serafim Batzoglou, What is the expectation maximization algorithm?)，硬币实验的一个例子。

![](http://ww1.sinaimg.cn/mw690/6bf0a364ly1g3x1kv0o7fj20j10dt79h.jpg)

$E$ (Evidence）: 我们已经观察到的结果

$A$ : 选择A硬币 

$\bar{A}$ ：选择B硬币

### E-Step

在E- step: 我们使用Bayes公式获取latent varible（隐变量）的估计值：

$$
P(A|E) = \frac{P(E,A)}{P(E)} = \frac{P(E|A)*P(A)}{P(E,A) + P(E,\bar{A})}= \frac{P(E|A) * P(A)}{P(E|A)*P(A) + P(E|\bar{A})*P(\bar{A})}
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

## 附录

图1代码：
```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
N = 100
mu_1 = 3
mu_2 = 3.5
sigma_1 = 0.3
sigma_2 = 0.5
scape = 3
np.random.seed(0)
x1 = np.random.normal(mu_1, sigma_1, N)
x2 = np.random.normal(mu_2, sigma_2, N)
y1 = np.zeros(N)
y2 = np.zeros(N) + 0.1
# guass 1
plt.scatter(x1, y1, alpha = 0.9, marker = "x", label = r"$\mu = {}, \sigma = {}$".format(mu_1, sigma_1))
plt.legend()
guass1 = stats.norm.pdf(np.linspace(mu_1-scape*sigma_1,mu_1+scape*sigma,100), loc = mu_1, scale = sigma_1)
plt.plot(np.linspace(mu_1-scape*sigma_1,mu_1+scape*sigma_1,100),guass1) 
# guass 2
plt.scatter(x2, y2, alpha = 0.9, marker = "*", label = r"$\mu = {}, \sigma = {}$".format(mu_2, sigma_2))
plt.legend()
guass2 = stats.norm.pdf(np.linspace(mu_2-scape*sigma_2,mu_2+scape*sigma_2,100), loc = mu_2, scale = sigma_2)
plt.plot(np.linspace(mu_2-scape*sigma_2,mu_2+scape*sigma_2,N), guass2) 
# sum
sum = guass1 + guass2
plt.plot(np.linspace(mu_1-scape*sigma_1,mu_2+scape*sigma_2,N),sum, label="sum")
plt.legend()
plt.show()
```



## 为什么使用loglikelihood?
参看图片[^4]

![](http://ww1.sinaimg.cn/mw690/6bf0a364ly1g4lbkgqm6cj20k907xjrm.jpg)


[^1]: 清华大学 公开课《数据挖掘：理论与算法》
[^3 ]: 徐亦达机器学习：Expectation Maximization EM算法 【2015年版-全集】
[^4]:https://www.cnblogs.com/en-heng/p/5994192.html
>>>>>>> 3229282e84c92b5aa47b4cadfa32aecf717c6c3a
