---
layout: posts
title: Bayesian Estimation
date: 2019-01-18 14:00:00
categories: 概率论
tags: 机器学习 数学
mathjax: true
---
## Bayesian Estimation

频率派和贝叶斯派对于参数$\theta$ 的态度区别是：

* 频率派：$\theta$ 是一个未知的常量

* 贝叶斯派：$\theta$ 是一个随机变量

贝叶斯估计通过一个example引入：

[柏松分部 统计出现车辆数](https://newonlinecourses.science.psu.edu/stat414/sites/onlinecourses.science.psu.edu.stat414/files/lesson52/147882_traffic/index.jpg)

考虑一个路口间隔时段T内通过某一区域的车辆数这个样一个问题，这种问题常用到的概率模型是泊松分布。

泊松分布（Poisson distribution）：

$$ P( \textrm {k  events in interval}) = e^{-r}\frac{\lambda^k}{k!} $$

<!-- more -->

其中：$\lambda$ 是平均个事件发生次数 per interval，可以看到这一模型只有**一个参数**$\theta = \lambda$，只要确定了$\lambda$ 就确定了模型。泊松分布有如下性质:

$\lambda = E(X) = Var(X)$

如果交通控制工程师认为通过这一区域平均数（mean rate) $\lambda$ 为3 或5。工程师在收集数据之前可能认为$\lambda = 3$ 比 $\lambda = 5$ 更可能发生先于（这是一个先验知识），先验概率是：

$P(\lambda = 3) = 0.7$ 和 $P(\lambda = 5) = 0.3$

某一天，工程师在随机的一个时段T观察到$x = 7$ 辆车通过指定区域。**在这个观察结果下**（即条件概率），$\lambda = 3$ 和 $\lambda = 5$ 的概率是多少？

通过条件概率我们知道：

$P(\lambda=3 | X=7) = \frac{P(\lambda=3, X=7)}{P(X=7)}$

贝叶斯展开如下：

$P(\lambda=3 | X=7) = \frac{P(\lambda=3)P(X=7| \lambda=3)}{P(\lambda=3)P(X=7| \lambda=3)+P(\lambda=5)P(X=7| \lambda=5)}$

通过查询Possion累计分布函数，得到如下结果：

$P(X=7|\lambda=3)=0.988-0.966=0.022$  和

$P(X=7|\lambda=5)=0.867-0.762=0.105$

最后计算得到目标后验概率（**posterior probability**)：

$P(\lambda=3 | X=7)=\frac{(0.7)(0.022)}{(0.7)(0.022)+(0.3)(0.105)}=\frac{0.0154}{0.0154+0.0315}=0.328 $

同样得到：

$P(\lambda=5 | X=7)=\frac{(0.3)(0.105)}{(0.7)(0.022)+(0.3)(0.105)}=\frac{0.0315}{0.0154+0.0315}=0.672$

对比上面的$P(\lambda = 3) = 0.7$  和 $P(\lambda = 5) = 0.3$ 我们发现，贝叶斯估计“修正“了先验知识，平均出现5辆的可能性更大。

上面我们关于$p(\lambda) = \widehat{\lambda}$ 的假设就是先验概率$p(\theta)$ , 在这个问题中 $\theta$ 被当作变量来看待，$p(\theta)$是一个关于变量$\theta $ 的p.m.f（离散概率）。$p(\theta) * p(D|\theta) = p(X, \theta)$ ，这是一个联合一个关于变量 $\theta$ 和 $X$ 的joint p.d.f（联合概率分布），通过对$\theta$ 积分，我们可以获取$X$ 的概率分布：

$$
p(x)=\int_{-\infty}^{\infty}p(y,\theta)d\theta=\int_{-\infty}^{\infty}p(y|\theta)p(\theta)d\theta
$$

通过Bayes's theorem我们可以得到$\theta$的后验概率：

$p(\theta|y)=\frac{p(y, \theta)}{p(y)}=\frac{p(y|\theta)p(\theta)}{p(y)}$

参考文献：

https://newonlinecourses.science.psu.edu/stat414/node/241/
