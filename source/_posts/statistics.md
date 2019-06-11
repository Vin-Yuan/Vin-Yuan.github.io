---
title: statistics
date: 2019-02-19 12:22:07
categories: statistic
tags: statistic
mathjax: true
---
## 1.期望 

李航统计与机器学习书中理论模型中的风险函数 (risk function)定义如下，代表一种理想情况下计算误差代价的方法。

​              $R_{exp}(f) = E[Y, L(X)] = \sum_{X,Y}L(y, f(x))P(x, y) = \int_{X,Y}L(y,f(x))P(x,y)dxdy$

而在4.1.2 章后验概率最大化的地方有个公式：

​               $R_{exp}(f) = E_x \sum_{i=1}^K[L(c_k, f(X))]P(c_k|X)$ 

初看比较费解，怎么变成条件概率了？

<!-- more -->

其实这个公式要从多重积分的角度考虑就明显了，相当于：

​                             $\int_X[\int_Y f(x,y)p(x,y)dy]dx$.

在对$y$积分的时候，$x$相当于一个常量，从这一角度考虑，正好就是 $p(y|x)$ ，即一个条件概率形式。