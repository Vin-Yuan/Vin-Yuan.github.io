---
title: covariance and correlation
mathjax: true
date: 2019-07-13 17:45:16
categories:
tags: math
---

## covariance

协方差：

> 还要考虑一点，每个点的概率是不一样的，因此各个矩形的面积并非是平等的，或者说权重是不一样的，所以需要对面积和进行加权平均，也就是对面积和计算数学期望，这就得到了：

$$
Cov(X, Y) = E[(X-\mu_{X})(Y-\mu_{Y})]
$$

## Variance

Variance是covariance的特例：
$$
Var(X) = Cov(X, X)
$$


## correlation

$$
\rho_{X,Y} = \frac{Cov(X,Y)}{\sigma(X) \sigma(Y)}
$$

对于分母下面的形式，通常我们都知道是消除scale影响的系数，但是为什么这样scale就一样了？

<!-- more -->

资料[^1]很好的解释了原因，高中物理就经常用到这种消除单位的因素，进而比较两个变量的trick。
$$
\begin{aligned}
\rho_{X,Y} &= \frac{Cov(X,Y)(\text{厘米.公斤})}{\sigma(X)(\text{厘米}) \sigma(Y)(\text{公斤})} \\
\rho_{Y,Z} &= \frac{Cov(Y,Z)(\text{年龄.公斤})}{\sigma(Y)(\text{年龄}) \sigma(Z)(\text{公斤})}
\end{aligned}
$$

当它们都约掉单位后，变成为可以比较的一致变量了

[^1]:https://www.matongxue.com/madocs/568.html
