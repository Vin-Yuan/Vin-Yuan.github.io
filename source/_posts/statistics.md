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

## 2.Estimator 估计量

 a function of the data that is used to infer the value of an unknown parameter in a statistical model,can be writed like $\hat{\theta}(X)$."估计量"是样本空间映射到样本估计值的一个函数 (Then an estimator is a function that maps the sample space to a set of sample estimates.)估计量用来估计未知总体的参数，它有时也被称为估计子；一次估计是指把这个函数应用在一组已知的数据集上，求函数的结果。对于给定的参数，可以有许多不同的估计量。

### Estimand

 The parameter being estimated,like $\theta$.
 Estimate: a particular realization of this random variable $\hat{\theta}(X)$  is called the "estimate",like $\hat{\theta}(x)$.

Bias: The bias of $\widehat{\theta}$ is defined as $$B(\widehat{ \theta }) = \operatorname{ E }(\widehat{ \theta }) - \theta$$.
It is the distance between the average of the collection of estimates, and the single parameter being estimated. It also is the expected value of the error, since
$$\operatorname{E}(\widehat{\theta}) - \theta = \operatorname{E}(\widehat{ \theta } - \theta)$$
The estimator $\widehat{\theta}$ is an unbiased estimator of  $\theta$  if and only if $B(\widehat{ \theta }) = 0$.*example*: If the parameter is the bull's-eye of a target, and the arrows are estimates, then a relatively high absolute value for the bias means the average position of the arrows is off-target, and a relatively low absolute bias means the average position of the arrows is on target. They may be dispersed, or may be clustered.

### Variance(方差)

 The variance of $\widehat{\theta}$ is simply the expected value of the squared sampling deviations; that is, $$\operatorname{var}(\widehat{ \theta }) = \operatorname{E}[(\widehat{ \theta } - \operatorname{E}(\widehat{\theta}) )^2]$$. It is used to indicate how far, on average, the collection of estimates are from the expected value(期望) of the estimates.

#### example

If the parameter is the bull's-eye of a target, and the arrows are estimates, then a relatively high variance means the arrows are dispersed, and a relatively low variance means the arrows are clustered. Some things to note: even if the variance is low, the cluster of arrows may still be far off-target, and even if the variance is high, the diffuse collection of arrows may still be unbiased. Finally, note that even if all arrows grossly miss the target, if they nevertheless all hit the same point, the variance is zero.
The relationship between bias and variance is analogous to the relationship between accuracy and precision.

 note:the sample mean ${\overline{X}}=\frac{1}{N}\sum_{i=1}^{N}{X}_i$ is an unbiased estimator of $μ$,and the sample variance

$s^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X}\,)^2$ is an unbiased estimator of $σ^2$,(not the
$S^2=\frac{1}{n}\sum_{i=1}^n\left(X_i-\overline{X}\right)^2$,it's a baised estimator of $σ^2$,proof is [here][1])

### Mean squared error

In statistics, the mean squared error (MSE) of an estimator measures the average of the squares of the "errors", that is, the difference between the estimator and what is estimated.MSE is a risk function, corresponding to the expected value of the squared error loss or quadratic loss.(损失函数or代价函数？)
$\operatorname{MSE}(\hat{\theta})=\operatorname{Var}(\hat{\theta})+ \left(\operatorname{Bias}(\hat{\theta},\theta)\right)^2$
$=\operatorname{E}[(\widehat{\theta} - \operatorname{E}(\widehat{\theta}) )^2]+ {\left( \operatorname{E}(\widehat{\theta}) - \theta\right)}^2$
[proof][2]
ps:
In statistics, the bias (or bias function) of an estimator is the difference between this estimator's expected value and the true value of the parameter being estimated. An estimator or decision rule with zero bias is called unbiased. Otherwise the estimator is said to be biased.

$n = mq + r $
$a = b + c$
$b \equiv r_1 \pmod{9} $
$c \equiv r_2 \pmod{9} $
$a \equiv r_1 + r_2\pmod{9}$

  [1]: http://upload.wikimedia.org/math/5/5/7/5570b43693f45e8bba75ba5702a8fea5.png
  [2]: http://upload.wikimedia.org/math/b/0/5/b05d66203821b091ba1ea862fe8ee898.png
