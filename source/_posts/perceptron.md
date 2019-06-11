---
title: 感知器和逻辑回归的思考
mathjax: true
date: 2019-04-21 12:34:25
categories:
tags: perceptron, logistic regression
---

感知器的学习算法是误分类驱动的，更新规则如下
$$
w_{i,j} = w_{i,j} + \eta(\hat{y_j} - y_j)x_i
$$
看起来很简单。但是这个结果是怎么得到的呢？其实源自随机梯度下降法（stochastic gradient descent)。

根据《统计机器学习》中所述，具体的，假设所有样本集合为 $D = \{(x_i, y_i)| x_i \in train\ set\}$, **误分类样本**集合$M = \{(x_i, y_i)|x_i \in 误分类样本\}$

- 基于误分类驱动的

loss function: 
$$
L(w,b) = -\sum\limits_{x_i \in M} y_i(wx_i+b)
$$


最小化$L(w,b)$是优化目标，求导后：
$$
\triangledown_{w}L(w,b) = -\sum\limits_{x_i\in M}y_ix_i
$$

- 基于所有训练样本的

变得到如上答案。

这里要说明的是感知器的step function 是 {-1, 1}​ 的情况，当step function是{0, 1}的情况时，我们可以修改 loss function: $L(w,b) = -\sum\limits_{x\in D}[y_i(wx_i+b) + (1-y_i)(1-(wx_i+b))]​$

对比逻辑回归到loss function:
$$
J(\theta) = -\frac{1}{m}\sum\limits_{i=1}^{m}[y^{i}log(\hat{p}^{i})+(1-y^{i})log(1-\hat{p}^{i})]
$$


发现两者有相似之处，对其求导后：
$$
\frac{\partial}{\partial \theta_j} = \frac{1}{m}\sum\limits_{i=1}^{m}(\sigma(\theta^Tx^{i})-y^{i})x_j^{i}
$$
