---
title: Perceptron
date: 2019-04-22 10:56:44
categories: machine learning
tags: perceptron
mathjax: true
---

![](http://ww1.sinaimg.cn/thumbnail/6bf0a364ly1g2bd89fz9lj20kf0e4q36.jpg)

感知器的分类函数为 $f(x) = sign(wx + b) $，在阅读书籍时发现有两种update rule：
$$
\triangle w = \eta y_ix_i \tag{1}
$$

$$
\triangle w = \eta(\hat{y_i}-y_i)x_i \tag{2}
$$

为什么会产生两种形式呢？

<!--more-->

- 第一种其实是误分类驱动的，对于错误集 $M =\lbrace x_i | f(x_i) \ne y_i \rbrace $， 其损失函数为：

$$
\triangledown_w L(w,b) = -\sum\limits_{x_i \in M} y_ix_i
$$

我们采用随机梯度下降法，**每次随机选取一个误分类点$(x_i, y_i)$** ,得到更新法则(1)。这种对错误分类结果进行修正的方法可以通过迭代解释。假设有误分类实例 $x_i$，更新前$w^{t}$，更新后$w^{t+1}$，迭代前后的函数关系为：


$$
\begin{equation}
\begin{aligned}  
f^{(t+1)}(x_i) &=  w^{(t+1)}x_i + b \\\\
&= (w^{i}+\eta y_ix_i)x_i+b \\\\
&= w^ix_i + \eta y_i||x_i||_2 + b \\\\
&= f^t(x_i) +  \eta y_i||x_i||_2 \\\\
&= f^t(x_i) + \eta y_i \delta
\end{aligned}
\end{equation}
$$



其中$\delta > 0$，现在分情况讨论：

1. $y_i = +1, f^t(x_i) = -1$

   修正后函数值向+1方向改变

2. $y_i = -1, f^t(x_i) = +1$

   修正后函数值向-1方向改变

   

- 第二种是MSE推理得到的，，也被称作*Hebb's rule*，考虑损失函数：
  $$
  L(w,b) = \frac{1}{m} \sum\limits_{i=1}^m (\hat{y_i} - y_i)^2
  $$
  

求导后使用**随机梯度下降法**更新便得到式子(2)

