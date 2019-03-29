---
title: bootstrap
date: 2019-03-16 16:21:22
categories:
tags: machine learning
---



一些机器学习方法对数据的变更比较敏感，Bagging是一种减少variace的方法（why?)。Bagging依赖于多次boostrap，然后用一种分类器在多个这样bootstrap生成的训练集上训练。将最终得到的多个分类器ensemble，从而对新数据进行投票方式预测。

**Bagging = (Bootstrap Aggregating)**

关于为什么Bagging方法有 0.368 是**oob**  (out of bagging)？

<!-- more -->

Bootstrap Sampling 过程伪代码：定义D为原数据集， D`为抽样的数据集：

initialize D' as an empty set

repeat N times:

​	random select an instance x from D

​	add x to D'

注意到所有抽样实例都是从原数据D集抽取，即 sampling with **replacement **（放回抽样）。因为是放回抽样，每次抽样时样本总体都是一样的，所以每个实例x没被抽中的概率时 P = (N-1)/N。因为时抽样了N次。

let $A$ = 实例 $x$ 在 $N$ 次实验中均没被抽中

$P(A) = (1-1/N)^N = 1/e  \approx 0.368$ .

$P(\overline{A}) = 1-0.368=0.632$



为什么Bagging 起作用：

* 分类器自身内在的"bias"，例如：其错误率 error rate 当测试集无穷大时保持不变，这是一钟算法or分类器本身的偏差bias。

* 在数据集有限的情况下，数据集本身的方差variance。

  分类器的错误率是上述两部分相加造成的



[1]. http://www.cs.bc.edu/~alvarez/ML/bagging.html