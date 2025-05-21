---

title:  HMM（Hidden Markov Model)
mathjax: true
date: 2019-07-19 09:28:13
categories: machine_learning
tags: HMM
---

隐马尔科夫模型的一个例子是输入发的提示功能。
![](http://ww1.sinaimg.cn/mw690/6bf0a364ly1g54w8od3o3j20gw02gdfw.jpg)

其实就是，观测序列越长，模型能得到的信息越多，自然推断的准确性就越高。除了推断隐藏序列，HMM还可用作预测，即给定一段观测序列，预测下一个隐藏序列是什么，拿输入法来说，这就是所谓的联想输入法。不仅如此，HMM还能进一步推断下一个甚至未来多个观测值是什么，只不过这种能力在卡尔曼滤波中应用的比较多，即目标跟踪。



## 马尔可夫模型

了解HMM首先要介绍一下markov model。已知N个有序随机变量，根据贝叶斯定理，他们的联合分布可以写成条件分布的连乘积：，
$$
p\left(x_{1}, x_{2}, \cdots, x_{N}\right)=\prod_{n=1}^{N} p\left(x_{n} | x_{n-1}, \cdots, x_{1}\right)		\tag{1}
$$
注意这只是markov model而非hidden markov model, marlov model是指符合markov特性的模型，markov特性假设序列中的任何一个随机变量在给定它的前一个变量时的分布与更早的变量无关：
$$
p\left(x_{n} | x_{n-1}, \cdots, x_{1}\right)=p\left(x_{n} | x_{n-1}\right)
$$
这样对于联合概率就可以简单处理，变成如下形式：
$$
p\left(x_{1}, x_{2}, \cdots, x_{N}\right)=p\left(x_{1}\right) \prod_{n=2}^{N} p\left(x_{n} | x_{n-1}\right)
$$
这是一阶markov模型的形式，一阶的意思就是当前状态只与之前状态相关。如果我们想将当前状态和更早之前的状态联系起来就需要高阶markov，比如说和前M个状态相关：
$$
p\left(x_{n} | x_{n-1}, \cdots, x_{1}\right)=p\left(x_{n} | x_{n-1}, \cdots, x_{n-M}\right)
$$
但是这样会有一个问题，参数会指数级增加，对于上面这一个M阶模型，如果$x_n$可以取$K$个观察值，其参数个数为：$K^M\cdot(k-1)$

$k-1$是指条件概率和为$\int_{x_n} p(x_n |...) = 1$，所以最后一个概率可由其他求得。

$K^M$ 是指条件概率的condition排列组合的可能个数。

对于这样一个指数级爆炸的问题，很显然是不好解决的。

所以重点来了！为了不割断和之前状态的联系，又想避免指数级参数问题，一个新的模型被提了出来：

![](http://ww1.sinaimg.cn/large/6bf0a364ly1g54yi60wrvj20k0069t8y.jpg)

该类模型的关键是隐藏变量之间满足如下条件独立性，即在给定$z_n$时，$z_{n-1}$和$z_{n+1}$ 条件独立 <条件独立参见附录> ：
$$
p(z_{n+1}|z_{n-1}, z_n) = p(z_{n+1}|z_n) \\
\mathbf{z}_{n+1} \perp \mathbf{z}_{n-1} | \mathbf{z}_{n}
$$

这样一来，对于(1)就可以化简为：
$$
p\left(\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{N}, \boldsymbol{z}_{1}, \cdots, \boldsymbol{z}_{N}\right)=p\left(\boldsymbol{z}_{1}\right)\left[\prod_{n=2}^{N} p\left(\boldsymbol{z}_{\boldsymbol{n}} | \boldsymbol{z}_{n-1}\right)\right]\left[\prod_{n=1}^{N} p\left(\boldsymbol{x}_{n} | \boldsymbol{z}_{\boldsymbol{n}}\right)\right]
$$

<!-- more -->

## 附录

### 条件独立[^2]

以下两个定义时等价的：
$$
\begin{aligned}
&p(a|b,c) = p(a|c) \\ 
&p(a,b|c) = p(a|c) \cdot p(b|c)
\end{aligned}
$$
只要将第二个式子转换一下，左边除右边第二项变更可得到第一个式子

![](http://ww1.sinaimg.cn/large/6bf0a364ly1g5501jzlsbj20m157gb29.jpg)