---
layout: posts
title: Entropy
date: 2018-05-10 17:15:21
tags: 机器学习 数学
mathjax: true
---
## 一个例子

$H(y_i) = \sum_i y_ilog(\frac{1}{y_i}) = -\sum_iy_ilog(y_i)$

例子：现在有两枚硬币，抛出有四种情况，正正、正负、负正、负负。如果用熵来计算需要几维表示信息的话，计算如下：

- 2进制
  $H(y_i) = 4.({\frac{1}{4}.log_24}) = 2$，即00，01，10，11就可表示
- 4进制
  $H(y_i) = 4.({\frac{1}{4}.log_44}) = 1$，即1，2，3，4就可表示

[交叉熵代价函数][1]

<!-- more -->

## 交叉熵理论

交叉熵与熵相对，如同协方差与方差。

熵考察的是单个的信息（分布）的期望：


$H(p)=-\sum_{i=1}^n p(x_i)\log p(x_i)$
交叉熵考察的是两个的信息（分布）的期望： 

$H(p,q)=-\sum_{i=1}^np(x_i)\log q(x_i)$

详见 wiki Cross entropy
交叉熵代价函数


$L_H(\mathbf x,\mathbf z)=-\sum\limits_{k=1}^dx_k\log z_k+(1-x_k)\log(1-z_k)$

$x$ 表示原始信号，$z$ 表示重构信号，以向量形式表示长度均为 $d$，又可轻易地将其改造为向量内积的形式。

## 神经网络中的交叉熵代价函数

为神经网络引入交叉熵代价函数，是为了弥补 sigmoid 型函数(sigmoid, tanh ...)的导数形式易发生饱和（saturate，梯度更新的较慢）的缺陷。

首先来看平方误差函数（squared-loss function），对于一个神经元（单输入单输出），定义其代价函数： 

$C=\frac{1}{2}(a−y)^2$

其中$a = \sigma(z)$，$ z = wx+b$ ，然后计算对权值 $w$ 和偏置 $b$ 的偏导（为说明问题的需要，不妨将 $x=1$, $y=0$）： 

$\frac{\partial C}{\partial w}  = (a - y) \sigma'(z)x  =(a - 0) \sigma'(z) \cdot1 = a\sigma'(z)$

$\frac{\partial C}{\partial b} = (a-y)\sigma'(z) = (a-0)\sigma'(z) = a\sigma'(z)$

根据偏导计算权值和偏置的更新： 

$w = w -\eta \frac{C}{w} = w - \eta a \sigma'(z)$

$b = b - \eta \frac{C}{b} = w - \eta a \sigma'(z)$

无论如何简化，sigmoid 型函数的导数形式 $\sigma'(z)$ 始终存在，上文说了$\sigma(z)$ 较容易达到饱和，当输出处于饱和区时，$\sigma '(z) = 0$, 上面的参数将不会更新。这对整个训练过程来说会严重降低参数更新的效率。

为了解决参数更新效率下降这一问题，我们使用交叉熵代价函数替换传统的平方误差函数。

对于多输入单输出的神经元结构而言，如下图所示： 

![](http://ww1.sinaimg.cn/large/6bf0a364ly1g1h822vi52j20p7094dgb.jpg)
我们将其损失函数定义为： 

$C=-\frac1m \sum\limits_xy\ln a+(1-y)\ln(1-a)$

其中 $a=\sigma(z),\;z=\sum\limits_jw_jx_j+b$
最终求导得： 

$\frac{\partial\,C}{\partial\,w}=\frac1n\sum\limits_xx_j(\sigma(z)-y)\\ \frac{\partial\,C}{\partial\,b}=\frac1n\sum\limits_x(\sigma(z)-y)$
这样就避免了$\sigma'(z)$参与参数更新、影响更新效率的问题；



## Entropy[[2]]

### 1.什么是信息量？
假设$\mathcal{X}$是一个离散型随机变量，其取值集合为$D$，概率分布函数为 $p(x)=Pr(X=x),x \in \mathcal{X}$，我们定义事件$X=x_0$ 的信息量为： $I(x_0)=−log(p(x_0))$。

可以理解为，一个事件发生的概率越大，则它所携带的信息量就越小，而当 $p(x_0)=1$时，熵将等于$0$，也就是说该事件的发生不会导致任何信息量的增加。举个例子，小明平时不爱学习，考试经常不及格，而小王是个勤奋学习的好学生，经常得满分，所以我们可以做如下假设： 

- 事件A:小明考试及格:
	对应的概率$P(x=A)=0.1$，信息量为$I(x=A)=−log(0.1)=3.3219$ 
- 事件B:小王考试及格:
	对应的概率$P(x=B)=0.999$，信息量为$I(x=B)=−log(0.999)=0.0014$ 

可以看出，结果非常符合直观：小明及格的可能性很低(十次考试只有一次及格)，因此如果某次考试及格了（大家都会说：XXX竟然及格了！），必然会引入较大的信息量，对应的$I$值也较高。而对于小王而言，考试及格是大概率事件，在事件B发生前，大家普遍认为事件B的发生几乎是确定的，因此当某次考试小王及格这个事件发生时并不会引入太多的信息量，相应的$I$值也非常的低。

### 2.什么是熵？
那么什么又是熵呢？还是通过上边的例子来说明，假设小明的考试结果是一个0-1分布$D$只有两个取值$D = \lbrace x|0:不及格,1:及格 \rbrace$，在某次考试结果公布前，小明的考试结果有多大的不确定度呢？你肯定会说：十有八九不及格！因为根据先验知识，小明及格的概率仅有0.1,90%的可能都是不及格的。怎么来度量这个不确定度？求期望！不错，我们对所有可能结果带来的额外信息量求取均值（期望），其结果不就能够衡量出小明考试成绩的不确定度了吗。 
即： 
$H_A(x)=−[p(x_A)log(p(x_A))+(1−p(x_A))log(1−p(x_A))]=0.4690$
对应小王的熵： 
$H_B(x)=−[p(x_B)log(p(x_B))+(1−p(x_B))log(1−p(x_B))]=0.0114$ 
虽然小明考试结果的不确定性较低，毕竟十次有9次都不及格，但是也比不上小王（1000次考试只有一次才可能不及格，结果相当的确定） 
我们再假设一个成绩相对普通的学生小东，他及格的概率是$P(x_C)=0.5$,即及格与否的概率是一样的，对应的熵： 
$H_C(x)=−[p(x_C)log(p(x_C))+(1−p(x_C))log(1−p(x_C))]=1$
其熵为1，他的不确定性比前边两位同学要高很多，在成绩公布之前，很难准确猜测出他的考试结果。 
可以看出，熵其实是信息量的期望值，它是一个随机变量的确定性的度量。熵越大，变量的取值越不确定，反之就越确定。

对于一个随机变量X而言，它的所有可能取值的信息量的期望（$E[I(x)]$）就称为熵。 
$X$的熵定义为： 
$H(X)=E_p\log\frac1{p(x)}=−\sum\limits_{x\in X}p(x)\log p(x)$ 
如果$p(x)$是连续型随机变量的$pdf$，则熵定义为： 
$H(X)=-\int\limits_{x∈X}p(x)\log p(x)dx$ 
为了保证有效性，这里约定当$p(x)→0$时,有$p(x)\log p(x)→0$ 
当X为0-1分布时，熵与概率p的关系如下图： 

可以看出，当两种取值的可能性相等时，不确定度最大（此时没有任何先验知识），这个结论可以推广到多种取值的情况。在图中也可以看出，当$p=0$或1时，熵为0，即此时X完全确定。 
熵的单位随着公式中log运算的底数而变化，当底数为2时，单位为“比特”(bit)，底数为e时，单位为“奈特”。

### 3.什么是相对熵？
相对熵(relative entropy)又称为KL散度（Kullback-Leibler divergence），KL距离，是两个随机分布间距离的度量。记为DKL(p||q)DKL(p||q)。它度量当真实分布为p时，假设分布q的无效性。 
$$
\begin{equation}
\begin{aligned}
D_{KL}(p||q) &= E_p[\log \frac{p(x)}{q(x)}] \\\\
&=\sum\limits_{x∈\mathcal{X}} p(x)\log \frac{p(x)}{q(x)} \\\\
&=\sum\limits_{x∈\mathcal{X}} [p(x)\log p(x)-p(x)\log q(x)] \\\\
&=\sum\limits_{x∈\mathcal{X}} p(x)\log p(x)-\sum\limits_{x∈\mathcal{X}} p(x)\log q(x) \\\\
&=-H(p)-\sum\limits_{x∈\mathcal{X}} p(x)\log q(x) \\\\
&=-H(p)+E_p[-\log q(x)] \\\\
&=H_p(q)-H(p) 
\end{aligned}
\end{equation}
$$

并且为了保证连续性，做如下约定： 
$0\log \frac{0}{0}=0，0\log \frac{0}{q}=0，p\log \frac{p}{0}=∞$
显然，当$p=q$时,两者之间的相对熵$D_{KL}(p||q)=0$ 
上式最后的$Hp(q)$表示在$p$分布下，使用$q$进行编码需要的bit数，而$H(p)$表示对真实分布$p$所需要的最小编码bit数。基于此，相对熵的意义就很明确了：$D_{KL}(p||q)$表示在真实分布为$p$的前提下，使用$q$分布进行编码相对于使用真实分布$p$进行编码（即最优编码）所多出来的bit数。

### 4. 什么是交叉熵？
交叉熵容易跟相对熵搞混，二者联系紧密，但又有所区别。假设有两个分布p，qp，q，则它们在给定样本集上的交叉熵定义如下： 
$CEH(p,q)=E_p[-\log q]=-\sum\limits_{x∈\mathcal{X}} p(x)\log q(x)=H(p)+D_{KL}(p||q)$
可以看出，交叉熵与上一节定义的相对熵仅相差了$H(p)$,当pp已知时，可以把$H(p)$看做一个常数，此时交叉熵与$KL$距离在行为上是等价的，都反映了分布$p$，$q$的相似程度。最小化交叉熵等于最小化KL距离。它们都将在$p=q$时取得最小值$H(p)$（$p=q$时$KL$距离为$0$），因此有的工程文献中将最小化KL距离的方法称为Principle of Minimum Cross-Entropy (MCE)或Minxent方法。 
特别的，在logistic regression中， 
p:真实样本分布，服从参数为p的0-1分布，即$X \sim B(1,p)$ 
q:待估计的模型，服从参数为q的0-1分布，即$X \sim B(1,q)$ 
两者的交叉熵为：


$$
\begin{equation}
\begin{aligned}
CEH(p,q) &=-\sum\limits_{x\in\mathcal{X}} p(x)\log q(x) \\\\
&=-[P_p(x=1)\log P_q(x=1)+P_p(x=0)\log P_q(x=0)] \\\\
&=-[p\log q+(1-p)\log (1-q)] \\\\
&=-[y\log h_{\theta}(x)+(1-y)\log (1-h_(x))] \\\\ 
\end{aligned}
\end{equation}
$$

对所有训练样本取均值得： 

$$
-\frac{1}{m}\sum\limits_{i=1}^m[y^{(i)}\log h_{\theta}(x^{(i)})+(1-y^{(i)})\log (1-h_{\theta}(x^{(i)}))]
$$
这个结果与通过最大似然估计方法求出来的结果一致。

[1]: http://blog.csdn.net/lanchunhui/article/details/50970625
[2]: https://blog.csdn.net/rtygbwwwerr/article/details/50778098
