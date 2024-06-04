---
title: svm 随时感想
mathjax: true
date: 2019-07-16 09:28:31
categories:
tags: svm, machine learning

---



以前常会疑惑：
$$
\begin{equation}
w^T\cdot x + b = 0 \tag{1}
\end{equation}
$$

为什么他可以确定一条直线，以及为什么其作为分割面后 > 0 和 <0 就可以作为分类？

首先我们考虑如何确定一条直线，给定一个法向量$w$，会有无数个直线与其垂直正交，我们要的那一条如何唯一表示呢？其实很简单，找一个点就行，只需要这一个点，再加这一个法向量，一条直线就完全确定了。

假设  $a = (a_1, a_2，a_3)^T$ 是三维空间的一个点:
$$
w^T\cdot (x-a) = 0 \tag{2}
$$

可以确定一条直线，这是两个向量的乘积。$w$和$a$都是常量，所以展开后会生成一个常数项，即（1）式的 $b$， 最后形式就是 (1)。

值得注意的是：n维空间的一个分割超平面是n-1维的，减少了一维的降维打击。即3维立体空间：分割面为2维平面；2维平面：分割面为一维直线，1维直线：分个面为一个点。例如：$y = kx + b$是一维的，原因是$y$受$x$控制。$ax + by + cz = 0$是2维的是因为任选一维都是受另外两维控制，非自由的，这一点和线性代数的最大无关向量组很像。

接上面的说，对于式子（1），其确定一条直线，两边的点带入要么大于零，要么小于零，直观去想为什么呢？

其实很简单，对于给定点$a$， 所有**基于a的向量**，可以分为三类：

- 与$w^T$相乘等于零的，过a点且垂直与法向量$w$

- 与$w^T$相乘小于零的，过a点且与法向量$w$夹角小于90度的，比如说postive sample
- 与$w^T$相乘大于零的，过a点且与法向量$w$夹角大于90度的，比如说negative sample

这样就很直观明显了。

<!-- more -->

## Hinge Loss

在机器学习中，**hinge loss**作为一个损失函数(loss function)，通常被用于最大间隔算法。在SVM定义损失函数
$$
\ell(y) = \max(0, 1-y \cdot \hat y)
$$

定义这样的损失函数是因为svm算法有一个需求，**样本被分对且离分割面越远越好**。

对于分类正确且远离分割面的样本，我们希望损失贡献最小或为0；而对于分类器难以分辨的样本，我们希望其损失贡献最大。基于这些需求，我们采用了hinge loss function.

## Old Notes from zybuluo

&emsp;当通过最小化损失函数后学习到W，找到一个适合的超平面时，可能会出现无数个符合要求的W矩阵，即任何的$\lambda W$都代表这个平面，然而Loss function却不这样，他会随着$\lambda W$而称量级或倍数改变，例如：当$\lambda =2$ 时，假设一个“正确类”和一个离其最近的“非正确类”的距离$L$是15，由于所有$W$均乘以一个值为2的倍数，导致新距离$L$变为了30，但其实这两个$W$代表的分割面（或超平面）是一个东东。
可以在loss function中加入regularization penalty ->R(W)，以去除歧义.
$$L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2  \tag{1}$$
$R(W)$有几类，此处使用了$L2$ norm(应该是$L^2$ [norm][1])
 &emsp;上式中，对于多分类的SVM加入了一个regularization作为惩罚penalty.从我的理解来看，由于加入了regularization,任何$\lambda >1$的W都将被剔除，因为在 $L$ 函数中，这些同义的$W$会加大regularization因子。
 &emsp;*ps:*对于hyperparameter $\lambda$ 的取值需要使用cross-validation来确定。
 &emsp;The most appealing property is that penalizing large weights tends to improve generalization此段描述不明确？

    #向量化的lost function 
    def L_i_vectorized(x, y, W):
      """ 
      A faster half-vectorized implementation. half-vectorized
      refers to the fact that for a single example the implementation contains
      no for loops, but there is still one loop over the examples (outside this function)
      """ 
      delta = 1.0
      scores = W.dot(x)
      # compute the margins for all classes in one vector operation
      margins = np.maximum(0, scores - scores[y] + delta)
      # on y-th position scores[y] - scores[y] canceled and gave delta. We want
      # to ignore the y-th position and only consider margin on max wrong class
      margins[y] = 0 
      loss_i = np.sum(margins)
      return loss_i
 &emsp;"Additionally, making good predictions on the training set is equivalent to minimizing the loss."

## Practical Considerations

### delta的设置

&emsp;delta和 $\lambda$ 有些不同,delta的含义是 "the exact value of the margin between the scores." 在损失函数中，测试$\Delta = 1.0$ or $\Delta = 100.0$并没有太大意义，因为W是可以缩放的，这也导致在表达式中
$$L_i = \sum_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$$
&emsp;预测值scores的差异量 $w_j^T x_i - w_{y_i}^T x_i$是可以缩放到任意值的，$\lambda$只是确保了权重W最大可以扩展到什么量级。比如说，当 $\lambda=1$ 时，所有 $\lambda > 1$ 的 $\lambda W$ 取值都会被剔除，而当$\lambda < 1$是可以缩小差异量的，

### 和2分类的SVM

$$L_i = C \max(0, 1 - y_i w^Tx_i) + R(W) $$
&emsp;二分类的SVM可以看做是多分类SVM的特例,在这里 $y_i \in { -1,1 }$。上述式子是第i个example的Loss,描述一下，例如，当 $x_i$ 被正确分类是，$y_i$和 $x_i$ 同号，相乘后 $y_i w^Tx_i$ 为正数，从而max()第二项是小于1且应该趋向于零的。在$L_i$中，$C$与(1)中的 $\lambda$ 成反比。

[1]: http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Euclidean_norm
