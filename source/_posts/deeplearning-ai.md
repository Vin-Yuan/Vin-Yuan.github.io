---
title: deeplearning.ai
date: 2019-03-08 18:46:52
categories: machine-learning
tags:
---
## [Locally Connected Networks][1]

&emsp;However, with larger images (e.g., 96x96 images) learning features that span the entire image (fully connected networks) is very computationally expensive--you would have about $10^4$ input units, and assuming you want to learn `100 features`, you would have on the order of $10^6$ parameters to learn.
&emsp;One simple solution to this problem is to restrict the connections between the hidden units and the input units, allowing each hidden unit to connect to only a small subset of the input units. Specifically, each hidden unit will connect to only a small contiguous region of pixels in the input. (For input modalities different than images, there is often also a natural way to select "contiguous groups" of input units to connect to a single hidden unit as well; for example, `for audio`, a hidden unit might be connected to only the input units corresponding to a certain time span of the input audio clip.)
&emsp;This idea of having locally connected networks also draws inspiration from how the early visual system is wired up in biology. Specifically, neurons in the visual cortex have localized `receptive fields` (i.e., they `respond` only to `stimuli` in a certain location).

<!-- more -->

[Autoencoders and Sparsity][2]
 对于sparsity constraint ，为什么说
 >We would like to constrain the neurons to be inactive most of the time

 激活神经元越少越好？另外查看**Kullback-Leibler (KL）**
![此处输入图片的描述][3]

## Softmax

![softmax][4]
图片来自：<https://zhuanlan.zhihu.com/p/46816007（最初来源为台大李毅宏教程）>

## Batch Normalization

参考：<https://zhuanlan.zhihu.com/p/34879333>
一方面，当底层网络中参数发生微弱变化时，由于每一层中的**线性变换**与**非线性激活**映射，这**些微弱变化随着网络层数的加深而被放大**（类似蝴蝶效应）；另一方面，参数的变化导致每一层的输入**分布会发生改变**，进而上层的网络需要不停地去**适应这些分布变化**，使得我们的模型训练变得困难。上述这一现象叫做**Internal Covariate Shift**。

$$
Z^{[l]}=W^{[l]} \times i n p u t+b^{[l]} \\
A^{[l]}=g^{[l]}\left(Z^{[l]}\right)
$$

其中$l$代表层数，$g^{[l]}(\cdot)$代表非线性激活函数(sigmoid, relu)
公式] 代表层数；非线性变换为为第$l$ 层的激活函数。
随着梯度下降的进行，每一层的参数 $W^{[l]}$ 与 $b^{[l]}$ 都会被更新，那么 $Z^{[l]}$ 的分布也就发生了改 变，进而 $A^{[l]}$ 也同样出现分布的改变。而 $A^{[l]}$ 作为第 $l+1$ 层的输入，意味着 $l+1$ 层就需要 去不停适应这种数据分布的变化，这一过程就被叫做**Internal Covariate Shift**。
下面的图对于理解batch Normalization比较有帮助：
![此处输入图片的描述][5]
经过对每一个特征（神经元做BN候，得到如下结果）
![此处输入图片的描述][6]
通过上面的变换使得第 $l$ 层的输入每个特征的分布均值为0，方差为1。

BN的问题：
让每一层网络的输入数据分布都变得稳定，但却导致了**数据表达能力**的缺失。也就是我们通过变换操作**改变了原有数据的信息表达**（representation ability of the network），使得底层网络学习到的参数信息丢失。另一方面，通过让每一层的输入分布均值为0，方差为1，会使得输入在经过sigmoid或tanh激活函数时，容易陷入非线性激活函数的**线性区域**。(这句理解是说原本抑制的神经元被shift到了线性区域？）

应对这个问题加入两个可学习参数$\gamma$ 与 $\beta$，这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换，即$\tilde{Z}_{j}=\gamma_{j} \hat{Z}_{j}+\beta_{j}$。特别当：$\gamma^{2}=(\sigma+\epsilon)^{2}, \beta=\mu$时，起到恒等变换（identity transform)的作用
注：**这一种思想很类似resnet的残差网络和highway思想，寻求一些学习到的参数，当网络权重不想改变、维持原样的时候，至少有种方式使其恢复**）

更详细的参照：
<https://zhuanlan.zhihu.com/p/33173246>

## L1 和 L2 损失函数的比较

<http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/>

  [1]: http://ufldl.stanford.edu/wiki/index.php/Feature_extraction_using_convolution
  [2]: http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
  [3]: http://ufldl.stanford.edu/wiki/images/thumb/4/48/KLPenaltyExample.png/400px-KLPenaltyExample.png
  [4]: https://pic1.zhimg.com/80/v2-48456d984ea5741cb25a36acb0093a74_1440w.jpg
  [5]: https://pic1.zhimg.com/80/v2-084e9875d10896369e09af5a60e56250_1440w.jpg
  [6]: https://pic3.zhimg.com/80/v2-c37bda8f138402cc7c3dd62c509d36f6_1440w.jpg

**optimization:**
Optimization is the process of finding the set of parameters W that minimize the loss function.
>foreshadowing
3部分：score function,loss function,optimization
对于第一部分，后面会从线性分类器扩展到神经网络再到CNN，但是Loss function和和这节讲的optimization会保持不变

stepsize also called **learning rate**,是很重要的hyperparameter

## 计算梯度

[其他参考资料][2]

- numerical gradient：

$$\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}$$
是一种近似的方式计算梯度

- analytic gradient（ with Calculus ）
说明白点就是直接计算导数，Numercial gradient 计算的代价较大，而caculus虽然快，但可能在实现implement时出错（就是coding 错了），所以通常都做法是使用analytic gradient计算梯度，并用numercial gradient检查自己的implement是否正确，这种结合的方式通常叫做：<font color="red">gradient check</font>

## analytic gradient 举例

Example:
例如SVM的损失函数:
$$L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + \Delta) \right]$$
可如下计算其gradient（![gradient](#800080)是一个向量）:
> We are given some function $f(x)$ where $x$ is a **vector** of inputs and we are interested in computing the gradient of f at $x$ (i.e. $∇f(x)$).

- 目标类别所对应的的$w$,即目标类别对应于矩阵$W$的行$w_{y_i}$
$$\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i$$
说明：样本（$x_i,y_i$)输入预测函数$h_{\theta}$后的预测值是一个向量，每一维维对应类别的score,这些scores $(Wx_i)_j ,j\neq y_i$ 与目标类的间隔大于$\Delta$的，对cost function没有contribute,而小于$\Delta$的，才增加了cost function。这里通过这种方式，计算对应于目标类的参数$w$的梯度是多少，方式是统计有贡献的类的个数，然后按此比例扩大特征向量$x_i$，进而对其参数 $w_{y_i}，w_{y_i}\in R^{n+1}$做更新,从上面与下面两个梯度的式子符号的区别可以看出，对于$w_{y_i}$是要做增加，而对$w_{j\neq y_i}$要做的是减小，这样正负类的间隔才会最大化。
- 非目标类别对应的$w$参数,即非目标类别对应于矩阵$W$的行$w_j ,w_j\in R^{n+1}$  ，$j\neq y_i$.
$$\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i$$
非目标类对应w参数的梯度，则是看其预测下score值是否符合大于间隔$\Delta$这一目标，符合则不更新，不符合，则按$x_i$大小调整其对应参数$w_j$

```python
while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
```

```python
while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```

The reason this works well is that the examples in the training data are correlated
The extreme case of this is a setting where the mini-batch contains only a **single** example. This process is called Stochastic Gradient Descent (SGD) (or also sometimes on-line gradient descent)

mini-batch GD:
$${ {\theta}_j}^{'} = {\theta}_j + \frac1m\sum_{i=1}^m(y^i - h_{\theta}(x_i))x_j^i$$
Stochastic GD:
$${ {\theta}_j}^{'} = \theta_j + (y^i - h_\theta(x^i))x_j^i$$

>与批量梯度下降对比，随机梯度下降求解的会是最优解吗？
    （1）批量梯度下降---最小化所有训练样本的损失函数，使得最终求解的是全局的最优解，即求解的参数是使得风险函数最小。
    （2）随机梯度下降---最小化每条样本的损失函数，虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。[here][3]
    ps:随机梯度是增量性update:假设我们已经在数据库A上训练好一个分类器h了，那新来一个样本x。对非增量学习算法来说，我们需要把x和数据库A混在一起，组成新的数据库B，再重新训练新的分类器。但对增量学习算法，我们只需要用新样本x来更新已有分类器h的参数即可），所以它属于在线学习算法。[here][4]

  [4]: <http://blog.csdn.net/zouxy09/article/details/20319673> 2015-07-15 19:23:10
vin Introduction to Neural Networks, backpropagation #Introduction to Neural Networks, backpropagation

偏导数（微分）的含义：
$$\frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}$$
>The derivative on each variable tells you the sensitivity of the whole expression on its value

将微分定义转变一下后，可以这样理解，$f(x + h) = f(x) + h \frac{df(x)}{dx}$,即整个函数在变量$x$上施加一个微小的变化$h$，对于整个函数值的影响（增加多少或减少多少），对应灵敏度sensitivity。

$$f(x,y) = \max(x, y) \hspace{0.1in} \rightarrow \hspace{0.1in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.3in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)$$

## backprogagation

```
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
dfdy = 1.0 * dfdq # dq/dy = 1
```

每一个gate分两部分：output gradient和local gradient,例如$f(x) = (x+y)*z$,令$q = x + y$,则对于add gate其output gradient是$z = \frac{df}{dq}$,而local gradient是（$\frac {dq}{dx}$ or $\frac {dq}{dy}$）.

## Unintuitive effects and their consequences

 Note that in linear classifiers where the weights are dot producted $w^Tx_i$ (multiplied) with the inputs, this implies that the scale of the data has an effect on the magnitude of the gradient for the weights.

# Neural network

## part 1

![3-layer neural network][1]
>Notice that when we say N-layer neural network, we do not count the input layer
>Notice also that instead of having a single input column vector, the variable x could hold an entire **batch** of training data (where each input example would be a column of x) and then all examples would be efficiently evaluated in **parallel**.
>The forward pass of a fully-connected layer corresponds to one matrix multiplication followed by a bias **offset** and an activation function.
> given any **continuous function f(x)** and some ϵ>0, there exists a Neural Network **g(x)** with **one hidden layer** (with a reasonable choice of non-linearity, e.g. sigmoid) such that **∀x,∣f(x)−g(x)∣<ϵ.** In other words, the neural network can **approximate any continuous function**.
>Similarly, the fact that deeper networks (with multiple hidden layers) can work better than a single-hidden-layer networks is an **empirical observation**, **despite** the fact that their representational power is **equal**.
>there are many other preferred ways to prevent overfitting in Neural Networks that we will discuss later (such as L2 regularization, dropout, input noise).
>The takeaway is that you should not be using smaller networks because you are afraid of overfitting. Instead, you should use as big of a neural network as your computational budget allows, and use other regularization techniques to control overfitting.

## part 2

In the previous section we introduced a model of a Neuron, which computes a dot product **following a non-linearity**, and Neural Networks that arrange neurons into layers. Together, these choices define the new form of the **score function**, which we have extended from the simple linear mapping that we have seen in the Linear Classification section. In particular, a Neural Network performs a sequence of linear mappings with interwoven **non-linearities**.
![Common data preprocessing pipeline][2]

### Mean subtraction

It involves subtracting the mean across **every individual feature** in the data, and has the geometric interpretation of centering the cloud of data around the origin along every dimension.

### Normalization

It only makes sense to apply this preprocessing if you have a reason to believe that different input **features** have different **scales** (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.

### PCA and Whitening

**variance**:$Var(X) = Σ ( Xi - X )^2 / N = Σ x_i^2 / N$
**covariance**:$Cov(X, Y) = Σ ( X_i - X ) ( Y_i - Y ) / N = Σ x_iy_i / N$
[**Variance-Covariance Matrix][3]**:
$V=\left[\begin{matrix}
\sum x_1^2/N&\sum x_1x_2/N&\cdots&\sum x_1x_c/N\\
\sum x_{21}/N&\sum x_2^2/N&\cdots&\sum x_2x_c/N\\
\cdots&\cdots&\cdots&\cdots&\\
\sum x_{c1}/N&\sum x_cx_2/N&\cdots&\sum x_c^2/N\\
\end{matrix}\right]
$
创建一个协方差矩阵：

1. $x = X - 11^TX(\frac1N)$
2. $V = x^Tx(\frac1N)$
协方差矩阵特点：对角线元素为第i组数据的放方差（variance)，非对角线元素为不同组数据（数据集）之间的协方差(covariance)。

PCA,Whited,...

### Common pitfall
>
>Instead, the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).
>
## Regularization
>
>The L2 regularization has the intuitive interpretation of heavily **penalizing peaky weight vectors** and **preferring diffuse weight vectors**
>
## part3

### Gradient check

#### Use **relative error** for the comparison

$$relative\_error = \frac{\mid f'_a - f'_n \mid}{\mid f'_a \mid + \mid f'_n \mid}$$
解释：如果两函数梯度都大约为1时，绝对误差1e-4是很小的误差，这时候看起来合适，但如果两个导数大约为1e-5时，这时1e-4的绝对误差就很大了，所以才用相对误差的方式更为合理一些。在实际中，误差准则如下：

- relative error > 1e-2 usually means the gradient is probably wrong
- 1e-2 > relative error > 1e-4 should make you feel uncomfortable
- 1e-4 > relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high.
- 1e-7 and less you should be happy.
当网络层次越深是，相对误差就越大，所以，1e-2的误对于单独的可微分函数来说已经很大，但在多层网络时是可以认可的误差。

### Before learning: sanity checks Tips/Tricks

1. Look for correct loss at chance performance
“Make sure you're getting the loss you expect when you initialize with small parameters.”
初始参很小的情况下，输出为10个类的softmax预期的loss 应该为2.302。因为预期是diffused的分布，每个类的Probility应是$0.1 = 1 / 10 $，而$-log(0.1) = 2.302$
SVM则应该9，因为在此种情况下所有的margins都是违反的,根据：$L_i = \sum_{j\neq y_i} \max(0, f(x_i, W)_j - f(x_i, W)_{y_i} + \Delta)$可知当每个类的margin为1时，loss为9。
2. As a second sanity check, increasing the regularization strength should
3. increase the loss Overfit a tiny subset of data.

### Babysitting the learning process

在训练网络时需要关注的特性

#### 1. loss function(学习曲线等）

![此处输入图片的描述][4]
&emsp;学习曲线通常x-轴代表epoch周期，epoch表示每个example第几次迭代入网络，之所以不用迭代次数作为横轴，是因为数据大多是批量batch进入系统地，比如样本$x_i$第一次输入系统到第二次输入系统中间可能已经经历了好几次迭代（e.g.SGD)。
学习曲线的震荡与batch size有很大关系，试想当batch size设置为1，即SGD时，学习曲线会波动很大，但若batch size是所有数据集是，学习曲线回事单调递减的，而非上下震荡。

#### 2.  Train/Val accuracy

![此处输入图片的描述][5]
&emsp;training accuracy和validation accuracy差别很大时，例如蓝线和红线，意味着Overfiting很严重，应该适当增减regularization(e.g.L2 weight penalty, more dropout, etc.)或增加数据量。然而当training accuracy和validation accuracy轨迹非常匹配时也不好，那意味着你的model capacity不够高，应试着增加更多参数来扩大模型。

#### 3. Ratio of weights:updates

统计和观察“更新”的变化情况。
此处需要强调的是Update指的是w更新的部分，而非raw gradient，例如SGD中乘了learning rate的部分。比率是指$\frac{\Delta W}W $。
![此处输入图片的描述][6]
上图中选取了变量max(W1)和min(W1),随着训练进行，这两个统计量在不断扩张直到平缓。值域范围大约0.02。
![此处输入图片的描述][7]
上图则描绘了对应变量（max(W1),min(W1))update的变化情况，每一次更新的量大约为0.0002.则更新的量级为$\frac{\Delta W}W = \frac{0.0002} {0.02} = 1e-2$

#### 4.Activation / Gradient distributions per layer

不正确的初始化会减缓甚至阻碍学习过程，解决的方法很简答，就是统计每一层计激活神经元或梯度的方差variance，好的初始化会使得方差较一致，而非随着层数上升而减为零。

#### 5.First-layer Visualizations

将第一层的权值可视化来观察是否正常

### Parameter updates

#### 1.SGD and bells and whistles

1.Vanilla update.
`x += - learning_rate * dx`
2.Momentum update

```python
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

3.Nesterov Momentum

```python
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```

<font color='orange'>$\Delta w_{ij}(t) = \mu_i\sigma_iy_j + m\Delta w_{ij}(t-1)$[资料查看][8]</font>
[资料二][9]

#### 2.Annealing the learning rate

随着训练的进行，适当的减少学习率是有必要的，主要分以下几种:

- Step decay:每5 epochs减一半或每20 epochs减0.1，这些都取决于问题类型和模型，一个启发式的方式是观察validation errors(初始固定学习率),当validation errors不再改进时再减少学习率（例如：0.5）
- Exponential decay:$\alpha = \alpha_0 e^{-k t}$
- 1/t decay:$\alpha = \alpha_0 e^{-k t}$
实际中，step decay用到较多一些

#### 3.Second order methods

牛顿法:
$$x \leftarrow x - [H f(x)]^{-1} \nabla f(x)$$
求hessian矩阵的逆代价较大，所以不好实行，例如一个百万参数的神经网络的会有一个[1,000,000 x 1,000,000]的Hessian矩阵，计算逆时需要内存3725GB。目前有种近似方式，即 L-BFGS 来求解近似的Hessian矩阵的逆。但是L-BFGS需要使用所有的训练样例，目前以批量的数据方式计算还处于研究阶段。

  [7]: http://cs231n.github.io/assets/nn3/updates.jpeg
  [8]: http://www.willamette.edu/~gorr/classes/cs449/momrate.html
  [9]: <http://www.willamette.edu/~gorr/classes/cs449/Momentum/momentum.html> 2015-07-17 19:51:24
vin Convolutional Neural Networks # Convolutional Neural Networks

标签（空格分隔）： CNN

---

[TOC]

## FAST R-CNN

Two inputs
• Conv feature map: 512 × ? × ?
(512&?&?: ???? ???? ????? ????)
• RoI: ? × 5
• 5 from ?, ?, ?, ℎ, ?
• ? ∈ 0, ? − 1 : image batch index
• Adaptive max pooling
• Pooled to fixed size feature vector

## ResNet

[参考资料](https://www.cnblogs.com/xiaoboge/p/10539884.html)
考虑一个训练好的网络结构，如果加深层数的时候，不是单纯的堆叠更多的层，而是堆上去一层使得堆叠后的输出和堆叠前的输出相同，也就是恒等映射/单位映射（identity mapping），然后再继续训练。这种情况下，按理说训练得到的结果不应该更差，因为在训练开始之前已经将加层之前的水平作为初始了，然而实验结果结果表明在网络层数达到一定的深度之后，结果会变差，这就是退化问题。**这里至少说明传统的多层网络结构的非线性表达很难去表示恒等映射**。

残差这里是指$H(x) + x = \bar{x}$，$H(x)$是残差块，如果$x=\bar{x}$，则残差为零，如果不等，学习到的就是这部分**残差**，相当于$\Delta$，这也是“残差”网络的含义。

Resnet创新的点是用残差学习来解决深层网络的退化问题。

对于一个堆积层结构（几层堆积而成）当输入为$x$时其学习到的特征记为$H(x)$，现在我们希望其可以学习到残差$F(x) = H(x) - x$，这样其实原始的学习特征是$H(x)$。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为$F(x) = 0$ 时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层**在输入特征基础上学习到新的特征**，从而拥有更好的性能。

为什么残差学习相对更容易，从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。

## Activation Function

![此处输入图片的描述][1]

## Receptive Field 感受野计算

参照：<https://zhuanlan.zhihu.com/p/35708466>

![此处输入图片的描述][2]

从上面的图可以看出，卷积的过程可以看作a) 金字塔递减的图层结构，也可以看作 b) 保持尺寸的感受野图层结构。通过第二种方式，我们可以衡量一个特征（这里特征即是feature map上的一个神经元，其负责response一片区域Path上是否包含某种特征，有这种特征则neural activation会被激活）所能"看到"的区域，这个看到的区域是对于原始输入图片而言的，我么看可以理解为能看到多少像素单位。通过种方式我们发现，相邻特征之间是有一个距离的，这个距离是多少？
在这里要说明距离的概念：

1. **特征图上的距离**：特征图上的相邻特征距离是零，因为他们是紧密相连的，无间隙的。其它特征则直接数相隔特征个数即可
2. **映射到输入图像上的距离**：由于特征映射到原图上是一片区域，区域和区域之间还有可能有overlap，所以这是计算距离是计算中心点的距离，即这个特征在原图上会标记为一个点，即其所映射的感受野的中心坐标。

这个距离就是：
$$
\prod_{i=1}^{k-1} s_i
$$
其中$s_i$是第i层的stride，是指一个特征跳到另一个特征走的步数（看出是距离+1）。对于上面的等尺度视野图，我们也可以这样理解，如果每次前进一步稠密的计算卷积，则得到结果中的"白色"特征是我们跳过的地方，跳过几个地方呢，则正好是stride-1。
从上面的狮子可以看出，经历过一些列卷积后，这个距离是呈指数增长的。

基于上面的推算，不难理解感受野的递推公式：
$$
l_k = l_{k-1} + (f_k-1)\prod_{i=1}^{k-1} s_i
$$
和式可以看作两部分，基于上一层的**被卷积特征**：

1. 第一部分是**左上角特征**的感受野size
2. 第二部分是**右上角特征**和**左上角特征**的**感受野距离**
  [1]: <https://www.researchgate.net/profile/Vivienne_Sze/publication/315667264/figure/fig3/AS:669951052496900@1536740186369/Various-forms-of-non-linear-activation-functions-Figure-adopted-from-Caffe-Tutorial.png>
  [2]: <https://pic2.zhimg.com/80/v2-bf2e71cf68c1f1af5921087c3f928781_1440w.jpg> 2015-08-01 16:31:17
vin mean substractino & PCA # mean substractino & PCA

标签（空格分隔）： mean_substraction PCA

---

<https://chrisjmccormick.wordpress.com/2014/06/03/deep-learning-tutorial-pca-and-whitening/>
<http://stattrek.com/matrix-algebra/covariance-matrix.aspx>
<https://www.math.hmc.edu/calculus/tutorials/eigenstuff/>
<http://ufldl.stanford.edu/wiki/index.php/PCA#PCA_on_Images>
>Concretely, if $\textstyle x^{(i)}$ $\in$ $\Re^{n}$ are the (grayscale) intensity values of a 16x16 image patch ($\textstyle n=256$), we might normalize the intensity of each image $\textstyle x^{(i)}$ as follows:
$\mu^{(i)} := \frac{1}{n} \sum_{j=1}^n x^{(i)}_j $
$x^{(i)}_j := x^{(i)}_j - \mu^{(i)}$, for all $\textstyle j$

$2^{k-1}$

```

class Point
{
　　public:
　　　　Point(double xx, double yy) { x=xx; y=yy; }
　　　　void Getxy();
　　　　friend double Distance(Point &a, Point &b);
　　private:
　　　　double x, y;
};

void Point::Getxy()
{
　　cout<<"("<<<","<<<")"<< FONT>
}

double Distance(Point &a, Point &b)
{
　　double dx = a.x - b.x;
　　double dy = a.y - b.y;
　　return sqrt(dx*dx+dy*dy);
}

void main()
{
　　Point p1(3.0, 4.0), p2(6.0, 8.0);
　　p1.Getxy();
　　p2.Getxy();
　　double d = Distance(p1, p2);
　　cout<<"Distance is"<<< FONT>
}
```

 2015-08-13 11:32:04
vin scrapy 札记 "# scrapy 札记

标签（空格分隔）： scrapy  

---

## 1.[Response object][1]
>
>**Parameters**:
**url** (string) – the URL of this response
**headers** (dict) – the headers of this response. The dict values can be strings (for single valued headers) or lists (for multi-valued headers).
**status** (integer) – the HTTP status of the response. Defaults to 200.
**body** (str) – the response body. It must be str, not unicode, unless you’re using a encoding-aware Response subclass, such as TextResponse.
**meta** (dict) – the initial values for the Response.meta attribute. If given, the dict will be shallow copied.
**flags** (list) – is a list containing the initial values for the Response.flags attribute. If given, the list will be shallow copied.

## 2. 在回调函数添加参数

  [1]: <http://doc.scrapy.org/en/1.0/topics/request-response.html#response-objects>" 2015-08-28 18:37:34
vin matlab札记 # matlab札记

## [],(),{}的含义和用法

>(): locate the element, function call
1 a=[1 2]; a(1)
2 sin(2)

>[ ]: construct matrix and combine several strings
1 a = [1 2];
2 s = ['I love ' 'Matlab']

>{}: mixed element
a{1} = [1 2]
a{2} = [1 2 3]
a{3} = 'I love matlab'

 2015-10-17 11:54:59
vin Probability Theory # Probability Theory

标签（空格分隔）： math

---
[TOC]

## 1.Probability mass functions (pmf) and Probability density functions (pdf)

pmf 和 pdf 类似，但不同之处在于所适用的分布类型

PMF's are for <font color='green'>discrete distributions</font>, while pdf's are for <font color='green'>continuous distributions</font>

例如：

pmf: if P(X=5) = 0.2,则随机变量等于5时的概率是0.2
（pmf非负且sum等于1）

但是pdf就不能这么说了，因为pdf定义在point上，而他的Probability却定义在积分上，即：$$\int_A^Bf(x)dx \quad \textrm{and} \quad X\in [A,B] $$
若$A=B$则积分为0，给定点的"概率"永远是0。
因为我们只要确保积分后的结果是合法的概率值就可以，所以pdf可以大于1（离散分布就不可以了），但是pdf必须非负且积分区间是$(-\infty, +\infty)$

## 2.Cumulative Distribution Funtions

CDF是累积分布函数
$F(c) = P(X<c);\textrm{F is the CDF}$
离散分布：
$F(c) = \sum_{-\infty}^c p(c)$
连续分布：
$F(c) = \int_{-\infty}^c f(x)dx$

总结：PMF and PDF are almost the same, but one is for discrete distributions and one is for continuous distributions. CDF are different, but are the sum/integral of PMF/PDF and tell us **the probability that X is less than a certain value**.

## 3. Quantile (四分位)

它由五个数值点组成：最小值(min)，下四分位数(Q1)，中位数(median)，上四分位数(Q3)，最大值(max)。也可以往盒图里面加入平均值(mean)。
可以理解将数据从小到大排列好，然后等分位4分，这样在三个分割点就变成`Q1`（25%）, `Q2`（50%）, `Q3`（75%）三个分位。
密切相关的是箱线图（Box)
参照：<https://www.cnblogs.com/space-place/p/7643480.html>

## 4 Logit vs Logistic 2015-12-31 10:06:07

vin Generative and Discriminative Learning algorithms # Generative and Discriminative Learning algorithms

标签（空格分隔）： machine_learning

---

#### talked before

<font color='#728C00'>generative algorithm</font> models how the data was generated in order to categorize a signal. It asks the question: based on my generation assumptions, which category is most likely to generate this signal?

<font color='#FFA62F'>discriminative algorithm</font> does not care about how the data was generated, it simply categorizes a given signal.

#### now begin

**discriminative**试图找到class之间的差异，进而找到decision boundary,最大可能性的区分数据。他是通过直接**学习**到$p(y|x)$(*例如Logistic regress*)或者$X \rightarrow Y\in (0,1,...,k)$(*例如perceptron algrithm*)。
而**generative**采取另外一种方式，首先由先验知识prori-knowledge得到 $p(x|y),p(y)$ 然后，通过Bayes rule：$$p(y|x) = \frac{p(x|y)p(y)}{p(x)} \qquad (p(x)=p(x|y=1)p(y=1)+p(x|y=0)p(y=0))$$来求得$p(y|x)$。这个过程可以看做有先验分布去derive后验分布。当然，在只需要判断出可能性大小的情况下，分母无需考虑，即：$$\arg\max_yp(y|x) = \arg \max_y\frac{p(x|y)p(y)}{p(x)}\\=\arg\max_yp(x|y)p(y)$$
先验知识获取$p(x|y)和p(y)$的方式，是通过现有训练数据样本获得参数的过程。

 1. 首先假设一个模型，即样本分布的模型（是伯努利还是高斯分布）
 2. 然后通过似然估计likelihood function估计出参数
 3. 最后通过贝叶斯公式导出$p(y|x)$

#### example

数据集:$X=(x_1,x_2)$,$Y\in{0,1}$
将数据可视化后如下图：
![此处输入图片的描述][1]

 1. 首先我们假设数据的<font color='#F87217'>条件分布</font>$p(x|y)$服从多元高斯正态分布（multivariate normal distribution）,则model形式如下：$$y\sim \textrm{Bernoulli}(\phi) \\ x|y=0 \sim \mathcal{N}(\mu_0,\Sigma) \\ x|y = 1\sim \mathcal{N}(\mu_1,\Sigma )$$
 2. 接着通过<font color='#F87217'>最大似然估计（max likelihood estimate）</font>估计参数。首先写出log似然函数：$$\ell(\phi,\mu_0,\mu_1,\Sigma) = log\prod_{i=1}^{m}p(x^{(i)},y^{(i)},\mu_0,\mu_1,\Sigma) \\ =log\prod_{i=1}^mp(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi).$$
然后似然函数$\ell$最大化,即求解似然函数对参数导数为零的点：$$\phi=\frac{1}{m}\sum_{i=1}^{m}1\{y^{(i)}=1\} \\ \mu_0= \frac{\sum_{i=1}^{m}1\{y^{(i)}=0\}x^{(i)}} {\sum_{i=1}^m1\{y^{(i)}=0\}} \\ \mu_1= \frac{\sum_{i=1}^{m}1\{y^{(i)}=1\}x^{(i)}} {\sum_{i=1}^m1\{y^{(i)}=1\}} \\ \Sigma = \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T$$得到参数的估计值$(\phi,\mu_0,\mu_1,\Sigma)$，亦即得到分布函数$p(x|y)$。对照上面的图，$\mu_0,\mu_1$是两个二维向量，在图中的位置是两个正态分布各自的中心点，$\Sigma$则决定者多元正态分布的形状。
![此处输入图片的描述][2]
**从这一步可以看出获取参数的方式是“学习”得到的，即从大量样本-先验知识去估计模型，这样想是很自然的逻辑.然而严格的依据却是大数定律law of large numbers (LLN)**，大数定律的证明很精彩，可自行查找资料。
 3. 通过贝叶斯公式比较$p(y=1|x)$和$p(y=0|x)$,来判别类属性。

  $$\ell(\theta|x) = P(x|\theta)$$
  $$\ell(\theta|x) = p_{\theta}(x)=P_{\theta}(X=x)$$
  $\ell(\theta|x)=f_\theta(x)$ 2015-12-31 18:22:02
vin self-introduction # self-introduction

标签（空格分隔）： face2face_interview

---

" 2016-04-20 12:38:07
vin Cross Entropy # Cross Entropy

标签（空格分隔）： machine_learning

$H(y_i) = \sum_i y_ilog(\frac{1}{y_i}) = -\sum_iy_ilog(y_i)$
---

例子：现在有两枚硬币，抛出有四种情况，正正、正负、负正、负负。如果用熵来计算需要几为表示信息的话，计算如下：

- 2进制
$H(y_i) = 4.({\frac{1}{4}.log_24}) = 2$，即00，01，10，11就可表示
- 4进制
$H(y_i) = 4.({\frac{1}{4}.log_44}) = 1$，即1，2，3，4就可表示

[交叉熵代价函数][1]

交叉熵理论

交叉熵与熵相对，如同协方差与方差。

熵考察的是单个的信息（分布）的期望：

$H(p)=-\sum_{i=1}^n p(x_i)\log p(x_i)$
交叉熵考察的是两个的信息（分布）的期望：

$H(p,q)=-\sum_{i=1}^np(x_i)\log q(x_i)$

详见 wiki Cross entropy
交叉熵代价函数

$L_H(\mathbf x,\mathbf z)=-\sum_{k=1}^dx_k\log z_k+(1-x_k)\log(1-z_k)$

x 表示原始信号，z 表示重构信号，以向量形式表示长度均为 d，又可轻易地将其改造为向量内积的形式。
神经网络中的交叉熵代价函数

为神经网络引入交叉熵代价函数，是为了弥补 sigmoid 型函数的导数形式易发生饱和（saturate，梯度更新的较慢）的缺陷。

首先来看平方误差函数（squared-loss function），对于一个神经元（单输入单输出），定义其代价函数：

C=(a−y)22

其中 a=σ(z),z=wx+b，然后根据对权值（w）和偏置（b）的偏导（为说明问题的需要，不妨将 x=1,y=0）：
∂C∂w=(a−y)σ′(z)x=aσ′(z)∂C∂b=(a−y)σ′(z)=aσ′(z)
根据偏导计算权值和偏置的更新：

w=w−η∂C∂w=w−ηaσ′(z)b=b−η∂C∂b=b−ηaσ′(z)
无论如何简化，sigmoid 型函数的导数形式 σ′(z) 始终阴魂不散，上文说了 σ′(z) 较容易达到饱和，这会严重降低参数更新的效率。

为了解决参数更新效率下降这一问题，我们使用交叉熵代价函数替换传统的平方误差函数。

对于多输入单输出的神经元结构而言，如下图所示：

这里写图片描述
我们将其损失函数定义为：

C=−1n∑xylna+(1−y)ln(1−a)

其中 a=σ(z),z=∑jwjxj+b
最终求导得：

∂C∂w=1n∑xxj(σ(z)−y)∂C∂b=1n∑x(σ(z)−y)
就避免了 σ′(z) 参与参数更新、影响更新效率的问题；

  [1]: <http://blog.csdn.net/lanchunhui/article/details/50970625> 2016-06-07 10:29:14
vin 更换gcc/g++ # 更换gcc/g++

标签（空格分隔）： linux gcc

---

vin detection 札记 # detection 札记

标签（空格分隔）： detection

---

rcnn  fast-rcnn faster-rcnn

## Spatial Pyramid Pooling

<http://www.cnblogs.com/venus024/p/5590044.html>
<http://blog.csdn.net/myarrow/article/details/51878004>
<http://blog.csdn.net/liyaohhh/article/details/50614380>

## Region of interest (roi pooling map feature)

roi 映射到 feature map 的方式
<http://caffecn.cn/?/question/135&utm_source=tuicool&utm_medium=referral>
<http://www.caffecn.cn/?/article/25>

>在CNN网络中roi从原图映射到feature map中的计算方法
在使用fast rcnn以及faster rcnn做检测任务的时候，涉及到从图像的roi区域到feature map中roi的映射，然后再进行roi_pooling之类的操作。
比如图像的大小是（600,800），在经过一系列的卷积以及pooling操作之后在某一个层中得到的feature map大小是（38,50），那么在原图中roi是（30,40,200,400），
在feature map中对应的roi区域应该是
roi_start_w = round(30 *spatial_scale);
roi_start_h = round(40* spatial_scale);
roi_end_w = round(200 *spatial_scale);
roi_end_h = round(400* spatial_scale);
其中spatial_scale的计算方式是spatial_scale=round(38/600)=round(50/800)=0.0625，所以在feature map中的roi区域[roi_start_w,roi_start_h,roi_end_w,roi_end_h]=[2,3,13,25];
具体的代码可以参见caffe中roi_pooling_layer.cpp

## RPN

<http://blog.csdn.net/XZZPPP/article/details/51582810>
<http://closure11.com/rcnn-fast-rcnn-faster-rcnn%E7%9A%84%E4%B8%80%E4%BA%9B%E4%BA%8B/>
>上面说完了三种可能的训练方法，可非常神奇的是作者发布的源代码里却傲娇的用了另外一种叫做4-Step Alternating Training的方法，思路和迭代的Alternating training有点类似，但是细节有点差别（rbg大神这样介绍训练方式我也是醉了），具体来说：

>第一步：用ImageNet模型初始化，独立训练一个RPN网络；
第二步：仍然用ImageNet模型初始化，但是使用上一步RPN网络产生的proposal作为输入，训练一个Fast-RCNN网络，至此，两个网络每一层的参数完全不共享；
第三步：使用第二步的Fast-RCNN网络参数初始化一个新的RPN网络，但是把RPN、Fast-RCNN共享的那些卷积层的learning rate设置为0，也就是不更新，仅仅更新RPN特有的那些网络层，重新训练，此时，两个网络已经共享了所有公共的卷积层；
第四步：仍然固定共享的那些网络层，把Fast-RCNN特有的网络层也加入进来，形成一个unified network，继续训练，fine tune Fast-RCNN特有的网络层，此时，该网络已经实现我们设想的目标，即网络内部预测proposal并实现检测的功能。

faster rcnn 算法详解与实践
<http://blog.csdn.net/shenxiaolu1984/article/details/51036677>
<http://blog.csdn.net/Gavin__Zhou/article/details/52052915>
<http://blog.csdn.net/shenxiaolu1984/article/details/51152614>

Sumary:
线性分类器，score function，两种线性分类器（SVM,softmax ) 分别使用不同的loss function来衡量学习到的参数模型与真实的预测模型的差异
![此处输入图片的描述][1]
$$f(x_i, W, b) =  W x_i + b$$
上面的函数也称作score function
如上图所述，加入偏置b的目的是使得直线不通过原点，不然的话，对于$x_i=[0,0,...,0]$来说，$f(x_i) = 0$,即预测结果总是位于直线上，不利于分类，加入“截距”后，会使直线偏离原点。当然，bias也可以放入$W$,类似：
$$f(x_i, W) =  W x_i$$
只需将$b$以列向量附在$W$最后一列，并将$x_i$多加一维保持常数1即可。这样在只需学习一个矩阵($W$)而非两个矩阵（$W$ and $b$)就可以了。

## Interpretation of linear classifiers as template matching

<li>对于线性分类器的一个解释。$W$的每一行可以看作一个类的模板template(or prototype)。对于一张图片，其输入模型后再每一个类上的score是他和每一个template的inner product(or dot product)，借由此来寻找最合适的“类”。这样理解后，线性分类器所做工作就是在匹配模板template,当然template是通过学习获得的。</li>
<li>还有另一种解释，我们可以认为他仍在做Nearst Neighbor，但是不同于和成千上万的训练图片最近邻搜索，这一次，只是用一张“图片”（这张图片是通过学习获得的
，他不一定是训练集中的任何一张图片），并且使用的“distance”不再是$L1$或$L2$距离，而是inner product。</li>
![此处输入图片的描述][2]
> Skipping ahead a bit: Example learned weights at the end of learning for CIFAR-10. Note that, for example, the ship template contains a lot of blue pixels as expected. This template will therefore give a high score once it is matched against images of ships on the ocean with an inner product.

## Image data preprocessing

上面使用的都是像素值都是[0...255]，在实践中，通常会对输入的特征做normalization（in the case of images,every pixel is thought of as a feature).实际上，“集中”center你的数据很重要，比如从每个特征中减去均值mean.在图像中，相关工作是通过计算整个训练集来获取均值mean然后再对每一个图片减去mean，以获得每个像素取值范围为[-127,127]的图片集。更进一步的处理是scale每一个输入的feature以使得值范围为[-1,1]。

## Multiclass Support Vector Machine loss

$$L_i = \sum_{j\neq y_i} \max(0, f(x_i, W)_j - f(x_i, W)_{y_i} + \Delta)$$
&emsp;**Example**. This expression may seem daunting if you're seeing it for the first time, so lets unpack it with an example to see how it works. Suppose that we have three classes that receive the scores $f(xi,W)=[13,−7,11]$, and that the first class is the true class (i.e. $yi=0$). Also assume that Δ (a hyperparameter we will go into more detail about soon) is 10. The expression above sums over all incorrect classes ($j≠yi$), so we get two terms:

$$Li=max(0,−7−13+10)+max(0,11−13+10)$$
因为我们使用的score functions是($f(x_i; W) =  W x_i$),所以上面的loss function可以等价于如下式子：（意味着当score functions 不同是，可做替换？）

$$L_i = \sum_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$$
$w_j$ 在这里是**列向量** ($w_j$is the j-th row of $W$reshaped as a column.)
$max(0,-)$通常被称作hinge loss.更强烈的惩罚会使用$max(0,-)^2$

### Regularization Note

&emsp;当通过最小化损失函数后学习到W，找到一个适合的超平面时，可能会出现无数个符合要求的W矩阵，即任何的$\lambda W$都代表这个平面，然而Loss function却不这样，他会随着$\lambda W$而称量级或倍数改变，例如：当$\lambda =2$ 时，假设一个“正确类”和一个离其最近的“非正确类”的距离$L$是15，由于所有$W$均乘以一个值为2的倍数，导致新距离$L$变为了30，但其实这两个$W$代表的分割面（或超平面）是一个东东。
可以在loss function中加入regularization penalty ->R(W)，以去除歧义.
$$R(W) = \sum_k\sum_l W_{k,l}^2$$
这样全部的Multicalss Support Vector Machine loss分为两部分：
<li>data loss</li>
<li>reqularization loss</li>

$$L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \lambda R(W) }_\text{regularization loss} \\$$
$$L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2  \tag{1}$$
$R(W)$有几类，此处使用了$L2$ norm(应该是$L^2$ [norm][3])

 &emsp;上式中，对于多分类的SVM加入了一个regularization作为惩罚penalty.从我的理解来看，由于加入了regularization,任何$\lambda >1$的W都将被剔除，因为在 $L$ 函数中，这些同义的$W$会加大regularization因子。

 The most appealing property is that penalizing large weights tends to improve generalization, because **it means that no input dimension can have a very large influence on the scores all by itself.**

## softmax calssifier

softmax 对于每个样本的损失函数**[cross-entropy loss][4]**如下所示：
$$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} $$
$f(x_i; W) =  W x_i$， $f_j$ to mean the j-th element of the vector of class scores $f$。 多分类中，$W$是一个矩阵，输出的$f$是一个scores向量，每一维是当前类i的预测值（分数值）。$f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$叫做<font color="red">softmax function</font>，其将score值（可正可负）映射到0-1之间的正数。$e^x$函数保证正数的同时也有不足，他会使得数值很大，所以在实际操作时会在分子分母同时减掉一个常数C:
$$\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
= \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}}
= \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}$$
这个常数通常设置成：$\log C = -\max_j f_j$,以使得向量$f$中的值偏移，最大值变为0，这样根据$e^x$函数的特性，会保证$e^{f_i}$总是在0到1之间。

## soft plus

$$
f(x)=\log \left(1+e^{x}\right)
$$

$$
\hat{f(x)}=\frac{e^{x}}{1+e^{x}}=\frac{1}{1+e^{-x}}=\sigma(x)
$$
