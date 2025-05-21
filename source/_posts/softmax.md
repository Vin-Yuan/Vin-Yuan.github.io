---
title: softmax
date: 2019-03-27 11:30:22
categories:
tags: machine learning
mathjax: true
---

# Softmax 与梯度下降

参考1：http://dlsys.cs.washington.edu/schedule lecture3



![](http://ww1.sinaimg.cn/large/6bf0a364ly1g1h79b85ybj20mg0bc0tm.jpg)

<!-- more -->

参考上图的网络结构，输入层，输出层。

$W_{i,j}$的含义：输出神经元 $a_i$ 关联的第 $j$ 个权重，要从输出神经元的角度去理解，这样比较清楚。

神经元的构造如下：

![](http://ww1.sinaimg.cn/large/6bf0a364ly1g1h822vi52j20p7094dgb.jpg)

$z_i = w_i \cdot x \rightarrow z_i = \sum_j w_{i,j} \cdot x_j $ ，意味着$a_i$ 的每一个权重和输入$x$的相应feature相乘

$W^T = \begin{bmatrix} -&-& -\\-&w_i&- \\-&-&- \end{bmatrix} ​$，$x = \begin{bmatrix} | \\ x_j\\| \end{bmatrix} ​$

logic regression 的损失函数为

$$\begin{equation} C = -\sum_i y_i lna_i  \end{equation}​$$ ，注意这里表达的含义，softmax有k个输出，准确的表达式应该是$C = -\sum_k y_k ln(\hat{a}_k) ​$，即对当前输入$x​$，其label为第 $i​$ 个类，则 $y_i​$ 为1，其它为0。所有样本的损失函数如下：

$J(\theta) = -\frac{1}{m}\sum\limits_{i=1}^m \sum\limits_{k=1}^K y^{(i)}_k log(\hat{p}^{(i)}_k) \tag{1}​$

 $\hat{p}^{(i)}_k$ 为 $softmax(x^{(i)})$的第$k$个输出，

$\hat{p}_k = \sigma(Z(x))_k = \frac{exp(z_k(x))}{\sum\limits_{j=1}^{K} exp(z_j(x))} \tag{2}$



$$\begin{equation} \frac{\partial C}{\partial z_i} = a_i - y_i \end{equation}  ​$$

损失函数对每个权重$w_{i,j}​$的导数：

$\frac{\partial C}{\partial w_{i,j}} = \frac{\partial C}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_{i,j}} = （a_i - y_i)\cdot x_j ​$

loss function对每个参数的导数构成梯度向量，即**标量对矩阵的求导**：$\frac {\partial L}{\partial W}​$

在下面的代码中，```W_grad = np.dot(batch_xs.T, y_grad)``` 这一步正是利用的梯度向量。思考这段代码是如何形成两个矩阵相乘形式的。batch_xs 原始布局如下：

$X = \begin{bmatrix} -&-& -\\-&x_i&- \\-&-&- \end{bmatrix} $，其中 $x_i$为行向量$(x_{i,1}, x_{i,2}, ... , x_{i,n})$ （备注：也可以写为 $x_1^{(i)}$，取决于样本 $i$ 的表示方式，用数组numpy表达时为前者）。Loss function是对**所有的本批次样本计算**的，所以 $Loss = \sum_i loss(f(x^i)， y^i)$ ，其中 ​$i$ 为batch_size , 综合(1) 和（2）可以得出Loss要对所有样本loss1 + loss2 + loss3+ ...求梯度，所以是一个加和。

```python
import numpy as np
from tinyflow.datasets import get_mnist
def softmax(x):
x = x - np.max(x, axis=1, keepdims=True)
x = np.exp(x)
x = x / np.sum(x, axis=1, keepdims=True)
return x
# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)
learning_rate = 0.5 / 100
W = np.zeros((784, 10))
for i in range(1000):
batch_xs, batch_ys = mnist.train.next_batch(100)
# forward
y = softmax(np.dot(batch_xs, W))
# backward
y_grad = y - batch_ys
W_grad = np.dot(batch_xs.T, y_grad)
# update
W = W - learning_rate * W_grad
```

重构程 tensorflow API 方式代码如下：

```python
import tinyflow as tf
from tinyflow.datasets import get_mnist
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y = tf.nn.softmax(tf.matmul(x, W))
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Update rule
learning_rate = 0.5
W_grad = tf.gradients(cross_entropy, [W])[0]
train_step = tf.assign(W, W - learning_rate * W_grad)
# Training Loop
sess = tf.Session()
sess.run(tf.initialize_all_variables())
mnist = get_mnist(flatten=True, onehot=True)
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}) #Real execution happens here
```

注释：

```cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))```

- 这里面的reduce_mean, 第二个参数是reduction_indices，可以这样理解，0 代表对第一个维度规约，即$\sum\limits_{i=0}^m a_{i,j}$ ,1则代表对第二个维度规约$\sum\limits_{j=0}^n a_{i,j}$，以下是tensorflow的官方文档示例：

```
# 'x' is [[1,1,1],
			[1,1,1]]
tf.reduce_sum(x) = 6
tf.reduce_sum(x, 0) = [2, 2, 2]
tf.reduce_sum(x, 1) = [3,3]
tf.reduce_sum(x, 1, keep_dims=True) = [[3], [3]]
tf.reduce_sum(x, [0,1]) = 6
```

- 这里的y_ * tf.log(y) 是两个 （m, 10)的矩阵 “点乘” ，区别于矩阵乘法，这里是对应元素相乘。具体参考 [2] [Python 之 numpy 和 tensorflow 中的各种乘法（点乘和矩阵乘）](https://www.cnblogs.com/liuq/p/9330134.html)

