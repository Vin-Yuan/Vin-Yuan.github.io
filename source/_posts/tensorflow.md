---
title: tensorflow
mathjax: true
date: 2024-06-04 15:14:36
categories:
tags:
---

## 基本元素

### tf.Variable(……)

```python
import tensorflow as tf
A = tf.Variable(3, name="number")
B = tf.Variable([1,3], name="vector")
C = tf.Variable([[0,1],[2,3]], name="matrix")
D = tf.Variable(tf.zeros([100]), name="zero")
E = tf.Variable(tf.random_normal([2,3], mean=1, stddev=2, dtype=tf.float32))
```

我们可以把函数variable()理解为构造函数，构造函数的使用需要初始值，而这个初始值是一个任何形状、类型的Tensor。
变量有两个重要的步骤，先后为：

- 创建
- 初始化

变量在使用前一定要进行初始化，且变量的初始化必须在模型的其它操作运行之前完成，通常，变量的初始化有三种方式：

- 1.初始化全部变量
`init = tf.global_variables_initializer()`
global_variables_initializer()方法是不管全局有多少个变量，全部进行初始化，是最简单也是最常用的一种方式；
- 2.初始化变量的子集
`init_subset=tf.variables_initializer([b,c], name="init_subset")`
variables_initializer()是初始化变量的子集，相比于全部初始化化的方式更加节约内存
- 3.初始化单个变量

<!-- more -->
```python
nit_var = tf.Variable(tf.zeros([2,5]))
with tf.Session() as sess:
    sess.run(init_var.initializer)
```

Variable()是初始化单个变量，函数的参数便是要初始化的变量内容。

### 为什么要使用tf.global_variables_initializer()？

参考博客[【任意门】](https://blog.csdn.net/qq_37285386/article/details/89054090)

```python
import tensorflow as tf
# 必须要使用global_variables_initializer的场合
# 含有tf.Variable的环境下，因为tf中建立的变量是没有初始化的，也就是在debug时还不是一个tensor量，而是一个Variable变量类型
size_out = 10
tensor = tf.Variable(tf.random_normal(shape=[size_out]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # initialization variables
    print(sess.run(tensor))
# 可以不适用初始化的场合
# 不含有tf.Variable、tf.get_Variable的环境下
# 比如只有tf.random_normal或tf.constant等
size_out = 10
tensor = tf.random_normal(shape=[size_out])  # 这里debug是一个tensor量哦
#init = tf.global_variables_initializer()
with tf.Session() as sess:
    # sess.run(init)  # initialization variables
    print(sess.run(tensor))
```

需要注意的是 tf.placeholder也是tensor，可以这样理解，tf.Variable是需要申请存储（显存/内存）的变量，而tensor:

1. 计算图上计算的中间结果，比如operation
2. 常量，比如tf.random_normal, tf.constant
3. 等待输入的placeholder（不需要初始化，等待feed data)
常见的计算系统，无非是操作数，运算符，然后是存储器，如果施加运算符的步骤不再立刻执行，而是最后计算，那么这些中构结果就没必要一开始申请存储，这便是tensor的由来。

### 获取graph的名称

参考[stackoverflow](https://stackoverflow.com/questions/36883949/in-tensorflow-get-the-names-of-all-the-tensors-in-a-graph)

- To get all nodes:

```python
all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
These have the type tensorflow.core.framework.node_def_pb2.NodeDef
```

- To get all ops:

```python
all_ops = tf.get_default_graph().get_operations()
These have the type tensorflow.python.framework.ops.Operation
```

- To get all variables:

```python
all_vars = tf.global_variables()
These have the type tensorflow.python.ops.resource_variable_ops.ResourceVariable
```

- And finally, to answer the question, to get all tensors:

```python
all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
```

## 方法

### tf.reduce_sum的理解
<https://www.jianshu.com/p/30b40b504bae>

```
tf.reduce_sum(
    input_tensor, 
    axis=None, 
    keepdims=None,
    name=None,
    reduction_indices=None, 
    keep_dims=None)
```

- 0维，又称0维张量，数字，标量：1
- 1维，又称1维张量，数组，vector：[1, 2, 3]
- 2维，又称2维张量，矩阵，二维数组：[[1,2], [3,4]]
- 3维，又称3维张量，立方（cube），三维数组：[ [[1,2], [3,4]], [[5,6], [7,8]] ]
- n维：你应该get到点了吧~

**再多的维只不过是是把上一个维度当作自己的元素**
**越往里axis就越大，依次加1**
下面举个多维tensor例子简单说明。下面是个 2 *3* 4 的tensor。

```
[[[ 1   2   3   4]
  [ 5   6   7   8]
  [ 9   10 11 12]],
 [[ 13  14 15 16]
  [ 17  18 19 20]
  [ 21  22 23 24]]]
```

tf.reduce_sum(tensor, axis=0) axis=0 说明是按第一个维度进行求和。那么求和结果shape是3*4

```
[[1+13   2+14   3+15 4+16]
 [5+17   6+18   7+19 8+20]
 [9+21 10+22 11+23 12+24]]
```

依次类推，如果axis=1，那么求和结果shape是2*4，即：

```
[[ 1 + 5 + 9   2 + 6+10   3 + 7+11   4 + 8+12]
 [13+17+21     14+18+22   15+19+23   16+20+24]]
```

如果axis=2，那么求和结果shape是2*3，即：

```
[[1+2+3+4          5+6+7+8          9+10+11+12]
 [13+14+15+16      17+18+19+20      1+22+23+24]]
```

### tf.stack的理解
<https://stackoverflow.com/questions/50820781/quesion-about-the-axis-of-tf-stack/50821422#50821422>
tf.stack可以理解为先对需要stack的tensor做expand_dims，添加一维，添加的位置即axis，然后在这一axis上做concate

```
def tf.stack(tensors, axis=0):
    return tf.concatenate([tf.expand_dims(t, axis=axis) for t in tensors], axis=axis)
```

### 具有先后顺序，synchronize的计算

- tf.GraphKeys.UPDATE_OPS
- tf.control_dependencies

[参考资料](https://blog.csdn.net/huitailangyz/article/details/85015611)
`tf.GraphKeys.UPDATE_OPS` 和 `tf.control_dependencies` 搭配使用，用来限制一些有先后关系的节点运算
`tf.control_dependencies`，该函数保证其**作用域中的操作**必须要在该函数所传递的**参数中的操作**完成后再进行，比如：

```python
# 第一个运算
import tensorflow as tf
a_1 = tf.Variable(1)
b_1 = tf.Variable(2)
update_op = tf.assign(a_1, 10)
add = tf.add(a_1, b_1)

# 第二个运算
a_2 = tf.Variable(1)
b_2 = tf.Variable(2)
update_op = tf.assign(a_2, 10)
with tf.control_dependencies([update_op]):
    add_with_dependencies = tf.add(a_2, b_2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ans_1, ans_2 = sess.run([add, add_with_dependencies])
    print("Add: ", ans_1)
    print("Add_with_dependency: ", ans_2)

输出：
Add:  3
Add_with_dependency:  12
```

可以看到上面例子中，第一个update_op 对变量做了加一操作，
**但正常的计算图在计算add时是不会经过update_op操作**，所以没有生效。
于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用。
至于`tf.GraphKeys.UPDATE_OPS`的作用，可以在Batch Normalization的例子中理解其作用：

```python
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean)
……
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print(update_ops)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```

两个`tf.add_to_collection`在这里是将需要先计算的Mean和var加入UPDATE_OPS中，这样
>如果不在使用时添加tf.control_dependencies函数，即在训练时(training=True)每批次时只会计算当批次的mean和var，并传递给tf.nn.batch_normalization进行归一化，由于mean_update和variance_update在计算图中并不在上述操作的依赖路径上，因为并不会主动完成，也就是说，在训练时mean_update和variance_update并不会被使用到，其值sfsfafsfafafdafa一直是初始值。因此在测试阶段(training=False)使用这两个作为mean和variance并进行归一化操作，这样就会出现错误。而如果使用tf.control_dependencies函数，会在训练阶段每次训练操作执行前被动地去执行mean_update和variance_update，因此moving_mean和moving_variance会被不断更新，在测试时使用该参数也就不会出现错误。

### embedding 和 lookupTable [[link](https://gshtime.github.io/2018/06/01/tensorflow-embedding-lookup-sparse/)]

feature_num : 原始特征数
embedding_size: embedding之后的特征数
[feature_num, embedding_size] 权重矩阵shape
[m, feature_num] 输入矩阵shape，m为样本数
[m, embedding_size] 输出矩阵shape，m为样本数

embedding_lookup不是简单的查表，[params 对应的向量是可以训练的](https://gshtime.github.io/2018/06/01/tensorflow-embedding-lookup-sparse/)，训练参数个数应该是 feature_num * embedding_size，即前文表述的embedding层权重矩阵，就是说 lookup 的是一种全连接层。

```python
# 当输入单个tensor时，partition_strategy不起作用，不做 id（编号） 的切分
a = np.arange(20).reshape(5,4)
print (a)

# 前面的编号是我手动加的，意思是不做切分的时候就顺序编号就行
# 0#[[ 0  1  2  3]
# 1# [ 4  5  6  7]
# 2# [ 8  9 10 11]
# 3# [12 13 14 15]
# 4# [16 17 18 19]]

tensor_a = tf.Variable(a)
embedded_tensor = tf.nn.embedding_lookup(params=tensor_a, ids=[0,3,2,1])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    embedded_tensor = sess.run(embedded_tensor)
    print(embedded_tensor)
# 根据 ids 参数做选择
#[[ 0  1  2  3]  选择了 id 0
# [12 13 14 15]  选择了 id 3
# [ 8  9 10 11]  选择了 id 2
# [ 4  5  6  7]] 选择了 id 1
```

### loss function
<https://zhuanlan.zhihu.com/p/44216830>

#### 回归

tf.losses.mean_squared_error
tf.losses.absolute_difference
tf.losses.huber_loss：Huber loss

#### 分类

tf.nn.sigmoid_cross_entropy_with_logits
tf.losses.log_loss
tf.nn.softmax_cross_entropy_with_logits_v2
tf.nn.sparse_softmax_cross_entropy_with_logits
tf.nn.weighted_cross_entropy_with_logits
tf.losses.hinge_loss

##### tf.softmax_cross_entropy_with_logits

tf.softmax_cross_entropy_with_logits()的计算过程一共分为两步:

- 1）将logits转换成概率 $$ l_k = \frac{e^k}{\sum_{i=1}^{n}{e^i}} $$
- 2）计算交叉熵损失 $$ -\Sigma y'* log(y)$$

注意事项：
般训练时batch size不会为设为1,所以要使用tf.reduce_mean()来对tf.softmax_cross_entropy_with_logits()的结果取平均,得到关于样本的平均交叉熵损失.

## 调试

### tf.Print

```
……
var = tf.concat([var_a, var_b])
var = tf.Print(var, [var_a, var_b], message="print message in there", summarized=10000)
```

tf.Print 类似identity，挂载到图上，但不影响图结构，所以即使是checkpoint也可以打印计算的中间结果，方便诊断问题。需要注意的是待打印的变量需是图中流过var的上端节点tensor

### tf.cond

```
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.constant(4)
y = tf.constant(5)
z = tf.multiply(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
with tf.Session() as session:
    print(result.eval())
```

## same 和 padding

$$
\left\lceil\frac{n_{i}+p_{i}-k+1}{s}\right\rceil=n_{o}
$$
$n_i$ 为input_size
$n_o$ 为output_size
$k$   为 kernel size
$s$   为 stride
$p_i$ 为 padding size
<https://www.jianshu.com/p/b9eb4758118d>

## Session.run([a,b,c])中变量的顺序

```python
loss_val, _ = sess.run([loss, optimizer])
```

对于上面遇到的问题，可能会产生怀疑，这个Loss到底是back propgation之前的还是之后的？在查看[stack overflow](https://stackoverflow.com/questions/53165418/order-of-sess-runop1-op2-in-tensorflow) 上的解答，发现sess.run中的变量求解是不确定的。上面的代码求解的是BP之前的，tensorflow为了保证不重复计算，图中节点已经计算过的会直接取出，若想获取BP之后的Loss, 可通过如下方式：

1. 再次sess.run([loss])
2. 定义一个loss_end tensor,
3. 使用tf.control_dependencies([optimizer])来规定依赖

## Session 和 Graph 的关系
<https://www.easy-tensorflow.com/tf-tutorials/basics/graph-and-session>
网络经过定义然后训练后得的参数保存在session中而非graph中，graph只是网路结构的表述。

## Loss Function
<https://zhuanlan.zhihu.com/p/44216830>

### 1. tf.nn.sigmoid_cross_entropy_with_logits

先 sigmoid 再求交叉熵,二分类问题首选,使用时，一定不要将预测值（y_pred）进行 sigmoid 处理，因为这个函数已经包含了sigmoid过程

```
# Tensorflow中集成的函数
sigmoids = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
sigmoids_loss = tf.reduce_mean(sigmoids)

# 利用Tensorflow基础函数手工实现
y_pred_si = 1.0/(1+tf.exp(-y_pred))
sigmoids = -y_true*tf.log(y_pred_si) - (1-y_true)*tf.log(1-y_pred_si)
sigmoids_loss = tf.reduce_mean(sigmoids)
```

### 2. tf.losses.log_loss

预测值（y_pred）计算完成后，若已先行进行了 sigmoid 处理，则使用此函数求 loss ，若还没经过 sigmoid 处理，可直接使用 sigmoid_cross_entropy_with_logits。

```
# Tensorflow中集成的函数
logs = tf.losses.log_loss(labels=y, logits=y_pred)
logs_loss = tf.reduce_mean(logs)

# 利用Tensorflow基础函数手工实现
logs = -y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)
logs_loss = tf.reduce_mean(logs)
```

## 模型导出和恢复

[TensorFlow：保存和提取模型](https://www.jianshu.com/p/c3a7f5c47b83)

### 模型恢复

#### savedModel

[如何查看Tensorflow SavedModel格式模型的信息](https://blog.csdn.net/mogoweb/article/details/83054861)
signature并非是为了保证模型不被修改的那种电子签名。类似于编程语言中模块的输入输出信息，比如函数名，输入参数类型，输出参数类型等等。

```python
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:
  model_filename ='./model/saved_model.pb'
  with gfile.FastGFile(model_filename, 'rb') as f:

    data = compat.as_bytes(f.read())
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(data)

    if 1 != len(sm.meta_graphs):
      print('More than one graph found. Not sure which to write')
      sys.exit(1)

    g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
LOGDIR='./logdir'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
```

另外可参考stackoverflow的总结
<https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model>

## Tensorflow 主流程

### 梯度更新部分

```python
with tf.control_dependencies(update_ops):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# 计算梯度，grads_and_vars 是一个list：(gradient, variable)，变量和变量对应的梯度    
grads_and_vars = optimizer.compute_gradients(loss)
# 
# ...... (此处可以做 梯度修建等操作，然后再对变量更新梯度）
#
# 执行对应变量的更新梯度操作
train_op = optimizer.apply_gradients(grad_vars, global_step=global_step)
```

### tf.Summary
<https://zhuanlan.zhihu.com/p/31459527>
<https://www.cnblogs.com/lyc-seu/p/8647792.html>

## Keras backend tensorflow

keras 和 tensorflow-gpu版本兼容列表
<https://docs.floydhub.com/guides/environments/>
应对不同tensorlow-gpu好cuda版本的安装
```conda install tensorflow-gpu==1.9.0 cudatoolkit==8.0```

## Tensorboard

多个model对比train, validation效果

```
tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
```

demo 例子: <https://blog.csdn.net/qiu931110/article/details/80137287>
讲解：<https://cloud.tencent.com/developer/section/1475708>

# 分布式

## 分布式

[参考资料:浅显易懂的分布式TensorFlow入门教程](https://yq.aliyun.com/articles/603370)

### 系统会包含三种类型的节点

- **一个或多个参数服务器（ps server)**，用来存放模型
- **一个主worker**，用来协调训练操作，负责模型的初始化，为训练步骤计数，保存模型到checkpoints中，从checkpoints中读取模型，向TensorBoard中保存summaries（需要展示的信息）。主worker还要负责分布式计算的容错机制（如果参数服务器或worker服务器崩溃）。
- **worker服务器（包括主worker服务器）**，用来执行训练操作，并向参数服务器发送更新操作。(worker服务器在这里是指多个worker节点，集群的意思，见上面结构图）

> 也就是说最小的集群需要包含一个主worker服务器和一个参数服务器。可以将它扩展为一个主worker服务器，多个参数服务器和多个worker服务器。
最好有多个参数服务器，**因为worker服务器和参数服务器之间有大量的I/O通信**。如果只有2个worker服务器，可能1个参数服务器可以扛得住所有的读取和更新请求。但如果你有10个worker而且你的模型非常大，一个参数服务器可能就不够了。

**在分布式TensorFlow中，同样的代码会被发送到所有的节点**。虽然你的main.py、train.py等会被同时发送到worker服务器和参数服务器，但
每个节点会依据自己的环境变量来执行不同的代码块。

### 分布式TensorFlow代码的准备包括三个阶段

1. 定义tf.trainClusterSpec和tf.train.Server
2. 将模型赋给参数服务器和worker服务器
3. 配置和启动tf.train.MonitoredTrainingSession

## PS and Worker

参考:
[【1】分布式TensorFlow入门教程](https://zhuanlan.zhihu.com/p/35083779)
[【2】Distributed TensorFlow](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md)
[【3】Distributed TensorFlow](https://www.oreilly.com/content/distributed-tensorflow/)

### Client

A client is typically a program that builds a TensorFlow graph and constructs a tensorflow::Session to interact with a cluster. Clients are typically written in Python or C++. **A single client process can directly interact with multiple TensorFlow servers** (see "Replicated training" above), **and a single server can serve multiple clients**.
server在这里是服务者的角色，无论是ps还是worker都是server,我们可以建立多个server,服务与多个client。比如有个worker server已经建立，A用client建立一个regression任务是使用这个server训练，B用client建立了一个CNN任务也使用了这个server，这不冲突，在资源充足情况下是可以先后使用同一个server的。
### Job
A job comprises a list of "tasks", which typically serve a common purpose. For example, a job named ps (for "parameter server") typically hosts nodes that store and update variables; while a job named worker typically hosts stateless nodes that perform compute-intensive tasks. The tasks in a job typically run on different machines. The set of job roles is flexible: for example, a worker may maintain some state.
注意原则上ps和worker两种job功能不同

- ps (parameter server)：hosts nodes that **store** and **update** variables;
- worker：hosts **stateless** nodes that perform **compute-intensive tasks**.
虽然规则上这样各司其职，但实际上并不一定需要严格这样执行，job的角色是灵活的，比如，worker也可以维护一些状态（state)
### Master service
An RPC service that provides remote access to a set of distributed devices, and **acts as a session target.** The master service implements the tensorflow::Session interface, and is responsible for coordinating work across one or more "worker services". **All TensorFlow servers implement the master service.**
具体理解参照下面的例子：

在【1】的例子中：

```
# example.py
import tensorflow as tf

tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # create cluster
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # create the server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    server.join()

if __name__ == "__main__":
    tf.app.run()
```

exmple.py用来建立执行不同功能的server，执行上面的example.py来生成不同的**server**

```
python example.py --job_name=ps --task_index=0
python example.py --job_name=worker --task_index=0
python example.py --job_name=worker --task_index=1
```

以work-0为例子，打印日志如下：

```
2020-03-20 15:50:24.761196: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:252] Initialize GrpcChannelCache for job ps -> {0 -> localhost:2222}
2020-03-20 15:50:24.761240: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:252] Initialize GrpcChannelCache for job worker -> {0 -> localhost:2223, 1 -> localhost:2224}
2020-03-20 15:50:24.762191: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:391] Started server with target: grpc://localhost:2223
```

日志里显示了当前server的grpc地址，以及server所知道的其他集群信息，这是理所应当，如果我们需要协调分布式任务，必然需要知道其他服务的信息，这样才可以通信协调工作。
到目前为止，server已经建立，也就意味着资源已经建立，而接下来我们就可以通过client使用这些资源来完成分布式任务了。
我们创建一个client来执行一个计算图，并且采用/job:worker/task:0这个server所对应的**master**，即grpc://localhost:2223来创建Session，如下所示：

```
#client.py
import tensorflow as tf

if __name__ == "__main__":
    with tf.device("/job:ps/task:0"):
        x = tf.Variable(tf.ones([2, 2]))
        y = tf.Variable(tf.ones([2, 2]))

    with tf.device("/job:worker/task:0"):
        z = tf.matmul(x, y) + x

    with tf.device("/job:worker/task:1"):
        z = tf.matmul(z, x) + x

    with tf.Session("grpc://localhost:2223") as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(z)
        print(val)
```

其实这个client就是一个进程，但是其在计算时需要依靠cluster中的device来执行部分计算子图。这时候各个server的日志中，只有2223的日志发生了变化，多了一行：

```
……
2020-03-20 15:55:50.796022: I tensorflow/core/distributed_runtime/master_session.cc:1192] Start master session b7f2e9eb1f8c5548 with config:
```

## Between-graph replication

在Between-graph replication中，各个worker都包含一个client，它们构建相同的计算图，然后把参数放在ps上，TensorFlow提供了一个专门的函数tf.train.replica_device_setter来方便Graph构建，先看代码【1】：

```
# cluster包含两个ps 和三个 worker
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
cluster = tf.train.ClusterSpec(cluster_spec)
with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
               cluster=cluster)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
  # Run compute
```

使用**tf.train.replica_device_setter**可以自动把Graph中的Variables放到ps上，而同时将Graph的计算部分放置在当前worker上，省去了很多麻烦。由于ps往往不止一个，这个函数在为各个Variable分配ps时默认采用简单的round-robin方式，就是按次序将参数挨个放到各个ps上，但这个方式可能不能使ps负载均衡，如果需要更加合理，可以采用tf.contrib.training.GreedyLoadBalancingStrategy策略。
采用Between-graph replication方式的另外一个问题，由于各个worker都独立拥有自己的client，但是对于一些公共操作比如模型参数初始化与checkpoint文件保存等，如果每个client都独立进行这些操作，显然是对资源的浪费。为了解决这个问题，一般会指定一个worker为chief worker，它将作为各个worker的管家，协调它们之间的训练，并且完成模型初始化和模型保存和恢复等公共操作。在TensorFlow中，可以使用tf.train.MonitoredTrainingSession创建client的Session，并且其可以指定哪个worker是chief worker。