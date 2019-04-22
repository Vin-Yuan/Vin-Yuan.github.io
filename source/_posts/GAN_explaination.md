# 机器之心GitHub项目：GAN完整理论推导与实现，Perfect！







> 本文是机器之心第二个 GitHub 实现项目，上一个 GitHub 实现项目为从头开始构建卷积神经网络。在本文中，我们将从原论文出发，借助 Goodfellow 在 NIPS 2016 的演讲和台大李宏毅的解释，而完成原 GAN 的推导与证明。本文主要分四部分，第一部分是描述 GAN 的直观概念，第二部分描述概念与优化的形式化表达，第三部分将对 GAN 进行详细的理论推导与分析，最后我们将实现前面的理论分析。

GitHub 实现地址：<https://github.com/jiqizhixin/ML-Tutorial-Experiment>

本文更注重理论与推导，更多生成对抗网络的概念与应用前查看：

- [生成对抗网络初学入门：一文读懂GAN的基本原理](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650730721&idx=2&sn=95b97b80188f507c409f4c72bd0a2767&chksm=871b349fb06cbd891771f72d77563f77986afc9b144f42c8232db44c7c56c1d2bc019458c4e4&scene=21#wechat_redirect)
- [深入浅出：GAN原理与应用入门介绍](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650730028&idx=1&sn=21d57cf54f257aeab15ebd4058671a2b&chksm=871b2a52b06ca3449f255549a914e8ab8d85bb4d43e0487a95fd9ffd97e708d9eac7a1f9943b&scene=21#wechat_redirect)
- [宅男的福音：用GAN自动生成二次元萌妹子](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650729957&idx=4&sn=bdeb666588a1e926e2802b2daf3c2836&chksm=871b299bb06ca08db37816a9b8bce1d38ddf957f25f37bd39fb6e9ce2da5c1120373a2a1e888&scene=21#wechat_redirect)
- [一文帮你发现各种出色的GAN变体](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650724769&idx=2&sn=6fa540106cf6a5fd55fc39d057092888&chksm=871b1ddfb06c94c9e11d3a8281f60c0fce06a4e021fcd8eaab858c7f08ab9c939c4ad130e4b2&scene=21#wechat_redirect)
- [直观理解GAN背后的原理：以人脸图像生成为例](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650723168&idx=2&sn=68b21b815688443a0dd7caa115cc13fa&chksm=871b171eb06c9e085ab0f2223e6bab04d2eecfa430ca071b19d5c377d909d46724269126dbf3&scene=21#wechat_redirect)
- [萌物生成器：如何使用四种GAN制造猫图](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650728794&idx=2&sn=426052e5cd7d952902b6623e40df16de&chksm=871b2d24b06ca4323d055668db51590703c45966149e0849d1dd92771bc28d2b5e24c1b4ec5d&scene=21#wechat_redirect)
- [GAN之父NIPS 2016演讲现场直击：全方位解读生成对抗网络的原理及未来](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650721284&idx=1&sn=427e7f45c8253ab22a3960978409f5d1&chksm=871b087ab06c816c424ad03810be3e1b3aa9d6e99a5f325047796f110d178a07736f667d1a10&scene=21#wechat_redirect)
- [看穿机器学习（W-GAN模型）的黑箱](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650723168&idx=3&sn=41fcf2fb0408c7b6a9b82d55d91c2b9c&chksm=871b171eb06c9e082c4083ff32748104a617e5cb1e6bd4d296b4db431358b8a41f40908ea8a5&scene=21#wechat_redirect)

### 生成对抗网络基本概念

要理解生成对抗模型（GAN），首先要了解生成对抗模型可以拆分为两个模块：一个是判别模型，另一个是生成模型。简单来说就是：两个人比赛，看是 A 的矛厉害，还是 B 的盾厉害。比如，我们有一些真实数据，同时也有一把随机生成的假数据。A 拼命地把随手拿过来的假数据模仿成真实数据，并揉进真实数据里。B 则拼命地想把真实数据和假数据区分开。

这里，A 就是一个生成模型，类似于造假币的，一个劲地学习如何骗过 B。而 B 则是一个判别模型，类似于稽查警察，一个劲地学习如何分辨出 A 的造假技巧。

如此这般，随着 B 的鉴别技巧的越来越厉害，A 的造假技巧也是越来越纯熟，而一个一流的假币制造者就是我们所需要的。虽然 GAN 背后的思想十分直观与朴素，但我们需要更进一步了解该理论背后的证明与推导。

总地来说，Goodfellow 等人提出来的 GAN 是通过对抗过程估计生成模型的新框架。在这种框架下，我们需要同时训练两个模型，即一个能捕获数据分布的生成模型 G 和一个能估计数据来源于真实样本概率的判别模型 D。生成器 G 的训练过程是最大化判别器犯错误的概率，即判别器误以为数据是真实样本而不是生成器生成的假样本。因此，这一框架就对应于两个参与者的极小极大博弈（minimax game）。在所有可能的函数 G 和 D 中，我们可以求出唯一均衡解，即 G 可以生成与训练样本相同的分布，而 D 判断的概率处处为 1/2，这一过程的推导与证明将在后文详细解释。

当模型都为多层感知机时，对抗性建模框架可以最直接地应用。为了学习到生成器在数据 $x$ 上的分布 $P_g$，我们先定义一个先验的输入噪声变量 $P_z(z)$，然后根据 $G(z;θ_g)$ 将其映射到数据空间中，其中 $G$ 为多层感知机所表征的可微函数。我们同样需要定义第二个多层感知机 $D(s;θ_d)$，它的输出为单个标量。$D(x)$ 表示 $x$ 来源于真实数据而不是 $P_g$ 的概率。我们训练 $D$ 以最大化正确分配真实样本和生成样本的概率，因此我们就可以通过最小化 $log(1-D(G(z)))$ 而同时训练 $G$。也就是说判别器 $D$ 和生成器对价值函数 $V(G,D)$ 进行了极小极大化博弈：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/24307image.png)

我们后一部分会对对抗网络进行理论上的分析，该理论分析本质上可以表明如果 $G$ 和 $D$ 的模型复杂度足够（即在非参数限制下），那么对抗网络就能生成数据分布。此外，Goodfellow 等人在论文中使用如下案例为我们简要介绍了基本概念。

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/52960image%20(1).png)

如上图所示，生成对抗网络会训练并更新判别分布（即 $D​$，蓝色的虚线），更新判别器后就能将数据真实分布（黑点组成的线）从生成分布 $P_g(G)​$（绿色实线）中判别出来。下方的水平线代表采样域 $Z​$，其中等距线表示 $Z​$ 中的样本为均匀分布，上方的水平线代表真实数据 $X​$ 中的一部分。向上的箭头表示映射 $x=G(z)​$ 如何对噪声样本（均匀采样）施加一个不均匀的分布 $P_g​$。

- （a）考虑在收敛点附近的对抗训练：$P_g$ 和 $P_{data}$ 已经十分相似，$D$ 是一个局部准确的分类器。

- （b）在算法内部循环中训练 $D​$ 以从数据中判别出真实样本，该循环最终会收敛到:$D(x)=P_{data}(x)/(P_{data}(x)+P_g(x))​$。

- （c）随后固定判别器并训练生成器，在更新 $G$ 之后，$D$ 的梯度会引导 $G(z)$流向更可能被 $D$ 分类为真实数据的方向。

- （d）经过若干次训练后，如果 $G​$ 和 $D​$ 有足够的复杂度，那么它们就会到达一个均衡点。这个时候 $P_g=P_{data}​$，即生成器的概率密度函数等于真实数据的概率密度函数，也即生成的数据和真实数据是一样的。在均衡点上 $D​$ 和 $G​$ 都不能得到进一步提升，并且判别器无法判断数据到底是来自真实样本还是伪造的数据，即 $D(x)= 1/2​$。

上面是比较精简地介绍了生成对抗网络的基本概念，下一节将会把这些概念形式化，并描述优化的大致过程。

### 概念与过程的形式化

#### 理论完美的生成器

该算法的目标是令生成器生成与真实数据几乎没有区别的样本，即一个造假一流的 A，就是我们想要的生成模型。数学上，即将随机变量生成为某一种概率分布，也可以说概率密度函数为相等的：$P_G(x)=P_{data}(x)$。这正是数学上证明生成器高效性的策略：即定义一个最优化问题，其中最优生成器 $G$ 满足 $P_G(x)=P_{data}(x)$。如果我们知道求解的 $G$ 最后会满足该关系，那么我们就可以合理地期望神经网络通过典型的 SGD 训练就能得到最优的 $G$。

#### 最优化问题

正如最开始我们了解的警察与造假者案例，定义最优化问题的方法就可以由以下两部分组成。首先我们需要定义一个判别器$ D$ 以判别样本是不是从 $P_{data}(x)​$ 分布中取出来的，因此有：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/29837image%20(2).png)

其中 $E​$ 指代取期望。这一项是根据「正类」（即辨别出 x 属于真实数据 data）的对数损失函数而构建的。最大化这一项相当于令判别器$ D​$ 在 $x​$ 服从于 data 的概率密度时能准确地预测$ D(x)=1​$，即：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/53394image%20(3).png)

另外一项是企图欺骗判别器的生成器$ G$。该项根据「负类」的对数损失函数而构建，即：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/74077image%20(4).png)

因为 $x<1$ 的对数为负，那么如果最大化该项的值，则需要令均值 $D(G(z))≈0$，因此 $G$ 并没有欺骗 $D$。为了结合这两个概念，判别器的目标为最大化：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/10759image%20(5).png)

给定生成器 $G​$，其代表了判别器 $D​$ 正确地识别了真实和伪造数据点。给定一个生成器 $G​$，上式所得出来的最优判别器可以表示为 ![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/36731image%20(6).png) （下文用 $D_G^*​$表示）。定义价值函数为：



![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/72972image%20(7).png)

然后我们可以将最优化问题表述为：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/05150image%20(8).png)![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/32595image%20(8).png)

现在 $G​$  的目标已经相反了，当 $D=D_G^*​$ 时，最优的 $G​$ 为最小化前面的等式。在论文中，作者更喜欢求解最优化价值函的 $G​$ 和 $ D ​$ 以求解极小极大博弈：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/56239image%20(9).png)

对于 $D​$ 而言要尽量使公式最大化（识别能力强），而对于 $G​$ 又想使之最小（生成的数据接近实际数据）。整个训练是一个迭代过程。其实极小极大化博弈可以分开理解，即在给定 G 的情况下先最大化 V(D,G) 而取 D，然后固定 $D​$，并最小化 $V(D,G)​$ 而得到 G。其中，给定 $G​$，最大化 $V(D,G)​$ 评估了 $P_G​$ 和 $P_{data}​$ 之间的差异或距离。

最后，我们可以将最优化问题表达为：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/72295image%20(10).png)

上文给出了 GAN 概念和优化过程的形式化表达。通过这些表达，我们可以理解整个生成对抗网络的基本过程与优化方法。当然，有了这些概念我们完全可以直接在 GitHub 上找一段 GAN 代码稍加修改并很好地运行它。但如果我们希望更加透彻地理解 GAN，更加全面地理解实现代码，那么我们还需要知道很多推导过程。比如什么时候 D 能令价值函数 $V(D,G) $取最大值、$G$ 能令 $V(D,G)$ 取最小值，而 $D$ 和 $G$ 该用什么样的神经网络（或函数），它们的损失函数又需要用什么等等。总之，还有很多理论细节与推导过程需要我们进一步挖掘。

### 理论推导

在原 GAN 论文中，度量生成分布与真实分布之间差异或距离的方法是 $JS$ 散度，而 $JS$ 散度是我们在推导训练过程中使用 $KL$ 散度所构建出来的。所以这一部分将从理论基础出发再进一步推导最优判别器和生成器所需要满足的条件，最后我们将利用推导结果在数学上重述训练过程。这一部分为我们下一部分理解具体实现提供了强大的理论支持。

#### KL 散度

在信息论中，我们可以使用香农熵（Shannon entropy）来对整个概率分布中的不确定性总量进行量化：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/07326image%20(11).png)

如果我们对于同一个随机变量 x 有两个单独的概率分布 $P(x)$ 和 $Q(x)$，我们可 以使用 $KL$ 散度（Kullback-Leibler divergence）来衡量这两个分布的差异：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/30228image%20(12).png)

在离散型变量的情况下，$KL$ 散度衡量的是，当我们使用一种被设计成能够使 得概率分布 $Q$ 产生的消息的长度最小的编码，发送包含由概率分布 $P$ 产生的符号 的消息时，所需要的额外信息量。

KL 散度有很多有用的性质，最重要的是它是非负的。$KL$ 散度为 0 当且仅当 P 和 Q 在离散型变量的情况下是相同的分布，或者在连续型变量的情况下是 『几乎 处处』 相同的。因为 KL 散度是非负的并且衡量的是两个分布之间的差异，它经常 被用作分布之间的某种距离。然而，它并不是真的距离因为它不是对称的：对于某 些 P 和 Q，$D_{KL}(P||Q)$ 不等于 $D_{KL}(Q||P)$。这种非对称性意味着选择 $D_{KL}(P||Q)$ 还是 $D_{KL}(Q||P)$ 影响很大。

在李弘毅的讲解中，KL 散度可以从极大似然估计中推导而出。若给定一个样本数据的分布 $P_{data}(x)$ 和生成的数据分布 $P_G(x;θ)$，那么 GAN 希望能找到一组参数θ使分布 $P_g(x;θ)$ 和 $P_{data}(x)$ 之间的距离最短，也就是找到一组生成器参数而使得生成器能生成十分逼真的图片。

现在我们可以从训练集抽取一组真实图片来训练 $P_G(x;θ)$ 分布中的参数θ使其能逼近于真实分布。因此，现在从 $P_{data}(x)$ 中抽取 m 个真实样本 ${𝑥^1,𝑥^2,…,𝑥^𝑚}$，上标$i$代表第 $i$ 个样本。对于每一个真实样本，我们可以计算 $P_G(x^i;θ)$，即在由θ确定的生成分布中，$x^i$ 样本所出现的概率。因此，我们就可以构建似然函数：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/92642image%20(13).png)

其中 $∏​$ 代表累乘、$P_G(x^i;θ)​$ 代表第 i 个样本在生成分布出现的概率。从该似然函数可知，我们抽取的 $m​$ 个真实样本在 $P_G(x;θ)​$ 分布中全部出现的概率值可以表达为 $L​$。又因为若 $P_G(x;θ)​$ 分布和 $P_{data}(x)​$ 分布相似，那么真实数据很可能就会出现在 $P_G(x;θ)​$ 分布中，因此 $m​$ 个样本都出现在 $P_G(x;θ)​$ 分布中的概率就会十分大。

下面我们就可以最大化似然函数 $L​$ 而求得离真实分布最近的生成分布（即最优的参数$θ​$）：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/28796image%20(14).png)

在上面的推导中，我们希望最大化似然函数 L。若对似然函数取对数，那么累乘∏就能转化为累加 $∑​$，并且这一过程并不会改变最优化的结果。因此我们可以将极大似然估计化为求令 $log[P_G(x;θ)]​$ 期望最大化的$θ​$，而期望$ E[logP_G(x;θ)]​$ 可以展开为在 $x​$ 上的积分形式：$∫P_{data}(x)logP_G(x;θ)dx​$。又因为该最优化过程是针对θ的，所以我们添加一项不含θ的积分并不影响最优化效果，即可添加 $-∫P_{data}(x)logP_{data}(x)dx​$。添加该积分后，我们可以合并这两个积分并构建类似 $KL​$ 散度的形式。该过程如下：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/22020%E7%B2%98%E8%B4%B4%E5%9B%BE%E7%89%87_20170930194024.jpg)

这一个积分就是 $KL$ 散度的积分形式，因此，如果我们需要求令生成分布 $P_G(x;θ)$ 尽可能靠近真实分布 P_{data}(x) 的参数$\theta$，那么我们只需要求令 $KL$ 散度最小的参数$θ$。若取得最优参数$θ$，那么生成器生成的图像将显得非常真实。

#### 推导存在的问题

下面，我们必须证明该最优化问题有唯一解 $G*$，并且该唯一解满足 $P_G=P_{data}$。不过在开始推导最优判别器和最优生成器之前，我们需要了解 Scott Rome 对原论文推导的观点，他认为原论文忽略了可逆条件，因此最优解的推导不够完美。

在 GAN 原论文中，有一个思想和其它很多方法都不同，即生成器 $G$ 不需要满足可逆条件。Scott Rome 认为这一点非常重要，因为实践中 $G$ 就是不可逆的。而很多证明笔记都忽略了这一点，他们在证明时错误地使用了积分换元公式，而积分换元却又恰好基于 $G$ 的可逆条件。Scott 认为证明只能基于以下等式的成立性：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/03195image%20(15).png)

该等式来源于测度论中的 Radon-Nikodym 定理，它展示在原论文的命题 1 中，并且表达为以下等式：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/70934image%20(16).png)

我们看到该讲义使用了积分换元公式，但进行积分换元就必须计算 $G^{(-1)}$，而 $G$ 的逆却并没有假定为存在。并且在神经网络的实践中，它也并不存在。可能这个方法在机器学习和统计学文献中太常见了，因此我们忽略了它。

#### 最优判别器

在极小极大博弈的第一步中，给定生成器 $G$，最大化 $V(D,G)$ 而得出最优判别器 $D$。其中，最大化 $V(D,G)$ 评估了 $P_G$ 和 $P_{data}$ 之间的差异或距离。因为在原论文中价值函数可写为在 $x$ 上的积分，即将数学期望展开为积分形式：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/98621image%20(16)1.png)

其实求积分的最大值可以转化为求被积函数的最大值。而求被积函数的最大值是为了求得最优判别器 D，因此不涉及判别器的项都可以看作为常数项。如下所示，$P_{data(x)}$ 和 $P_G(x)$ 都为标量，因此被积函数可表示为 $a*D(x)+b*log(1-D(x))$。

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/46879image%20(17).png)

若令判别器 $D(x)$ 等于 $y$，那么被积函数可以写为：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/85286image%20(18).png)

为了找到最优的极值点，如果 $a+b≠0$，我们可以用以下一阶导求解：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/09657image%20(19).png)

如果我们继续求表达式 $f(y)​$ 在驻点的二阶导：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/24515image%20(20).png)

其中 $a,b∈(0,1)$。因为一阶导等于零、二阶导小于零，所以我们知道 a/(a+b) 为极大值。若将 $a=P_{data}(x)$、$b=P_G(x)$ 代入该极值，那么最优判别器 $D(x)=P_{data}(x)/(P_{data}(x)+P_G(x))$。

最后我们可以将价值函数表达式写为：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/51044image%20(21).png)

如果我们令 $D(x)=P_{data}/(P_{data}+p_G)$，那么我们就可以令价值函数 $V(G,D)$ 取极大值。因为 f(y) 在定义域内有唯一的极大值，最优 $D$ 也是唯一的，并且没有其它的 $D$ 能实现极大值。

其实该最优的 $D$ 在实践中并不是可计算的，但在数学上十分重要。我们并不知道先验的 $P_{data(x)}$，所以我们在训练中永远不会用到它。另一方面，它的存在令我们可以证明最优的 $G$ 是存在的，并且在训练中我们只需要逼近 $D​$。

#### 最优生成器

当然 GAN 过程的目标是令 $P_G=P_{data}$。这对最优的 $D$ 意味着什么呢？我们可以将这一等式代入 $D_G^*$的表达式中：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/75323image%20(22).png)

这意味着判别器已经完全困惑了，它完全分辨不出 $P_{data}$ 和 $P_G$ 的区别，即判断样本来自 P_{data} 和 P_G 的概率都为 1/2。基于这一观点，GAN 作者证明了 $G$ 就是极小极大博弈的解。该定理如下：

「当且仅当 $P_G=P_{data}$，训练标准 $C(G)=maxV(G,D)$ 的全局最小点可以达到。

以上定理即极大极小博弈的第二步，求令 $V(G,D*)$ 最小的生成器 $G$（其中 $G*$代表最优的判别器）。之所以当 $P_G(x)=P_{data}(x)$ 可以令价值函数最小化，是因为这时候两个分布的 $JS$ 散度$ [JSD(P_{data}(x) || P_G(x))]$ 等于零，这一过程的详细解释如下。

原论文中的这一定理是「当且仅当」声明，所以我们需要从两个方向证明。首先我们先从反向逼近并证明 $C(G)​$ 的取值，然后再利用由反向获得的新知识从正向证明。设 $P_G​$=$P_{data}​$（反向指预先知道最优条件并做推导），我们可以反向推出：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/25267image%20(23).png)

该值是全局最小值的候选，因为它只有在 $P_{data}$ 的时候才出现。我们现在需要从正向证明这一个值常常为最小值，也就是同时满足「当」和「仅当」的条件。现在放弃 $P_G=P_{data}$ 的假设，对任意一个 $G$，我们可以将上一步求出的最优判别器 $D*$ 代入到 $C(G)=maxV(G,D)$ 中： 

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/68126image%20(24).png)

因为已知 $-log4$ 为全局最小候选值，所以我们希望构造某个值以使方程式中出现 $log2$。因此我们可以在每个积分中加上或减去 $log2$，并乘上概率密度。这是一个十分常见并且不会改变等式的数学证明技巧，因为本质上我们只是在方程加上了 0。

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/88447image%20(25).png)

采用该技巧主要是希望能够构建成含 log2 和 JS 散度的形式，上式化简后可以得到以下表达式：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/59249image%20(26).png)

因为概率密度的定义，$P_G$ 和 $P_{data}$ 在它们积分域上的积分等于 1，即：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/85412image%20(27).png)

此外，根据对数的定义，我们有：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/25220image%20(28).png)

因此代入该等式，我们可以写为：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/48004image%20(29).png)

现在，如果读者阅读了前文的 KL 散度（Kullback-Leibler divergence），那么我们就会发现每一个积分正好就是它。具体来说：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/68250image%20(30).png)

$KL$ 散度是非负的，所以我们马上就能看出来-log4 为 $C(G)$ 的全局最小值。

如果我们进一步证明只有一个 $G$ 能达到这一个值，因为 $P_G=P_{data}$ 将会成为令 $C(G)=−log4$ 的唯一点，所以整个证明就能完成了。

从前文可知 $KL​$ 散度是非对称的，所以 $C(G)​$ 中的 $KL(P_{data} || (P_{data}+P_G)/2)​$ 左右两项是不能交换的，但如果同时加上另一项 $KL(P_G || (P_{data}+P_G)/2)​$，它们的和就能变成对称项。这两项 $KL​$ 散度的和即可以表示为 $JS​$ 散度（Jenson-Shannon divergence）：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/99310image%20(31).png)

 假设存在两个分布 $P$ 和 $Q$，且这两个分布的平均分布 $M=(P+Q)/2$，那么这两个分布之间的 $JS$ 散度为 $P$ 与 $M$ 之间的 $KL$ 散度加上 $Q$ 与 $M$ 之间的 $KL$ 散度再除以 2。

$JS$ 散度的取值为 0 到 $log2$。若两个分布完全没有交集，那么 JS 散度取最大值 $log2$；若两个分布完全一样，那么 $JS$ 散度取最小值 0。

因此 $C(G)​$ 可以根据 $JS​$ 散度的定义改写为：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/35494image%20(32).png)

这一散度其实就是 Jenson-Shannon 距离度量的平方。根据它的属性：当 $P_G=P_{data}$ 时，$JSD(P_{data}||P_G)$为 ​$0$。综上所述，生成分布当且仅当等于真实数据分布式时，我们可以取得最优生成器。

#### 收敛

现在，该论文的主要部分已经得到了证明：即 $P_G=P_{data}​$ 为 $maxV(G,D)​$ 的最优点。此外，原论文还有额外的证明白表示：给定足够的训练数据和正确的环境，训练过程将收敛到最优 $G​$，我们并不详细讨论这一块。

#### 重述训练过程

下面是推导的最后一步，我们会重述整个参数优化过程，并简要介绍实际训练中涉及的各个过程。

1.参数优化过程

若我们需要寻找最优的生成器，那么给定一个判别器 $D​$，我们可以将 $maxV(G,D)​$ 看作训练生成器的损失函数$ L(G)​$。既然设定了损失函数，那么我们就能使用 SGD、Adam 等优化算法更新生成器 $G​$ 的参数，梯度下降的参数优化过程如下：

$$
\theta_{G} \leftarrow \theta_{G}-\eta \partial L(G) / \partial \theta_{G}
$$

其中求 $L(G)​$ 对$θ_G​$ 的偏导数涉及到求 $max{V(G,D)}​$ 的偏导数，这种对 max 函数求微分的方式是存在且可用的。

现在给定一个初始 $G_0​$，我们需要找到令 $V(G_0,D)​$ 最大的 $D_0*​$，因此判别器更新的过程也就可以看作损失函数为-$V(G,D)​$ 的训练过程。并且由前面的推导可知，$V(G,D)​$ 实际上与分布 $P_{data}(x)​$ 和 $P_G(x)​$ 之间的 $JS​$ 散度只差了一个常数项。因此这样一个循环对抗的过程就能表述为：



- 给定 $G_0$，最大化 $V(G_0,D)$ 以求得 $D_0^*$，即 $max[JSD(P_{data}(x)||P_{G_0}(x)]$；
- 固定 $D_0^*$，计算 $\theta G_1 \leftarrow \theta G_0 - \eta \frac{ \partial(G, D_0^*)}{\partial \theta_G}$  以求得更新后的 $G_1$；
- 固定 $G_1​$，最大化 $V(G_1,D_0^*)​$ 以求得 $D_1^*​$，即 $max[JSD(P_{data}(x)||P_{G_1}(x)]​$；
- 固定 $D_1^*$，计算 $\theta G_2 \leftarrow \theta G_1 - \eta \frac{\partial V(G, D_0^*)}{\partial \theta_G} $ 以求得更新后的 $G_2$；
- ……

2.实际训练过程

根据前面价值函数 $V(G,D)​$ 的定义，我们需要求两个数学期望，即 E[log(D(x))] 和 E[log(1-D(G(z)))]，其中 $x​$ 服从真实数据分布，$z​$ 服从初始化分布。但在实践中，我们是没有办法利用积分求这两个数学期望的，所以一般我们能从无穷的真实数据和无穷的生成器中做采样以逼近真实的数学期望。

若现在给定生成器 $G​$，并希望计算 $maxV(G,D)​$ 以求得判别器 $D​$，那么我们首先需要从 $P_{data}(x)​$ 采样 m 个样本$ {𝑥^1,𝑥^2,…,𝑥^𝑚}​$，从生成器 $P_G(x)​$ 采样 $m​$ 个样本 $\left\{\tilde{x}^{1}, \tilde{x}^{2}, \ldots, \tilde{x}^{m}\right\}​$。因此最大化价值函数 $V(G,D)​$ 就可以使用以下表达式近似替代：
$$
Maxmize \quad \tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log D\left(x^{i}\right)+\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(\tilde{x}^{i}\right)\right)
$$
 若我们需要计算上述的极大化过程，可以采用等价形式的训练方法。若我们有一个二元分类器 $D​$（参数为 $θ_d​$），当然该分类器可以是深度神经网络，那么极大化过程的输出就为该分类器 $D(x)​$。现在我们从 $P_{data}(x)​$ 抽取样本作为正样本，从 $P_G(x)​$ 抽取样本作为负样本，同时将逼近负 $V(G,D)​$ 的函数作为损失函数，因此我们就将其表述为一个标准的二元分类器的训练过程：
$$
\left\{x^{1}, x^{2}, \ldots, x^{m}\right\} \text { from } P_{\text {data}}(x) \rightarrow 正样本
$$

$$
\left\{\tilde{x}^{1}, \tilde{x}^{2}, \ldots, \tilde{x}^{m}\right\} \text { from } P_{G}(x) \rightarrow 负样本
$$

$$
Minimize \quad L=-\frac{1}{m} \sum_{i=1}^{m} \log D\left(x^{i}\right)-\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(\tilde{x}^{i}\right)\right)
$$

在实践中，我们必须使用迭代和数值计算的方法实现极小极大化博弈过程。在训练的内部循环中完整地优化 D 在计算上是不允许的，并且有限的数据集也会导致过拟合。因此我们可以在 k 个优化 D 的步骤和一个优化 G 的步骤间交替进行。那么我们只需慢慢地更新 G，D 就会一直处于最优解的附近，这种策略类似于 SML/PCD 训练的方式。

综上，我们可以描述整个训练过程，对于每一次迭代：

- 从真实数据分布 $P_{data}$ 抽取 m 个样本
- 从先验分布$P_{prior}(z)​$ 抽取 m 个噪声样本
- 将噪声样本投入 G 而生成数据：$\left\{\tilde{x}^{1}, \tilde{x}^{2}, \ldots, \tilde{x}^{m}\right\}, \tilde{x}^{i}=G\left(z^{i}\right)$ ，通过最大化 $V$ 的近似而更新判别器参数$θ_d$，即极大化 $\tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log D\left(x^{i}\right)+\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(\tilde{x}^{i}\right)\right)$ ，且判别器参数的更新迭代式为$\theta_{d} \leftarrow \theta_{d}+\eta \nabla \tilde{V}\left(\theta_{d}\right)$

以上是学习判别器 $D​$ 的过程。因为学习 $D​$ 的过程是计算 $JS​$ 散度的过程，并且我们希望能最大化价值函数，所以该步骤会重复 $k​$ 次。

- 从先验分布 $P_{prior}(z)​$ 中抽取另外 m 个噪声样本 ${z^1,...,z^m}​$

- 通过极小化 $\widetilde{V}$ 而更新生成器参数$θ_g$，即极小化 
  $$
  \tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(G\left(z^{i}\right)\right)\right)
  $$
  且生成器参数的更新迭代式为
  $$
  \theta_{g} \leftarrow \theta_{g}-\eta \nabla \tilde{V}\left(\theta_{g}\right)
  $$

以上是学习生成器参数的过程，这一过程在一次迭代中只会进行一次，因此可以避免更新太多而令$ JS$ 散度上升。

### 实现

在上一期机器之心 GitHub 项目中，我们[从零开始使用 TensorFlow 实现了简单的 CNN](https://www.jiqizhixin.com/articles/2017-08-29-14)，我们不仅介绍了 TensorFlow 基本的操作，并从全连接神经网络开始简单地实现了 LeNet-5。在第一期 GitHub 实现中，我们陆续上传了三段实现代码，第二次上传补充的是全连接网络进行 MNIST 图像识别，我们逐行注释了该模型的所有代码。第三次上传补充的是使用 Keras 构建简单的 CNN，我们同样添加了大量注释。本文是第二期 GitHub 实现，首先提供的是 GAN 实现代码与注释，随后我们会将以上的理论分析与实现代码相结合并展示在 Jupyter Notebook 中。虽然首次实现使用的是比较简单的高级 API（Keras），但后面我们会补充使用 TensorFlow 构建 GAN 的代码与注释。

GitHub 实现地址：<https://github.com/jiqizhixin/ML-Tutorial-Experiment>

机器之心首先使用基于 TensorFlow 后端的 Keras 实现了该生成对抗网络，并且我们在 MNIST 数据集上对模型进行训练并生成了一系列手写字体。这一章节只简要解释部分实现代码，更完整与详细的注释请查看 GitHub 项目地址。

#### 生成模型

首先需要定义一个生成器 $G$，该生成器需要将输入的随机噪声变换为图像。以下是定义的生成模型，该模型首先输入有 100 个元素的向量，该向量随机生成于某分布。随后利用两个全连接层接连将该输入向量扩展到 1024 维和 128*7*7 维，后面就开始将全连接层所产生的一维张量重新塑造成二维张量，即 MNIST 中的灰度图。我们注意到该模型采用的激活函数为 tanh，所以也尝试过将其转换为 relu 函数，但发现生成模型如果转化为 relu 函数，那么它的输出就会成为一片灰色。

由全连接传递的数据会经过几个上采样层和卷积层，我们注意到最后一个卷积层所采用的卷积核为 1，所以经过最后卷积层所生成的图像是一张二维灰度图像，更详细的分析请查看机器之心 GitHub 项目。

```
def generator_model():
    #下面搭建生成器的架构，首先导入序贯模型（sequential），即多个网络层的线性堆叠
    model = Sequential()
    #添加一个全连接层，输入为100维向量，输出为1024维
    model.add(Dense(input_dim=100, output_dim=1024))
    #添加一个激活函数tanh
    model.add(Activation('tanh'))
    #添加一个全连接层，输出为128×7×7维度
    model.add(Dense(128*7*7))
    #添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #Reshape层用来将输入shape转换为特定的shape，将含有128*7*7个元素的向量转化为7×7×128张量
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    #2维上采样层，即将数据的行和列分别重复2次
    model.add(UpSampling2D(size=(2, 2)))
    #添加一个2维卷积层，卷积核大小为5×5，激活函数为tanh，共64个卷积核，并采用padding以保持图像尺寸不变
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    #卷积核设为1即输出图像的维度
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model
```

#### 拼接

前面定义的是可生成图像的模型 $G(z;θ_g)​$，而我们在训练生成模型时，需要固定判别模型 D 以极小化价值函数而寻求更好的生成模型，这就意味着我们需要将生成模型与判别模型拼接在一起，并固定 D 的权重以训练 G 的权重。下面就定义了这一过程，我们先添加前面定义的生成模型，再将定义的判别模型拼接在生成模型下方，并且我们将判别模型设置为不可训练。因此，训练这个组合模型才能真正更新生成模型的参数。

```
def generator_containing_discriminator(g, d):
    #将前面定义的生成器架构和判别器架构组拼接成一个大的神经网络，用于判别生成的图片
    model = Sequential()
    #先添加生成器架构，再令d不可训练，即固定d
    #因此在给定d的情况下训练生成器，即通过将生成的结果投入到判别器进行辨别而优化生成器
    model.add(g)
    d.trainable = False
    model.add(d)
    return model
```

#### 判别模型

判别模型相对来说就是比较传统的图像识别模型，前面我们可以按照经典的方法采用几个卷积层与最大池化层，而后再展开为一维张量并采用几个全连接层作为架构。我们尝试了将 tanh 激活函数改为 relu 激活函数，在前两个 epoch 基本上没有什么明显的变化。

```
def discriminator_model():
    #下面搭建判别器架构，同样采用序贯模型
    model = Sequential()
    
    #添加2维卷积层，卷积核大小为5×5，激活函数为tanh，输入shape在‘channels_first’模式下为（samples,channels，rows，cols）
    #在‘channels_last’模式下为（samples,rows,cols,channels），输出为64维
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    #为空域信号施加最大值池化，pool_size取（2，2）代表使图片在两个维度上均变为原长的一半
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Flatten层把多维输入一维化，常用在从卷积层到全连接层的过渡
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    #一个结点进行二值分类，并采用sigmoid函数的输出作为概念
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
```

#### 训练

训练这一部分比较长，也值得我们进行详细的探讨。总的来说，以下训练过程可简述为：

- 加载 MNIST 数据
- 将数据分割为训练与测试集，并赋值给变量
- 设置训练模型的超参数
- 编译模型的训练过程
- 在每一次迭代内，抽取生成图像与真实图像，并打上标注
- 随后将数据投入到判别模型中，并进行训练与计算损失
- 固定判别模型，训练生成模型并计算损失，结束这一次迭代

以上是下面训练过程的简要介绍，我们将结合上文的理论推导在 GitHub 中展示更详细的分析。

```
def train(BATCH_SIZE):
    
    # 国内好像不能直接导入数据集，我们试了几次都不行，后来将数据集下载到本地'~/.keras/datasets/'，也就是当前目录（我的是用户文件夹下）下的.keras文件夹中。
    #下载的地址为：https://s3.amazonaws.com/img-datasets/mnist.npz
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #iamge_data_format选择"channels_last"或"channels_first"，该选项指定了Keras将要使用的维度顺序。
    #"channels_first"假定2D数据的维度顺序为(channels, rows, cols)，3D数据的维度顺序为(channels, conv_dim1, conv_dim2, conv_dim3)
    
    #转换字段类型，并将数据导入变量中
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    
    #将定义好的模型架构赋值给特定的变量
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    
    #定义生成器模型判别器模型更新所使用的优化算法及超参数
    d_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    
    #编译三个神经网络并设置损失函数和优化算法，其中损失函数都是用的是二元分类交叉熵函数。编译是用来配置模型学习过程的
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    
    #前一个架构在固定判别器的情况下训练了生成器，所以在训练判别器之前先要设定其为可训练。
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    
    #下面在满足epoch条件下进行训练
    for epoch in range(30):
        print("Epoch is", epoch)
        
        #计算一个epoch所需要的迭代数量，即训练样本数除批量大小数的值取整；其中shape[0]就是读取矩阵第一维度的长度
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        
        #在一个epoch内进行迭代训练
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            
            #随机生成的噪声服从均匀分布，且采样下界为-1、采样上界为1，输出BATCH_SIZE×100个样本；即抽取一个批量的随机样本
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            
            #抽取一个批量的真实图片
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
            #生成的图片使用生成器对随机噪声进行推断；verbose为日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录
            generated_images = g.predict(noise, verbose=0)
            
            #每经过100次迭代输出一张生成的图片
            if index % 100 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "./GAN/"+str(epoch)+"_"+str(index)+".png")
            
            #将真实的图片和生成的图片以多维数组的形式拼接在一起，真实图片在上，生成图片在下
            X = np.concatenate((image_batch, generated_images))
            
            #生成图片真假标签，即一个包含两倍批量大小的列表；前一个批量大小都是1，代表真实图片，后一个批量大小都是0，代表伪造图片
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            
            #判别器的损失；在一个batch的数据上进行一次参数更新
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            
            #随机生成的噪声服从均匀分布
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            
            #固定判别器
            d.trainable = False
            
            #计算生成器损失；在一个batch的数据上进行一次参数更新
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            
            #令判别器可训练
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            
            #每100次迭代保存一次生成器和判别器的权重
            if index % 100 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)
```

 

#### 试验

在实践中，我们训练 30 个 epoch 后能得到如下不错的生成结果：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/15268image%20(39)1.png)

当然，中间我们还发现很多训练上的问题，比如说学习率、批量大小、激活函数等。学习率一般我们设置为 0.001 到 0.0005，其它的学习率还有很多没有测试。批量大小我们使用的比较小，例如 16、32、64 等，较小的批量大小可能训练的 epoch 数就不需要那么多。我们发现若将生成模型的激活函数修改为 relu，那么生成的图像很可能会显示为一片灰色，生成模型和判别模型的训练损失可能会表现为：

```
#batch_size=32
batch 2000 d_loss : 0.000318
batch 2000 g_loss : 7.618911
```

以上是在迭代 2000 次后所出现的情况，判别模型的损失一直在下降，而生成模型的损失一直在上升。而正常情况下，我发现生成模型的损失和判别模型的损失会在一定范围内交替上升与下降，而迭代 2000 次后训练损失情况为：

```
#batch_size=36
batch 2000 g_loss : 1.663281
batch 2000 d_loss : 0.483616
```

此外，我们还发现很多出现问题的生成模式，比如说如下生成结果更多是倾向于 0 与 1：

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/99121image%20(39)2.png)

最后，附上我们结束训练的标志。

![img](https://image.jiqizhixin.com/uploads/wangeditor/a9b99e30-7d09-49fc-a400-79a57434c20d/13589image%20(40).png)

参考文献：

- 生成对抗网络原论文：<https://arxiv.org/pdf/1406.2661.pdf>
- Goodfellow NIPS 2016 Tutorial：<https://arxiv.org/abs/1701.00160>
- 李宏毅 MLDS17：<http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html>
- Scott Rome GAN 推导：<http://srome.github.io//An-Annotated-Proof-of-Generative-Adversarial-Networks-with-Implementation-Notes/>