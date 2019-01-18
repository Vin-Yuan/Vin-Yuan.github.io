# Bayesian Estimation

参考文献：https://newonlinecourses.science.psu.edu/stat414/node/241/

频率派和贝叶斯派对于参数$\theta$ 的态度区别是：

频率派：$\theta$ 是一个未知的常量

贝叶斯派：$\theta$ 是一个随机变量

贝叶斯估计通过一个example引入：

考虑一个路口间隔时段T内通过某一区域的车辆数这个样一个问题，可以用到的概率模型是泊松分布。

如果交通控制工程师认为通过这一区域平均数（mean rate) $\lambda$ 为3 或5。prior 于收集数据之前，工程师认为$\lambda = 3$ 比 $\lambda = 5$ 更可能发生，先验概率是：

$P(\lambda = 3) = 0.7$ 和 $P(\lambda = 5) = 0.3$

莫一天，工程师在随机的一个时段T观察到$x = 7$ 辆车通过指定区域。**在这个观察结果下**（即条件概率下），$\lambda = 3$ 和 $\lambda = 5$ 的概率是多少？

通过条件概率我们知道：

 $P(\lambda=3 | X=7) = \frac{P(\lambda=3, X=7)}{P(X=7)}$ 

贝叶斯展开如下：

$P(\lambda=3 | X=7) = \frac{P(\lambda=3)P(X=7| \lambda=3)}{P(\lambda=3)P(X=7| \lambda=3)+P(\lambda=5)P(X=7| \lambda=5)}$

通过查询Possion累计分布函数，得到如下结果：

$P(X=7|\lambda=3)=0.988-0.966=0.022$ 和 $P(X=7|\lambda=5)=0.867-0.762=0.105$

最后目标后验概率：

$P(\lambda=3 | X=7)=\frac{(0.7)(0.022)}{(0.7)(0.022)+(0.3)(0.105)}=\frac{0.0154}{0.0154+0.0315}=0.328 $

