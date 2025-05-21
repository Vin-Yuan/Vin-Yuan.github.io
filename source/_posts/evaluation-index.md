---
title: 机器学习评价指标
mathjax: true
date: 2019-07-20 13:40:09
categories: machine_learning
tags: ROC,  AUC, PR
---

![](http://ww1.sinaimg.cn/mw690/6bf0a364ly1g5692r9t8fj20b40k7jtd.jpg)
以下术语可以参照图[^1]，一目了然：

- Precision(查准率):  $\text{Precision} = \frac{TP}{TP+FP}$
- Recall(查全率):  $\text{Recall} = \frac{TP}{TP+FN} = \text{TPR}$
- AUC : area under curve
- ROC : receiver operating characteristic (**TPR** vs FPR)
  - $TPR = \frac{TP}{TP+FN}$ 图[1] 左半部分
  - $FPR=\frac{FP}{FP+TN}$ 图[1]右半部分
  - $TNR = 1-FPR = \frac{TN}{TN+FP}$
- PR: precision vs recall

如果有人和你说他的model 准确率有91%，那你最好问一下他recall多少！

查全率的大小也是一个很重要的指标，如果一个分类器准确率很高查全率却很低的是没有意义的。

例如你有一个行人识别的分类器，来了50个object，你将10个识别为行人，而ground truth中这10个也确定是行人，从准确率来说你会觉得很赞，100%耶:smile:。但是，实际情况是这50个都是行人，如果这是一个自动驾驶的识别系统的话，:scream:那就很糟糕了。形成这样的原因很可能是模型过拟合了。

往往对于一个问题我们的关注点不同，侧重的指标也就不同。

<!--more-->

垃圾邮件的识别过滤，有时候会更关注查全率，再加之与用户甄别。但如果准确率太低，把重要邮件过滤掉，那就不妙了:joy:

就癌症识别的初步阶段来说，我们更关注查全率，以起到防患于未然，毕竟世界上最美好的事情莫过于虚惊一场。

“宁可错杀一千也不放过一个”这句话也是更关注查全率。

所以，问题不同，侧重的也会不同。

基于precision和recall，我们可以绘制一个曲线，即PR曲线，基于TPR和FPR可以绘制ROC曲线，如下图[^2]：

![](http://ww1.sinaimg.cn/large/6bf0a364ly1g56bsodiryj20kf0djq4e.jpg)

PR曲线横轴是precision纵轴是recall。

AUC 并不是指某种指标，而是指曲线下面的面积（Area Under Curve)，所以对于PR曲线，ROC曲线，都有相对应的AUC，即

PR-AUC, ROC-AUC。

ROC 越靠左上角模型效果越好，PR则是右上角，同时两种趋向也会使得AUC接近1。

通过观察PR曲线的表达式，发现PR指标更**关注正例（Focus Positive)**，如果样本不均衡则更关注正例。

在观察ROC曲线的表达式，发现ROC同时考虑了正例和负例，比较公平均衡。

e.g.
举了个例子，(比如说positive 20, negative 1000), 负例增加了10倍，ROC曲线没有改变，而PR曲线则变了很多。作者认为这是ROC曲线的优点，即具有鲁棒性，在类别分布发生明显改变的情况下依然能客观地识别出较好的分类器 (可以画个图更直观)。

## 总结

ROC同时考虑了postivie和negative, 适用于评估整体分类器的性能，而PR则更侧重于positive

如果有多份数据且存在**不同**的类别分布，比如信用卡欺诈问题中每个月正例和负例的比例可能都不相同，这时候如果只想单纯地比较分类器的性能且剔除类别分布改变的影响，则ROC曲线比较适合，因为类别分布改变可能使得PR曲线发生变化时好时坏，这种时候难以进行模型比较；反之，如果想测试不同类别分布下对分类器的性能的影响，则PR曲线比较适合[^3]

如果想要评估在同一分布下正例的预测情况，选PR曲线。

类别不平衡问题中，ROC曲线通常会给出一个乐观的效果估计，所以大部分时候还是PR曲线更好。

最后可以根据具体的应用，在曲线上找到最优的点，得到相对应的precision，recall，f1 score等指标，去调整模型的阈值，从而得到一个符合具体应用的模型[^3]。

如果你关注positive多于negative，则使用PR，otherwise 使用 ROC[^4]


[^1]: https://en.wikipedia.org/wiki/F1_score
[^2]:[The Relationship Between Precision-Recall and ROC Curves](http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)

[^3]: https://www.imooc.com/article/48072>
[^4]: Hands-On Machine Learning with Scikit-Learn & TensorFlow p92