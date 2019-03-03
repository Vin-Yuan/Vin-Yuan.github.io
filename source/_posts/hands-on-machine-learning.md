---
title: hands-on-machine-learning
date: 2019-03-03 16:58:25
categories:
tags:
---

# 03 classification

- 对数据进行**shuffling**, 因为某些算法对数据的顺序比较敏感。但有些数据则需要保持这种顺序，比如说股票或天气数据。

- **np.random.seed(n)**，确保每次随机数相同，经过测试，即使关闭python重新打开程序，使用同样的seed情况下，获取到的随机数仍然保持一致，这对于重现实验结果很有帮助，上一次对trian set 如何shuffling，这一次依旧保持一致。

- 样本的标签使用**tuple**保证在处理数据时不被更改。
- **Precision** and **Recall**

  ![IMG_9F902B26592B-1](https://ws3.sinaimg.cn/large/006tKfTcly1g0pr8bjq49j30dw061q3y.jpg)

recall“召回率”这个含义很好理解，例如召回有质量问题汽车这一情况，算法在判别有问题时偏重检测了发动机，因此最后的结果只召回了发动机有问题的样本，而其他零件有问题的没有召回，最后在统计召回率这一指标中便可发现问题。

权衡准确率和召回率要应对不同问题，比如说过判别疑犯，寻求的是high recall，precison低一些没关系，俗话说宁可错杀千万不可漏掉一个便是如此，当然在这个情况中，进一步二次审查即可，通俗点说，即尽可能找出（“召回”）目标正例；再比如判定一个视频是否对儿童安全，重视的是high precision，recall低一些没关系。