---
title: matplotlib
mathjax: true
date: 2024-02-29 19:10:11
categories:
tags:
---



## matplotlib

### pyplot

&emsp;pyplot负责绘制图像，修饰图像figure。此处应强调的是，其保持matlab的风格，总是跟踪当前figure,绘制函数直接指向当前axes.
&emsp;figure()函数负责创建一个图像，默认不用调用此函数，并且一个subplot(111)也会默认被创建如果不手动指定axes的话。figure(i)创建标号为i的figure

### 结构知识

[plt, axes, figure之间的关系](https://zhuanlan.zhihu.com/p/93423829)

![此处输入图片的描述][1]
![此处输入图片的描述][2]

### 绘制子图

<https://blog.csdn.net/You_are_my_dream/article/details/53439518>
dataFrame绘制在子图上
<https://blog.csdn.net/htuhxf/article/details/82986440?spm=1001.2014.3001.5502>

```python
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle("whole title")
# 等价于：
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1)
fig.suptitle("whole title")

```

- 使用subplots (**Template** !!!)

```python
import matplotlib.pyplot as plt

# data
curve_list = []
x = np.arange(1,100)
curve_list.append((x, x))
curve_list.append((x, -x))
curve_list.append((x, x**2))
curve_list.append((x, np.log(x)))

# sub window nums
n = len(curve_list)
column_num = 2
row_num = int(np.ceil(n/column_num))

# construct figure and pre split
fig, axes = plt.subplots(row_num, column_num, figsize=(10,10))
fig.suptitle("curve show")

# draw sub plot on figure
index = 0
for row in axes:
    for ax in row:
        x, y = curve_list[index]
        ax.set_title("curve_{}".format(index))
        # plot 时设置 label, 颜色, 线宽度等
        ax.plot(x, y, 'C{}'.format(index), label='curve{}'.format(index), linewidth=2)
        ax.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
        index += 1 
        # ax 显示label, 可设置显示位置
        ax.legend(loc='lower right', fontsize=10)   
```

- subplots和DataFrame结合
<https://blog.csdn.net/htuhxf/article/details/82986440?spm=1001.2014.3001.5502>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
fig, axes = plt.subplots(2, 2)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot.bar(ax=axes[1,1], color='b', alpha = 0.5)
data.plot.barh(ax=axes[0,1], color='k', alpha=0.5)
```

- 使用add_subplot

```python
import numpy as np
import matplotlib.pyplot as plt
 
x = np.arange(1, 100)
# first you have to make the figure
fig = plt.figure()
# now you have to create each subplot individually
ax1 = fig.add_subplot(221)
ax1.plot(x, x)
ax2 = fig.add_subplot(222)
ax2.plot(x, -x)
ax3 = fig.add_subplot(223)
ax3.plot(x, x ** 2)
ax4 = fig.add_subplot(224)
ax4.plot(x, np.log(x))
plt.show()
```

多个子图的模板

```python
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

code [link][3]
![此处输入图片的描述][4]

- 使用subplot

```python
#!/usr/bin/python
#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 100)
# first you have to make the figure
fig = plt.figure(1) 
# now you have to create each subplot individually
plt.subplot(221)
plt.plot(x, x)
plt.subplot(222)
plt.plot(x, -x)
plt.subplot(223)
plt.plot(x, x ** 2)
plt.subplot(224)
plt.plot(x, np.log(x))
# or
# ax1 = plt.subplot(221)
# ax1.plot(x, x)
plt.show()
```

### 定制legend

场景，例如在绘制曲线的是后想绘制每个曲线的极值点，同事legend是"曲线名+极值点value"这种需求，可以使用custom legend
普通plt.plot(x,y)会返回绘制的曲线，是Line2D object, 获取这些objects后，使用ax.legend(lines, legends, loc='lower right')可以定制legend
例如

```python
# simple example
lines = plt.plot(data)
plt.legend(lines, ['line1','line2','lin3'])

# another simple example
from numpy.random import randn
z = randn(10)
red_dot, = plt.plot(z, "ro", markersize=15)
# Put a white cross over some of the data.
white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)
plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])

# practice example
# 注意dataframe.plot不返回Line2D对象，如果需要曲线对象，需要使用dataframe.plot.line(), 最好还是使用原始的ax.plot()分column绘制，这样好控制一些
lines = []
legends = []
for idx, metric_type in enumerate(metrics_type_column):
    data = df[metric_type]
    x, y = data.idxmax(), data.max()
    line = ax.plot(data, c='C%d'%(idx))
    lines.extend(line)
    ax.scatter(x, y, c='C%d'%(idx), alpha=0.5)
    legends.append("%s max=%.4f"%(metric_type.replace("bf_", ""), y))
ax.legend(lines, legends, loc='lower right')
```

#### Legend for Size of Points

<https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html>

### 绘制gauss曲线

<http://emredjan.github.io/blog/2017/07/19/plotting-distributions/>

### 绘制activation 函数曲线

```python
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
# data
x = np.arange(-3,3,0.1)
y1 = np.log(np.exp(x)+ 1) 
y2 = np.maximum(0, x)
# draw
fig = plt.figure()
fig.suptitle("activation function")
axes = fig.add_subplot()
axes.plot(x, y1, color='C0', label='softplus')
axes.plot(x, y2, color='C1', label='relu')
plt.legend()
#axes.set_ylim((-1,3))
#ax.grid()
#axes.set_yticks(np.arange(-1, max(y+1), 1))
#axes.set_xticks(np.arange(-3, max(x+1), 1))
# 注意set_yticks会造成图空出一部分，函数说明也很明显：If necessary, the view limits of the Axis are expanded so that all given ticks are visible.
axes.axis([-3,3,-1,3])
# axis 函数则直接会截取到指定区间，不会空余出margin
axes.grid(linestyle='--', linewidth=1,alpha=0.3)
axes.axvline(x=0, linewidth=1, color='black',alpha=0.6)
axes.axhline(y=0, linewidth=1,  color='black',alpha=0.6)
```
