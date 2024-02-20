---
title: object-detection
mathjax: true
date: 2019-07-22 19:10:32
categories: object_detection
tags:  object_detection
---

## Fast-RCNN 中region of interest的映射

首先在原图上用selective search得到某个推荐框，假设推荐目标在一个区域内，这个区域的左上点坐标为（x1,y1），右下角坐标为（x2,y2）。那么按照ZF-5的网络模型
求出 $$S=2*2*2*2$$ ( 所有stride的连续乘积，在conv5之前，包括conv5 ) [^1] 
所以具体落实到feature map上，就应该是（(x1/16)+1,(y1/16)+1）;（(x2/16)-1,(y2/16)-1）。

### Caffe中的实现
我们可以打开roi_pooling_layer层： 
这里的源码是

```c++
int roi_start_w = round(bottom_rois[1] * spatial_scale_);
int roi_start_h = round(bottom_rois[2] * spatial_scale_);
int roi_end_w = round(bottom_rois[3] * spatial_scale_);
int roi_end_h = round(bottom_rois[4] * spatial_scale_);
```

spatial_scale_其实就是stride连续乘积的倒数。这里用的这个round()有点意思，得到小数的最邻近整数，就可以理解为四舍五入，并没有像Spp-Net中所述的左上角+1，右下角-1。我认为这两种方式其实都是可行的。+1或-1更多的是代表防止过界（或者是取整方式的不同），关键还是除以S。

可以理解为在不同维度上对图像的缩放，而stride正好代表了这个缩放因子（看前面特征图大小的计算），所以相应的roi也需要跟着图像缩小这么多倍。

#### 示例：[^2]

在使用fast rcnn以及faster rcnn做检测任务的时候，涉及到从图像的roi区域到feature map中roi的映射，然后再进行roi_pooling之类的操作。
比如图像的大小是（600,800），在经过一系列的卷积以及pooling操作之后在某一个层中得到的feature map大小是（38,50），那么在原图中roi是（30,40,200,400），
在feature map中对应的roi区域应该是

```c++
roi_start_w = round(30 * spatial_scale);
roi_start_h = round(40 * spatial_scale);
roi_end_w = round(200 * spatial_scale);
roi_end_h = round(400 * spatial_scale);
```

其中spatial_scale的计算方式是spatial_scale=round(38/600)=round(50/800)=0.0625，所以在feature map中的roi区域[roi_start_w,roi_start_h,roi_end_w,roi_end_h]=[2,3,13,25];



## Faster R-CNN

论文中提到如果用3x3 的 slice window，其对应到原图的感受野(effetctive receptive field)在VGG和ZF模型上分别是228 pixels，171 pixels。
对于VGG16来说（图片来源:[ kaggle](https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl))
![](http://ww1.sinaimg.cn/large/6bf0a364ly1g59wophostj20m808bac7.jpg)
In Faster-rcnn, the effective receptive field can be calculated as follow (VGG16):
Img->
Conv1(3)->Conv1(3)->Pool1(2) ==>
Conv2(3)->Conv2(3)->Pool2(2) ==>
Conv3(3)->Conv3(3)->Conv3(3)->Pool3(2) ==>
Conv4(3)->Conv4(3)->Conv4(3)->Pool4(2) ==>
Conv5(3)->Conv5(3)->Conv5(3) ====>
a 3 * 3 window in feature map.
Lets take one dimension for simplicity. If we derive back from size 3, the original receptive field:
1). in the beginning of Conv5: 3 + 2 + 2 + 2 = 9
2). in the beginning of Conv4: 9 * 2 + 2 + 2 + 2 = 24
3). in the beginning of Conv3: 24 * 2 + 2 + 2 + 2 = 54
4). in the beginning of Conv2: 54 * 2 + 2 + 2 = 112
5). in the beginning of Conv1 (original input): 112 * 2 + 2 + 2 = 228

[^1]:<http://www.voidcn.com/article/p-rrugdtwl-bps.html>
[^2]: <https://www.cnblogs.com/ymjyqsx/p/7592590.html>