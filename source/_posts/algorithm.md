---
title: algorithm
date: 2019-01-22 10:40:40
categories:
tags:
---

## 一个十进制数分成左右两部分

Palindrome[[1]]回文应用：左右镜像

example: 1123613

result:  left:112， right:316

这里写出回文Palindrome关键代码，x是目标整数。核心思想是分成左右两部分的**位数**是一样的，循环截至时一定是righ**位数**大于left，所以可以界定 when rigntNum > leftNum 即可

```python
revertHalfNum = 0;
while revertHalfNum < x:
    revertHalfNum = revertHalfNum * 10 + x % 10
    x = x // 10
```

注意点：

* 整数为基数位数字
* 提前过滤关键整数：3， 10， 0等

[1]: https://leetcode.com/problems/palindrome-number/	"palindrome"

