---
title: algorithm
date: 2019-01-22 10:40:40
categories:
tags:
---

## 十进制数分成左右两部分

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



## 列出数学式子

有些时候列出数学表达式可以更明了的解决问题

对于罗马数字[[2]]这道题：

XCVII = 97 = -10 + 100 + 5 + 1 + 1

很明了，循环到当前字符s[i]，查询对应数字如果：

roma[s[i]] > roma[s[i+1]]， + roma[s[i]]

roma[s[i]] < roma[s[i+1]]，  - roma[s[i]]



[1]: https://leetcode.com/problems/palindrome-number/	"palindrome"
[2]: https://leetcode.com/problems/roman-to-integer/

