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

<!-- more -->

## 列出数学式子

有些时候列出数学表达式可以更明了的解决问题

对于罗马数字[[2]]这道题：

XCVII = 97 = -10 + 100 + 5 + 1 + 1

很明了，循环到当前字符s[i]，查询对应数字如果：

roma[s[i]] > roma[s[i+1]]， + roma[s[i]]

roma[s[i]] < roma[s[i+1]]，  - roma[s[i]]

## common

### ASCI 码值

在一些需要字符加减的地方获取asci码值

```python
ord('A') - ord('B')
>> 1
```

### map函数

```python
a = [1,2,3,4]
b = map(lambda x: x ** 2, a)
```

b在这里是一个map对象，可迭代对象，如果直接回显变量或print变量是无法看到值的，如果想查看内部值，使用list,这里list更像一个“大散”的操作：

```python
>>> list(b)
>>> [1,2,3,4]
```

### sort and sorted

- sort 原地排序 inplace
* sorted 返回一个新列表

### 赋值语句

```python
# error 不可以这样赋值
i = 1, j = 2 #  SyntaxError: can't assign to literal
# right 
i = 1 
j = 2 
```

### if 条件语句

```python
# error python if 
if !s.isnumeric() or !s.isalpha():
  ……
# right python if 
if not s.isnumeric() or not s.isalpha():
  ……
```

### bit位操作

#### 移位

bit << 1
bit <<= 1

#### 按位与

bit= 0010000
判断num某一位是否是1要用
if bit & num != 0:
**不要使用是否等于1作判断**，相关题目剑指 Offer 56 - I. 数组中数字出现的次数

#### 按位或

bit|num

#### 按位亦或

bit^num
num ^= bit

#### 结合

查看遇到的第一个为1的bit位( **注意使用<<=而非<<** )

```
bit = 1
num = 10
while num & bit == 0:
    bit <<= 1
```

12 = 00001100
12 & 00000001 = 0
12 & 00000010 = 0
**12 & 00000100 = 4**

### 循环完毕的标志

如果需要判断子串是否比较完毕，可以使用while i < len(s)，这样当都比较完的时候，i就越界了，然后可以拿 i == len(s) 来判断是否比较完毕，这里使用for i in range(0, len(s)) 不合适。参见28

### python bit 操作

```python
# 按位与 AND
a & b
# 按位或 OR
a | b
# 按位异或 XOR
a ^ b
# 按位取反
~ a
```

### python 二维数组

```
n = 3
m = 3
dp = [[0 for i in range(n)] for j in range(m)]
```

arr2 = [arr1] * 3操作中，只是创建3个指向arr1的引用，所以一旦arr1改变，arr2中3个list也会随之改变，故这种方式不合适。

### 双段队列

参照 剑指 Offer 59 - I. 滑动窗口的最大值
调整双端队列时：

```
while len(deque) > 0 and '不满足条件'：
    deque.pop()
队列调整完毕，可以放入当前元素
```

### 双指针解法(快慢指针）

设计题目：
5. 3sum
11. Container With Most Water
这种题目的特点是涉及到求两个数组元素的函数$f(a_i, a_j)$（和，max等）
可以使用左右指针向中间移动，**符合趋向预期目标的指针移动**
剑指offer 22

* 左右指针

```c
int left = 0, right = nums.size() - 1;
while (left < right) {
    if ((nums[left] & 1) != 0) {
        left ++;
        continue;
    }
    if ((nums[right] & 1) != 1) {
        right --;
        continue;
    }
    swap(nums[left++], nums[right--]);
}
```

* 快慢指针

```c
int low = 0, fast = 0;
while (fast < nums.size()) {
    if (nums[fast] & 1) {
        swap(nums[low], nums[fast]);
        low ++;
    }
    fast ++;
}
```

#### 快排

```python
def partition(nums, start, end):
    index = random(start, end)
    swap(nums[index], nums[end])
    small = start-1
    index = start
    while index < end:
        if nums[index] < nums[end]:
            small += 1
            if small != index:
                swap(nums[small], nums[index])
        index += 1
    small += 1
    swap(nums[small], nums[index])
    return small
    
def quickSort(nums, start, end):
    index = partition(nums, start, end)
    if index > start:
        quickSort(nums, start, index-1)
    if index < end:
        quickSort(nums, index+1, end)
```

### 可以hash的类型

tuple可以作为hash的键值（set等数据结构）,list 不可以，**不可变等可以作为hasn的键值**
defaultdict可以在没有的时候返回默认值

```python
res = collecitons.defaultdict()
```

### 集合的排序

```python
x = [[1,3],[2,6],[8,10],[15,18]]
x.sort(key=lambda x:x[0])
```

### 栈和队列

```python
# 栈
stack = []
stack.append(1)
top = stack.pop()

# 队列1
queue = []
queue.append(1)
forehead = queue.pop(0)
# 队列2
queue = collections.deque()
queue.append(root)
node = queue.popleft()
len(queue) == 0判断是否为空
```

### 树

二叉树的遍历分为前序、后序、中序：

#### 1.前序遍历

```python
while stack:
    root = stack.pop()
    if root is not None:
        output.append(root.val)
        if root.right is not None:
            stack.append(root.right)
        if root.left is not None:
            stack.append(root.left)
```

#### 2.后续遍历

```python
while stack:
    root = stack.pop()
    output.append(root.val)
    if root.left is not None:
        stack.append(root.left)
    if root.right is not None:
        stack.append(root.right)
return output[::-1]
```

#### 3.中序遍历

值得注意的几个点：

* 判断条件设置为只要**栈不空或者当前节点不为空**，则继续**while**循环
* 刚开始栈为空，初始化cur = root
* 中序遍历和其他两个遍历稍有不同，分为两个condition: 如果当前节点为空，说明要弹出栈顶元素；如果不为空，就一直压栈其left节点

```python
stack = []
cur = root
while len(stack) > 0 or cur != None:
    while cur != None:
        stack.append(cur)
        cur = cur.left
    cur = stack.pop()
    # visited current node
    cur = cur.right
return -1
```

涉及到的题目：
[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal)
[230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)
[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)
[145. 145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal)

### 深度优先搜索

1.dfs
2.dfs + 剪枝

```
dfs(i,j):
    if 终止条件（越界or不符合探测条件)
        return False
    if 找到解
        return True
    set flag[i][j] = use
    res = 对i,j的相邻结点做dfs
    set flag[i][j] = free
    return 
```

剑指offer 12,13,14

### 动态规划

动态规划通常用来求最优解。能用动态规划解决的求最优解问题，必须满足最优解的每个**局部解也都是最优的**

### 递归（回溯，深度优先，层次优先，树的遍历）

#### 1. 不需要返回值

递归的过程只应对当前处理元素，比较独立，不涉及到前面和后面递归的结果
字符串全排列
最小k个数
快排

#### 2. 需要返回值

## Problem

### 2. Add Two Numbers

注意点：

1. 迭代方式，head节点只作为记录标志，不存储实际值
2. 链表已到头则操作数设置为0，可以避免判断链表是否已经到头
3. **每生成一个结果存储一个节点，处理的是currentNode**
4. while循环最后的遍历操作
5. 收尾操作，最后溢出情况
6. 返回的是实际节点，head节点要丢弃

```python
while  p != None and q != None:
    sum = (p.val + q.val + carry)
    carry = (1 if sum // 10 > 0 else 0)
    node.next = ListNode(sum % 10)
    node = node.next
    p = p.next
```

### 3. Longest Substring Without Repeating Characters

两个条件分支要明确功能，（1）在Map表里的做什么，（2）不在Map表里的做什么
并且在各自分支中又要分成什么功能等等。
（1） **只更新在滑动窗口里的值**
（2）比较最长结果
python的解法：

1. set对象存储不重复字符子串，方便之处是可以直接删除指定字符而不需要只到具体位置，因为set是无序的
2. 依次遍历的是right字符串，核查重复的条件也是right字符串，调整窗口大小的是left，如果不使用set而是list，需从头部删除字符，还需要一个变量记录子串的index

### 5. 最长回文子串

如果使用更新dp二维数组，需要考虑动态规划的边界条件
dp[i, j] = (df[i+1, j-1]) and (s[i]==s[j])
中的递归项：df[i+1, j-1]
试着画一下图，看一下那些递归时出现问题，明显会看到dp[i,i+1]递归项是invalid的，因为我们只初始化了右上角元素，左下角元素是invalid的，不能作为判断条件

### 6. Z字形变换

注意点：
一个flag控制+1还是-1，初始flag是True还是False
考虑边界情况：
（1）行数为1或行数为2
（2）字符数小于行数

### 7. 整数反转

重点：需要考虑边界值
由于python的求余数在负数的时候有所不同，可以考虑将x转换成整数abs(x)然后分正负号判断处理是否溢出
一个很好的点子：求余数以的过程类似栈的弹出

```
//pop operation:
pop = x % 10;
x /= 10;

//push operation:
ans = ans * 10 + pop;
```

### 8. 字符串转换整数 (atoi)

当涉及到读字符然后分情况处理时可以考虑状态机
状态机要素：
（1）状态：start, sign, digital, end
（2）事件：读到字符char
（3）动作：遇到字符char后需要做的动作，0：空字符，1：正负号，2：数字，3：其他字符
（4）状态转移矩阵：action 0转移到那个状态，action 1转移到那个状态……

action|' '|+/-|number|other
--|--|--|--|--|--
start|start|signed|in_number|end
signed|end|end|in_number|end
in_number|end|end|in_number|end
end|end|end|end|end

### 15. 3Sum

 1. 当产生循环的时候，每一个if条件都要记得更新指针 i, j 等，注意边界条件
 2. 题目要求要明确，唯一的triplets!
 3. 可以证明寻求 num[l]+num[r] = num[i] 的2sum要开始于i的下一个元素i+1而不是i=1，因为已经排序了，从头开始会有重复
 4. 考虑重复问题不仅要考虑开头元素num[i],还要靠后面两个元素num[l], nums[r]都可能出现重复元素！

### 20 Valid Parentheses

最后要检查stack是否为空，空才是合法的。

也可以往栈里放 "), ], }"，只判断closed character

```python
map = ["(":")", "{":"}", "[":"]"]
# 遍历、查看索引直接用 
# for x in map
# if x in map ...
```

### 22. Generate Parentheses

注意一点，右括号一定要小于左括号个数，对于这种序列
"())(()"是可以剪枝避免的

### 28. Implement strStr()

如果需要判断子串是否比较完毕，可以使用`while i < len(s)`，这样当都比较完的时候，i就越界了，然后可以拿 i == len(s) 来判断是否比较完毕，这里使用`for i in range(0, len(s))`不合适。

### 29. Divide Two Integers

dvd = dvs * n + remain
原始的想法是从1开始直到n次来用dvs减去dvd，直到dvd <  dvs后, 这时候已经不够减，商也就得到了。
但是优化的角度去想，n其实可以用指数逼近（这样会将问题减少一半）
如果迭代到 $n > 2^k \text{ and } n < 2^{(k+1)}$ 这时候， $n-2^k$ 大致会减少一半，可以理解成二分法，这样分治策略后达到简化问题的目的。

减去一般后在同样进行如此操作（是一个递归过程，不过可以用循环去做)

溢出只有一种情况：当被除数是最小负数，除数是-1时。

**移位操作一定要赋值!**

```python
i = i << 1
# 或者
i <<= 1
```

另外判断分子分母是否同号可以如下，很巧妙：

```python
isNegative = (up > 0) == (down > 0)
```

### 33. Search in Rotated Sorted Array

注意边界条件！
题目中已经声明没有重复元素
可能有等号的情况一定要注意！如果遗漏会使得元素被跳过引发错误
注意分类细节：
每一个if分支，首先要明确主要矛盾，确定mid的位置区域是首要任务，然后再确定target的未知区域，不要混淆了先后！
再次做题更新的领悟点：
主要两个，且这些都是基于题目的约束条件，**nums中的每个值都独一无二**

1. mid位置是在左边还是右边
2. target是在左有序部分还是右有序部分

例如：

```
if nums[0] <= target and target <= nums[mid]:
    r = mid-1
else:
    l = mid+1
```

这个条件，成立target则在mid左面，**不成立则target一定在mid右边**，不是因为别的，只因为这两个情况是互斥的，同时nums中每个值独一无二。

### 34. Find First and Last Position of Element in Sorted Array

这道题巧妙的地方在于我们寻求的目的：寻找边界。
所以二分查找并不是为了target的位置，而是begin和end
另外

```python
# 这种方式求得中间值是下取整
mid = (low + high) // 2
(2+5)//2 = 3
(2+4)//2 = 3
# 这种方式求得中间值是上取整
mid = (low + high + 1) // 2
(2+5+1)//2 = 4
(2+4+1)//2 = 3
```

### 46. permute

#### 回溯算法

回溯算法：通过探索所有可能的候选解来找出问题的解。
回溯会在每一步进行一些改变，尝试递归到下一步：
如果是可行解，便将其加入结果集。
如果不是则回溯尝试下一个改变。
至于回退的方式也会分两种：1.值传递 2.恢复现场

##### 1. 值传递

值传递的好处是不用恢复由于回溯而造成的修改，缺点是调用传递参数拷贝所带来的效率问题，这里要注意**不要传递对象的引用**

```
changeSth(trySolution)
backtracking(solution=trySolution)
unchangeSth(trySolution)
```

有两种回溯选择，虽然回溯的本质一样，但可以看到第二种，内存使用会小一些，同时由于是在change in place，传参数的时候效率更高一些
回溯递归一定要注意传参数，回溯的本质就是向下探索可能的道路，所以有些变量是需要传值，生成新的变量，而不是在原有变量上作改动。

```python
for i,v in enumerate(remain):
 self.backtracking(path+[v], remain[:i]+remain[i+1:], res)
```

和下面这种结果会是截然相反的！

```python
for i,v in enumerate(remain):
 path.append(v)
 self.backtracking(path, remain[:i]+remain[i+1:], res)
```

##### 2.恢复现场

```python
nums[index], nums[i] = nums[i], nums[index]
self.backtracking(index+1, nums)
nums[index], nums[i] = nums[i], nums[index]
```

恢复现场在共享对象上作了恢复现场的处理，这也意味着在保存找到的解上要格外注意，要保存副本而不是共享对象，因为如果保存的对象则会导致结果是一样的。
下面这种：

```python
if index == n-1:
  self.res.append(nums)
  return
```

是有问题的，这样每次保存的是同一个共享对象，所有递归都会更改着个nums对象，导致最后发现结果都是一个解：最后一次修改的结果！
正确的做法应该是使用：

```python
self.res.append(nums)
```

### 48. rotate image

#### 旋转矩阵

有两种方法，一种是先转置，然后垂直镜像；另一种是直观的顺时针旋转，但其实这两种后面的数学本质是一样的：

#### 1.转置+镜像

经历了两次函数变换$f=\text{transposition}, g=\text{mirroring}$
$$
(i,j)\xrightarrow{f}(j,i)\xrightarrow{g}(j,n-1-i)
$$

#### 2.直观旋转

$$\begin{align}
t &= m_{i,\;j} \\
m_{i,\;j} &= m_{n-1-j,\; i} \\
m_{n-i-j,\;i} &= m_{n-1-i,\;n-1-j} \\
m_{n-1-i,\;n-1-j} &= m_{j,\;n-1-i} \\
m_{j,\;n-1-i} &= t
\end{align}
$$
对比发现其实两者是一样的

### 49. 字母异位词分组

目的是让“abc", "acb", "bac" 放入同一个集合
做到这样需要一个hash方法，将字母异位词映射到同一个hash值
$$
f(\text{"abc"}) = f(\text{"bac"}) = \text{sth}
$$

* bag of character
对于bag of character，相当于统计了字母的“字频”，形成了一个唯一的向量来表示hash的键值
* code(abc) 唯一
给每一个字母一个编码，对每个单词字母编码进行运算能保证唯一$g(h('a'), h('b'), h('c'))$，这里用到素数相关的“正整数的唯一分解定理”：

>算术基本定理，又称为正整数的唯一分解定理，即：每个大于1的自然数，要么本身就是质数，要么可以写为2个或以上的质数的积，而且这些质因子按大小排列之后，写法仅有一种方式。

### 50. Pow(x, n)

$x^n = x^{(n)_2} = x^{110...1}$
其中
$n = a_k\cdot2^k + a_{k-1}\cdot2^{k-1} + ...+a_0\cdot2^0$
**这样就将一个问题吹处理成循环所有bit位，而数字n刚好有$\log n$位bit，所以问题的规模也就变成了$O(\log n)$**
值得注意的是递归式子（草稿纸上写好数学表达式会非常有帮助，例如roate image)：
$$
f(k) = X^{2^{k+1}} = X^{(2^k\cdot 2)} = (X^{2^k})^2 = f^2(k-1)
$$

### 53. Maximum Subarray

求最大和连续子序列，考虑动态规划的思想
$$
f(i)=\left\{

\begin{aligned}

&data[i] &i = 0 \text{ 或者 } f(i-1) \le 0\\

&data[i] + f(i-1) &i > 0 \text{并且} f(i-1) > 0\\

\end{aligned}

\right.
$$
并且，由于只使用f(i-1), 所以我们可以只声明一个变量来存f(i-1)，而不必申请一个数组来存动态规划的值。

### 54. Spiral Matrix

注意边界条件，内部最后一圈：

* 情况1

.|.|.
-|-|-
.|a|.
.|.|.

* 情况2

.|.|.|.
-|-|-|-
.|a|b|.
.|.|.|.

* 情况3

.|.|.
-|-|-
.|a|.
.|b|.
.|.|.

### 55. Jump Game

这是一个动态规划问题，通常解决并理解一个动态规划问题需要以下 4 个步骤：

* 利用递归回溯解决问题
* 利用记忆表优化（自顶向下的动态规划）
* 移除递归的部分（自底向上的动态规划）
* 使用技巧减少时间和空间复杂度

> 一个快速的优化方法是我们可以从右到左的检查 nextposition ，理论上最坏的时间复杂度复杂度是一样的。但实际情况下，对于一些简单场景，这个代码可能跑得更快一些。**直觉上，就是我们每次选择最大的步数去跳跃，这样就可以更快的到达终点**。

>底向上和自顶向下动态规划的区别就是消除了回溯，在实际使用中，自底向下的方法有更好的时间效率因为我们不再需要栈空间，可以节省很多缓存开销。更重要的事，这可以让之后更有优化的空间。回溯通常是通过反转动态规划的步骤来实现的。

从自顶向下到自底向上到优化过程中会发现：
> 底向上和自顶向下动态规划的区别就是消除了回溯，在实际使用中，自底向下的方法有更好的时间效率因为我们不再需要栈空间，可以节省很多缓存开销。更重要的事，这可以让之后更有优化的空间。**回溯通常是通过反转动态规划的步骤来实现的**。
这是由于我们每次只会向右跳动，意味着如果我们从右边开始动态规划，每次查询右边节点的信息，都是已经计算过了的，不再需要额外的递归开销，因为我们每次在 memo 表中都可以找到结果。

实际试验时：不优化的自顶向下会超时，但自底向上会accept，
贪心算法可能一时不会想到，动态规划可以一步步优化攻破。

**这个问题是不是动态规划问题？**

1. 问题的最优解包含子问题的最优解(从这道题看出，**最优**不需要太严格）
当前点在其可到达范围内的点是否有能到达终点的？
2. 重叠子问题

$$
\text{OPT[i]} = \{ \text{OPT(i+1)},...,\text{OPT(i+maxStep)} \}\; \text{if has True}
$$

### 56. Merge Intervals

python可迭代对象排序

```python
x = [[1,3],[2,6],[8,10],[15,18]]
intervals.sort(key=lambda x:x[0])
```

### 66. Plus One

在纸上演示一下情况，然后判断变化条件，模拟思维流程，只要不进位（小于9）就结束，否则，一直进位下去。

反向便利数组、或实现`for(int i = n-1; i >= 0; i—)`

```python
# 取负数，python中可以使用负数索引，代表倒数第几个。
for i in range(len(digits)):
  if digits[~i] < 9:
    digits[~i] += 1
    return digits
  digits[~i] = 0
```

### 69. Sqrt(x)

Newton 迭代法的原理其实是直线逼近曲线的方法：[^1]

直线方程：知道斜率k，一点$(x_0, y_0)$，求直线方程：
$$
\begin{aligned}
y = k(x - x_0) + y_0
\end{aligned}
$$

方法是 $k(x - x_0) = y - y_0 \\$ 将斜率等式化即可。

### 75. Sort Colors

用两个指针p0, p2实现的时候要注意, curr在移动的时候
----p0------p2-----
p0左面是0， p2右面是2，**$A_i$ 发现是0到时候不光移动p0，还需要移动 $i$**, 这里面值得注意的是 $A_i = 1$的时候不作关注，处理的只是0和2

### 78. Subsets

两个方法：

1. backtracking
2. 枚举
**需要注意的是，如果写成下面会形成死循环**，需要借助一个临时变量！

```python
 for x in res:
    res.append(c+x)
```

谨慎对待循环修改自身的情况

### 79. Word Search

注意几点：
上下左右递归之前可以筛掉一些，过滤不必要的递归，虽然递归之后有条件判断会返回，但递归进函数也是调用效率的浪费。

返回条件一定要想全了，如果直到word没有了才返回就要注意了，如果最后一个字符word[cur] 判断成功，在递归cur+1的时候要先判断word的cur而不是越界了，然后才是 i,j是否合法。

* **递归的首要条件是cur是否已经到判断完了，所以递归出口的第一个条件一定要判断cur >= len(word)**，主次之分要明确，然后才是i, j 合法性！

* 还有就是当上下左右只要有一个可行，就立马返回true,不需要递归下面的尝试，问题性质：**只有所有都不行才返回False, 只要有一个可行就返回True**
的可以用如下结构

```python
 for i, j in attemp_set:
  if backtracking(i,j...):
   return True
 return False
```

* 设置访问标志后一定要记得恢复现场！

### 88. Merge Sorted Array

原地置换，合并排序。还是遵循while循环每一次只做一件事，当两个序列都没截止时一直比较。边界条件时退出，额外处理剩余子序列，剩余子序列替换可以使用`nums1[:n] = nums2[:n]`

### 91. Decode Ways

特殊点需要仔细观察处理：分情况明确，最好在纸上把条件逐步分细致了，不要担心代码过长。
0  x
1
00 x
01 x
10 -
20 -
26
27
30 x

```python
if s[i-1] == '0':
    if s[i] == '0':
        return 0
    else:
        = step[i-1]
elif s[i-1] == '1':
    if s[i] == '0':
        = step[i-2]
    else:
        = step[i-2] + step[i-1]
elif s[i-1] == '2':
    if s[i] == '0':
        = step[i-2]
    else if int(s[i]) <= 6:
        = step[i-2] + step[i-1]
    else:
        = step[i-1]
else:
    if s[i] == 0:
        return 0
    else:
        = step[i-1]
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

根据前序和中序序列来构建二叉树：
(1,2,4,3,5,6,7)            preorder
 ^ l l r r r r
(4,2,1,5,3,7,6)            inorder
 l l ^ r r r r
 在中序序列中找到根节点后，左右子树的size也就知道了，分别是mid和n-1-mid+1=n-mid, 对应到前序序列里是可以界定哪部分是左子树，哪部分是右子树。

```python
root.left = self.buildTree(preorder[1:mid+1], inorder[0:mid])
root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])   
```

### 121. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock)

动态规划可以这样想：
$$
OPT(i) =
\left\{
\begin{aligned}
&\text{prices}[i]-min(\text{prices}[0:i]),&\text{sell}\ \text{stock}[i] \\
&OPT(i-1),& \text{do nothing}
\end{aligned}
\right.
$$

### 125. Valid Palindrome

判断是数字和字符

python

```python
s = "123AbcD"
x = s[0]
a.isalnum() # 字母or数字
a.isnumeric() # 数字
a.isalpha() # 字母
a.lower() # 转换小写
a.upper() # 转换大写
```

java

```java
Character.isLetterOrDigit(s.charAt(l))
Character.toLowerCase(s.charAt(l))
```

对于这类问题，一个大的while 里面做循环，**每次循环只做一个操作**

1. 非数字、字符，右移
2. 非数字、字符，左移
3. 可比较，比较

这样就不必考虑while(++ch)知道遇到可比较字符的问题，条理清晰一些

### 127. Word Ladder

对于广度优先搜索 BFS(breadth first search)来说，但虽然都是使用队列来辅助搜索，视问题不同代码逻辑稍有不同：

#### 1. 需要对同一层做批次操作

```python
while queue is not empty:
    level_next = []
    for node in queue:
        #process...
        level_next.append(node.relate_nodes())
    queue = level_next
```

相关题目
102. Binary Tree Level Order Traversal
116. Populating Next Right Pointers in Each Node

#### 2. 仅仅是遍历

例如本题以及二叉树层次遍历

```python
while queue is not empty:
    top = queue.pop()
    # process top
    # 必要时需要visited判断是否访问过top
    queue.append(top.relate_nodes())
```

当然本题也可以用上面对批次处理，这样处理一层depth+1，省去每次将深度存入queue里。
初始做这道题时会有个疑问，遍历某一层时如果将其中一个元素visited置为访问过，那之后是否需要释放？不释放的话如果后来的节点可以到达此节点不是不能用了吗？其实不然，广度优先搜索的特点就是不可能出现这样情况，可以用反证法来证明：如果有的话那就不是最短路径了！

### 131. Palindrome Partitioning

草稿纸上画一下回溯递归的过程

```python
s = "aab"
a|ab
a|a|b
aa|b

s = "aaa"
aaa
a|aa
a|a|a
aa|a
```

即使当前字符串已经valid加入结果集后，还需要对其划分

```python
if reamin is Palindrome:
(1). valid + remain
(2). valid + partition(remain)
```

valid是一样的，但后半段的划分也是解的一部分，直接在（1）返回会丢失解。

### 134. Gas Station

```python
Input: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]
```

注意题目 cost[i]代表从i到i+1的油量消耗
定义：$\alpha_i = \text{gas}[i]-\text{cost}[i]$
证明：如果一个数组和大于
$$
\sum_{t=0}^{n}\alpha_t >= 0 \tag{1}
$$
令其作为环形，那么一定可以找到一个起点，从此起点开始转一圈，连续和大于零。
**证明如下**：
首先一定可以找到一段$i$到$j$和是最大的而且是连续的（反证法，如果$i$到中间某点$k$累计和小于零，那么丢弃前半段$i$～$k$，后半段$k$+1~$j$一定大于$i$~$j$,矛盾）

$$
i,j = \arg \max_{a,b} \sum_{t=a}^{b}\alpha_t \tag{2}
$$

这样从$i$出发到达$j$是可行的。假设到达$m$的时候和小于零:
$$
\sum_{t=i}^{m}\alpha_t < 0 \tag{3}
$$
那么根据总和(1)大于0，可以得出
$$
\sum_{t=m+1}^{n-1}\alpha_t + \sum_{t=0}^{i-1}\alpha_t  > 0 \tag{4}
$$
这样一来，会得出($j$~$i$是循环折返的区间)：
$$
j,i = \arg \max_{a,b} \sum_{t=a}^{b}\alpha_t \tag{5}
$$
这和（2）是矛盾的。

### 162. Find Peak Element

```python
while i < j:
    mid = (i + j)//2
    if nums[mid] < nums[mid+1]:
        i = mid+1
    else:
        j = mid
    print(i,j,mid)
```

因为是和nums[i+1]比较大小：

* if nums[mid] > nums[mid+1]，**mid位置可能是峰值**
* if nums[mid] < nums[mid+1]，**mid一定不是峰值**

### 190. Reverse Bits

bit操作，可以循环`n&1`来获取某一位是否为1

这道题相当于流水线，先准备位置，然后放置值

1 (  )  结果左移1

1 (1)  放置此位置应该有的值

1 1 (  ) 结果左移1

1 1 (0) 放置此位置应该有的值

### 139. Word Break
>
>Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

dfs：(回溯法)带记忆的回溯，时间复杂度是$O(n^2)$，
bfs：宽度优先搜索可以用来找最小划分90. Reverse Bits

bit操作，可以循环`n&1`来获取某一位是否为1

这道题相当于流水线，先准备位置，然后放置值

1 (  )  结果左移1

1 (1)  放置此位置应该有的值

1 1 (  ) 结果左移1

1 1 (0) 放置此位置应该有的值

### 204. Count Primes

1. 构造标质数组 notPrimes[n]
2. 初始化，令p = 2 (最小Prime Number)
3. 枚举 2p, 3p, 4p, ……, Mark them int array，notPrimes[kp] = True
4. 找大于p 的第一个没有marked数字（**即下一个Prime Number， 略过合数！**)，如果没有stop;有的话令p = 这个新数字，重复第三步。

### 206. Reverse Linked List

反转链表的递归算法：

起初考虑递归函数如下写：

```python
node = reverseList(head.next)
node.next = head
return ?
```

发现return的时候没法返回了，其实可以换一个思路，既然最后要返回反转后链表的头节点，那我们也可以同样对待递归函数。**这里要注意在递归时我们传入节点head.next，这是不会影响当前节点的next指向的。这一层关联并未断掉**。

```python
newHead = reverseList(head.next)
head.next.next = head
head.next = None
return newHead
```

### 371. Sum of Two Integers

& 运算后是carry进位，如果没有进位，直接异或即可, 只要有进位，就递归调加法运算

关键点：对于负数运算来说：
-1+2 = ?

迭代次序|运算|二进制|十进制
-|-|-|-
1|XOR | 1111| -1
1|&<<1|0010 | 2
2|XOR|1101|-3
2|&<<1|0100|4
3|XOR|1001|-7
3|&<<1|1000|8
4|XOR|0001|1
4|&<<1|0000|0

可以发现进位数字carry是越来越大的, python的二进制可能没有限制，如果一直这样下去会出现超时溢出，所以要使用0xffffffff来限制最大迭代次数，且要限制在carry上。

## 剑指offer

### 剑指 Offer 07. 重建二叉树

关键点：

1. 前序遍历的理解（先root,后**left子树**,right子树)，特别区分开层次遍历(先root,后**left子节点**,right子节点)
2. 需要一种数据结构，**定位先序列表中某一节点在中序列表中的位置**, 即k,v反置，hashMap或dict
3. 递归调用划分左右子树时右子树传入参数的理解

```python
node.right = recursive(root+i-left+1, i+1, right)
# root+i-left+1: 根节点索引 + 左子树长度 + 1
```

### 剑指 Offer 11. 旋转数组的最小数字

```python
mid = (low+high)/2
if numbers[mid] < numbers[high]:
    high = mid
elif numbers[mid] > numbers[high]:
    low = mid+1
else:
    high = high-1
```

此题主要一个点，nums[high]是个分界点
1.numbers[mid]大于nums[high]，一定在左边
2.numbers[mid]小于nums[high]，一定在右边
3.numbers[mid]等于nums[high], **不确定**，既然不确定，可以看low,high有没调整的可能

### 剑指 Offer 12. 矩阵中的路径

要注意递归边界情况的先后顺序

1. （i, j 不合法）或（board[i][j] != word[k]）返回false
2. 剩下的情况一定是**合法且符合匹配**的，**这时候才检验k是否已到头**
tips:

$$
\begin{align} A &= i, j 不合法 \notag \\
B &= board[i][j] 不等于 word[k] \notag \\
\overline{A\vee B} &= \overline{A}\wedge\overline{B} \notag
\end{align}
$$

3. 递归深度搜索可行解

### 剑指 Offer 13. 机器人的运动范围

此题一个重要理解的点：
1.这是一个**搜索可到达点个数的题，而非有多少种解的题**
2.向下和向右即可覆盖题解，无需考虑向上向左
3.python可以使用set来做访问记录

### 剑指 Offer 14- I. 剪绳子

最优质子结构

### 剑指 Offer 16. 数值的整数次方

$$
x^n = {(x^2)}^{n/2}
$$
二分推导，可通过循环 $x := x^2$ 操作，每次把幂从 $n$ 降至 $n//2$,直至将幂降为0
这样理解，每次 $x := x^2$ 都会导致 $n$ 减少一半，直到 $n$ 减少到0
这里有个小技巧：通过判断`n&1`来确认是否为奇数，只有奇数需要单独乘上一个$x$

### 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面

* 快慢指针的用法模板

```python
int low = 0, fast = 0;
while (fast < nums.size()) {
    if (nums[fast] & 1) {
        swap(nums[low], nums[fast]);
        low ++;
    }
    fast ++;
}
```

* 左右指针的用法模板

```python
while left < right:
    if nums[left] & 1 == 1:
        left += 1
        continue

    if nums[right] & 1 == 0:
        right -= 1
        continue
    tmp = nums[left]
    nums[left] = nums[right]
    nums[right] = tmp
    left += 1
    right -= 1
```

注意此处的代码，左右指针在移动时只移动一步，不要试想在里面多次移动来增加逻辑的复杂性

### 剑指 Offer 31. 栈的压入、弹出序列

典型题
给定一个压入序列 pushedpushed 和弹出序列 poppedpopped ，**则压入和 弹出操作的顺序（即排列）是唯一确定的**。
写逻辑手动模拟入栈出栈的过程，需要用到一个辅助栈。如果合法，基于上面出入栈序列唯一的结论，最后栈中元素一定正好全部弹出。

### 剑指 Offer 32 - III. 从上到下打印二叉树 III

标准的层次遍历or广度遍历BFS

```python
queue = collections.deque()
while len(queue) != 0:
    node = queue.pop()
    # 处理node .....
    if node.left is not None:
        queue.append(node.left) 
    if node.right is not None:
        queue.append(node.right)
```

如果涉及到对单独一层顺序的改变，我们可以使用len(queue)控制个数来只操作一层，因为每次进入while循环，**队列里当前存放的就是上一层的所有节点**，通过个数我们就能卡到只属于上一层的节点而不用担心本层节点的加入。
tips:
注意变换奇偶性的时候使用~或not来反转！

### 剑指 Offer 33. 二叉搜索树的后序遍历序列

后续遍历：左子节点→右子节点→根节点
二叉搜索树：左子树的值<根节点<右子树的值
判断后续遍历序列中某一根节点是否合法：即对当前节点，右子树的所有节点都大于当前节点，左子树的所有节点都小于当前节点，这里的关键点即是**找这个节点的左右子树序列**
<https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/solution/di-gui-he-zhan-liang-chong-fang-shi-jie-jue-zui-ha/>

### 剑指 Offer 35. 复杂链表的复制

链表有next指针和random指针，**其实可以看作一个图**，如果看作图则可以考虑

1. 深度优先遍历
2. 广度优先遍历
两种遍历的过程中均会涉及到记录是否访问过的操作，即visited的hashMap，这是一个标准的图遍历算法

### 剑指 Offer 36. 二叉搜索树与双向链表

递归的理解：
中序遍历的时候，很自然会想到用递归返回之后选出来右子树最小元素，curNode.right可以递归调用来返回，即`curNode.right = inorder(curNode.right)`关键点就在curNode.left去连接谁？貌似信息丢掉了
其实理解方式是不对的，要把中序遍历写出来仔细观察，**每一次调用并不需要返回值**。

```python
def inorder(curNode):
    if curNode is None:
        return
    inorder(curNode.left)
    print(curNode.val)
    inorder(curNode.right)
```

上面是中序遍历，符合递增排序数组，可以看到这种递归方式是**只处理当前节点**
我们只需要只到，当前节点已遍历到，**把其连接到前一个节点**，就可以了，至于前一个节点怎么获取，**可以使用全局变量 self.pre**

### 剑指 Offer 48. 最长不含重复字符的子字符串

滑动窗口的应用, i, j分别为左右窗口，循环遍历的时候检查i-j的最大值
这里需要注意的点：

* 边界条件：str是一个字符的情况
* i的初始值
i的初始值可以考虑loop invariant:
左闭右开
[-1,0)
[i, j)
loop invariant: [i,j)是目前不含重复字符的最长子串？

```python
index = {}
res = 0
i = -1
for j in range(len(s)):
    if s[j] in index:
        i = max(i, index[s[j]])
    index[s[j]] = j
    res = max(res, j-i)
return res
```

* 左窗口i的调整策略

### 剑指 Offer 56 - I. 数组中数字出现的次数

判断第一个不为0的bit

```python
pivot = 1
while res & pivot == 0:
    pivot <<= 1
```

注意：if (num & pivot) != 0
pivot=$(0000100000)_2$
这个pivot和任意一个数字取&, **需要用 != 0判断，不可以使用 != 1**

### 剑指 Offer 59 - I. 滑动窗口的最大值

1. 判断当前要移出的nums[i-1]是否位队列最大元素（与deque[0]判等)
  若为最大则popleft()
2. 将当前right = i+k-1 放入递减队列合适位置
3. 找到当前窗口最大值

## 取模运算

余数在数学上的定义始终是大于等于零，即按照Euclidean division的定义：

给定两个整数 $a$ 和 $b$, 其中 $b \neq 0$，存在**唯一**的整数 $q$ 和 $r$使得：

​ $a = bq + r$ 和

​ $0 \leq r < |b|$ 成立

取模运算(Modulo operation)类似数学上求余数（reminder)的过程，但丁略有不同，一般满足下面的式子：

​   $q \in Z​$

​   $a = nq + r​$

​   $|r| < |n|​$

对比数学上的定义，由于最后一个约束的不同，会造成两种计算结果：

### 1. truncate

截断小数部分，取整数部分，<u>C/C++，JAVA， C#等语言中，"%"是取余运算</u>。

​  $r = a - n\  \textrm{trunc}(\frac{a}{b})​$

比如3/2 = 1 , -3/2 = -1

C 和 JAVA 使用的是 truncate 的方式，所以计算 -6 % 5如下：

> -6 - (5*trunc(-6/5))= -6 - (5 * -1) = -1

### 2. floor

向下取整，在正数的时候和truncate一样，但是在**负数**的时候，向下取整就会出现和truncate不一样的结果。

<u>Python 中 "%" 是取模运算</u>。

​		$ r = a = n\lfloor \frac{a}{n} \rfloor​$

比如：3/2 = 1 -3/2 = -2

python使用的floor除法的方式

> -6 - (5*floor(-6/5))= -6 - (5 * -2) = 4

注：

简单来说，求余的结果应该与a的符号保持一致；而取模的结果应该与b的符号保持一致。

```python
python　　　　a%n的符号与n相同
-11//4          #值为-3
-11%4  ->  (-11) -4*(-11//4) =1     #值为1
```

```c
C语言　　　　　　a%n的符号与a相同
-11/4         //值为-2
-11%4      (-11) - 4*(-11/4) =-3   //值为-3
```

### 辗转相除法

辗转相除法是用来计算两个整数的最大公约数。假设两个整数为`a`和`b`，他们的公约数可以表示为`gcd(a,b)`。如果`gcd(a,b) = c`,则必然`a = mc`和`b = nc`。a除以b得商和余数，余数r可以表示为`r = a - bk`，`k`这里是系数。因为`c`为 `a`和`b`的最大公约数，所以`c`也一定是`r`的最大公约数，因为`r = mc - nck = (m-nk)c`。

因此`gcd(a,b) = gcd(b,r)`，相当于把较大的一个整数用一个较小的余数替换了，这样不断地迭代，直到余数为0，则找到最大公约数。

[1].https://blog.csdn.net/hk2291976/article/details/52775299

[2].https://www.jianshu.com/p/7876eb2dff89