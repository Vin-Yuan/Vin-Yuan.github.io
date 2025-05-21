---
title: linear-algebra
mathjax: true
date: 2024-02-29 17:24:28
categories: math
tags: math
---



在学习SVM时，遇到了dot product的问题，一时忘了在algebra下定义的向量内积和在geometry下定义的向量内积为何相等，查找了一下资料，发现很有趣，故记录如下。
This operation can be defined either algebraically or geometrically.

- Algebraically, it is the sum of the products of the corresponding entries of the two sequences of numbers.
- Geometrically, it is the product of the Euclidean magnitudes of the two vectors and the cosine of the angle between them.

## Algebraic definition

The dot product of two vectors A = [A1, A2, ..., An] and B = [B1, B2, ..., Bn] is defined as:
$$\mathbf{A}\cdot \mathbf{B} = \sum_{i=1}^n A_iB_i = A_1B_1 + A_2B_2 + \cdots + A_nB_n$$

## Geometric definition

In Euclidean space, a Euclidean vector is a geometrical object that possesses both a magnitude and a direction. A vector can be pictured as an arrow. Its magnitude is its length, and its direction is the direction that the arrow points. The magnitude of a vector A is denoted by $\|\mathbf{A}\|$.
The dot product of two Euclidean vectors A and B is defined by

$$\mathbf{A}\cdot\mathbf{B} = |\mathbf{A}||\mathbf{B}|\cos\theta $$

where θ is the angle between A and B.

## Equivalence of the definitions

If e1,...,en are the standard basis vectors in Rn, then we may write
$$
\begin{align}
\mathbf A &= [A_1,\dots,A_n] = \sum_i A_i\mathbf e_i\\
\mathbf B &= [B_1,\dots,B_n] = \sum_i B_i\mathbf e_i.
\end{align}
$$

The vectors $e_i$ are an orthonormal basis, which means that they have unit length and are at right angles to each other. Hence since these vectors have unit length
$\mathbf e_i\cdot\mathbf e_i=1$
and since they form right angles with each other, if $i ≠ j$,
$\mathbf e_i\cdot\mathbf e_j = 0.$
Also, by the geometric definition, for any vector ei and a vector A, we note
$\mathbf A\cdot\mathbf e_i = \|\mathbf A\|\,\|\mathbf e_i\|\cos\theta = \|\mathbf A\|\cos\theta = A_i$,
where Ai is the component of vector A in the direction of ei.
Now applying the distributivity of the geometric version of the dot product gives
$$\mathbf A\cdot\mathbf B = \mathbf A\cdot\sum_i B_i\mathbf e_i = \sum_i B_i(\mathbf A\cdot\mathbf e_i) = \sum_i B_iA_i$$
which is precisely the algebraic definition of the dot product. So the (geometric) dot product equals the (algebraic) dot product.

  [1]: http://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Dot_Product.svg/330px-Dot_Product.svg.png