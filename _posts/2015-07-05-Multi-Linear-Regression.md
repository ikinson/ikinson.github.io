---
layout: post
title: '机器学习-多变量线性回归'
date: 2015-07-05
author: Kinson
categories: 机器学习
tags: 线性回归
---

* content
{:toc}

## 多变量线性回归

多变量线性回归是在单变量线性回归的基础上增加变量的个数，多变量回归的假设函数为：

$$h_\theta(x)=\theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$$

其中：

- $n$为变量个数（特征数）
 
- $x^{(i)}$第i个训练样本的特征数据

- $x^{(i)}_j$为第i个训练样本的第j个特征的值

- 为了方便，定义$x_0 = 1$ 

### 向量表示

$$x = [x_0,x_1,x_2,...x_n]^T$$

$$\theta = [\theta_0,\theta_1,\theta_2,...\theta_n]^T$$

$$h_\theta(x)=\theta^Tx$$

### 代价函数

类似于单变量线性回归的代价函数格式，多变量线性回归的代价函数可写成：

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$$

### 梯度下降法求解

$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta_0,...\theta_n)$$

迭代过程：

$$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$

Simultaneously update $\theta_j$ for $j=0,1...n$.

## 多项式回归

若数据基本符合多项式假设，函数形如：

$$h_\theta(x) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + ...$$

可将多项式回归问题转化为多变量线性回归问题：

$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ...$$

其中：

$$x_1=x, x_2=x^2, x_3 = x^3....$$

## 标准方程法

梯度下降法是通过多次迭代的方法求代价函数最小时的参数组合，除此之外我们还可以用**标准方程法（Normal Equation）**直接一步求出最优的参数值。假设有m个样本，n个特征，则可一构建一个m行n+1列特征矩阵X（包含添加的特征$x_0$）；参数向量$\theta$为n+1维向量；y为m维结果向量。

$$X_{m*(n+1)}\theta_{(n+1)*1} = y_{m*1}$$

根据矩阵方程求解，可得出参数向量$\theta$的值：

$$\theta = (X^TX)^{-1}X^Ty$$
