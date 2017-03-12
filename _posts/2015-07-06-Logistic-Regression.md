---
layout: post
title: '机器学习-Logistic回归'
date: 2015-07-06
author: Kinson
categories: 机器学习
tags: 逻辑回归 分类
---

* content
{:toc}

## Logistic回归

### 模型假设

根据线性回归可以预测连续的值，对于分类问题，我们需要输出0或者1。所以，在分类模型中需要将连续值转换为离散值。我们可以预测:

- 当$h_\theta$大于等于0.5时，输出为y=1；

- 当$h_\theta$小于0.5时，输出为y=0。

Logistic回归模型的输出变量范围始终在0和1之间，Logistic回归模型的假设为：

$$h_\theta(x) = g(\theta^Tx)$$

其中：

- $x$代表特征向量
- $g$代表逻辑函数（Logistic Function），常用逻辑函数为S形函数（Sigmoid Function），函数公式为：

$$g(z) = \frac{1}{1+e^{-z}}$$

该函数的图像为：

<div align="center"><img src="https://dn-kinson.qbox.me/ML/logistic/sigmoid.jpg" /><br /></div>

所以，整个模型的假设为：

$$h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$$

该假设函数$h_\theta(x)$的作用是，对于给定的输入变量，根据已经训练好的模型参数计算出输出变量=1的可能性（estimated probability），即

$$h_\theta(x)=P(y=1|x;\theta)$$

### 判定边界

在Logistic回归中，我们预测：

- 当$h_\theta$大于等于0.5时，预测 y=1；

- 当$h_\theta$小于0.5时，预测 y=0。
 
根据上面绘制的S形函数图象，当

- z=0时，g(z)=0.5；z>0时，g(z)>0.5；z<0时，g(z)<0.5

其中，$z=\theta^Tx$,即：

- $\theta^Tx \geq 0$时，预测 y=1；
- $\theta^Tx < 0$时，预测 y=0。

可以观察到$z=\theta^Tx$与线性回归非常相似，该函数所表示的线（面）就是Logistic回归中分界线，即判定边界（Decision Boundary）。针对不同的数据分布，我们可以用非常复杂的模型来适应形状判定边界。

### 代价函数

在线性回归中，我们将代价函数定义为模型所有误差的平方和。而在逻辑回归中沿用这个定义得到的代价函数是一个非凸函数，这难以用梯度下降法求局部最小值，因此需重新定义逻辑回归的代价函数：

$$J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})$$

其中Cost()函数定义为：

$$ Cost(h_\theta(x),y)=\left\{\begin{array}{rcl}-log(h_\theta(x))       &      & {y=1}\\-log(1-h_\theta(x))     &      & {y=0}\end{array} \right. $$

将构建的Cost()函数简化如下：

$$Cost(h_\theta(x),y) = -ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))$$

带入代价函数可得到代价函数表达式为：

$$J(\theta)=-\frac{1}{m}[\sum_{i=1}^my^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$$

### 代价函数求解

仍然采用梯度下降法求代价函数的局部最小值：

$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$$

求导后得到迭代过程：

$$\theta_j := \theta_j - \alpha\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})x_j^{(i)}$$

Simultaneously update $\theta_j$ for $j=0,1…n$.

注：虽然得到的梯度下降算法表面上看去与线性回归一样， 但是这里的$h_\theta(x)=g(\theta^Tx)$与线性回归中不同，所以实际上是不一样的。**另外，在运行梯度下降算法之前，进行特征缩放依旧是非常必要的**。
