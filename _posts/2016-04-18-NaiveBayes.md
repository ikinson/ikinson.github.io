---
layout: post
title: '朴素贝叶斯分类原理及其实现'
date: 2016-04-18
author: Kinson
categories: 机器学习
tags: 分类 贝叶斯 Python Spark MlLib
---

* content
{:toc}

## 贝叶斯原理

理论上来说，贝叶斯原理是根据已知的先验概率*P(A\|B)*，利用贝叶斯公式:

$$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$$

求出后验概率*P(B\|A)*。即该样本属于某一类的概率，然后选择具有最大后验概率的类作为该样本所属的类。

在这里，x是一个特征向量，将设x维度为M。因为朴素的假设，即特征条件独立，根据全概率公式展开可以得到：

$$p(y=c_k|x) = \frac{\prod_{i=1}^{M}p(x^i|y=c_k)p(y=c_k)}{\sum_{M}p(y=c_k)\prod_{i=1}^{M}p(x^i|y=c_k)}$$

这里，只要分别估计出，特征$x^i$在每一类的条件概率就可以了。类别*y*的先验概率可以通过训练集算出，同样通过训练集上的统计，可以得出对应每一类上的，条件独立的特征对应的条件概率向量。



## 实现：朴素贝叶斯用于文本分类

### 数据准备

```python
def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec
```

### 创建词汇表

```python
def createVocabList(dataSet):#创建词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建并集
    return list(vocabSet)
```

### 文本向量化

```python
def bagOfWord2VecMN(vocabList,inputSet):#根据词汇表，讲句子转化为向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
```

### 进行训练

```python
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive
```

### 构建分类器

```python
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
```

### 模型测试

```python
listTxt, listClasses = loadData()
myVocab = createVocab(listTxt)
trainMat = []
for txt in listTxt:
    trainMat.append(Word2Vec(myVocab, txt))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
testEntry = ['love', 'my', 'him']
thisDoc = Word2Vec(myVocab,testEntry)
print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
```

## Spark MlLib 实现

``` scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object NaiveBayesian {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local","NaiveBayesian")
    
    val data = sc.textFile("E:\\data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    
    val splits = parsedData.randomSplit(Array(0.7,0.3),seed = 11L)
    val training =splits(0)
    val test =splits(1)
    
    val model = NaiveBayes.train(training, lambda = 1.0)
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1==x._2).count() / test.count()

    println("Accuracy => " + accuracy)
  }
}
```
