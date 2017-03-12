---
layout: post
title: '初识Spark-WordCount'
date: 2017-01-18
author: Kinson
categories: 大数据
tags: Spark Scala
---

* content
{:toc}

## 实例描述
- 读文本
- 分词
- 去标点
- 词频统计
- 排序

## 代码片段

```scala
val conf = new SparkConf().setAppName("WordCount").setMaster("local[4]")

val sc = new SparkContext(conf)

val res = sc.textFile("E:\\The_Godfather.txt", 2)                    //读文件
            .flatMap(line => line.split(" "))                        //以空格分词
            .map{w =>
                 val lower = w.toLowerCase()                         //将字符串转为全小写
                 val pattern = "[a-z]".r
                 val clean = (pattern findAllIn lower).mkString("")  //去掉其他特殊字符
                 (clean, 1)                                          //返回键值对
                }
            .filter(_._1.length >= 1)                                //过滤掉空单词
            .reduceByKey(_+_)                                        //词频统计
            .sortBy(_._2, ascending = false)                         //排序
            .take(100)                                               //取前100位

res.foreach(println)
```
