# VGG16-tranfer
Distinguishing Cats and Dogs by CNN Neural Network
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)




## 环境

* Python3.6
* TensorFlow 1.8

## DataSet

文件下载
[train](https://s3-us-west-2.amazonaws.com/lintcode/ml/problems/3/train.zip)
[test](https://s3-us-west-2.amazonaws.com/lintcode/ml/problems/3/test.zip)
[VGG16-model](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)

训练集有20000张猫狗图，测试集有5000张，测试集不带`label`

## 文件介绍

* train.py 训练网络
* tfrecord_conversion.py 将数据集转化成tfrecord文件
* acc-cat-dog.py 验证网络在测试集上的accuracy
* 其他的都是烂尾网络

## 使用须知
数据集的相对路径需要注意
