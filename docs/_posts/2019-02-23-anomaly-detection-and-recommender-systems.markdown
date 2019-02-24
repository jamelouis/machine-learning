---
layout: post
title:  "机器学习 - Anomaly Detection and Recommender Systems"
date:   2019-02-23 19:43:01 +0800
categories: machine-learning
---

本篇博文主要是使用python的scikit-learn库重新实现吴恩达机器学习课程的
程序设计8的练习。该练习主要是关于Anomaly Detection 和 Recommender Systems.

## Anomaly detection 

有两组训练数据，一组是简单的二维数据，另一组是高维数据，维度为12.

第一个步骤，是从octave中到处csv格式。可以用如下代码导出：

```octave
load('ex8data1.m')
csvwrite('ex8data1.csv',X)
csvwrite('ex8data1_cs.csv',[Xval,yval])

load('ex8data2.m')
csvwrite('ex8data2.csv',X)
csvwrite('ex8data2_cs.csv',[Xval, yval])
```

* 通过函数*load*加载数据到octave终端
* 然后，通过函数*csvwrite*导出训练数据到文件*ex8data1.csv*
* 接着，导出交叉验证数据到文件*ex8data1_cs.csv*
* 对数据*ex8data2.m*做同样的处理

首先，导入python机器学习相关的模块库。

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
```

总共用到了四个库，pandas，numpy，matploblit，sklearn。

接着，加载课程数据。

```python
ex8data1 = pd.read_csv('ex8data1.csv', header = None)
ex8data1_cs = pd.read_csv('ex8data1_cs.csv', header=None)

print(ex8data1.describe())
X = ex8data1.as_matrix()
Xval = ex8data1_cs.as_matrix([0,1])
yval = ex8data1_cs[2]
yval = [ 1 - 2*y for y in yval]
```

* 加载练习数据*ex8data1*，提取训练数据X，交叉验证数据Xval，yval
* yval数据，0表示为正常数据，1表示为异常数据，需要预处理成，-1表示异常数据，1表示正常数据

然后，可视化训练数据，蓝色点表示；并可视化交叉验证数据，绿色表示正常数据，红色表示异常数据。

```python
def plot(X, color= "r"):
    plt.scatter(X[:,0],X[:,1], color=color, marker='o')
    
plot(X, 'b')
Xpos = ex8data1_cs[ex8data1_cs[2] == 0].as_matrix([0,1])
plot(Xpos,'g')
Xneg = ex8data1_cs[ex8data1_cs[2] == 1].as_matrix([0,1])
plot(Xneg, 'r')
```

接着，尝试简单使用sklearn.covariance.EllipticEnvelope模型训练数据，并做一下简单的交叉数据验证。

```python
anomal_detection = EllipticEnvelope()
clf = anomal_detection.fit(X)
print(clf.score(Xval,yval)) # output: 0.908
```

* 创建模型*EllipticEnvelope()*
* 用训练数据训练模型
* 做交叉验证，由输出可知，该模型对交叉数据的准确率为90.8%

然后，可视化模型。

```python
def visualize_anomaly_result(proportion_of_outliers = [0.1]):
    xx, yy = np.meshgrid(np.linspace(0, 30, 150),
                         np.linspace(0, 30, 150))

    plt.figure(figsize=(4*len(proportion_of_outliers),6))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.2,
                        hspace=.4)

    for i, prob_outlier in enumerate(proportion_of_outliers):
        anomal_detection = EllipticEnvelope(contamination = prob_outlier)
        clf = anomal_detection.fit(X)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(3, len(proportion_of_outliers), i+1 )
        plt.contour(xx, yy, Z, linewidths=2, colors='black')
        plot(X,'b')

        plt.subplot(3, len(proportion_of_outliers), i+1+len(proportion_of_outliers))
        plt.contour(xx, yy, Z, linewidths=2, colors='black')
        plot(Xpos,'g')
        plot(Xneg,'r')

        ypredict = clf.predict(Xval)
        Xpre_correct = ex8data1_cs[ypredict == yval].as_matrix([0,1])
        Xpre_wrong = ex8data1_cs[ypredict != yval].as_matrix([0,1])

        plt.subplot(3, len(proportion_of_outliers), i+1+2*len(proportion_of_outliers))
        plt.contour(xx, yy, Z, linewidths=2, colors='black')
        plot(Xpre_correct,'c')
        plot(Xpre_wrong, 'm')

        print("when the proportion of outliers is", prob_outlier, ", the score of cross valid data is", clf.score(Xval, yval))

    
visualize_anomaly_result()
```

![01]( {{ "assets/images/machine-learning-ex8/01.png" | relative_url }})

图表上图显示训练数据和模型，位于黑色线条内部的为正常数据，反之，为异常数据。中图显示了交叉验证数据和模型。
下图显示了模型，青色表示预测准确的数据，反之则为预测不准的数据。

由上图可以看出，该模型存在很大的偏差。接下来，我们试着调整一下模型的参数。

```python
proportion_of_outliers = [0.1, 0.05, 0.02, 0.01]
visualize_anomaly_result(proportion_of_outliers)
```

* 分别尝试参数 0.1， 0.05， 0.02, 0.01 四个参数
* 同时，可视化这四个参数的图表，并输出交叉验证数据集的准确率

> when the proportion of outliers is 0.1 , the score of cross valid data is 0.9087947882736156
> when the proportion of outliers is 0.05 , the score of cross valid data is 0.9641693811074918
> when the proportion of outliers is 0.02 , the score of cross valid data is 0.9869706840390879
> when the proportion of outliers is 0.01 , the score of cross valid data is 0.9804560260586319

![02]({{ "assets/images/machine-learning-ex8/02.png" | relative_url }})

由上图可知，当参数为 0.02 时，模型的准确率最高，达到了98.6%。

最后，我们在高维数据上做同样的模型处理。

```python
ex8data2 = pd.read_csv('ex8data2.csv', header=None)
ex8data2_cs = pd.read_csv('ex8data2_cs.csv', header=None)
X = ex8data2.as_matrix()

Xval = ex8data2_cs.as_matrix(np.arange(0,11))
yval = ex8data2_cs[11]
yval = [ 1 - 2*y for y in yval]

anomaly_detection = EllipticEnvelope(contamination = 0.01)
clf = anomaly_detection.fit(X)
ypredict = clf.predict(Xval)
print(sum(yval == ypredict)) # output: 95
print(clf.score(Xval, yval)) # output: 0.95

sum(clf.predict(X)==1) # output: 990
```

当参数为0.01时，训练数据中990个数据归为正常数据，10个数据为异常数据。而交叉验证数据集的准确率为95%。