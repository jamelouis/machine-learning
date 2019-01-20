---
layout: post
title:  "机器学习 - Regularized Linear Regression"
date:   2019-01-19 16:43:01 +0800
categories: machine-learning
---

本篇博文主要是讲述如何用python的Scikit-Learn库来重新实现吴恩达机器学习中第五周的课程作业。
该作业有两大目标：
* 实现 Regularized Linear Regression 算法
* 研究该算法的 bias-Variance 属性。

## 问题描述

> implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir.

即给定水库水位(x)变化，预测出水量(y)。该问题是一个单变量的线性回归问题。


## 准备数据

课程中的数据是 octave 的 mat 格式，存储在 ex5data.mat 中。通过 ocatave 内置函数 load 加载到内存中，并定义了3对变量存储了样例数据。样例数据为：

* X，y: 12个训练数据集
* Xval，yval: 21个交叉验证数据集
* Xtest，ytest: 21个测试数据集

通过 octave 内置函数 csvwrite 可以将数据导出为 csv 格式。因为有3对数据集，所以导出为三份数据，分别为 training.csv, cross_validation.csv, test.csv。

具体的 octave 代码如下:

```octave
% export dataset in octave
> load('ex5data.mat')
> csvwrite('training.csv',[X,y])
> csvwrite('cross_validation.csv',[Xval,yval])
> csvwrite('test.csv',[Xtest,ytest])
```

## 用 pandas 库加载数据

用 pandas 的函数 read_csv 函数分别加载 training.csv, cross_validation.csv, test.csv，因为 csv 文件没有头部信息（water_level,water_amount)，所以要设置 header 参数为 
None。

```python
# load dataset
import pandas as pd
training_data = pd.read_csv('training.csv',header=None)
cross_validation = pd.read_csv('cross_validation.csv',header=None)
test= pd.read_csv('test.csv',header=None)
```

从加载的数据集中，提取出训练数据集，交叉验证数据集和测试数据集。因为 SciKit-learn 的 LinearRegression 的输入需要矩阵类型的数据。

```python
x = training_data.as_matrix([0])
y = training_data.as_matrix([1])
x_cv = cross_validation.as_matrix([0])
y_cv = cross_validation.as_matrix([1])
x_test = test.as_matrix([0])
y_test = test.as_matrix([1])
```

## 数据可视化

因为有三对数据可以可视化，所以先定义个通用绘制函数 plotData。

```python
from matplotlib import pyplot as plt

def plotData(x,y,title):
    plt.plot(x,y,'r+')
    plt.xlabel('change in water level(x)')
    plt.ylabel('water flowing out of the dam(y)')
    plt.title(title)
    plt.show()
```

### 可视化训练数据

```python   
plotData(x,y,'training data')
```

![training data]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_3_0.png" | relative_url}})

### 可视化交叉验证数据

```python   
plotData(x_cv,y_cv,'cross validation data')
```

![cross validation data]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_3_1.png" | relative_url}})

### 可视化测试数据

```python   
plotData(x_test,y_test, 'test data')
```

![test data]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_3_2.png"| relative_url}})

## 用 SciKit-Learn 构建线性回归模型

第三方库大大简化了机器学习应用的开发工作量，这里调用简单的几个 API 就可以把模型训练出来。具体代码如下：

```python
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(x,y)
print('coef: ', reg.coef_)
print('intercept: ', reg.intercept_)
```

    coef:  [[0.36777923]]
    intercept:  [13.08790351]

即模型的数学形式为：**y = 13.09 + 0.368 * x**.

## 可视化模型和训练数据

> *predict*: 给定输入水位(x)，根据训练的模型，给出出水量(y)。

首先，找到训练数据的最小值和最大值，在该区间构建测试数据，用该测试数据集，用 **predict** 求出对应的出水量。

然后在之前的训练数据的可视化中，绘制出该测试集的曲线。

```python
x_min = min(x.flatten())
x_max = max(x.flatten())
x_predict = np.arange(x_min,x_max)
x_predict = x_predict.reshape(-1,1)
y_predict = reg.predict(x_predict)

def plotFit(x,y,x_predict, y_predict):
    plt.plot( x, y,'rx')
    plt.plot( x_predict, y_predict, '-')
    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water following out of the dam(y)')
    plt.title('Linear Fit')
    plt.show()
plotFit(x,y,x_predict,y_predict)
```

![png]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_5_0.png"| relative_url}})

## 学习曲线

学习曲线利于调试学习算法。回顾一下学习曲线的定义：

> a learning curve plots training and cross validation error as a function of training set size.

简单的讲，就是x轴是训练集大小，y轴是误差；学习曲线实际是一条训练误差曲线和对应的交叉验证集误差曲线。

针对本文的问题，训练数据集的大小是12，因此，绘制学习曲线的过程大致如下：

1. i = 1
2. 从训练数据集中取i个数据，训练模型
3. 分别根据训练的模型，计算出训练误差和交叉验证误差
4. 当i不等于训练集数据大小12时，重复步骤1，反之结束

```python
def calcError(ypredict, y):
    m = len(y)
    return 1/(2*m) * sum((ypredict-y)**2)[0]

def calcErrorForTrainingAndCV(x,y,x_cv,y_cv):
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    
    y_predict = reg.predict(x)
    y_cv_predict = reg.predict(x_cv)
    
    training_error = calcError(y_predict,y)
    cv_error = calcError(y_cv_predict, y_cv)
    
    return training_error, cv_error

def calcLearnCurveErrors(x,y,x_cv,y_cv):
    n = len(x)
    t_errors = []
    cv_errors = []
    for i in range(1,n+1):
        x_range = x[:i]
        y_range = y[:i]
        t_error,cv_error = calcErrorForTrainingAndCV(x_range,y_range,x_cv,y_cv)
        t_errors.append(t_error)
        cv_errors.append(cv_error)
    return t_errors, cv_errors

def plotLearnCurve(t_errors,cv_errors):
    x = np.arange(1,len(t_errors)+1)
    plt.plot(x,t_errors,'b',x,cv_errors,'g')
    plt.xlabel('Number of training examples')
    plt.axis([0,13,0,150])
    plt.ylabel('Error')
    plt.legend(['Train','Cross Validation'])
    plt.title('Linear regression learning curve')
    plt.show()
    
t_errors, cv_errors = calcLearnCurveErrors(x,y,x_cv,y_cv)
plotLearnCurve(t_errors, cv_errors)
```


![png]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_6_0.png" | relative_url}})

从学习曲线可以看出，当训练数据增加时，训练误差和交叉验证误差都很高，在20-40区间。因此，可以得出该模型存在 high bias 问题，属于欠拟合问题。增加特征值，可以减弱 high bias 的影响。

## Polynomial regression

> add more features using the higher powers of the existing feature x in the dataset.

分别对训练数据、预测测试集和交叉验证数据集添加高次幂的特征。

```python
def polynomial(x,p):
    x_poly = x
    for i in range(2,p+1):
        xx = x**i
        x_poly = np.insert(x_poly, i-1, xx.flatten(), axis=1)
    return x_poly
        
x_poly_8 = polynomial(x,8)
x_poly_predict_8 = polynomial(x_predict,8)
x_cv_poly_8 = polynomial(x_cv,8)
```

添加高次幂，需要对数据做一个特征缩放，从而归一化训练区间。

```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_poly_8)
x_poly_8 = scaler.transform(x_poly_8)
x_poly_predict_8 = scaler.transform(x_poly_predict_8);
x_cv_poly_8 = scaler.transform(x_cv_poly_8);
```

训练模型并绘制 Fit 曲线

```python
poly_reg = linear_model.LinearRegression()
poly_reg.fit(x_poly_8,y)

y_poly_predict = poly_reg.predict(x_poly_predict_8)
plotFit(x,y,x_predict,y_poly_predict)
```

![png]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_9_0.png" | relative_url}})

由图可以看出，欠拟合的现象大大缓减，测试数据基本都在模型曲线的附近。

再来计算一下 polynomial regression 的学习曲线。

```python
t_errors, cv_errors = calcLearnCurveErrors(x_poly_8,y,x_cv_poly_8,y_cv)
plotLearnCurve(t_errors, cv_errors)
```

![png]({{ "assets/images/machine-learning-ex5/regulared_linear_regression/output_10_0.png" | relative_url}})

由图可知，训练的误差基本为0，而交叉验证的误差出现了比较大的变化。

## 小结

* 使用 pandas 库加载了格式为 csv 的样例数据
* 使用 matplotlib 库绘制了数据曲线、Fit曲线和学习曲线
* 使用 Scikit-learn 库训练了单变量线性模型
* 使用了 Learn Curve 曲线调试了学习算法
* 通过添加特征减轻了欠拟合/*high bias* 现象