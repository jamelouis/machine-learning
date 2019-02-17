---
layout: post
title:  "机器学习 - KMeans And PCA"
date:   2019-02-17 19:43:01 +0800
categories: machine-learning
---

本篇博文主要是使用python的scikit-learn库重新实现吴恩达机器学习课程的
程序设计7的练习。该练习主要是关于kmeans和pca的实践，通过这些实践可以比较清晰地理解相关的原理，然后用sklearn高级别的API，通过相同的测试样例，了解相关API的调用。

程序设计练习7，主要是对 K-means 聚类和Principal Component Analysis 的练习实践。让我们简单的回顾一下相关的知识点。

> The K-means algorithm is a method to automatically cluster similar data examples together.

1. 随机初始化 K 中心点
2. 根据测试样例位置到中心点的距离分组
3. 根据分组的测试样例集，重新计算新的 K 中心点
4. 如果迭代一定的误差范围内，则退出；否则重复步骤 2），3）

在课程设计中，有如下的任务：

* Finding closest centroids
* Computing centroid means
* Random initialization
* Image comppression with K-means

PCA. Principal Component Analysis. 主成分分析是最常用的一种降维的方法。

在课程设计中，有如下相关的任务：

* 实现PCA 算法
* 投影数据集到主成分
* 对投影数据重建数据
* 可视化投影
* PCA在face上的运用
* PCA for visualization

## 数据准备

在octave软件，通过如下代码，导出相应的数据集，存储为csv格式。

```
load('ex7data1.mat')
csvwrite('ex7data1.csv',X)

load('ex7data2.mat')
csvwrite('ex7data2.csv',X)

load('ex7faces.mat')
csvwrite('ex7faces.csv',X)
```

数据集如下：

* ex7data1.csv
* ex7data2.csv
* ex7faces.csv
* bird_small.png

## K-means

K-means的练习部分，主要分三部分：

1. K-means在简单的二位数据集上的聚类练习
2. K-means在图像压缩上的运用，并用pca降维可视化主要特征

### K-means on example data

首先，导入相关的python库。pandas库来读取数据集，sklearn库来构建K-means模型，matplotlib库来可视化相关的图标，使过程和结果更直观。

```python
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
```

接着，准备数据。从 ex7data2.csv 中读取数据，并转为矩阵 X，以便下一步数据处理。
然后，可视化该数据集。

```python
ex7data2 = pd.read_csv('ex7data2.csv', header=None)
X = ex7data2.as_matrix()
plot(X)
```

```python
def plot(X):
    plt.plot(X[:,0],X[:,1])
    plt.show()
```

![ex7-01]({{ "assets/images/machine-learning-ex7/ex7-01.png" | relative_url}})

接着，训练K-means模型。

```python
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
visualize_kmeans(X, kmeans)
```

其中，*visualize_kmeans*函数如下：

```python
def visualize_kmeans(X, kmeans):
    color = ['ro','go','bo']
    for i in range(3):
        ind = kmeans.labels_ == i
        plt.plot(X[ind][:,0],X[ind][:,1],color[i])
    plt.show()
```

![ex7-02]({{ "assets/images/machine-learning-ex7/ex7-02.png" | relative_url}})


### KMeans on image compression

该练习使用了图片数据：bird_small.png。因此，需要额外使用 PIL 库，来获取图片的像素数据。

```python
from PIL import Image
```

首先，读取数据并可视化, 并将数据从 (128, 128, 3) 转换为 (16384,3) 的数据集。

```python
bird_small = Image.open('bird_small.png')
image_array = np.array(bird_small)
plt.imshow(image_array)
w,h,d = image_array.shape
image_array = np.reshape(image_array, (w*h, d))
```

![ex7-03]({{ "assets/images/machine-learning-ex7/ex7-03.png" | relative_url}})


接着，训练kmeans模型。

```python
kmeans = KMeans(n_cluster=16, random_state=0).fit(image_array)
```

然后，重建图像。

```python
labels = kmeans.predict(image_array)
bird_small_16 = recreate_image(kmeans.cluster_centers_, labels, w, h)
bird_small_16 = np.array(bird_small_16, dtype=np.uint8)
```

```python
# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image
```

最后，可视化原始图片和压缩重建的图片。

```python
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.title('Origin')
plt.imshow(bird_small)

plt.subplot(122)
plt.title('compressed, with 16 colors')
plt.imshow(bird_small_16)

plt.show()
```

![ex7-04]({{ "assets/images/machine-learning-ex7/ex7-04.png" | relative_url }})

### PCA for visualizaiton

```python
import random 

ind = np.arange(0,len(image_array))
random.shuffle(ind)
ind = ind[:3000]
```

导入随机库random, 从数据中随机采样3000个像素点。

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(kmeans.cluster_centers_)):
    ia = image_array[ind][kmeans.labels_[ind]==i]
    ax.scatter(ia[:0],ia[:,1], ia[:,2])
plt.show()
```

创建一个三维散点图表。结果如下图：

![ex7-05]({{ "assets/images/machine-learning-ex7/ex7-05.png" | relative_url }})

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(image_array)
x_norm = scaler.transform(image_array)
pcas2 = PCA(n_components=2).fit(x_norm)

Z = pca2.fit_transform(x_norm)

plt.figure(figsize=(12,12))
for i in range(len(kmeans.cluster_centers_)):
    z = Z[ind][kmeans.labels_[ind]==i]
    plt.scatter(z[:,0],z[:,1], marker='o')
plt.show()
```

首先，对数据做特征归一化。接着，对归一化的数据集，做PCA处理。然后可视化投影数据集。

![ex7-06]({{ "assets/images/machine-learning-ex7/ex7-06.png" | relative_url}})


## PCA

### pca 在简单数据上的应用

和以前一样，首先导入相关的python库。

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

首先，加载样例数据集 ex7data1.csv，并矩阵化。

```python
ex7data1 = pd.read_csv('ex7data1.csv', header=None)
X = ex7data1.as_matrix()
```

接着，可视化样例数据集。

```python
plt.plot(X[:,0],X[:,1], "ko")
plt.show()
```

![ex7-07]({{ "assets/images/machine-learning-ex7/ex7-07.png" | relative_url}})

然后，对数据做归一化处理。

```python
scaler = StandardScaler()
scaler.fit(X)
x_std = scaler.transform(X)

plt.plot(x_std[:,0], x_std[:,1], "ko")
plt.show()
```

归一化的数据如下图：

![ex7-08]({{ "assets/images/machine-learning-ex7/ex7-08.png" | relative_url}})

接着，做PCA处理。

```python
pca2 = PCA(n_components=2).fit(x_std)
print(pca2.sigular_values_)
print(pca2.components_)
mu = scaler.mean_
U = pca2.components_
S = pca2.singular_values_

line1 = mu+0.2 * S[0] * U[0]
line2 = mu+0.2 * S[1] * U[1]

plt.axis('equal')
plt.plot(X[:,0],X[:,1],"ko")
plt.plot([mu[0],line1[0]],[mu[1],line1[1]], "r-")
plt.plot([mu[0],line2[0]],[mu[1],line2[1]], "r-")
plt.show()
```

![ex7-09]({{ "assets/images/machine-learning-ex7/ex7-09.png" | relative_url }})

最后，可视化投影。

```python
def drawLine(point1, point2):
    plt.plot([point1[0],point2[0]], [point1[1], point2[1]], "k--")

pca1 = PCA(n_components=1).fit(x_std)
Z = pca1.fit_transform(x_std)
x_rec = pca.inverse_transform(Z)

plt.figure(figsize=(8,8))
plt.axis('equal')
plt.axis([-4, 3, -4, 3])
plt.plot(x_std[:,0], x_std[:,1], "bo")
plt.plot(x_rec[:,0], x_rec[:,1], "ro")

for i in range(len(X[:,0])):
    drawLine(x_std[i], x_rec[i])

plt.show()
```

![ex7-10]({{ "assets/images/machine-learning-ex7/ex7-10.png" | relative_url }})

### pca 在 faces 数据集上的应用

首先，导入相关的库。

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
```

接着，准备数据，并矩阵化。

```python
ex7faces = pd.read_csv('ex7faces.csv',header=None)
X = ex7faces.as_matrix()
```

然后，可视化数据集。

```python
def displayData(X,row=None,col=None):
    m,n=X.shape
    w = h = int(np.sqrt(n))
    if row==None or col == None:
        row = col = int(np.sqrt(m))
    data = []
    index = 0
    for r in range(0,row):
        rows = []
        for c in range(0,col):
            rows.append(np.reshape(X[index,:],(w,h),order='F'))
            index = index + 1
        data.append(rows)
    
    rows = []
    for r in range(0,row):
        t = np.concatenate(tuple(data[r]),axis=1)
        rows.append(t)
    
    c = np.concatenate(tuple(rows))
    plt.imshow(c,cmap="gray")
    plt.axis('off')
    plt.show()

displayData(X[0:25,:])
```

![ex7-11]( {{ "assets/images/machine-learning-ex7/ex7-11.png" | relative_url }})

接着，做PCA处理。

```python
scaler = StandardScaler().fit(X)
x_norm = scaler.fit_transform(X)

pca36 = PCA(n_components=36).fit(x_norm)
displayData(pca36.components_)
```

![ex7-12]({{ "assets/images/machine-learning-ex7/ex7-12.png" | relative_url }})

最后，对比原图和PCA后重构的图片。

```python
pca100 = PCA(n_components=100).fit(x_norm)
Z = pca100.fit_transform(x_norm)
x_rec = pca100.inverse_transform(Z)

displayData(x_norm[0:16])
displayData(x_rec[0:16])
```

![ex7-13-01]({{ "assets/images/machine-learning-ex7/ex7-13-01.png" | relative_url }})
![ex7-13-02]({{ "assets/images/machine-learning-ex7/ex7-13-02.png" | relative_url }})

## 小结

* K-means 聚类算法可以运用在图片的压缩
* PCA是一种常用降低维度的算法
* PCA可以用于kmeans的主要成分的可视化