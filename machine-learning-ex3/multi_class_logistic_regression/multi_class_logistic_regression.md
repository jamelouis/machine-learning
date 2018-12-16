

```python
import pandas as pd
train_data  = pd.read_csv('handwritten_digits.csv',header=None)
train_data.head()
```


```python
import numpy as np
index = list(range(0,400))
X = train_data.as_matrix(index);
y = train_data[400];
```


```python
from matplotlib import pyplot as plt
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
        t = np.concatenate(tuple(data[r]),axis=1);
        rows.append(t);
    
    c = np.concatenate(tuple(rows));
    plt.imshow(c,cmap="gray")
    plt.axis('off')
    plt.show();
        
m,n=X.shape
rand_indice = np.random.permutation(m)
displayData(X[rand_indice[:49]]);

```


![png](output_2_0.png)



```python
from sklearn.linear_model import LogisticRegression
linear_model = LogisticRegression(solver='lbfgs')
clf = linear_model.fit(X,y);
print("Training Set Accuracy: ",clf.score(X,y))
```

    Training Set Accuracy:  0.9446



```python
m,n=X.shape
rand_indice = np.random.permutation(m)
row=4
col=3
test_train = X[rand_indice[:row*col]]
displayData(test_train,row,col)
print(np.reshape(clf.predict(test_train),(row,col)))
```


![png](output_4_0.png)


    [[ 4  4  6]
     [10  6  4]
     [ 6  6  5]
     [ 5  2  8]]

