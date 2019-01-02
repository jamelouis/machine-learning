---
layout: post
title:  "机器学习 - Linear Regression (iOS)"
date:   2019-01-02 22:45:00 +0800
categories: machine-learning-ios
---

## 导出 mlmodel

```python
import coremltools
input_features = ["populations"]
output_feature = "profits"

model = coremltools.converters.sklearn.convert(reg, input_features, output_feature)
model.save("populations_profits.mlmodel")
```

## iOS App

![]({{ "assets/images/ios/linear-regression.gif" | relative_url}})

## Reference

* [Begining Machine Learning with Scikit Learn][1]

[1]:https://www.raywenderlich.com/174-beginning-machine-learning-with-scikit-learn "Begining Machine Learning with Scikit Learn"