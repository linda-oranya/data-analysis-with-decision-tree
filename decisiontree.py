#!/usr/bin/env python
# coding: utf-8


# DECISION TREE ALGORITHM


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()


df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

print(df.head())

#data preprocessing
print(df.isnull().any())


print(df.dtypes)

#To get a summary of the entire data sets
print(df.describe())

#To visualize the relationship between data sets
sns.pairplot(df, hue = 'target')

data1 = df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].values
data2 = df['target'].values

(train_data1,test_data1,train_data2,test_data2) = train_test_split(data1,data2,train_size = 0.7, random_state=1)


dtc = DecisionTreeClassifier()
dtc.fit(test_data1,test_data2)
print(dtc.score(test_data1,test_data2))




