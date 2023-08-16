# -*- coding: utf-8 -*-
"""Diabetes Prediction.ipynb
Diabetes Prediction

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""data extraction"""

df=pd.read_csv('diabetes.csv')

df.head()

df.describe()

df.shape

df['Outcome'].value_counts()

df['Age'].value_counts()

df.groupby('Outcome').mean()

X=df.drop(columns='Outcome',axis=1)
Y=df['Outcome']

"""Data Standardization"""

scaler=StandardScaler()
scaler.fit(X)

std=scaler.transform(X)

X=std
Y=df['Outcome']

"""Train Test Split"""

Xtrain, Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

Xtrain.shape
Xtest.shape

"""Training the Model"""

model=svm.SVC(kernel='linear')

model.fit(Xtrain,Ytrain)

"""Evaluate the Model"""

X_pred=model.predict(Xtrain)
train_acc=accuracy_score(X_pred,Ytrain)

print("Accuracy Score",train_acc)

Xtest_pred=model.predict(Xtest)
acc=accuracy_score(Xtest_pred,Ytest)

print("Actual Accuracy is: ", acc)

"""Prediction System"""

input_data=(5,166,72,19,175,25.8,0.587,51)
inp=np.asarray(input_data)

# reshape the array as we are predicting for one instance

inp_reshape=inp.reshape(1,-1)

std_inp=scaler.transform(inp_reshape)

pred=model.predict(std_inp)
if(pred[0]==0):
  print('No ur safe')
else:
  print('Sorry')

