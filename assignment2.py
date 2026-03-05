import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

trainData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
testData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

X = trainData.drop(['meal', 'id', 'DateTime'], axis=1)
Y = trainData['meal']

Xt = testData.drop(['meal', 'id', 'DateTime'], axis=1)
Yt = testData['meal']

# x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

model = DecisionTreeClassifier()
modelFit = model.fit(X, Y)

pred = modelFit.predict(Xt)

pred = [int(i) for i in pred]