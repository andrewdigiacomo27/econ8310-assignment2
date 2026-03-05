import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

trainData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
testData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

X = trainData.drop(['meal', 'id', 'DateTime'], axis=1)
Y = trainData['meal']

Xt = testData.drop(['meal', 'id', 'DateTime'], axis=1)
Yt = testData['meal']

# x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

# model = DecisionTreeClassifier()
# modelFit = model.fit(X, Y)

# pred = modelFit.predict(Xt)

# pred = [int(i) for i in pred]

from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# modelFit = model.fit(X, Y)
# pred = modelFit.predict(Xt)

# pred = [int(i) for i in pred]

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, max_depth=75, learning_rate=0.5)
modelFit = model.fit(X, Y)
pred = modelFit.predict(Xt)

pred = [int(i) for i in pred]