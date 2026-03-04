import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/mnistTrain.csv")

#upper case before split
Y = data['Label']
#make sure you drop a column with the axis=1 argument
X = data.drop('Label', axis=1)

x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

#create the model

from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=50, max_depth=3,
    learning_rate=0.5, objective='multi:softmax')

#fit, predict, and check accuracy

xgb.fit(x, y)
pred = xgb.predict(xt)
print(accuracy_score(yt, pred)*100)

#cross validation

from sklearn.model_selection import KFold

#if we have imported data and created x, y already:
kf = KFold(n_splits=10) #10 "Folds"

models = [] #we will store our models here

for train, test in kf.split(x): #iterate over folds
    model = model.fit(x[train], y[train]) #fit model
    accuracy = accuracy_score(y[train], #store accuracy
        model.predict(x[test]))
    print("Accuracy:", accuracy_score(y[test], model.predict(x[test]))) #print results
    models.append([model, accuracy])

print("Mean Model Accuracy: ",                     #print aggregate
    np.mean([model[1] for  model in models]))