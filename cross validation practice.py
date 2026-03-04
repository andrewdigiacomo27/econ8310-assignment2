import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/mnistTrain.csv")

#upper case before split
Y = data['Label']
#make sure you drop a column with the axis=1 argument
X = data.drop('Label', axis=1)

x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

kf = KFold(n_splits=10)

models = []

for train, test in kf.split(x):
    model = XGBClassifier(n_estimators = 50, max_depth = 3,
        learning_rate = 0.5, objective = 'multi:softmax'
            ).fit(x.values[train], y.values[train])
    accuracy = accuracy_score(y.values[test],
        model.predict(x.values[test]))
    print("Accuracy: ", accuracy)
    models.append([model, accuracy])

print("Mean Model Accuracy: ", np.mean([model[1] for model in models]))
print("Model Accuracy Standar Deviation: ", np.std([model[1] for model in models]))