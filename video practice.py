import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/refs/heads/master/DataSets/mnistTrain.csv")

label = data.iloc[1, 0]
image = data.iloc[1, 1:].values.reshape(28,28)

px.imshow(image)

#separate labels from inputs
Y = data['Label']
X = data.drop('Label', axis=1)

#randomly create train and test data
x, xt, y, yt = train_test_split(X, Y, test_size=0.8,
    random_state=42)

#generate the tree model
tree = DecisionTreeClassifier(max_depth = 5, min_samples_leaf= 10)
# fit the tree to the training data
tclf = tree.fit(x, y)
# make prediction
tpred = tclf.predict(xt)
# print the accuracy score of the fitted model
print("\nThe decision tree has an accuracy of : %s\n"
    % str(accuracy_score(tpred, yt)))

from sklearn.ensemble import BaggingClassifier

#generate the bagging model
bag = BaggingClassifier(n_estimators=100, n_jobs=-1,
    random_state=42)
# fit the model to the training data
baclf = bag.fit(x, y)
# make predictions
bapred = baclf.predict(xt)
# print the accuracy score fo the fitted model
print("The bagging algorthm has an accuracy of : %s\n"
    % str(accuracy_score(bapred, yt)))

from sklearn.ensemble import RandomForestClassifier
#generate the random forest model
forest = RandomForestClassifier(n_estimators=300, n_jobs=-1,
    random_state=42, max_depth=14)
#fit the model to the training data
fclf = forest.fit(x, y)
#make predictions
fpred = fclf.predict(xt)
#preint the accuracy score of the fitted model
print("The random forest has an accuracy of : %s\n"
    % str(accuracy_score(fpred, yt)))