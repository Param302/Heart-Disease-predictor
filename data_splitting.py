import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("./data/heart.csv")
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=0)

X_train["target"] = Y_train
training_data = X_train
test_data = X_test

training_data.to_csv("./data/train.csv", index=False)
test_data.to_csv("./data/test.csv", index=False)