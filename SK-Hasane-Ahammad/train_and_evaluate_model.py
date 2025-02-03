from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Model
from sklearn import metrics
from sklearn.svm import SVC
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import keras

# load data
data0 = pd.read_csv('processed_data/urldata.csv')
print(data0.head())

# display data information
print(data0.shape)
print(data0.columns)
print(data0.info())

"""
# plotting the data distribution
data0.hist(bins=50, figsize=(15,15))
plt.show()

# correlation heatmap
plt.figure(figsize=(15,13))
sns.heatmap(data0.corr())
plt.show()
"""

# data preprocessing & EDA
print(data0.describe())
data = data0.drop(['Domain'], axis = 1).copy()      # drop 'Domain' column
print(data.isnull().sum())                          # check for null values
data = data.sample(frac=1).reset_index(drop=True)   # shuffle the rows
print(data.head())

# split train and test data
y = data['Label']
X = data.drop('Label', axis=1)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=12)

print(X_train.shape, X_test.shape)

# machine learning models & training
ML_Model = []
acc_train = []
acc_test = []

def storeResults(model, a, b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))

# decision tree
tree = DecisionTreeClassifier(max_depth = 5)
tree.fit(X_train, y_train)

y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

acc_train_tree = accuracy_score(y_train, y_train_tree)
acc_test_tree = accuracy_score(y_test, y_test_tree)

print("Decision Tree: Accuracy on training data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test data: {:.3f}".format(acc_test_tree))

"""
# feature importance in tree model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
"""

storeResults('Decision Tree', acc_train_tree, acc_test_tree)

# random forest
forest = RandomForestClassifier(max_depth=5)
forest.fit(X_train, y_train)

y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test data: {:.3f}".format(acc_test_forest))

"""
# feature importance in random forest model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
"""

storeResults('Random Forest', acc_train_forest, acc_test_forest)

# multilayer perceptrons
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))
mlp.fit(X_train, y_train)

y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)

acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test data: {:.3f}".format(acc_test_mlp))

storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

# XGBoost model
xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
xgb.fit(X_train, y_train)

y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)

print("XGBoost: Accuracy on training data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test data: {:.3f}".format(acc_test_xgb))

storeResults('XGBoost', acc_train_xgb, acc_test_xgb)

# autoencoder neural network
input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)

encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim-2), activation='relu')(code)

decoder = Dense(int(encoding_dim), activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2)

acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

print('\nAutoencoder: Accuracy on training data: {:.3f}' .format(acc_train_auto))
print('Autoencoder: Accuracy on test data: {:.3f}' .format(acc_test_auto))

storeResults('AutoEncoder', acc_train_auto, acc_test_auto)

# support vector machines
svm = SVC(kernel='linear', C=1.0, random_state=12)
svm.fit(X_train, y_train)

y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)

acc_train_svm = accuracy_score(y_train,y_train_svm)
acc_test_svm = accuracy_score(y_test,y_test_svm)

print("SVM: Accuracy on training data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test data: {:.3f}".format(acc_test_svm))

storeResults('SVM', acc_train_svm, acc_test_svm)

# comparison of models
results = pd.DataFrame({ 'ML Model': ML_Model,
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
print(results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False))
