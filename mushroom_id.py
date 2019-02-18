import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
df = pd.read_csv('./mushrooms.csv', header=0)

df_random = df.sample(frac=1)

df_total = pd.get_dummies(df_random)

#Train Validate and Test Sets
df_train = df_total[:4875]
df_validate = df_total[4875:6500]
df_test = df_total[6500:]

#Getting target dataset:

y_train = df_train.iloc[:,0].values
y_validate = df_validate.iloc[:,0].values
y_test = df_test.iloc[:,0].values

X_train = X_train.values
X_validate = X_validate.values
X_test = X_test.values

# Functions

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def cross_entropy(y, p):
    return -np.sum(y*np.log(p) + (1 - y)*np.log(1 - p))

def accuracy(y, p):
    return np.mean(y == np.round(p))

def precision (y, p, thres):
    y_hat = (p > thres)
    tp = y_hat.dot(y)
    fp = y_hat.dot(1-y)
    return tp/(tp+fp) 

#Starting with random weights
w_0 = np.random.randn(X_train.shape[1])

#Training model and plotting the J_train
w = w_0.copy()

J_train = []
eta = 1e-3
epochs = int(1e5)
l2 = 8

for t in range(epochs):
    p = sigmoid(X_train.dot(w))
    J_train.append(cross_entropy(y_train, p))
    w -= eta*(X_train.T.dot(p - y_train) + l2*w)

plt.figure(figsize = (8,6))
plt.plot(J_train)

#Test Accuracy and Precision:
print("Accuracy: {}".format(accuracy(y_train, p)))
print("Precision: {}".format(precision(y_train, p, .95)))

#Cross Validation
p_val = sigmoid(X_validate.dot(w))
    
print("Accuracy: {}".format(accuracy(y_validate, p_val)))
print("Precision: {}".format(precision(y_validate, p_val, .95)))

#Test
p_test = sigmoid(X_test.dot(w))
    
print("Accuracy: {}".format(accuracy(y_test, p_test)))
print("Precision: {}".format(precision(y_test, p_test, .95)))
