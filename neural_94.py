#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:59:08 2024

@author: irisnese
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing

lr = 0.1
epochs = 200
def ReLU(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    A = 0.5 * (1 + np.tanh(Z/2))
    return A

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))
    
def predict(A2):
    temp = A2.reshape(A2.size)
    arr = np.array([np.argmax([0.5, el]) for el in temp])
    return arr
        
fh=open("data.csv", 'r')
   
data_all = []
classes =[]


for line in fh:
    temp_list=line.strip().split(",")
    for i in range(len(temp_list)):
        if temp_list[i]=="One" or temp_list[i]=="one":
            temp_list[i]=1
        elif temp_list[i]=="zero" or temp_list[i]=="Zero":
            temp_list[i]=0            
        elif temp_list[i]=="phishing":
            temp_list[i]=1
        elif temp_list[i]=="legitimate":
            temp_list[i]=0
    temp_list = [float(el) for el in temp_list]
    data_all.append(temp_list)
fh.close()     



random.shuffle(data_all)
size_train = 6858    # 60% approximately
size_val = 2285      # 20% approx
size_test = 2286     # 20% approx
train = np.array(data_all[0:size_train])
val = np.array(data_all[size_train : size_train + size_val])
test = np.array(data_all[size_train+size_val : size_train+size_val+size_test])


classes_train = train[:,-1]
classes_val = val[:,-1]
classes_test = test[:,-1]

data_train = np.delete(train, -1, axis=1)
data_val = np.delete(val, -1, axis=1)
data_test = np.delete(test, -1, axis=1)

data_train = preprocessing.scale(data_train)
data_test = preprocessing.scale(data_test)
data_val = preprocessing.scale(data_val)


data_train = data_train.T

W1 = np.random.randn(100,84)
b1 = np.random.randn(100,1)

W2 = np.random.randn(100,100)
b2 = np.random.randn(100,1)

W3 = np.random.randn(1,100)
b3 = np.random.randn(1,1)

for i in range(epochs):
    Z1= W1.dot(data_train) + b1
    A1 = sigmoid(Z1)
    
    Z2=W2.dot(A1) + b2
    A2=sigmoid(Z2)
    
    Z3=W3.dot(A2) + b3
    A3=sigmoid(Z3)
    
    m= len(classes_train)
    
    
    dZ3 = A3-classes_train
    dW3 = (1/m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    dZ2 = W3.T.dot(dZ3) + (Z2>0)
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) + (Z1>0)  # (Z1>0) is the derivate of Relu, either 0 or 1
    dW1 = (1/m) * dZ1.dot(data_train.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    W1 = W1- lr*(dW1)
    b1 = b1-lr*(db1)
    W2 = W2 - lr*(dW2)
    b2 = b2-lr*(db2)
    W3 = W3 - lr*(dW3)
    b3 = b3-lr*(db3)
    pred = predict(A3)
    accuracy = np.sum(pred == classes_train) / classes_train.size
    print("Iteration ", i, "Accuracy = ", accuracy)


data_val = data_val.T

Z1= W1.dot(data_val) + b1
A1 = sigmoid(Z1)

Z2=W2.dot(A1) + b2
A2=sigmoid(Z2)

Z3=W3.dot(A2) + b3
A3=sigmoid(Z3)

val_pred = predict(A3)
accuracy_val = np.sum(val_pred == classes_val) / classes_val.size
print("Validation: ", "Accuracy = ", accuracy_val)


data_test = data_test.T
Z1= W1.dot(data_test) + b1
A1 = sigmoid(Z1)

Z2=W2.dot(A1) + b2
A2=sigmoid(Z2)

Z3=W3.dot(A2) + b3
A3=sigmoid(Z3)

test_pred = predict(A3)
accuracy_test = np.sum(test_pred == classes_test) / classes_test.size
print("Test set: ", "Accuracy = ", accuracy_test)

TP=0
TN=0
FN=0
FP=0
for sample in range(len(classes_test)):
    if test_pred[sample] == 1 and classes_test[sample]==1.0:
        TP +=1
    elif test_pred[sample] == 0 and classes_test[sample]==0.0:
        TN +=1
    elif test_pred[sample] ==1 and classes_test[sample]==0.0:
        FP +=1
    elif test_pred[sample] ==0 and  classes_test[sample]==1.0:
        FN +=1


print("\nPerformance scores for test set:")
accuracy = (TP+TN)/(TP+TN+FP+FN)
print(f"Accuracy for test set = {accuracy:.2f} when lr = {lr}")

prec_p = TP/(TP+FP)
prec_n = TN/(TN+FN)
macro_prec = (prec_p+prec_n)/2
micro_prec = (TP+TN)/(TP+FP+TN+FN)
recall = TP/(TP+FN)
NPV = TN / (TN+FN)
FPR = FP / (FP+TN)
FDR = 1-prec_p
f1 = 2*prec_p*recall / (prec_p+recall)
f2 = 5*prec_p*recall / (4*prec_p + recall)
print("Confusion matrix:")
print(f"{TP}\t\t{FP}")
print(f"{FN}\t\t{TN}")
print(f"Macro average of precision={macro_prec:.2f}")
print(f"Micro average of precision={micro_prec:.2f}")
print(f"Recall = {recall:.2f}")
print(f"Negative predictive value(NPV)={NPV:.2f}")
print(f"FPR={FPR:.2f}")
print(f"FDR={FDR:.2f}")
print(f"f1={f1:.2f}")
print(f"f2={f2:.2f}")
