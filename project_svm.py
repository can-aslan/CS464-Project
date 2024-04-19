#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:36:47 2024

@author: irisnese
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing

C_hyper = 50
D_hyper=2
Gamma= 0.015
SAMPLE_SIZE = 11429
fh=open("dataset_link_phishing_modified.csv", 'r')
   
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



classifier = svm.SVC(kernel = "rbf", C = C_hyper, gamma=Gamma)
classifier.fit(data_train, classes_train)
pred = classifier.predict(data_val)



TP=0
TN=0
FP=0
FN=0
for i in range(len(pred)):
    if pred[i] == 1 and classes_val[i] == 1:
        TP += 1
    elif pred[i] == 1 and classes_val[i] == 0:
        FP += 1
    elif pred[i] == 0 and classes_val[i] == 1:
        FN += 1
    elif pred[i] == 0 and classes_val[i] == 0:
        TN += 1
    else:
        print("no")
print("\nPerformance scores for validation set:")
accuracy = (TP+TN)/(TP+TN+FP+FN)
print(f"Accuracy for validaton set = {accuracy:.3f} when C = {C_hyper}")
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



pred = classifier.predict(data_test)
TP=0
TN=0
FP=0
FN=0
for i in range(len(pred)):
    if pred[i] == 1 and classes_test[i] == 1:
        TP += 1
    elif pred[i] == 1 and classes_test[i] == 0:
        FP += 1
    elif pred[i] == 0 and classes_test[i] == 1:
        FN += 1
    elif pred[i] == 0 and classes_test[i] == 0:
        TN += 1
    else:
        print("no")
print("\nPerformance scores for test set:")
accuracy = (TP+TN)/(TP+TN+FP+FN)
print(f"Accuracy for test set = {accuracy:.3f} when C = {C_hyper}")
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




























