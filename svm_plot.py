#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:01:02 2024

@author: irisnese
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

C_hyper = 40
Gamma = 0.03

fh = open("dataset_link_phishing_modified.csv", 'r')
data_all = []
classes = []

for line in fh:
    temp_list = line.strip().split(",")
    for i in range(len(temp_list)):
        if temp_list[i] == "One" or temp_list[i] == "one":
            temp_list[i] = 1
        elif temp_list[i] == "zero" or temp_list[i] == "Zero":
            temp_list[i] = 0
        elif temp_list[i] == "phishing":
            temp_list[i] = 1
        elif temp_list[i] == "legitimate":
            temp_list[i] = 0
    temp_list = [float(el) for el in temp_list]
    data_all.append(temp_list)
fh.close()

random.shuffle(data_all)
size_train = 6858    # 60% approximately
size_val = 2285      # 20% approx
size_test = 2286     # 20% approx
train = np.array(data_all[0:size_train])
val = np.array(data_all[size_train:size_train + size_val])
test = np.array(data_all[size_train + size_val:size_train + size_val + size_test])

classes_train = train[:, -1]
classes_val = val[:, -1]
classes_test = test[:, -1]

data_train = np.delete(train, -1, axis=1)
data_val = np.delete(val, -1, axis=1)
data_test = np.delete(test, -1, axis=1)

data_train = preprocessing.scale(data_train)
data_test = preprocessing.scale(data_test)
data_val = preprocessing.scale(data_val)

classifier = svm.SVC(kernel="rbf", C=C_hyper, gamma=Gamma)
classifier.fit(data_train, classes_train)

# Prediction and evaluation on the validation set
pred_val = classifier.predict(data_val)

# Performance metrics for the validation set
print("\nPerformance scores for validation set:")
print("Confusion matrix:")
conf_mat_val = confusion_matrix(classes_val, pred_val)
print(conf_mat_val)

accuracy_val = np.diag(conf_mat_val).sum() / conf_mat_val.sum()
precision_val = precision_score(classes_val, pred_val)
recall_val = recall_score(classes_val, pred_val)
f1_val = f1_score(classes_val, pred_val)

print(f"Accuracy for validation set: {accuracy_val:.3f} when C = {C_hyper}")
print(f"Precision: {precision_val:.3f}")
print(f"Recall: {recall_val:.3f}")
print(f"F1 Score: {f1_val:.3f}")

# Plotting confusion matrix for the validation set
plt.imshow(conf_mat_val, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(conf_mat_val.shape[0]):
    for j in range(conf_mat_val.shape[1]):
        plt.text(j, i, format(conf_mat_val[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat_val[i, j] > conf_mat_val.max() / 2 else "black")
plt.title('Confusion Matrix - Validation Set')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Prediction and evaluation on the test set
pred_test = classifier.predict(data_test)

# Performance metrics for the test set
print("\nPerformance scores for test set:")
print("Confusion matrix:")
conf_mat_test = confusion_matrix(classes_test, pred_test)
print(conf_mat_test)

accuracy_test = np.diag(conf_mat_test).sum() / conf_mat_test.sum()
precision_test = precision_score(classes_test, pred_test)
recall_test = recall_score(classes_test, pred_test)
f1_test = f1_score(classes_test, pred_test)

print(f"Accuracy for test set: {accuracy_test:.3f} when C = {C_hyper}")
print(f"Precision: {precision_test:.3f}")
print(f"Recall: {recall_test:.3f}")
print(f"F1 Score: {f1_test:.3f}")

# Plotting confusion matrix for the test set
plt.imshow(conf_mat_test, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(conf_mat_test.shape[0]):
    for j in range(conf_mat_test.shape[1]):
        plt.text(j, i, format(conf_mat_test[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat_test[i, j] > conf_mat_test.max() / 2 else "black")
plt.title('Confusion Matrix - Test Set')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
