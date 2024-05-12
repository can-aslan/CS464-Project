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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd
import pickle

C_hyper = 40
Gamma = 0.03

# Read the training data
train_df = pd.read_csv("training.csv")

# Read the validation data
val_df = pd.read_csv("validation.csv")

# Read the test data
test_df = pd.read_csv("test.csv")

# Separate features and classes for training data
data_train = train_df.drop(columns=['status'])
classes_train = train_df['status']

# Separate features and classes for validation data
data_val = val_df.drop(columns=['status'])
classes_val = val_df['status']

# Separate features and classes for test data
data_test = test_df.drop(columns=['status'])
classes_test = test_df['status']

data_train = preprocessing.scale(data_train)
data_test = preprocessing.scale(data_test)
data_val = preprocessing.scale(data_val)

classifier = svm.SVC(kernel="rbf", C=C_hyper, gamma = Gamma)
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

print(f"Accuracy for validation set: {accuracy_val:.3f} when C = {C_hyper} and gamma = {Gamma}")
print(f"Precision: {precision_val:.3f}")
print(f"Recall: {recall_val:.3f}")
print(f"F1 Score: {f1_val:.3f}")

# Plotting confusion matrix for the validation set
plt.imshow(conf_mat_val, interpolation='nearest', cmap=plt.cm.Blues)
tick_marks = np.arange(len(np.unique(classes_val)))
plt.xticks(tick_marks, ['Phishing', 'Legitimate'])
plt.yticks(tick_marks, ['Phishing', 'Legitimate'])
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

print(f"Accuracy for test set: {accuracy_test:.3f} when C = {C_hyper} and gamma = {Gamma}")
print(f"Precision: {precision_test:.3f}")
print(f"Recall: {recall_test:.3f}")
print(f"F1 Score: {f1_test:.3f}")

# Plotting confusion matrix for the test set
plt.imshow(conf_mat_test, interpolation='nearest', cmap=plt.cm.Blues)
tick_marks = np.arange(len(np.unique(classes_test)))
plt.xticks(tick_marks, ['Phishing', 'Legitimate'])
plt.yticks(tick_marks, ['Phishing', 'Legitimate'])
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


# Get the decision function scores
scores_val = classifier.decision_function(data_val)
scores_test = classifier.decision_function(data_test)

# Compute ROC curve and ROC area for the validation set
fpr_val, tpr_val, _ = roc_curve(classes_val, scores_val)
roc_auc_val = auc(fpr_val, tpr_val)

# Compute ROC curve and ROC area for the test set
fpr_test, tpr_test, _ = roc_curve(classes_test, scores_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curve
plt.figure()
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Validation Set')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Test Set')
plt.legend(loc="lower right")
plt.show()



# Compute precision-recall curve for the validation set
precision_val, recall_val, _ = precision_recall_curve(classes_val, scores_val)

# Compute precision-recall curve for the test set
precision_test, recall_test, _ = precision_recall_curve(classes_test, scores_test)

# Plot precision-recall curve for validation set
plt.figure()
plt.plot(recall_val, precision_val, color='blue', lw=2, label='Precision-Recall curve (validation)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Validation Set')
plt.legend(loc="lower left")
plt.show()

# Plot precision-recall curve for test set
plt.figure()
plt.plot(recall_test, precision_test, color='blue', lw=2, label='Precision-Recall curve (test)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Set')
plt.legend(loc="lower left")
plt.show()

# Define the filename for the pickle file
filename = 'svm_model.pkl'

# Save the trained model to the pickle file
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)


