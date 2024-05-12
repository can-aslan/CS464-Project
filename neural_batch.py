import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd
import pickle

lr = 4
epochs = 200

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def predict(A3):
    return np.round(A3)

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


scaler = preprocessing.StandardScaler().fit(data_train)
data_train = scaler.transform(data_train)
data_val = scaler.transform(data_val)
data_test = scaler.transform(data_test)

data_train = data_train.T

W1 = np.random.randn(20, 79)
b1 = np.random.randn(20, 1)

W2 = np.random.randn(20, 20)
b2 = np.random.randn(20, 1)

W3 = np.random.randn(1, 20)
b3 = np.random.randn(1, 1)

train_accuracies = []
val_accuracies = []

for i in range(epochs):
    #classes_train = classes_train.reshape(1, -1)

    Z1 = W1.dot(data_train) + b1
    A1 = sigmoid(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = sigmoid(Z3)
    

    m = len(classes_train)

    dZ3 = A3 - classes_train.to_numpy().reshape(A3.shape)
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * sigmoid(Z2) * (1 - sigmoid(Z2))
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1 / m) * dZ1.dot(data_train.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    W1 = W1 - lr * (dW1)
    b1 = b1 - lr * (db1)
    W2 = W2 - lr * (dW2)
    b2 = b2 - lr * (db2)
    W3 = W3 - lr * (dW3)
    b3 = b3 - lr * (db3)
    pred_train = predict(A3)
    pred_train = predict(A3.reshape(-1))

    train_accuracy = accuracy_score(classes_train.squeeze(), pred_train.squeeze())
    train_accuracies.append(train_accuracy)

    Z_val1 = W1.dot(data_val.T) + b1
    A_val1 = sigmoid(Z_val1)

    Z_val2 = W2.dot(A_val1) + b2
    A_val2 = sigmoid(Z_val2)

    Z_val3 = W3.dot(A_val2) + b3
    A_val3 = sigmoid(Z_val3)

    pred_val = predict(A_val3)
    val_accuracy = accuracy_score(classes_val, pred_val.squeeze())
    val_accuracies.append(val_accuracy)

    print("Iteration ", i, "Train Accuracy = ", train_accuracy, "Val Accuracy = ", val_accuracy)

plt.plot(range(epochs), train_accuracies, label='Training Accuracy')
plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Epochs')
plt.legend()
plt.show()

pred_val = predict(A_val3).squeeze()  # Squeeze to match shape
conf_mat_val = confusion_matrix(classes_val, pred_val)

# Performance metrics for validation set
accuracy_val = accuracy_score(classes_val, pred_val)
precision_val = precision_score(classes_val, pred_val)
recall_val = recall_score(classes_val, pred_val)
f1_val = f1_score(classes_val, pred_val)

print("\nValidation set: ", "Accuracy = ", accuracy_val)
print("Precision: ", precision_val)
print("Recall: ", recall_val)
print("F1 Score: ", f1_val)

# Confusion matrix for validation set
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

data_test = data_test.T
Z_test1 = W1.dot(data_test) + b1
A_test1 = sigmoid(Z_test1)

Z_test2 = W2.dot(A_test1) + b2
A_test2 = sigmoid(Z_test2)

Z_test3 = W3.dot(A_test2) + b3
A_test3 = sigmoid(Z_test3)
pred_test = predict(A_test3)

accuracy_test = accuracy_score(classes_test, pred_test.squeeze())
precision_test = precision_score(classes_test, pred_test.squeeze())
recall_test = recall_score(classes_test, pred_test.squeeze())
f1_test = f1_score(classes_test, pred_test.squeeze())

print("\nTest set: ", "Accuracy = ", accuracy_test)
print("Precision: ", precision_test)
print("Recall: ", recall_test)
print("F1 Score: ", f1_test)


pred_test = predict(A_test3).squeeze()  # Squeeze to match shape
conf_mat_test = confusion_matrix(classes_test, pred_test)
# Confusion matrix for test set
conf_mat_test = confusion_matrix(classes_test, pred_test)
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

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(classes_test, A_test3.squeeze())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(classes_test, A_test3.squeeze())
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

model = {
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2,
    'W3': W3,
    'b3': b3
}

with open('trained_model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
