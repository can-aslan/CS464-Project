import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler as SS
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

def main():
    # Load the data and drop the url and status columns
    train_data = pd.read_csv("training.csv", low_memory=False)
    val_data = pd.read_csv("validation.csv", low_memory=False)
    test_data = pd.read_csv("test.csv", low_memory=False)

    # Extract features and target variable from each dataset
    train_x = train_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]

    val_x = val_data.iloc[:, :-1]
    val_y = val_data.iloc[:, -1]

    test_x = test_data.iloc[:, :-1]
    test_y = test_data.iloc[:, -1]

    # Standardize the features
    scaler = SS()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    min_k = 1
    max_k = len(train_y) // 2

    # Generate k values using numpy logspace
    num_points = 20
    range_of_k = np.logspace(np.log10(min_k), np.log10(max_k), num=num_points, base=10).astype(int)
    range_of_k = np.unique(range_of_k) # ensures all k values are unique and covers up to max_k

    accuracies = []
    k_values = []
    auc_scores = []
    roc_data = []

    # Iterate over the generated range of k values
    for k in range_of_k:
        knn = KNN(n_neighbors=k)
        knn.fit(train_x, train_y)

        # Predict on the validation set
        val_pred_probs = knn.predict_proba(val_x)[:, 1]
        accuracy = accuracy_score(val_y, knn.predict(val_x))
        fpr, tpr, _ = roc_curve(val_y, val_pred_probs)
        roc_auc = auc(fpr, tpr)

        accuracies.append(accuracy)
        k_values.append(k)
        auc_scores.append(roc_auc)
        roc_data.append((fpr, tpr, roc_auc))

        print(f"K:{k} | Validation Accuracy: {accuracy * 100:.3f}% | AUC: {roc_auc:.5f}")

    # Plot the results
    plt.figure(figsize=(10, 8))
    for (fpr, tpr, roc_auc), k in zip(roc_data, k_values):
        plt.plot(fpr, tpr, label=f"K={k} (AUC = {roc_auc:.5f})")

    plt.title("ROC Curves for different values of K")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    best_auc_k = k_values[auc_scores.index(max(auc_scores))]
    best_val_k = k_values[accuracies.index(max(accuracies))]

    plt.figure(figsize=(10, 8))
    best_k_index = k_values.index(best_auc_k)
    k1_index = k_values.index(1)

    for index in [best_k_index, k1_index]:
        fpr, tpr, roc_auc = roc_data[index]
        plt.plot(fpr, tpr, label=f"K={k_values[index]} (AUC = {roc_auc:.5f})")
    
    plt.title(f"ROC Curves for Best K based on Validation Accuracy={best_val_k} and Best K based on AUC={best_auc_k}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracies, marker="o", linestyle="-", color="b")
    plt.title("K-Value vs. Validation Accuracy")
    plt.xlabel("Number of Neighbors: K")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.xscale("log")
    plt.ylim([0, 1])
    plt.show()

    # Select the best K
    best_k = k_values[accuracies.index(max(accuracies))]
    print(f"Best K from Validation Accuracies: {best_k} with Validation Accuracy: {max(accuracies) * 100:.3f}%")

    # Combine training and validation set for final model training
    combined_train_x = np.concatenate((train_x, val_x), axis=0)
    combined_train_y = np.concatenate((train_y, val_y), axis=0)

    # Standardize the combined features
    scaler.fit(combined_train_x) # Refitting to the combined data
    combined_train_x = scaler.transform(combined_train_x)
    test_x = scaler.transform(test_x) # Transform the test set with the new scaler fit

    # Train again with the best K using the combined training and validation set
    knn = KNN(n_neighbors=best_k)
    knn.fit(combined_train_x, combined_train_y)

    # Predict on the test set and display the confusion matrix
    test_pred_y = knn.predict(test_x)
    cm = confusion_matrix(test_y, test_pred_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for K={best_k} on Test Set")
    plt.show()

    # Select the best K from AUC
    best_auc_k_accuracy = accuracies[auc_scores.index(max(auc_scores))]
    print(f"Best K from AUC: {best_auc_k} with Validation Accuracy: {best_auc_k_accuracy * 100:.3f}%")

    # Combine training and validation set for final model training
    combined_train_x = np.concatenate((train_x, val_x), axis=0)
    combined_train_y = np.concatenate((train_y, val_y), axis=0)

    # Standardize the combined features
    scaler.fit(combined_train_x) # Refitting to the combined data
    combined_train_x = scaler.transform(combined_train_x)
    test_x = scaler.transform(test_x) # Transform the test set with the new scaler fit

    # Train again with the best K using the combined training and validation set
    best_auc_knn = KNN(n_neighbors=best_auc_k)
    best_auc_knn.fit(combined_train_x, combined_train_y)

    # Predict on the test set and display the confusion matrix
    test_pred_y = best_auc_knn.predict(test_x)
    cm = confusion_matrix(test_y, test_pred_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for K={best_auc_k} on Test Set")
    plt.show()

    return

main()