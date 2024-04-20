import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    # Load the data
    data = pd.read_csv("data.csv", low_memory=False)

    # Modify features
    data["status"] = data["status"].apply(lambda x: 1 if x == "phishing" else 0)
    data["domain_with_copyright"] = data["domain_with_copyright"].apply(lambda x: 1 if x == "one" else 0)

    if "url" in data.columns:
        data = data.drop("url", axis=1)

    X = data.drop("status", axis=1)
    y = data["status"]

    # Split data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = SS()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    min_k = 1
    max_k = len(train_y) // 2

    # Generate k values using numpy logspace
    num_points = 20
    range_of_k = np.logspace(np.log10(min_k), np.log10(max_k), num=num_points, base=10).astype(int)
    range_of_k = np.unique(range_of_k) # ensures all k values are unique and covers up to max_k

    accuracies = []
    k_values = []

    # Iterate over the generated range of k values
    for k in range_of_k:
        knn = KNN(n_neighbors=k)
        knn.fit(train_x, train_y)

        pred_y = knn.predict(test_x)
        accuracy = accuracy_score(test_y, pred_y)

        accuracies.append(accuracy)
        k_values.append(k)

        print(f"K:{k} | Accuracy: {accuracy * 100:.3f}%")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracies, marker="o", linestyle="-", color="b")
    plt.title("K-Value vs. Accuracy")
    plt.xlabel("Number of Neighbors: K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xscale("log")
    plt.ylim([0, 1])
    plt.show()

    # Select the best K
    best_k = k_values[accuracies.index(max(accuracies))]
    print(f"Best K: {best_k} with Accuracy: {max(accuracies) * 100:.3f}%")
    
    # Train again with the best K
    knn = KNN(n_neighbors=best_k)
    knn.fit(train_x, train_y)
    pred_y = knn.predict(test_x)

    # Display the confusion matrix for the best K
    cm = confusion_matrix(test_y, pred_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for K={best_k}")
    plt.show()
    return

main()
