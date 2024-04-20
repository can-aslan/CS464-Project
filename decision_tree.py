import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import validation_curve, cross_val_score
import matplotlib.pyplot as plt

def cross_val_depth(clf, X_train, y_train, depths):
    train_scores, valid_scores = validation_curve(
    clf, X_train, y_train,
    param_name='max_depth', param_range=depths, cv=5
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Validation Curve with Decision Tree (max_depth)")
    plt.xlabel("max_depth")
    plt.ylabel("Score")
    plt.ylim(0.8, 1.1)
    plt.plot(depths, train_scores_mean, label="Training score", color="darkorange", marker='o')
    plt.plot(depths, valid_scores_mean, label="Cross-validation score", color="navy", marker='o')
    plt.legend(loc="best")
    plt.show()

    best_depth = depths[np.argmax(valid_scores_mean)]
    print("Best Depth:", best_depth)
    return best_depth

def plot_tree_cm(clf, feature_names, target_names, predictions):
    fig = plt.figure(figsize=(15,10))
    _ = tree.plot_tree(clf, 
                    feature_names=feature_names,  
                    class_names=target_names,
                    filled=True)

    cm = confusion_matrix(y_test, predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = [0, 1])
    cm_display.plot()
    plt.show()

data = pd.read_csv('dataset_link_phishing.csv', low_memory = False)
data = pd.DataFrame(data)
unique_value_counts = {col: data[col].nunique() for col in data.columns}
columns_to_drop = [col for col, count in unique_value_counts.items() if count == 1]
data.drop(columns=columns_to_drop, inplace=True)
data = data.drop(columns=['Unnamed: 0'])
data.drop(columns=['url'], inplace=True)
data = pd.DataFrame(data)
data['status'] = data['status'].map({'legitimate': 1, 'phishing': 0})
data['domain_with_copyright'] = data['domain_with_copyright'].map({'one': 1, 'zero': 0, 'One': 1, 'Zero': 0, '1': 1, '0': 0})

x = data.copy()
X = x.drop('status', axis=1)
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20,shuffle=True)

depths = np.arange(1, 31, 1)  
depths[0] = 1

clf = DecisionTreeClassifier(random_state=42)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)

best_depth_gini = cross_val_depth(clf, X_train, y_train, depths)
best_depth_entropy = cross_val_depth(clf_entropy, X_train, y_train, depths)

feature_names = X.columns.tolist()
target_names = ['phishing', 'legitimate']

best_clf_gini = DecisionTreeClassifier(max_depth=best_depth_gini, random_state=42)
best_clf_gini.fit(X_train, y_train)
test_pred_gini = best_clf_gini.predict(X_test)
print("Accuracy with Best Depth(gini):", accuracy_score(y_test, test_pred_gini))
plot_tree_cm(best_clf_gini,feature_names, target_names, test_pred_gini)

best_clf_entropy = DecisionTreeClassifier(max_depth=best_depth_entropy, random_state=42)
best_clf_entropy.fit(X_train, y_train)
test_pred_entropy = best_clf_entropy.predict(X_test)
print("Accuracy with Best Depth(entropy):", accuracy_score(y_test, test_pred_entropy))
plot_tree_cm(best_clf_entropy,feature_names, target_names, test_pred_entropy)


