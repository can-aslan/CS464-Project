import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
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

test_data = pd.read_csv('test.csv', low_memory = False)
train_data = pd.read_csv('training.csv', low_memory = False)
val_data = pd.read_csv('validation.csv', low_memory = False)


x_test = test_data.copy()
X_test = x_test.drop('status', axis=1)
y_test = test_data['status']

x_train = train_data.copy()
X_train = x_train.drop('status', axis=1)
y_train = train_data['status']

x_val = val_data.copy()
X_val = x_val.drop('status', axis=1)
y_val = val_data['status']

depths = np.arange(15, 25, 1)  

best_depth = None
best_accuracy = 0.0

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Depth: {depth}, Validation Accuracy: {accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth

print(f"Best Depth: {best_depth}, Best Validation Accuracy: {best_accuracy}")
best_clf_gini = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_clf_gini.fit(X_train, y_train)
test_pred_gini = best_clf_gini.predict(X_test)
conf_matrix = confusion_matrix(y_test, test_pred_gini)

accuracy = accuracy_score(y_test, test_pred_gini)
precision = precision_score(y_test, test_pred_gini, average='macro')
recall = recall_score(y_test, test_pred_gini, average='macro')
f1 = f1_score(y_test, test_pred_gini, average='macro')

print("Accuracy:", accuracy)
print("Macro Precision:", precision)
print("Macro Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


target_names = ['phishing', 'legitimate']

cm = confusion_matrix(y_test, test_pred_gini)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = target_names)
cm_display.plot()
plt.show()

#plot_tree_cm(best_clf_gini,feature_names, target_names, test_pred_gini)




