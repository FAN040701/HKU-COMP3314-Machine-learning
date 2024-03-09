#--- Load packages for datasets---
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import pandas as pd
#--- Load packages for logistic regression and random forest---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Load packages for train/test split---
from sklearn.model_selection import train_test_split

# Now, we will start to train logistic regression models on the Iris and Wine datasets.

# TODO: Load the Iris dataset using sklearn.
Iris = load_iris()
X, y = Iris.data, Iris.target
# Split train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)

# TODO: Initialize a logistic regression model for the Iris dataset.
# Here, you only need to tune the inverse regularization parameter `C`. 
# Please set `random_state` to 3.
def logistic(C_list, X_train, y_train, X_test, y_test):
    #This function is to traverse C in C_list for the logistic regression so that we can try many values an one time.
    test_list = []
    train_list = []
    for C in C_list:
        lr = LogisticRegression(C=C, solver='liblinear', multi_class='ovr', random_state=3)
        lr.fit(X_train, y_train)
        train_list.append(1 - lr.score(X_train, y_train))
        test_list.append(1 - lr.score(X_test, y_test))
    return train_list, test_list

def show(C_list, train_list, test_list):
    #This is for plotting purpose, the x-axis is C and the y-axis is the error
    plt.figure(figsize=(16, 8), dpi=80)

    plt.plot(C_list, train_list, label="train error", color="orange", linewidth=3, alpha=0.8)
    plt.plot(C_list, test_list, label="test error", color="cyan", linewidth=3, alpha=0.8)

    plt.grid(alpha=0.8)
    plt.xlabel("C", fontsize=15)
    plt.ylabel("Error", fontsize=15)
    plt.legend(loc="best")
    plt.xticks(C_list)
    plt.yticks([0.0,0.05, 0.1,0.15, 0.2, 0.25, 0.3])
    plt.title("Error vs C")
    plt.show()
    
def form_table(C_list, train_list, test_list):
    # Create a DataFrame from the error rates and corresponding C values
    error_rates_df = pd.DataFrame({
        'C': C_list,
        'Training Error': train_list,
        'Testing Error': test_list
    })
    return error_rates_df

# main
C_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
train_list, test_list = logistic(C_list, X_train, y_train, X_test, y_test)
show(C_list, train_list, test_list)
# Create a DataFrame from the error rates and corresponding C values
print(form_table(C_list, train_list, test_list))

C_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
train_list, test_list = logistic(C_list, X_train, y_train, X_test, y_test)
show(C_list, train_list, test_list)
print(form_table(C_list, train_list, test_list))