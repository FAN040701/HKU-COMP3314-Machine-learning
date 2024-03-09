#--- Load packages for datasets---
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

#--- Load packages for logistic regression and random forest---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Load packages for train/test split---
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import pandas as pd

def show_depth(depth_list, train_list, test_list):
    #We can use this function to plot the error rate of the training and testing dataset.
    #This is for plotting purpose, the x-axis is maximum depth and the y-axis is the error
    plt.figure(figsize=(16, 8), dpi=80)

    plt.plot(depth_list, train_list, label="train error", color="orange", linewidth=3, alpha=0.8)
    plt.plot(depth_list, test_list, label="test error", color="cyan", linewidth=3, alpha=0.8)

    plt.grid(alpha=0.8)
    plt.xlabel("Maximum Depth", fontsize=15)
    plt.ylabel("Error", fontsize=15)
    plt.legend(loc="best")
    plt.xticks(depth_list)
    plt.yticks([0.0,0.05, 0.1,0.15, 0.2, 0.25, 0.3])
    plt.title("Error vs Maximum Depth")
    plt.show()

def show_samples(sample_list, train_list, test_list):
    #We can use this function to plot the error rate of the training and testing dataset.
    #This is for plotting purpose, the x-axis is maximum depth and the y-axis is the error
    plt.figure(figsize=(16, 8), dpi=80)

    plt.plot(sample_list, train_list, label="train error", color="orange", linewidth=3, alpha=0.8)
    plt.plot(sample_list, test_list, label="test error", color="cyan", linewidth=3, alpha=0.8)

    plt.grid(alpha=0.8)
    plt.xlabel("Bootstrap Sample Size", fontsize=15)
    plt.ylabel("Error", fontsize=15)
    plt.legend(loc="best")
    plt.xticks(sample_list)
    plt.yticks([0.0,0.05, 0.1,0.15, 0.2, 0.25, 0.3])
    plt.title("Error vs Bootstrap Sample Size")
    plt.show()

def form_table_dep(dep_list, train_list, test_list):
    # Create a DataFrame from the error rates and corresponding C values
    error_rates_df = pd.DataFrame({
        'Maximum Depth:': dep_list,
        'Training Error': train_list,
        'Testing Error': test_list
    })
    return error_rates_df

def form_table_sam(sam_list, train_list, test_list):
    # Create a DataFrame from the error rates and corresponding C values
    error_rates_df = pd.DataFrame({
        'Bootstrap Sample': sam_list,
        'Training Error': train_list,
        'Testing Error': test_list
    })
    return error_rates_df

# Now, we will start to train random forest models on the Iris and datasets.
# Load the Iris dataset for training a random forest model.
X, y = load_iris().data, load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)

# Initialize the four lists for the maximum depth and bootstrap sample size.
dep_list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
sam_list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
sam_list2 = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Initialize a random forest model using sklearn.
# Here, you need to take turns to tune max_depth/max_samples for showing cases of underfitting/overfitting.
# Note that when you tune max_depth, please leave max_samples unchanged!
# Similarly, when you tune max_samples, leave max_depth unchanged!
# Please set `random_state` to 3 and feel free to set the value of `n_estimators`.

for i in range(3): # Iterate through the 3 lists.
    train_error = []
    test_error = []
    if i == 0:
        for depth in dep_list1:
            rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=depth, n_jobs=2, random_state=3, max_samples=20)
            rf.fit(X_train, y_train) # Start training.
            train_error.append(1 - rf.score(X_train, y_train)) # Put the training error into the train_error list for plotting.
            test_error.append(1 - rf.score(X_test, y_test)) # Put the testing set error into the test_error list for plotting.
        show_depth(dep_list1, train_error, test_error)
        print(form_table_dep(dep_list1, train_error, test_error))
    elif i == 1:
        for sample in sam_list1:
            rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=10, n_jobs=2, random_state=3, max_samples=sample)
            rf.fit(X_train, y_train)
            train_error.append(1 - rf.score(X_train, y_train))
            test_error.append(1 - rf.score(X_test, y_test))
        show_samples(sam_list1, train_error, test_error)
        print(form_table_sam(sam_list1, train_error, test_error))
    else:
        for sample in sam_list2:
            rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=10, n_jobs=2, random_state=3, max_samples=sample)
            rf.fit(X_train, y_train)
            train_error.append(1 - rf.score(X_train, y_train))
            test_error.append(1 - rf.score(X_test, y_test))
        show_samples(sam_list2, train_error, test_error)
        print(form_table_sam(sam_list2, train_error, test_error))





