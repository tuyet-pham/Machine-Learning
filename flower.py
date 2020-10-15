from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
import pydot
from io import StringIO

iris = load_iris()

"""
    feature_names : describe what features per flower
    data[0] : the value of the feature for flower at index 0
"""
print(iris.feature_names)
print(iris.data[0])


"""
    target_names : showcases some name from the dataset
    target[0] : the name of the flower at index 0 - The LABEL
"""
print(iris.target_names)
print(iris.target[60])


# Looking at all the data
print("Label\t", iris.feature_names)
print("====================================================================================================")

for i in range(len(iris.target)):
    if iris.target[i] == 0: 
        print("setosa \t", iris.data[i])
    elif iris.target[i] == 1:
        print("versicolor \t", iris.data[i])
    else:
        print("virginica \t", iris.data[i])



# Splitting up the dataset! IMPORTANT

# A list that holds the index of obj to remove.
test_idx = [0,50,100]

"""
Training data: 

Here I am removing 3 items from the target (label) and feature dataset at a very specific index.
    1. I will take the 3 removed and use them as my testing data. 
    2. I will take the rest as training data.

"""

# Removing the index 0, 50, & 100 from the target dataset & feature dataset
train_label = np.delete(iris.target, test_idx)
train_feature_data = np.delete(iris.data, test_idx, axis=0)


# Creating the testing list with the 3 obj removed from the dataset.
test_label = iris.target[test_idx]
test_feature_data = iris.data[test_idx]



print("My testing targets \n", test_label)
print("\nMy testing feature data \n", test_feature_data)

# Creating my tree
clf = tree.DecisionTreeClassifier()

# Using the training sets
clf = clf.fit(train_feature_data, train_label)


# Here I am predicting the flowers using the test features from the list 'test_feature_data'
pred = clf.predict(test_feature_data)


# Printing out the retrun of the predict function. adding the label here manually.
for label in pred:
    if label == 0: 
        print("setosa")
    elif label == 1:
        print("versicolor")
    else:
        print("virginica")


dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf") 