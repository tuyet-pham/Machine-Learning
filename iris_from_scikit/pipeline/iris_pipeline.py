from sklearn import datasets
from sklearn.model_selection import train_test_split
import pydot
from io import StringIO
# from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

X = iris.data
y = iris.target


"""
    We are using train_test_split to make our lives of splitting data easier :')
    Here test_size is .5 = %50 of the data
    The function gives us back datasets that we will store in..

        X_train : Label Training set
        X_test :  Label Testing set
        y_train : Feature Training set
        y_test :  Feature Testing set
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# Training with 50% of data
# Method : Decision tree
from sklearn import tree

clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_train,y_train)

pred1 = clf1.predict(X_test)
print(pred1)
# for row in pred1:
#     print(row)

# Comparing the accuracy from the predictive labels to the real labels
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred1))

from sklearn.neighbors import KNeighborsClassifier

clf2 = KNeighborsClassifier() 
clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)

print(pred2)
# for row in pred1:
#     print(row)

# Comparing the accuracy from the predictive labels to the real labels using KNeighborsClassifier
print(accuracy_score(y_test,pred2))


dot_data = StringIO()
tree.export_graphviz(clf1,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris_pipline.pdf")