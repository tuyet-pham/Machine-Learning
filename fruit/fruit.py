from sklearn import tree

# smooth - 1 
# bumpy = 0
features = [[140, 1],[130, 1],[150,0],[170, 0]]

# Apples = 0 
# Orange = 1
labels = [0,0,1,1]

# Empty classifier
clf = tree.DecisionTreeClassifier()

# Separating features
clf = clf.fit(features, labels)

# 0 if apple 1 if orange
print(clf.predict([[150,0]]))