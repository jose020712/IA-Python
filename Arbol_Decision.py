
!pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


data = load_iris()


df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())


X = df[data.feature_names]
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

accuracy = clf.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
