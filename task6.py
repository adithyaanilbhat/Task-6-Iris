# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset (with headers)
data = pd.read_csv("iris.csv")

# Features and labels
X = data.iloc[:, :-1]  # all columns except last
y = data.iloc[:, -1]   # last column is class

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

# Try different K values and print accuracy + confusion matrix
for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy with K={k}: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion matrix with K={k}:\n{confusion_matrix(y_test, y_pred)}\n")

# Optional: Visualizing decision boundaries using first two features (K=3)
def plot_decision_boundary(X, y, model, k):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z_labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z, _ = pd.factorize(Z_labels)    # Convert labels to numbers
    Z = Z.reshape(xx.shape)

    y_codes, _ = pd.factorize(y)     # Convert true labels likewise

    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_codes, edgecolor='k')
    plt.legend(handles=scatter.legend_elements()[0], labels=set(y))
    plt.title(f'Decision boundary for K={k}')
    plt.show()


# Fit model on first two features for visualization
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train[:, :2], y_train)
plot_decision_boundary(X_train[:, :2], y_train, model, 3)
