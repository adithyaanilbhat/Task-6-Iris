# K-Nearest Neighbors (KNN) Classification on Iris Dataset

## Overview
This project implements the K-Nearest Neighbors (KNN) algorithm to classify the famous Iris flower dataset. KNN is a simple yet powerful instance-based learning algorithm that classifies data points based on the majority category of their nearest neighbors.

## Dataset
- **Dataset Name:** Iris Dataset
- **Source:** https://people.sc.fsu.edu/~jburkardt/data/csv/iris.csv
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Target:** Species (three classes: Iris-setosa, Iris-versicolor, Iris-virginica)

## Steps and Methodology
1. **Data Loading:** The Iris dataset CSV file is loaded using Pandas.
2. **Data Preprocessing:** Features are normalized using StandardScaler to bring all measurements to the same scale.
3. **Data Splitting:** The dataset is split into 70% training data and 30% test data.
4. **Model Training:** KNN classifiers with K=3, 5, and 7 are trained using the training dataset.
5. **Evaluation:** The models are evaluated on the test data using accuracy score and confusion matrix.
6. **Visualization:** Decision boundary plots for K=3 KNN classifier are displayed using the first two features.

## Results
| K Value | Accuracy | Confusion Matrix                              |
|---------|----------|-----------------------------------------------|
| 3       | 1.0      | [[19, 0, 0], [0, 13, 0], [0, 0, 13]]         |
| 5       | 1.0      | [[19, 0, 0], [0, 13, 0], [0, 0, 13]]         |
| 7       | 1.0      | [[19, 0, 0], [0, 13, 0], [0, 0, 13]]         |

*Perfect classification accuracy was achieved on the test set for all chosen K values.*

## Files Included
- `task6.py`: Python code implementing K-Nearest Neighbors classification and plotting.
- `iris.csv`: Dataset file.
- `README.md`: Project documentation.

## How to Run
1. Ensure Python 3.x is installed.
2. Install required libraries if not already installed:
3. Run the Python script in the folder containing `task6.py` and `iris.csv`:
4. View the output accuracy, confusion matrices, and decision boundary plot.


