
# Classical Machine Learning Algorithms

This repository contains the implementation of classical machine learning algorithms from scratch using Python and libraries such as `NumPy`, `Matplotlib`, and `SciPy`. The models included are:

- **Linear Regression** (with and without regularization)
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Naive Bayes**

Each of these models is implemented with gradient descent, normalization, and other relevant methods to illustrate the core concepts of machine learning. These algorithms were implemented as part of the coursework for **IE7300: Statistical Learning** at Northeastern University.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Models](#models)
  - [Linear Regression](#linear-regression)
  - [Logistic Regression](#logistic-regression)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Naive Bayes](#naive-bayes)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is intended to provide a hands-on implementation of classical machine learning models using Python. It aims to:

- Reinforce the understanding of machine learning algorithms by implementing them from scratch.
- Showcase how gradient descent, regularization, and evaluation metrics are used.
- Provide utility functions for normalization, evaluation, and plotting model performance.

These models were part of a course project for **IE7300: Statistical Learning** at Northeastern University, where the focus was on understanding the underlying mathematics and the use of different machine learning techniques.

## Models

### Linear Regression

The `LinearRegression` class implements a linear regression model with options for:
- **Closed-form solution**: Solving the normal equations to find the optimal weights.
- **Gradient Descent**: Iteratively optimizing the weights using the gradient of the loss function.
- **Regularization**: Ridge regularization (L2) can be applied to the model.
  
Key methods:
- `fit()`: Fits the model using the specified solution type.
- `predict()`: Predicts the output for a given input dataset.
- `gradient_descent()`: Runs gradient descent to minimize the loss function.
- `closed_form_solution()`: Solves the linear regression using the normal equation.

### Logistic Regression

The `LogisticRegression` class implements a binary logistic regression model using gradient descent:
- **Normalization**: Automatically normalizes input data.
- **Evaluation**: Computes metrics such as accuracy, precision, and recall.

Key methods:
- `fit()`: Trains the logistic regression model using gradient descent.
- `predict()`: Predicts the class probabilities for a given dataset.
- `evaluate()`: Computes and prints evaluation metrics such as accuracy, precision, and recall.

### Support Vector Machine (SVM)

The `SVM` class implements a binary Support Vector Machine using a linear kernel:
- **Gradient Descent**: Uses gradient descent to find the optimal hyperplane for classification.
- **Regularization**: Allows regularization to control the margin size and avoid overfitting.

Key methods:
- `fit()`: Trains the SVM using gradient descent.
- `predict()`: Predicts the class labels based on the trained model.
- `evaluate()`: Computes accuracy and prints evaluation metrics.

### Naive Bayes

The `NaiveBayes` class implements the Naive Bayes classification model:
- **Laplace Smoothing**: Optional parameter to apply Laplace smoothing to avoid zero probabilities.
- **Conditional Probabilities**: Calculates the conditional probabilities for each feature given the class label.

Key methods:
- `fit()`: Trains the Naive Bayes model by calculating the required conditional probabilities.
- `predict()`: Predicts the class label for a given dataset by computing the posterior probabilities.

## Evaluation Metrics

For each model, evaluation metrics such as **Root Mean Squared Error (RMSE)**, **Sum of Squared Errors (SSE)**, **Accuracy**, **Precision**, and **Recall** are provided. These metrics help assess model performance during training and testing.

### Example Metrics:
- **RMSE** (Root Mean Squared Error): Measures the difference between predicted and actual values.
- **SSE** (Sum of Squared Errors): Sum of the squared differences between predicted and actual values.
- **Accuracy**: Percentage of correctly predicted classes for classification models.
- **Precision & Recall**: Evaluate the performance for binary classification.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
