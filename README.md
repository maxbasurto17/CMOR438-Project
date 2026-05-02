# Custom Machine Learning Package & Football Skill Position Analysis
## CMOR 438 Final Project

## Overview

This repository houses a custom-built machine learning package developed for the **CMOR 438 (Data Science and Machine Learning)** final project. By building supervised and unsupervised learning algorithms from scratch, the core principles of ML become more accessible.

We've leveraged the package to analyze NFL skill position players (Running Backs, Wide Receivers, and Tight Ends). Ultimately, the project aims to identify the specific traits that drive player production and define the unique roles each position plays within an offense.

## Package Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci-test.yml
├── .gitignore
├── pyproject.toml
├── README.md
├── src/
│   └── rice_ml/
│       ├── __init__.py
│       ├── measures/
│       │   ├── __init__.py
│       │   ├── distances.py
│       │   └── validation.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── one_hot_encoder.py
│       │   └── standard_scaler.py
│       ├── supervised/
│       │   ├── __init__.py
│       │   ├── decision_tree.py
│       │   ├── gradient_boosting.py
│       │   ├── k_nearest_neighbors.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── multi_layer_perceptron.py
│       │   ├── perceptron.py
│       │   ├── random_forest.py
│       │   └── regression_tree.py
│       └── unsupervised/
│           ├── __init__.py
│           ├── dbscan.py
│           ├── k_means.py
│           └── pca.py
└── tests/
    ├── test_dbscan.py
    ├── test_decision_tree.py
    ├── test_distances.py
    ├── test_gradient_boosting.py
    ├── test_k_nearest_neighbors.py
    ├── test_linear_regression.py
    ├── test_logistic_regression.py
    ├── test_multi_layer_perceptron.py
    ├── test_one_hot_encoder.py
    ├── test_pca.py
    ├── test_perceptron.py
    ├── test_random_forest.py
    └── test_kmeans.py
```

## Capabilities

As defined above, this package can be used to perform standard data preprocessing, implement supervised and unsupervised machine learning methods, and measure the effectiveness of the implemented models. This creates a comprehensive ML package built for end-to-end data analysis.

**Supervised Learning**
* Linear & Logistic Regression: Baseline statistical models for predicting continuous values (Linear) and binary classifications (Logistic).
* Perceptron & MLP: Neural networks ranging implemting both a single-layer linear classifier (Perceptron) and a Multi-Layer Perceptron (MLP) for complex patterns.
* Decision & Regression Trees: Non-linear models that split data based on feature thresholds for both classification (Decision) and regression tasks.
* Random Forest & Gradient Boosting: Ensemble methods that combine multiple trees to reduce variance (Bagging) or bias (Boosting).
* K-Nearest Neighbors (KNN): An instance-based learner that classifies points based on the majority vote of their closest neighbors.

**Unsupervised Learning**
* Principal Component Analysis (PCA): A dimensionality reduction technique that projects data onto axes of maximum variance.
* K-Means Clustering: A centroid-based algorithm that partitions data into $K$ distinct, non-overlapping subgroups.
* DBSCAN: A density-based clustering method capable of finding arbitrary shapes and identifying outliers as noise.

**Measures & Metrics**
* Distances: Euclidean distance.
* Validation: Tools for model assessment, including performance metrics like root mean squared error, mean absolute error, and $R^2$.

**Preprocessing** 
* StandardScaler: Features scaling centering all variables at a mean of 0 with a standard deviation of 1.
* One-Hot Encoder: Converts a categorical string data into a binary numeric format suitable for machine learning.

## Package Usage

In this repository, we leverage our package to conduct analysis on NFL skill position (RB, WR, TE) players. We look at a variety of angles such as player evaluation, NFL Draft projections, and play style identification. Below is how we leverage each model in this repository.

**Supervised Learning**
* Linear Regression: Predict wide receiver catch totals based on their type of usage.
* Logistic Regression: Predicting touchdown probability on a play based on play context.
* Perceptron: Using pass game usage to distinguish running backs and wide receivers.
* MLP: Predicting playoff and non-playoff teams using platoon strength.
* Decision Tree: Leveraging wide receiver NFL career data to predict what round they were drafted in.
* Regression Trees: Utilizing in-game athleticism data to predict yards per carry.
* Random Forest: Expanding on example from decision tree example. 
* Gradient Boosting: Understanding how impactful skill is in determining tight end production.
* K-Nearest Neighbors (KNN): Utilizing player heights and weights to identify player positions. 

**Unsupervised Learning**
* Principal Component Analysis (PCA): Leveraged in multiple examples. Used in K-Means to compare model effectiveness between using and not using PCA in data prep for a model.
* K-Means Clustering: Identifying wide receiver archetypes using wide receiver bio and production data.
* DBSCAN: Classifying running back profiles using heights and weights.


## Installation

To install the package directly to your device, use the following code in your terminal:

```bash
pip install git+https://github.com/maxbasurto17/CMOR438-Project.git
```

## Authors & Liscense

**Authors:** Suhas Narra, Max Basurto

**License:** Project licensed under MIT License.