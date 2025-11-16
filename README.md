# 🏈 Athlete Injury Prediction with Restricted Stacking Classifier

## Project Overview

This repository hosts a machine learning project focused on **proactive athlete injury risk assessment**. Utilizing a synthetic dataset containing various physiological, training, and performance metrics, the goal is to predict an **Injury Indicator**, with a specific emphasis on quantifying the **ACL Risk Score**.

The core of this project is the implementation of a **Restricted Stacking Classifier**. This ensemble method leverages the strengths of multiple diverse models (known as base estimators) and combines their predictions through a meta-model. The "restricted" nature refers to a controlled architecture, ensuring a robust and interpretable prediction system for critical sports science applications.

## Key Features

* **Ensemble Modeling:** Implementation of a **Stacking Classifier** for superior predictive performance and robustness compared to single-model approaches.
* **Injury Risk Focus:** Specifically utilizes features like `Load_Balance_Score` and `ACL_Risk_Score` to target high-impact injuries.
* **Data Preparation:** Includes essential steps for handling categorical features (`Gender`, `Position`) via one-hot encoding.
* **Convergence Handling:** Demonstrates awareness and handling of convergence warnings from base estimators (e.g., Logistic Regression) often encountered with non-scaled, complex datasets.
* **Practical Application:** Includes a final prediction step on a hypothetical **new athlete's data** to demonstrate real-world utility in a sports management context.

## Technologies and Libraries

* **Python 3.x**
* **pandas** (For data manipulation and preprocessing)
* **scikit-learn** (For model creation, specifically `LogisticRegression`, `KNeighborsClassifier`, `SVC`, and `StackingClassifier`)
* **matplotlib** & **seaborn** (For data visualization and analysis—as shown in the accompanying notebook)

## Model Architecture

The **Restricted Stacking Classifier** is constructed as follows:

1.  **Base Estimators:**
    * **Logistic Regression:** Linear classification baseline.
    * **K-Nearest Neighbors (k-NN):** Instance-based non-linear classifier.
    * **Support Vector Machine (SVC):** Margin-based non-linear classifier.
2.  **Final Estimator (Meta-Model):**
    * **Logistic Regression:** Learns how to optimally combine the predictions of the base estimators.

## Getting Started

### Prerequisites

You will need Python 3.x and the following libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
