# Heart Attack Prediction

This project implements a machine learning model to predict the risk of heart attacks using the CDC BRFSS 2015 dataset. We explored the following models:
- **Linear regression using gradient descent**
- **Linear regression using stochastic gradient descent**
- **Least squares regression using normal equations**
- **Ridge regression using normal equations**
- **Logistic regression using gradient descent ($y ∈ {0, 1})**
- **Regularized logistic regression using gradient descent
(y ∈ {0, 1}, with regularization term λ$∥w∥^2$)**


The project is built using **basic Python libraries and NumPy** (no advanced machine learning libraries like Pandas or Scikit-learn are used).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Contributors](#contributors)

## Overview

This project aims to develop a predictive model for heart attacks by analyzing health-related data from the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset. We apply linear and logistic regression models, perform exploratory data analysis, and handle missing values using basic feature engineering techniques.

### Objectives:

1. Understand the dataset and distributions.
2. Handle missing values using feature engineering techniques.
3. Train and evaluate the implemented models using basic Python and NumPy.
4. Write a scientific report

## Dataset

The dataset is sourced from the **CDC BRFSS 2015**. It includes various health-related factors that could influence the likelihood of a heart attack. Key features include:
- **Age**
- **Gender**
- **Cholesterol**
- **Blood Pressure**
- **Smoking Status**
- **Physical Activity**
- **Body Mass Index (BMI)**
- **Diabetes**
... and many others features

The target variable is **Heart Attack Risk** (binary: -1 = No, 1 = Yes).

For more information, visit the [CDC BRFSS 2015 page](https://www.cdc.gov/brfss/annual_data/annual_2015.html).

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualizations)

## Installation

This implementation assumes that the data (x_train.csv, y_train.csv, x_test.csv) can be found in the path "../data/dataset_to_release/" which means right outside of the project repository.

1. Clone this repository:
   ```bash
   git clone https://github.com/rosbotmay/MLprojects_era.git


## Contributors
- Elias Mir 
- Amene Gafsi
- Rosa Mayila 


