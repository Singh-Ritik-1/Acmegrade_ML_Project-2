Name: Ritik Singh Company: Acmegrade Pvt Ltd Domain: Machine Learning Duration: July 9, 2024 to September 9, 2024

## Overview of cancer prediction prediction project :
# Breast Cancer Prediction Project

## Overview
This project aims to predict breast cancer diagnosis using machine learning techniques. The project employs various algorithms, including Support Vector Classification (SVC), Random Forest Classifier, and XGBoost, to analyze the dataset and classify tumors as benign or malignant.

## Features
- **Data Preprocessing**: Includes handling missing values, encoding categorical variables, and scaling features.
- **Exploratory Data Analysis (EDA)**: Visualization of correlations and distributions of features.
- **Model Training and Evaluation**: Utilizes multiple algorithms for classification, model tuning, and performance evaluation.
- **Feature Importance**: Evaluates the importance of various features in predicting the diagnosis.

## Dataset
- **Source**: The dataset is included in the project folder as `data.csv`.
- **Description**: The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image. 
- **Target Variable**: `diagnosis` (Malignant: 1, Benign: 0).
- **Features**: The dataset includes features such as `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, and many others, including standard errors and worst values for each feature.

## Libraries Used
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `seaborn`: For statistical data visualization.
- `matplotlib`: For creating visualizations.
- `sklearn`: For machine learning algorithms, model evaluation, and preprocessing.
- `xgboost`: For XGBoost classifier.
- `imblearn`: For handling imbalanced datasets using SMOTE.
- `pickle`: For saving and loading trained models.

## Process
1. **Data Loading**: The dataset is loaded using Pandas.
2. **Data Preprocessing**:
   - Convert the target variable `diagnosis` into a categorical type and then to numerical codes.
   - Handle missing values (if any).
   - Normalize or standardize the dataset as required.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize correlations using heatmaps.
   - Create box plots to analyze the distribution of features based on the diagnosis.
   - Use distribution plots to understand feature distributions for different classes.
4. **Model Training**:
   - Split the dataset into training and testing sets.
   - Train different classifiers (SVC, Random Forest, XGBoost) with hyperparameter tuning using GridSearchCV.
   - Evaluate the model performance using accuracy, confusion matrix, and classification report.
5. **Feature Importance Analysis**: Determine which features are most important in making predictions using Gini importance.
6. **Model Persistence**: Save the trained models using `pickle` for future predictions.

## Class Structures
- **FitModel**: A function that trains a given machine learning model with hyperparameter tuning and evaluates its performance.

## Usage
To run this project, follow these steps:

1. Clone the repository or download the project files.
2. Ensure you have the necessary libraries installed. You can use the following command to install them:
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn xgboost imbalanced-learn


## output :
![image](https://github.com/user-attachments/assets/a6533c87-8852-41d5-b514-d440df2c7f9f)

