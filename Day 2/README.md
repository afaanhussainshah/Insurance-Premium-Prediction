# Day 2: Data Preprocessing Pipeline - Insurance Premium Prediction

## Overview
On Day 2, we focused on establishing a robust, professional, and leakage-safe machine learning preprocessing pipeline. We ensured that data leakage is prevented by properly structuring our workflow before any transformations are applied.

## 1. Removing Duplicates
The first step in our data cleaning process was to identify and remove duplicate records from the dataset. This ensures that our model does not memorize redundant data and maintains generalization.

![Removing Duplicates](Screenshot%202026-07-29%20181759.png)
![Duplicates Removed](Screenshot%202026-07-29%20181811.png)

## 2. Train-Test Split
Before applying any transformations or imputations, we performed a train-test split. Splitting the data beforehand is critical for preventing data leakage, ensuring that information from the test set does not influence the training phase.

![Train-Test Split](Screenshot%202026-07-29%20182348.png)
![Train-Test Output](Screenshot%202026-07-29%20182432.png)

## 3. Building the ColumnTransformer Pipeline
We utilized a `ColumnTransformer` to handle different types of features simultaneously. This modular approach allows for clean, reproducible code. 

![Pipeline Setup](Screenshot%202026-07-29%20182604.png)

## 4. Handling Missing Values and Categorical Encoding
Within the `ColumnTransformer`, we implemented:
- **`SimpleImputer`**: To systematically handle missing numerical values (e.g., using mean/median).
- **`OneHotEncoder`**: To convert categorical variables into machine-readable numeric formats without imposing any ordinal relationship.

![Handling Missing Values](Screenshot%202026-07-29%20183253.png)
![Categorical Encoding](Screenshot%202026-07-29%20183320.png)

## Summary
By the end of Day 2, we successfully set up a complete data preprocessing pipeline that safely splits the data and efficiently processes missing values and categorical features, paving the way for model training in the upcoming steps.
