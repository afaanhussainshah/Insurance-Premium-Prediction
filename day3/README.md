# 📅 Day 3: Data Cleaning & Preprocessing Pipeline

Welcome to Day 3 of the Machine Learning project! Today's focus is all about transitioning our raw data into a clean, structured format that Machine Learning models can understand. We built a robust **Data Preprocessing Pipeline** to ensure that our models get the highest quality data without any data leakage.

## 🎯 Goal for the Day
Our objective is to take our raw `dataset.csv` and transform it by:
1. **Cleaning the data** (removing duplicates and unnecessary columns).
2. **Splitting the data** into training and testing sets.
3. **Building a Preprocessing Pipeline** to handle missing values and encode categorical data.
4. **Getting everything ready for Model Training** (Day 4).

---

## 🛠️ Step-by-Step Breakdown

Here is a detailed explanation of everything we accomplished today and the reasoning behind each decision.

### Step 1: Create a Working Copy
```python
data = df.copy()
```
**What we did:** We created a copy of the original dataset (`df`).
**Why we did it:** This is a crucial best practice. By working on a copy, we protect the original dataset from unintended modifications. If we make a mistake during preprocessing, we can always revert to the original `df` without having to reload the data from the CSV file.

### Step 2: Remove Duplicates
```python
print("Duplicates before:", data.duplicated().sum())
data = data.drop_duplicates()
print("Duplicates after:", data.duplicated().sum())
```
**What we did:** We checked for identical rows in our dataset and removed them.
**Why we did it:** Duplicate data doesn't add any new information for the model to learn from. In fact, if duplicates are heavily skewed towards certain outcomes, they can bias the model and lead to overfitting. Removing them ensures the model learns general patterns rather than memorizing repeated entries.

### Step 3: Remove Unnecessary Columns
```python
data = data.drop("Property_ID", axis=1)
```
**What we did:** We dropped the `Property_ID` column from our dataset.
**Why we did it:** A Machine Learning model looks for statistical relationships between features and the target variable. An ID column is just a unique identifier for administrative purposes; it holds no predictive power. Leaving it in might confuse the model or cause it to mistakenly find false patterns.

### Step 4: Separate Features and Target
```python
target = "Total_Annual_Premium_INR"
X = data.drop(target, axis=1)
y = data[target]
```
**What we did:** We split our dataset into two parts: `X` (the features/inputs) and `y` (the target/output we want to predict).
**Why we did it:** Supervised learning models require clear separation between the data they use to make predictions (features) and the actual value they are trying to predict (target).

### Step 5: Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```
**What we did:** We divided our features (`X`) and target (`y`) into a training set (80%) and a testing set (20%).
**Why we did it:** This is one of the most important rules in Machine Learning! We must split the data **before** applying any transformations (like finding the mean/median for missing values). If we don't, information from the test set leaks into the training set (a phenomenon called **Data Leakage**), making our model appear much better than it actually is.
> *By using `random_state=42`, we ensure that every time we run the code, the split remains exactly the same. This is great for reproducibility.*

### Step 6: Identify Feature Types
```python
numerical_features = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
```
**What we did:** We grouped our columns into two lists: numerical (numbers) and categorical (text/objects).
**Why we did it:** Numbers and text cannot be handled the same way. For example, you can calculate the average of numerical data, but you can't calculate the average of text data (like "City"). Separating them allows us to apply different preprocessing techniques to each group.

### Step 7: Build the Numerical Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
```
**What we did:** We created a mini-pipeline for numerical columns that fills in missing values using the median.
**Why we did it:** Insurance data is often heavily skewed (e.g., a few people with very high property values). If we used the `mean` (average) to fill missing values, those extreme outliers would pull the average up artificially. The `median` represents the middle value and is highly resistant to outliers, making it a much safer choice here.

### Step 8: Build the Categorical Pipeline
```python
from sklearn.preprocessing import OneHotEncoder
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```
**What we did:** We created a mini-pipeline for categorical columns. First, it fills missing text with the most common category (`most_frequent`). Then, it converts the text into numbers using `OneHotEncoder`.
**Why we did it:**
1. **Imputation:** If a category is missing, assuming it belongs to the most common group is a standard and safe statistical bet.
2. **Encoding:** Machine learning models are essentially mathematical equations; they can't multiply or add text like "Apartment". One-Hot Encoding creates a new binary column (0 or 1) for each category, allowing the model to interpret text mathematically. Using `handle_unknown="ignore"` ensures that if the model sees a brand-new category in the test set, it won't crash; it will just ignore it.

### Step 9: Combine Both Pipelines
```python
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])
```
**What we did:** We combined our `numeric_pipeline` and `categorical_pipeline` into a single, unified `ColumnTransformer` called `preprocessor`.
**Why we did it:** This makes our code clean and professional. Instead of manually applying transformations to different parts of the dataframe and gluing them back together, the `ColumnTransformer` handles the routing automatically.

### Step 10: Test the Preprocessor
```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```
**What we did:** We applied the pipeline to our data. We used `.fit_transform()` on the training data and only `.transform()` on the testing data.
**Why we did it:**
- **`fit_transform` (Training):** The pipeline looks at the training data, learns the medians and the most frequent categories (fits), and then applies the changes (transforms).
- **`transform` (Testing):** The pipeline applies the exact same rules it learned from the training data to the test data. This strictly prevents the test data from influencing the rules, preventing data leakage.

### Step 11: Check for Missing Values
```python
import numpy as np
print(np.isnan(X_train_processed.toarray()).sum() if hasattr(X_train_processed, "toarray") else np.isnan(X_train_processed).sum())
```
**What we did:** We ran a final check to ensure there are exactly `0` missing values left in our processed dataset.
**Why we did it:** Most machine learning algorithms (like Random Forest or XGBoost) will throw an error if they encounter even a single missing value. This is our safety check before moving to modeling.

### Step 12: Save the Preprocessor
```python
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
```
**What we did:** We exported our finished `preprocessor` pipeline to a file named `preprocessor.pkl`.
**Why we did it:** When we eventually deploy our model (e.g., in a web app or API), any new incoming customer data will need to go through the exact same transformations (same medians, same categorical columns) before the model can make a prediction. Saving the preprocessor ensures we can recreate this exact pipeline in production.

---

## 💡 Suggestions & Best Practices Moving Forward

1. **Feature Engineering Exploration (Optional for Future):** While our current pipeline is solid, we could later experiment with creating new features. For instance, combining `Age of Building` and `Renovation Status` might give the model a better understanding of structural integrity.
2. **Scaling the Data:** Depending on the model we choose on Day 4 (like Support Vector Machines, KNN, or Neural Networks), we might need to add a `StandardScaler` or `MinMaxScaler` to our `numeric_pipeline`. Tree-based models (like Random Forest/XGBoost) don't strictly need scaling, but it's good practice to keep it in mind.
3. **Handling Class Imbalance:** If we are trying to predict categorical outcomes (classification) and one category heavily dominates, we should look into SMOTE or class weights. Since we are predicting a continuous value (`Total_Annual_Premium_INR`), this is a Regression task, so standard loss functions should suffice.

## 🏆 Today's Deliverables Achieved
- [x] Duplicate records removed
- [x] Unnecessary identifiers (`Property_ID`) removed
- [x] Features (`X`) and target (`y`) successfully separated
- [x] Train/test split completed properly to prevent data leakage
- [x] Dedicated Numerical preprocessing pipeline created
- [x] Dedicated Categorical preprocessing pipeline created
- [x] Pipelines combined using `ColumnTransformer`
- [x] Training and test data successfully transformed without errors
- [x] Pipeline saved as a `.pkl` file for future deployment
