import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.max_columns", None)

# 1. Loading the dataset
df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Shape: ", df.shape)

display(df.head())

# 2. Dataset information
df.info()

# 3. Statistical summary
display(df.describe())

# 4. Missing values
print(df.isnull().sum())

# 5. Duplicate rows
print("Duplicates:", df.duplicated().sum())

# 6. Column names
print(df.columns.tolist())

# 7. Numerical columns
print("Numerical columns:")
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

# Note: this cell produces a FutureWarning about object dtypes
print(df.select_dtypes(include=["object"]).columns.tolist())

# 8. Categorical columns
categorical_columns = df.select_dtypes(include=["str"]).columns.tolist()

print("Categorical columns:")
print(categorical_columns)

# 9. Target variable analysis
target = "Total_Annual_Premium_INR"

display(df[target].describe())

# 10. Data Visualization
plt.figure(figsize=(10, 6))

plt.hist(df[target], bins=30)

plt.title("Distribution of Annual Insurance Premium")
plt.xlabel("Annual Premium (INR)")
plt.ylabel("Frequency")

plt.show()
