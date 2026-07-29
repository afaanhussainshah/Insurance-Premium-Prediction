import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Assuming df is loaded earlier
# For the script to be valid, we add a placeholder:
# df = pd.read_csv("dataset.csv")

# STEP 1 — Create a Backup
# Keep the original dataset untouched.
data = df.copy()

print("Original shape:", data.shape)

# STEP 2 — Check Missing Values
missing_summary = pd.DataFrame({
    "Missing_Count": data.isnull().sum(),
    "Missing_Percentage": (
        data.isnull().sum() / len(data) * 100
    )
})

missing_summary = missing_summary[
    missing_summary["Missing_Count"] > 0
].sort_values(
    by="Missing_Count",
    ascending=False
)

display(missing_summary)

# STEP 3 — Check Duplicates
print(
    "Duplicate rows:",
    data.duplicated().sum()
)

data = data.drop_duplicates()

print(
    "Shape after removing duplicates:",
    data.shape
)

# STEP 4 — Check Property_ID
print(
    "Unique Property IDs:",
    data["Property_ID"].nunique()
)

print(
    "Total rows:",
    len(data)
)


# STEP 6 — Create X and y
target = "Total_Annual_Premium_INR"

X = data.drop(
    columns=[
        target,
        "Property_ID"
    ]
)

y = data[target]

# STEP 7 — Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "Training data:",
    X_train.shape
)

print(
    "Testing data:",
    X_test.shape
)

# STEP 8 — Identify Numerical and Categorical Features
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

# STEP 9 — Build the Proper Preprocessing Pipeline
numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    ),
    
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])
