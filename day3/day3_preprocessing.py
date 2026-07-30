import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def main():
    # ==========================================
    # Prerequisites: Load the dataset
    # ==========================================
    try:
        df = pd.read_csv('dataset.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'dataset.csv' not found. Please ensure the file exists in the directory.")
        return

    # ==========================================
    # Step 1 — Create a Working Copy
    # ==========================================
    data = df.copy()

    # ==========================================
    # Step 2 — Remove Duplicates
    # ==========================================
    print("Duplicates before:", data.duplicated().sum())
    data = data.drop_duplicates()
    print("Duplicates after:", data.duplicated().sum())

    # ==========================================
    # Step 3 — Remove Unnecessary Columns
    # ==========================================
    # Property_ID is just an identifier and does not carry predictive power.
    if "Property_ID" in data.columns:
        data = data.drop("Property_ID", axis=1)
        print("Removed 'Property_ID' column.")

    # ==========================================
    # Step 4 — Separate Features and Target
    # ==========================================
    target = "Total_Annual_Premium_INR"
    if target not in data.columns:
        print(f"Error: Target column '{target}' not found in the dataset.")
        return
        
    X = data.drop(target, axis=1)
    y = data[target]
    print(f"Separated features (X) and target (y: {target}).")

    # ==========================================
    # Step 5 — Train/Test Split
    # ==========================================
    # This must happen before fitting any preprocessing to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
    
    print("\nData Split Shapes:")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # ==========================================
    # Step 6 — Identify Feature Types
    # ==========================================
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # ==========================================
    # Step 7 — Build the Numerical Pipeline
    # ==========================================
    # Using 'median' strategy as it works well with skewed insurance data and resists outliers.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # ==========================================
    # Step 8 — Build the Categorical Pipeline
    # ==========================================
    # Using 'most_frequent' for missing categories and OneHotEncoder for converting text to numbers.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ==========================================
    # Step 9 — Combine Both Pipelines
    # ==========================================
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # ==========================================
    # Step 10 — Test the Preprocessor
    # ==========================================
    # Fit and transform the training data, but ONLY transform the test data.
    print("\nFitting preprocessor on training data and transforming...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Processed Training shape:", X_train_processed.shape)
    print("Processed Testing shape:", X_test_processed.shape)

    # ==========================================
    # Step 11 — Check for Missing Values
    # ==========================================
    missing_values = np.isnan(X_train_processed.toarray()).sum() if hasattr(X_train_processed, "toarray") else np.isnan(X_train_processed).sum()
    print(f"\nMissing values after preprocessing: {missing_values}")

    # ==========================================
    # Step 12 — Save the Preprocessor (Optional)
    # ==========================================
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("Preprocessor successfully saved as 'preprocessor.pkl'.")

if __name__ == "__main__":
    main()
