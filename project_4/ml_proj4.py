from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import os

# Try importing TargetEncoder
try:
    from category_encoders import TargetEncoder
    print("category_encoders imported successfully.")
except ImportError:
    TargetEncoder = None

def main():

    print("=" * 60)
    print("CRICKET RUNS PREDICTION ")
    print("=" * 60)

    file_path = "odb.csv"

    if not os.path.exists(file_path):
        print("Dataset not found!")
        return

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully!")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns Available:")
    print(df.columns.tolist())

    
    # Required Columns
    

    required_columns = ['Mat', 'Inns', 'Runs', 'BF', 'SR', 'Span']

    for col in required_columns:
        if col not in df.columns:
            print(f"{col} column not found.")
            return

    
    # Convert Numeric Columns
    

    numeric_cols = ['Mat', 'Inns', 'Runs', 'BF', 'SR']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    
    # Missing Value Handling


    print("\n" + "=" * 60)
    print("Handling Missing Values")
    print("=" * 60)

    # Artificial Missing Values
    df.loc[0:20, 'SR'] = np.nan

    print(df[['SR']].head())

    imputer = SimpleImputer(strategy="median")

    df['SR'] = imputer.fit_transform(df[['SR']]).ravel()

    print("\nMissing Values after Imputation:")
    print(df['SR'].isnull().sum())

    
    # Log Transformation
    

    print("\n" + "=" * 60)
    print("Applying Log Transformation")
    print("=" * 60)

    if (df['Runs'] >= 0).all():

        print("Original Skewness :", df['Runs'].skew())

        df['LogRuns'] = np.log1p(df['Runs'])

        print("New Skewness :", df['LogRuns'].skew())

    
    # Target Encoding
    

    print("\n" + "=" * 60)
    print("Target Encoding")
    print("=" * 60)

    if TargetEncoder is not None:

        encoder = TargetEncoder()

        df['Span_Encoded'] = encoder.fit_transform(
            df[['Span']],
            df['Runs']
        ).iloc[:, 0]

        print("Target Encoding Applied Successfully.")

    else:

        print("Target Encoder not installed.")
        print("Skipping Target Encoding.")

    # Save Processed Dataset
    

    output_file = "processed_cricket_data.csv"

    df.to_csv(output_file, index=False)

    print(f"\nProcessed Dataset Saved as {output_file}")

    
    # Feature Selection
    

    print("\n" + "=" * 60)
    print("Feature Selection")
    print("=" * 60)

    features_to_test = ['Mat', 'Inns', 'BF', 'SR']

    X_features = df[features_to_test].fillna(0)

    y_target = df['Runs']

    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=2
    )

    selector.fit(X_features, y_target)

    winning_features = selector.get_support()

    best_features = X_features.columns[winning_features].tolist()

    print("Best Features :", best_features)

    
    # Train Test Split

    X = df[best_features]

    y = df['Runs']

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42

    )

    print("\nTraining Size :", X_train.shape)
    print("Testing Size :", X_test.shape)


    # Linear Regression
    

    print("\nTraining Linear Regression Model...")

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Model Trained Successfully!")

    
    # Prediction Comparison


    print("\n" + "=" * 60)
    print("Actual vs Predicted Runs")
    print("=" * 60)

    actual_runs = y_test.head(5).values

    predicted_runs = predictions[:5]

    for i in range(5):

        predicted = round(predicted_runs[i])

        actual = actual_runs[i]

        difference = abs(actual - predicted)

        print(f"\nPlayer {i+1}")

        print(f"Predicted Runs : {predicted}")

        print(f"Actual Runs    : {actual}")

        print(f"Difference     : {difference}")

    print("\nProject Completed Successfully!")


if __name__ == "__main__":
    main()