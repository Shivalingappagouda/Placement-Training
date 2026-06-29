from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer

# Try importing TargetEncoder
try:
    from category_encoders import TargetEncoder
    print("category_encoders imported successfully.")
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders is not installed.")
    print("Install it using: pip install category_encoders")


def main():

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    file_path = "data.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    # Read dataset
    df = pd.read_csv(file_path)

    print(f"Dataset Loaded Successfully!")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}\n")

    # Check required columns
    required_columns = ['H', 'R', 'W']

    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found.")
            print("Available columns are:")
            print(df.columns.tolist())
            return

    # ====================================================
    # Handling Missing Values
    # ====================================================

    print("=" * 50)
    print("Handling Missing Values")
    print("=" * 50)

    # Artificially create missing values
    df.loc[0:25, 'H'] = np.nan

    print("Dataset before imputation:\n")
    print(df.head())

    # Median Imputation
    imputer = SimpleImputer(strategy='median')

    df['H'] = imputer.fit_transform(df[['H']]).ravel()

    print("\nMissing values after imputation:")
    print(df['H'].isnull().sum())

    print("\nDataset after imputation:\n")
    print(df.head())

    # ====================================================
    # Log Transformation
    # ====================================================

    print("\n" + "=" * 50)
    print("Applying Log Transformation")
    print("=" * 50)

    if (df['R'] >= 0).all():

        original_skew = df['R'].skew()

        df['LogRuns'] = np.log1p(df['R'])

        transformed_skew = df['LogRuns'].skew()

        print(f"Original Skewness : {original_skew:.2f}")
        print(f"New Skewness      : {transformed_skew:.2f}")

    else:
        print("Cannot apply log transformation because column R contains negative values.")

    # ====================================================
    # Target Encoding


    print("\n" + "=" * 50)
    print("Target Encoding")
    print("=" * 50)

    # Create random Team IDs
    df['Team_ID'] = [
        f"Team_{np.random.randint(1,150)}"
        for _ in range(len(df))
    ]

    if TargetEncoder is not None:

        encoder = TargetEncoder()

        df['Team_ID_Encoded'] = encoder.fit_transform(
            df[['Team_ID']],
            df['W']
        ).iloc[:, 0]

        print("Target Encoding Applied Successfully.")

    else:

        print("TargetEncoder unavailable.")
        print("Skipping Target Encoding.")

    

    print("\n" + "=" * 50)
    print("Final Dataset")
    print("=" * 50)

    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    # Save processed dataset
    output_file = "processed_data.csv"
    df.to_csv(output_file, index=False)

    print(f"\nProcessed dataset saved as '{output_file}'.")

    features_to_test=['R','HR','SO','SB']

    X_features=df[features_to_test].fillna(0)

    y_target=df['W']

    selector=SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(X_features,y_target)

    winnig_features=selector.get_support()
    best_features=X_features.columns[winnig_features].tolist()

    print(best_features)

    X=df[best_features]
    y=df['H']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    print(f"Training Data size:{X_train.shape}")
    print(f"Testing Data size:{X_test.shape}")

#Training Model
    model=LinearRegression()
    model.fit(X_train,y_train)

    predictions= model.predict(X_test)
    print("predictions")

    
    

if __name__ == "__main__":
    main()