# pyrefly: ignore [missing-import]
import numpy as np 
import pandas as pd 
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

try: 
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder 
except ImportError:
        TargetEncoder=None
        print("Warning Category Encoders not installed. Target encoding will not work")



def main():
    print("Loading Dataset") 
    file='data.csv'
    if not os.path.exists(file):
        print(f"Error")
        return 

    df=pd.read_csv(file)
    print(f"Dataset loaded successfully. Rows {df.shape[0]},Features: {df.shape[1]} \n") 

    print("Handling missing Data:")
    print("Artificially deleting some Hits (h) data to demonstrate ")

    df.loc[0.25,'H']=np.nan 

    imputer=SimpleImputer(strategy="median")
    df['H']=imputer.fit_transform(df[['H']]) 

    print(f"Imputation complete.'Hits' (H) now has {df['H'].isnull().sum()}")
    
if __name__=="__main__":
    main()

