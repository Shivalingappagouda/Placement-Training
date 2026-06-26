import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os

print("Understanding dataset")
file = 'sales.csv'
if not os.path.exists(file):
    print(f"Error:{file} is not found")


df=pd.read_csv('sales.csv')
print("Successfully Loaded!")
print(f"Shape of dataset = Rows: {df.shape[0]},columns:{df.shape[1]}")
print(f"\nHead of the data set\n {df.head()}")
print(f"\nTail of the data set\n {df.tail()}")
print(f"\nDescription of the data set\n {df.describe()}")

print("\nHandling Missing Values:")

print(df.isnull().sum())

# With using Median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)