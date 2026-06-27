import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding Dataset")

file='dataset.csv'
if not os.path.exists(file):
    print(f"Error:{file} is not found")

df=pd.read_csv('dataset.csv')
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

median_spending=df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)
print(median_spending)

mean_age=df['Age'].mean()
df['Age']=df['Age'].fillna(mean_age)
print(mean_age) 

plt.figure(figsize=(8,6))
df['Spending'].hist(bins=18,color='skyblue',edgecolor='black')
plt.title('Spending Distribution')
plt.xlabel('Spending Amount')
plt.ylabel('Number of customers')
plt.show()

correlation = df.corr(numeric_only=True)
print(correlation)

print("Plotting Correlation Heatmap")
plt.figure(figsize = (7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("correlation Heatmap")
plt.show()

print("find the outliers in age")
outliers = df[df['Age']>100]
print(outliers)