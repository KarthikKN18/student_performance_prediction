
import pandas as pd

# Read the Mathematics student dataset
df = pd.read_csv("student-mat.csv", sep=";")

print("DATASET LOADED SUCCESSFULLY")
print("-" * 40)

print("Number of rows and columns:")
print(df.shape)

print("\nFirst five records:")
print(df.head())

print("\nAll column names:")
print(df.columns.tolist())

print("\nMissing values in each column:")
print(df.isnull().sum())