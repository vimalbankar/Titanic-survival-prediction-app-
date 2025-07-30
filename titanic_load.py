import pandas as pd

df = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Pictures\Desktop\codsoft project\Titanic prediction\titanic_cleaned.csv")

print(" Data loaded successfully!")
print(df.head())
print("Script finished. Data shape is:", df.shape)

# Step 2: Understand the structure
print(df.info())         # Shows data types and nulls
print(df.describe())     # Summary stats
print(df.isnull().sum()) # Count missing values
print(df['Survived'].value_counts())  # Count how many survived vs not

# Fix missing values in Embarked column
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)
df.to_csv("titanic_cleaned.csv", index=False)


