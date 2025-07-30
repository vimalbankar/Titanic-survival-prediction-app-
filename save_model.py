# save_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load the cleaned dataset
df = pd.read_csv("titanic_cleaned.csv")

# Step 2: Handle missing values
df = df.dropna()  # OR use df.fillna(method='ffill')

# Step 3: Split into features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Save the trained model
joblib.dump(model, "titanic_model.pkl")

print("✅ Model trained and saved as titanic_model.pkl")
