import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Advertising.csv")

# Remove unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Pairplot of all features
sns.pairplot(df)
plt.suptitle("Pairplot of Advertising Data", y=1.02)
plt.show()

# Pairplot of all features
sns.pairplot(df)
plt.suptitle("Pairplot of Advertising Data", y=1.02)
plt.show()


# Sales vs TV advertising
plt.figure(figsize=(6, 4))
plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.title("Sales vs TV Advertising")
plt.show()

# Sales vs Radio advertising
plt.figure(figsize=(6, 4))
plt.scatter(df["Radio"], df["Sales"])
plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.title("Sales vs Radio Advertising")
plt.show()

# Sales vs Newspaper advertising
plt.figure(figsize=(6, 4))
plt.scatter(df["Newspaper"], df["Sales"])
plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.title("Sales vs Newspaper Advertising")
plt.show()

# Correlation matrix
correlation = df.corr()

# Display correlation values
print("\nCorrelation Matrix:")
print(correlation)

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Machine Learning Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Display coefficients
print("\nModel Coefficients:")
print("TV:", model.coef_[0])
print("Radio:", model.coef_[1])
print("Newspaper:", model.coef_[2])
print("Intercept:", model.intercept_)

# Predict sales for new advertising spending

new_data = pd.DataFrame([[200, 40, 30]], columns=["TV", "Radio", "Newspaper"])

predicted_sales = model.predict(new_data)

print("\nPredicted Sales:", predicted_sales[0])