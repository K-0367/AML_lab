# ===============================
# Student Grade Prediction
# ===============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===========================================
# 1. DATASET LOADING
# ===========================================

# Replace with your dataset name
df = pd.read_csv("StudentsPerformance.csv")

print("First 5 Records")
print(df.head())

# ===========================================
# 2. DATASET DESCRIPTION
# ===========================================

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# ===========================================
# 3. DATA PREPROCESSING
# ===========================================

# Remove duplicate values
df.drop_duplicates(inplace=True)

# Fill missing numeric values
for column in df.select_dtypes(include=np.number):
    df[column].fillna(df[column].mean(), inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

for column in df.select_dtypes(include='object'):
    df[column] = label_encoder.fit_transform(df[column])

print("\nData After Preprocessing")
print(df.head())

# ===========================================
# 4. EXPLORATORY DATA ANALYSIS
# ===========================================

# Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Box Plot")
plt.xticks(rotation=90)
plt.show()

# Pair Plot
sns.pairplot(df)
plt.show()

# Correlation Matrix
correlation = df.corr()

print("\nCorrelation Matrix")
print(correlation)

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ===========================================
# 5. SIMPLE LINEAR REGRESSION
# ===========================================

# Replace 'Grade' with your target column

target = "SEM 5"

# Select first feature for Simple Linear Regression
feature = df.columns[0]

X = df[[feature]]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nSimple Linear Regression")

print("R2 Score :", r2_score(y_test, prediction))
print("MSE :", mean_squared_error(y_test, prediction))

# Plot Regression Line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, prediction, color='red')
plt.xlabel(feature)
plt.ylabel(target)
plt.title("Simple Linear Regression")
plt.show()

# ===========================================
# 6. MULTIPLE LINEAR REGRESSION
# ===========================================

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

multiple_model = LinearRegression()

multiple_model.fit(X_train, y_train)

prediction = multiple_model.predict(X_test)

print("\nMultiple Linear Regression")

print("R2 Score :", r2_score(y_test, prediction))
print("MSE :", mean_squared_error(y_test, prediction))

print("\nActual Values")
print(y_test.values[:10])

print("\nPredicted Values")
print(prediction[:10])