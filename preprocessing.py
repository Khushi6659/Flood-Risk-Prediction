#  Step 1: Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Step 2: Load dataset

df = pd.read_csv("flood_risk_dataset_india.csv")

# Show first 5 rows
df.head()

#  Step 3: Basic Exploration
# Dataset info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing values check
print("\nMissing Values:")
print(df.isnull().sum())

#  Step 4: Exploratory Data Analysis (EDA)

# Distribution of target variable
sns.countplot(x="Flood Occurred", data=df)
plt.title("Flood Occurrence Distribution")
plt.show()

# Correlation heatmap (numeric features only)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=["int64", "float64"]).corr(), 
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()


# Rainfall vs Flood Occurrence
sns.boxplot(x="Flood Occurred", y="Rainfall (mm)", data=df)
plt.title("Rainfall vs Flood Occurrence")
plt.show()

# Population Density vs Flood Occurrence
sns.boxplot(x="Flood Occurred", y="Population Density", data=df)
plt.title("Population Density vs Flood Occurrence")
plt.show()
