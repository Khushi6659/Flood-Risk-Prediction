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

