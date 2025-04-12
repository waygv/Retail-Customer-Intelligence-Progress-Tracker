import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Dataset

path = "/kaggle/input/customer-personality-analysis/marketing_campaign.csv"
df = pd.read_csv(path, sep='\t')
df.head()
df.info()
df.describe()
df.isnull().sum()

# Basic EDA

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Income"])
plt.title("Income Outliers")
plt.show()

df["Age"] = 2025 - df["Year_Birth"]
sns.boxplot(x=df["Age"])
plt.title("Age Outliers")
plt.show()

spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df["TotalSpend"] = df[spend_cols].sum(axis=1)

sns.histplot(df["TotalSpend"], kde=True)
plt.title("Total Spending Distribution")
plt.show()

# Heat Map

plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Imputation

df["Income"] = df["Income"].fillna(df["Income"].median())

# OHC

df = pd.get_dummies(df, columns=["Education", "Marital_Status"], drop_first=True)

# Standard Scaler

from sklearn.preprocessing import StandardScaler

features_to_scale = ['Income', 'Age', 'Recency', 'TotalSpend', 'Kidhome', 'Teenhome']

scaler = StandardScaler()

df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Saving the Cleaned-Data

df.to_csv('cleaned_customer_data.csv', index=False)