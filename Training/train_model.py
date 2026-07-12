"""
HDI Predictor - Training Pipeline
Follows Epics 2-7 of the project spec:
  Epic 2: Import libraries
  Epic 3: Load & understand dataset
  Epic 4: Preprocessing & label encoding
  Epic 5: Train/test split
  Epic 6: Fit Linear Regression model
  Epic 7: Save model with Pickle
"""

# ---- Epic 2: Importing Required Libraries ----
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- Epic 3: Dataset Download and Understanding ----
Development = pd.read_csv("../Dataset/HDI.csv")
print("Shape:", Development.shape)
print(Development.head())
print("\nUnique countries:", Development["Country"].nunique())

# ---- Data Visualization (EDA) ----
data1 = Development.head(20)

plt.figure(figsize=(8, 5))
sns.stripplot(x="Mean years of schooling", y="HDI", data=data1, jitter=True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../Dataset/eda_schooling_vs_hdi.png")
plt.close()

plt.figure(figsize=(8, 6))
heat_cols = ["HDI Rank", "HDI", "Life expectancy", "Mean years of schooling",
             "Gross national income (GNI) per capita", "Internet users"]
sns.heatmap(Development[heat_cols].corr(), annot=True, cmap="rocket_r", fmt=".2f")
plt.tight_layout()
plt.savefig("../Dataset/eda_correlation_heatmap.png")
plt.close()
print("Saved EDA plots to Dataset/")

# ---- Epic 4: Data Preprocessing and Label Encoding ----
# Independent variables: Country, Life expectancy, Mean years of schooling, GNI per capita, Internet users
X = Development[["Country", "Life expectancy", "Mean years of schooling",
                  "Gross national income (GNI) per capita", "Internet users"]].copy()
y = Development[["HDI"]].copy()

# Label encode Country (categorical -> numeric) so it can feed the model
le_country = LabelEncoder()
X["Country"] = le_country.fit_transform(X["Country"])

print("\nNull values before fill:\n", X.isnull().sum())

numeric_cols = ["Life expectancy", "Mean years of schooling",
                 "Gross national income (GNI) per capita", "Internet users"]
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

print("\nNull values after fill:\n", X.isnull().sum())

# ---- Epic 5: Dividing the Dataset into Train and Test Data ----
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"\nTrain size: {x_train.shape}, Test size: {x_test.shape}")

# ---- Epic 6: Fitting the Model ----
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print("\nPredicted values:\n", y_pred.flatten())
print("\nActual values:\n", y_test.values.flatten())

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nR-squared: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# quick sanity test with a single example (Country label-encoded manually below not needed;
# we just take the first row of x_test as an example)
sample = x_test.iloc[[0]]
sample_pred = reg.predict(sample)
print("\nSample prediction:", round(sample_pred[0][0], 3), "| Actual:", y_test.iloc[0].values[0])

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("Actual HDI")
plt.ylabel("Predicted HDI")
plt.title(f"Actual vs Predicted HDI (R2={r2:.3f})")
plt.tight_layout()
plt.savefig("../Dataset/actual_vs_predicted.png")
plt.close()

# ---- Epic 7: Saving the Model ----
with open("../Flask/HDI.pkl", "wb") as f:
    pickle.dump(reg, f)

with open("../Flask/country_encoder.pkl", "wb") as f:
    pickle.dump(le_country, f)

print("\nModel and country encoder saved to Flask/HDI.pkl and Flask/country_encoder.pkl")
