import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("dataset/loan_data.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df.drop("Loan_ID", axis=1, inplace=True)

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

# -----------------------------
# Encode Categorical Columns
# -----------------------------
label_encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("Encoded Dataset:\n")
print(df.head())

print("\nData Types:\n")
print(df.dtypes)


import matplotlib.pyplot as plt

# -----------------------------
# Loan Status Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df["Loan_Status"].value_counts().plot(kind="bar")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df["Gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Education Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df["Education"].value_counts().plot(kind="bar")
plt.title("Education Distribution")
plt.xlabel("Education")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Applicant Income Distribution
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(df["ApplicantIncome"], bins=20)
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
#plt.show()

from sklearn.model_selection import train_test_split

# -----------------------------
# Split Features and Target
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
joblib.dump(model, "model/loan_model.pkl")

print("Model saved successfully!")