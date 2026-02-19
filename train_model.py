import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("loan_data.csv")

# -------------------------------
# Drop unnecessary column
# -------------------------------
df = df.drop("Loan_ID", axis=1)

# -------------------------------
# Encode categorical columns
# -------------------------------
label_encoders = {}

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------------
# Split Features & Target
# -------------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------
# Save Model & Encoders
# -------------------------------
joblib.dump(model, "loan_prediction_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("Model trained and saved successfully!")
