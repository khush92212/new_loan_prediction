import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("loan_data.csv")

# Drop ID column
df = df.drop("Loan_ID", axis=1)

# Split features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Define categorical & numeric columns
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numeric_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "loan_prediction_model.pkl")

print("Model trained successfully!")
