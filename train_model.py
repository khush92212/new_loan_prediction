import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import sklearn
print(sklearn.__version__)

# Load dataset
data = pd.read_csv("loan_data.csv")

# Create Label Encoders dictionary
label_encoders = {}

# Encode categorical columns
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define X and y
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Save feature names
feature_columns = X.columns

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save encoders
pickle.dump(label_encoders, open("encoders.pkl", "wb"))

# Save feature columns
pickle.dump(feature_columns, open("features.pkl", "wb"))

print("Model and encoders saved successfully!")
