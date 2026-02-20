import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('fraud_oracle.csv')

print("Dataset loaded successfully!")
print(f"Total records: {len(data)}")

# Select features
selected_features = [
    'Deductible',              # claim_amount
    'Days_Policy_Accident',    # policy_age_days
    'PastNumberOfClaims',      # previous_claims
    'Days_Policy_Claim',       # days_since_last_claim
    'AgeOfPolicyHolder',       # claimant_age
    'PolicyType',              # policy_type
    'Fault',                   # incident_type
    'VehiclePrice',
    'DriverRating',
    'Make',
    'AccidentArea'
]

# Create a copy
df = data[selected_features + ['FraudFound_P']].copy()

# Encode ALL non-numeric columns
label_encoders = {}
for col in df.columns:
    if col != 'FraudFound_P' and df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded: {col}")

# Separate features and target
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

print(f"\nTarget distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

# Check accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save everything
pickle.dump(model, open('fraud_detection_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('feature_names.pkl', 'wb'))

print("\n✅ Files created:")
print("   - fraud_detection_model.pkl")
print("   - scaler.pkl")
print("   - label_encoders.pkl")
print("   - feature_names.pkl")
print("\n🎉 Step 1 Complete! Ready for Docker!")
