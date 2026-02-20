import pandas as pd

data = pd.read_csv('fraud_oracle.csv')

print("Checking actual values in dataset:\n")
print("="*60)

columns_to_check = [
    'Days_Policy_Accident',
    'PastNumberOfClaims', 
    'Days_Policy_Claim',
    'AgeOfPolicyHolder',
    'PolicyType',
    'Fault',
    'VehiclePrice',
    'DriverRating',
    'Make',
    'AccidentArea'
]

for col in columns_to_check:
    print(f"\n{col}:")
    print(data[col].unique()[:10])  # Show first 10 unique values
