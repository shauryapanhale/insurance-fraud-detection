import pandas as pd

data = pd.read_csv('fraud_oracle.csv')

# Get fraud cases
fraud_cases = data[data['FraudFound_P'] == 1]
legitimate_cases = data[data['FraudFound_P'] == 0]

print("="*60)
print("ACTUAL FRAUDULENT CLAIMS (First 3):")
print("="*60)
print(fraud_cases[['Deductible', 'Days_Policy_Accident', 'PastNumberOfClaims', 
                   'Days_Policy_Claim', 'AgeOfPolicyHolder', 'PolicyType', 
                   'Fault', 'VehiclePrice', 'Make', 'AccidentArea']].head(3))

print("\n" + "="*60)
print("ACTUAL LEGITIMATE CLAIMS (First 3):")
print("="*60)
print(legitimate_cases[['Deductible', 'Days_Policy_Accident', 'PastNumberOfClaims', 
                        'Days_Policy_Claim', 'AgeOfPolicyHolder', 'PolicyType', 
                        'Fault', 'VehiclePrice', 'Make', 'AccidentArea']].head(3))

print("\n" + "="*60)
print("FRAUD PATTERN ANALYSIS:")
print("="*60)
print("\nMost common patterns in FRAUD:")
print(f"  PolicyType: {fraud_cases['PolicyType'].value_counts().head(3).to_dict()}")
print(f"  Fault: {fraud_cases['Fault'].value_counts().to_dict()}")
print(f"  AccidentArea: {fraud_cases['AccidentArea'].value_counts().to_dict()}")
print(f"  Days_Policy_Accident: {fraud_cases['Days_Policy_Accident'].value_counts().head(3).to_dict()}")
