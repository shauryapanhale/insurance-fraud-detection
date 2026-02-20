import pandas as pd

# Load dataset
data = pd.read_csv('fraud_oracle.csv')
# Show first few rows
print("First 5 rows:")
print(data.head())

print("\n" + "="*50 + "\n")

# Show all column names
print("All columns in dataset:")
print(data.columns.tolist())

print("\n" + "="*50 + "\n")

# Show dataset shape
print(f"Total rows: {len(data)}")
print(f"Total columns: {len(data.columns)}")
