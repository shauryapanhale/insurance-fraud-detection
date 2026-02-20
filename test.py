import pickle
import numpy as np
import pandas as pd

# Load all saved files
model = pickle.load(open('models/fraud_detection_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))

print("Model loaded successfully!\n")
print("="*60)

# Test Case 1: Based on ACTUAL FRAUD pattern
test_case_1 = {
    'Deductible': 400,                       
    'Days_Policy_Accident': 'more than 30',  
    'PastNumberOfClaims': 'none',            
    'Days_Policy_Claim': 'more than 30',     
    'AgeOfPolicyHolder': '26 to 30',         
    'PolicyType': 'Sedan - All Perils',      # Most common in fraud
    'Fault': 'Policy Holder',                
    'VehiclePrice': 'more than 69000',       
    'DriverRating': 1,                       
    'Make': 'Honda',                         
    'AccidentArea': 'Urban'                  
}

# Test Case 2: Based on ACTUAL LEGITIMATE pattern
test_case_2 = {
    'Deductible': 300,                       
    'Days_Policy_Accident': 'more than 30',  
    'PastNumberOfClaims': 'none',            
    'Days_Policy_Claim': 'more than 30',     
    'AgeOfPolicyHolder': '31 to 35',         
    'PolicyType': 'Sedan - All Perils',      
    'Fault': 'Policy Holder',                
    'VehiclePrice': 'more than 69000',       
    'DriverRating': 1,                       
    'Make': 'Honda',                         
    'AccidentArea': 'Urban'                  
}

# Test Case 3: Clearly different (Third Party fault)
test_case_3 = {
    'Deductible': 500,
    'Days_Policy_Accident': '15 to 30',      
    'PastNumberOfClaims': '1',               
    'Days_Policy_Claim': '8 to 15',          
    'AgeOfPolicyHolder': '41 to 50',         
    'PolicyType': 'Utility - Liability',     
    'Fault': 'Third Party',                  # Different!
    'VehiclePrice': '30000 to 39000',        
    'DriverRating': 2,                       
    'Make': 'Toyota',                        
    'AccidentArea': 'Rural'                  
}

def predict_fraud(test_data, case_name):
    print(f"\n{case_name}")
    print("-"*60)
    
    # Encode categorical features
    encoded_data = test_data.copy()
    for col, value in encoded_data.items():
        if col in label_encoders:
            encoded_data[col] = label_encoders[col].transform([str(value)])[0]
    
    # Create dataframe
    input_df = pd.DataFrame([encoded_data], columns=feature_names)
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    fraud_probability = probability[1] * 100
    
    print(f"\n🎯 Prediction: {'🔴 FRAUDULENT' if prediction == 1 else '🟢 LEGITIMATE'}")
    print(f"📊 Fraud Probability: {fraud_probability:.2f}%")
    print(f"📊 Legitimate Probability: {probability[0]*100:.2f}%")
    
    if fraud_probability > 70:
        print("⚠️  Risk Level: HIGH - Manual review required!")
    elif fraud_probability > 40:
        print("⚠️  Risk Level: MEDIUM - Additional verification needed")
    else:
        print("✅ Risk Level: LOW - Can be auto-approved")
    
    print("="*60)

# Run predictions
predict_fraud(test_case_1, "TEST CASE 1: Fraud Pattern (Deductible=400)")
predict_fraud(test_case_2, "TEST CASE 2: Legit Pattern (Deductible=300)")
predict_fraud(test_case_3, "TEST CASE 3: Third Party Fault")

print("\n✅ Testing Complete!")
