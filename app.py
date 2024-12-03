from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
from faker import Faker
import os

app = Flask(__name__)

# Initialize Faker
fake = Faker()

# Generate fake data
def generate_fake_data(num_samples=100):
    data = {
        'age': [random.randint(1, 120) for _ in range(num_samples)],
        'gender': [random.choice(['Male', 'Female']) for _ in range(num_samples)],
        'height': [random.uniform(150, 200) for _ in range(num_samples)],  # Height in cm
        'weight': [random.uniform(50, 100) for _ in range(num_samples)],   # Weight in kg
        'systolic': [random.randint(90, 180) for _ in range(num_samples)],  # Systolic BP
        'diastolic': [random.randint(60, 120) for _ in range(num_samples)],  # Diastolic BP
        'diagnosis': [random.choice(['Healthy', 'Hypertension', 'Diabetes', 'Heart Disease']) for _ in range(num_samples)]
    }
    return pd.DataFrame(data)

# Load and preprocess data
def load_and_preprocess_data():
    # Generate fake data
    data = generate_fake_data()
    
    # Encode gender (convert Male/Female to 0/1)
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # Example preprocessing
    X = data.drop('diagnosis', axis=1)  # Ensure 'diagnosis' is the target column
    y = data['diagnosis']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode target labels if necessary
    label_encoders = {}
    if isinstance(y.iloc[0], str):  # Check if target is categorical
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_encoders['diagnosis'] = le
    
    return X_scaled, y, scaler, label_encoders, X.columns  # Return column names for prediction


# Train model
def train_model(X, y):
    model = RandomForestClassifier()  # Example model
    model.fit(X, y)
    return model

# Generate and preprocess data
X, y, scaler, label_encoders, feature_names = load_and_preprocess_data()
trained_model = train_model(X, y)

# Function to predict diagnosis
def predict_diagnosis(model, scaler, label_encoders, feature_names, age, gender, height, weight, systolic, diastolic):
    # Ensure that gender is encoded as numeric (0 for Female, 1 for Male)
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Prepare input data as DataFrame with feature names, applying the gender encoding
    input_data = pd.DataFrame([[age, gender_encoded, height, weight, systolic, diastolic]], columns=feature_names)
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Decode prediction if necessary
    diagnosis = label_encoders['diagnosis'].inverse_transform(prediction)[0] if 'diagnosis' in label_encoders else prediction[0]
    
    return diagnosis

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = int(data['age'])
    gender = data['gender']  # Assume gender is passed as 0 or 1 (Male=1, Female=0)
    height = float(data['height'])
    weight = float(data['weight'])
    systolic = int(data['systolic'])
    diastolic = int(data['diastolic'])
    
    diagnosis = predict_diagnosis(trained_model, scaler, label_encoders, feature_names, age, gender, height, weight, systolic, diastolic)
    
    return jsonify({'diagnosis': diagnosis})

if __name__ == '__main__':
    # Use '0.0.0.0' for public access and dynamically assigned port
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
