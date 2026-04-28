from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR.parent / 'public'

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path='')

try:
    parkinson_model = joblib.load(BASE_DIR / 'parkinson_model.pkl')
    parkinson_scaler = joblib.load(BASE_DIR / 'parkinson_scaler.pkl')

    diabetes_model = joblib.load(BASE_DIR / 'diabetes_model.pkl')
    diabetes_scaler = joblib.load(BASE_DIR / 'diabetes_scaler.pkl')

    heart_model = joblib.load(BASE_DIR / 'heart_disease_model.pkl')
    heart_scaler = joblib.load(BASE_DIR / 'heart_disease_scaler.pkl')

    print(" Models and scalers loaded successfully.")
except Exception as e:
    print(f" Error loading models or scalers: {e}")
    exit()

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/')
def home():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_public_file(filename):
    return send_from_directory(PUBLIC_DIR, filename)

@app.route('/predict-parkinson', methods=['POST'])
def predict_parkinson():
    try:
        data = request.json  
        print(f"Received data: {data}")  

        required_keys = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                         'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                         'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                         'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
                         'Spread1', 'Spread2', 'D2', 'PPE']
        parkinson_feature_order = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                   'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                                   'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                                   'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                                   'spread1', 'spread2', 'D2', 'PPE']

        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        normalized_data = {
            'MDVP:Fo(Hz)': float(data['MDVP:Fo(Hz)']),
            'MDVP:Fhi(Hz)': float(data['MDVP:Fhi(Hz)']),
            'MDVP:Flo(Hz)': float(data['MDVP:Flo(Hz)']),
            'MDVP:Jitter(%)': float(data['MDVP:Jitter(%)']),
            'MDVP:Jitter(Abs)': float(data['MDVP:Jitter(Abs)']),
            'MDVP:RAP': float(data['MDVP:RAP']),
            'MDVP:PPQ': float(data['MDVP:PPQ']),
            'Jitter:DDP': float(data['Jitter:DDP']),
            'MDVP:Shimmer': float(data['MDVP:Shimmer']),
            'MDVP:Shimmer(dB)': float(data['MDVP:Shimmer(dB)']),
            'Shimmer:APQ3': float(data['Shimmer:APQ3']),
            'Shimmer:APQ5': float(data['Shimmer:APQ5']),
            'MDVP:APQ': float(data['MDVP:APQ']),
            'Shimmer:DDA': float(data['Shimmer:DDA']),
            'NHR': float(data['NHR']),
            'HNR': float(data['HNR']),
            'RPDE': float(data['RPDE']),
            'DFA': float(data['DFA']),
            'spread1': float(data['Spread1']),
            'spread2': float(data['Spread2']),
            'D2': float(data['D2']),
            'PPE': float(data['PPE']),
        }

        input_data = pd.DataFrame([normalized_data])[parkinson_feature_order]
        input_scaled = parkinson_scaler.transform(input_data)  

        prediction = parkinson_model.predict(input_scaled) 
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f" Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/predict-diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json  
        print(f"Received data: {data}") 

        required_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        input_data = pd.DataFrame([data])[required_keys]
        input_scaled = diabetes_scaler.transform(input_data)  

        prediction = diabetes_model.predict(input_scaled)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f" Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/predict-heart', methods=['POST'])
def predict_heart():
    try:
        data = request.json  
        print(f"Received data: {data}") 

        required_keys = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs',
                         'restecg', 'thalach', 'exang', 'oldpeak',
                         'slope', 'ca', 'thal']
        heart_feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                               'restecg', 'thalach', 'exang', 'oldpeak',
                               'slope', 'ca', 'thal']

        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        normalized_data = {
            'age': float(data['age']),
            'sex': float(data['sex']),
            'cp': float(data['cp']),
            'trestbps': float(data['restbp']),
            'chol': float(data['chol']),
            'fbs': float(data['fbs']),
            'restecg': float(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': float(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': float(data['slope']),
            'ca': float(data['ca']),
            'thal': float(data['thal']),
        }

        input_data = pd.DataFrame([normalized_data])[heart_feature_order]
        input_scaled = heart_scaler.transform(input_data)  

        prediction = heart_model.predict(input_scaled)  
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
