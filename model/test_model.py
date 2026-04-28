import joblib
import numpy as np 
model = joblib.load('parkinson_model.pkl')
scaler = joblib.load('parkinson_scaler.pkl')

import pandas as pd

required_keys = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
                 'spread1', 'spread2', 'D2', 'PPE']

sample_input = pd.DataFrame([[119.992, 157.302, 74.997, 0.00784, 0.00007,
                          0.00370, 0.00554, 0.01109, 0.04374, 0.426,
                          0.02182, 0.03130, 0.02971, 0.06425, 0.02211,
                          21.033, 0.414783, 0.815285, -4.813031, 0.266482,
                          2.301442, 0.284654]], columns=required_keys)

scaled_input = scaler.transform(sample_input)

prediction = model.predict(scaled_input)

print(" Parkinson Prediction:", "Positive" if prediction[0] == 1 else "Negative")
