from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model
classifier = load_model('trained_model.h5')
# Load the StandardScaler
scaler = joblib.load('scaler.pkl')

def predict_exit(sample_values):
    # Convert list to numpy array
    sample_values = np.array(sample_values).astype(float)
    # Feature Scaling
    sample_values = scaler.transform(sample_values.reshape(1, -1))
    return classifier.predict(sample_values)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        input_values = [request.form[field] for field in ["CreditScore", "Age", "Tenure", "Balance","HasCrCard","IsActiveMember", "Geography", "EstimatedSalary",  "Gender", "Exited","NumOfProducts"]]

        # input_values = [credit_score, age, tenure, balance, card, active, geo_value,estimated_salary,gende,exite,num_of_products]

        # Make prediction
        prediction = predict_exit(input_values)
        # Determine prediction result
        if prediction > 0.5:
            result = 'High chance of exit!'
        else:
            result = 'Low chance of exit.'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
