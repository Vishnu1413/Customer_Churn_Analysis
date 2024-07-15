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
        credit_score = request.form['CreditScore']
        age = request.form['Age']
        tenure = request.form['Tenure']
        balance = request.form['Balance']
        num_of_products = request.form['NumOfProducts']
        estimated_salary = request.form['EstimatedSalary']

        has_cr_card = request.form['HasCrCard']
        is_active_member = request.form['IsActiveMember']
        exited = request.form['Exited']
    
        geography = request.form['Geography']
        gender = request.form['Gender']

        card = 1 if has_cr_card.lower() == 'yes' else 0
        active = 1 if is_active_member.lower() == 'yes' else 0
        exite = 1 if exited.lower() == 'yes' else 0
        gende = 1 if gender.lower() == 'male' else 0

        # Mapping for geography
        if geography.lower() == 'france':
            geo_value = 0
        elif geography.lower() == 'germany':
            geo_value = 1
        else:  # Assuming other cases represent 'Spain'
            geo_value = 2

        # Combine input values into a list
        input_values = [credit_score, age, tenure, balance, card, active, geo_value,estimated_salary,gende,exite,num_of_products]

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
