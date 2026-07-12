# importing the necessary dependencies
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)  # initializing a flask app

# Resolve paths relative to this file's own location, so the app works
# no matter which directory you launch "python app.py" from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'HDI.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'country_encoder.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find HDI.pkl at {MODEL_PATH}. "
        "Make sure HDI.pkl sits in the same folder as app.py "
        "(run Training/train_model.py first if it's missing)."
    )
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Could not find country_encoder.pkl at {ENCODER_PATH}. "
        "Make sure it sits in the same folder as app.py "
        "(run Training/train_model.py first if it's missing)."
    )

# loading the trained model and the country label-encoder
model = pickle.load(open(MODEL_PATH, 'rb'))
country_encoder = pickle.load(open(ENCODER_PATH, 'rb'))
COUNTRIES = sorted(list(country_encoder.classes_))


@app.route('/')  # route to display the home page
def home():
    return render_template('home.html')


@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('indexnew.html', countries=COUNTRIES)


@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def predict():
    try:
        country_name = request.form['Country']
        life_expectancy = float(request.form['Life expectancy'])
        mean_years_schooling = float(request.form['Mean years of schooling'])
        gni_per_capita = float(request.form['Gross national income (GNI) per capita'])
        internet_users = float(request.form['Internet users'])

        country_encoded = country_encoder.transform([country_name])[0]

        features_value = [[country_encoded, life_expectancy, mean_years_schooling,
                            gni_per_capita, internet_users]]
        features_name = ['Country', 'Life expectancy', 'Mean years of schooling',
                          'Gross national income (GNI) per capita', 'Internet users']

        df = pd.DataFrame(features_value, columns=features_name)

        # predicting using the loaded model file
        output = model.predict(df)
        y_pred = round(float(output[0][0]), 3)
        y_pred = min(max(y_pred, 0), 1)  # clamp to a valid HDI range

        if y_pred < 0.550:
            category = 'Low Human Development'
        elif y_pred < 0.700:
            category = 'Medium Human Development'
        elif y_pred < 0.800:
            category = 'High Human Development'
        else:
            category = 'Very High Human Development'

        prediction_text = f'Predicted HDI Score: {y_pred}  —  Category: {category}'
        return render_template('resultnew.html', prediction_text=prediction_text,
                                country=country_name, hdi_score=y_pred, category=category)

    except Exception as e:
        error_text = f'Could not generate a prediction. Please check your inputs. ({e})'
        return render_template('resultnew.html', prediction_text=error_text,
                                country=None, hdi_score=None, category=None)


if __name__ == '__main__':
    # running the app
    app.run(debug=True, port=5000)
