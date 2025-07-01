from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
         'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = dict.fromkeys(model_columns, 0)
    form = request.form

    # One-hot encode team
    input_data[f'bat_team_{form["batting_team"]}'] = 1
    input_data[f'bowl_team_{form["bowling_team"]}'] = 1

    # Numeric values
    input_data['overs'] = float(form['overs'])
    input_data['runs'] = int(form['runs'])
    input_data['wickets'] = int(form['wickets'])
    input_data['runs_last_5'] = int(form['runs_last_5'])
    input_data['wickets_last_5'] = int(form['wickets_last_5'])

    # Create input DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    return render_template('index.html', teams=teams, prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)
