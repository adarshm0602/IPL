# IPL First Innings Score Predictor

A machine learning-powered web app that predicts the first innings final score in an IPL match based on real-time match conditions like overs, runs, and wickets.

This project uses Flask, scikit-learn, and a Random Forest Regressor trained on historical IPL match data.

## Features

- Select Batting and Bowling teams
- Enter current match conditions:
  - Overs completed
  - Runs
  - Wickets
  - Last 5 overs stats
- Predict the first innings final score
- Simple HTML/CSS frontend, Python backend

## Project Structure

project/
├── app.py                  # Flask web server
├── model_creator.py        # Trains & saves the ML model
├── rf_model.pkl            # Trained ML model file
├── model_columns.pkl       # Column structure for prediction
├── ipl.csv                 # IPL dataset
├── requirements.txt        # Dependencies list
├── README.md
└── templates/
    └── index.html          # User input form

## How to Run the Project

1. Clone the Repository

git clone [https://github.com/adarshm0602/IPL.git](https://github.com/adarshm0602/IPL.git)
cd ipl-score-predictor

2. Create and Activate Virtual Environment (Optional but Recommended)

python -m venv venv
venv\Scripts\activate       # Windows
# or
source venv/bin/activate    # macOS/Linux

3. Install Required Packages

pip install -r requirements.txt

4. Train the Model (One-Time Setup)

python model_creator.py

5. Start the Flask Web App

python app.py

Then visit: http://127.0.0.1:5000

## Dataset Used

- Based on historical IPL match data
- Filtered to remove irrelevant columns and overs < 5
- Teams are one-hot encoded
- Trained to predict final first-innings score based on match context

## Model Details

- Algorithm: Random Forest Regressor
- Libraries: scikit-learn, pandas, joblib
- Input Features:
  - Batting/Bowling teams (one-hot encoded)
  - Overs completed
  - Current score
  - Wickets fallen
  - Last 5 overs performance

## Deployment Options

Deploy this app on:
- Render
- Replit
- Railway
- Or host locally using Flask

## UI Preview

(Add a screenshot or GIF of the web app here if available)

## License

MIT License - you are free to use, modify, and distribute this project.

## Acknowledgements

- Inspired by IPL analytics projects and ML bootcamps
- Built for learning and hands-on deployment of ML models
