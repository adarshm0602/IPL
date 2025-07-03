
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv("ipl.csv")


teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
         'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
         'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[df['bat_team'].isin(teams) & df['bowl_team'].isin(teams)]
df = df[df['overs'] >= 5.0]
df.drop(columns=['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], inplace=True)


df = pd.get_dummies(df, columns=['bat_team', 'bowl_team'])


X = df.drop(columns=['date', 'total'])
y = df['total']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor()
model.fit(X_train, y_train)


joblib.dump(model, 'rf_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')  
