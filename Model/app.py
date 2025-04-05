import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model, scaler, and columns used during training
model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Title
st.title("🏏 IPL Win Probability Predictor")

# Input fields
batting_team = st.selectbox("Select Batting Team", [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
])

bowling_team = st.selectbox("Select Bowling Team", [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
])

city = st.selectbox("Match City", [
    'Mumbai', 'Hyderabad', 'Delhi', 'Chennai', 'Jaipur', 'Kolkata', 'Bangalore',
    'Ahmedabad', 'Pune', 'Rajkot', 'Visakhapatnam'
])

runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)
total_runs = st.number_input("Target Score", min_value=0)

# Calculate current and required run rates
if balls_left > 0:
    required_runrate = (runs_left * 6) / balls_left
else:
    required_runrate = 0

current_score = total_runs - runs_left
overs_done = (120 - balls_left) / 6
current_runrate = current_score / overs_done if overs_done > 0 else 0

# Predict button
if st.button("Predict Win Probability"):
    # Prepare input
    input_dict = {
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs': [total_runs],
        'current_runrate': [current_runrate],
        'required_runrate': [required_runrate]
    }

    input_df = pd.DataFrame(input_dict)

    # One-hot encode
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex to match training columns (fill missing with 0)
    input_final = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale and predict
    input_scaled = scaler.transform(input_final)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0]

    st.subheader("🎯 Result:")
    if prediction[0] == 1:
        st.success(f"Batting team will likely WIN 🏆 (Confidence: {round(prob[1]*100, 2)}%)")
    else:
        st.error(f"Batting team will likely LOSE 😞 (Confidence: {round(prob[0]*100, 2)}%)")