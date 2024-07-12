import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the trained model
model = joblib.load('best_model.pkl')

# Function to predict the rating
def predict_rating(input_data):
    features = np.array(input_data).reshape(1, -1)
    rating = model.predict(features)[0]
    return rating

# Streamlit App
def main():
    st.title('FIFA Player Rating Prediction')
    st.write('Enter the player details to predict the overall rating.')

    # Input fields
    columns = ['potential', 'value_eur', 'wage_eur', 'passing', 'dribbling',
               'movement_reactions', 'mentality_composure']
    inputs = []
    for col in columns:
        inputs.append(st.number_input(f'Enter {col}', step=1))

    # Prediction
    if st.button('Predict'):
        rating = predict_rating(inputs)
        st.success(f'Predicted Rating: {rating:.2f}')

if __name__ == '__main__':
    main()
