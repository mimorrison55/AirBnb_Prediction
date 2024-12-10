import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from joblib import load
import os

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('LSTM_model.h5')

def load_scalers(directory='./model_files'):
    """Load scalers from files"""
    scaler_X = load(os.path.join(directory, 'scaler_X.joblib'))
    scaler_y = load(os.path.join(directory, 'scaler_y.joblib'))
    return scaler_X, scaler_y

# Cached model and scalers
model = load_model()

scaler_X, scaler_y = load_scalers()

def generate_feature_engineering(start_date, avg_price, min_nights, max_nights):
    """
    Generate additional features for prediction
    """
    # Generate 14 consecutive dates starting from start_date
    dates = [start_date + timedelta(days=i) for i in range(14)]
    
    # Create DataFrame with dates and additional features
    df = pd.DataFrame({'date': dates})
    
    # Feature engineering
    df['day_of_week'] = df['date'].dt.dayofweek
    df['Month'] = df['date'].dt.month
    
    # Season assignment
    def assign_season(month):
        if month in [12, 1, 2]:
            return 3  # Winter
        elif month in [3, 4, 5]:
            return 0  # Spring
        elif month in [6, 7, 8]:
            return 1  # Summer
        else:
            return 2  # Autumn
    
    df['Season'] = df['Month'].apply(assign_season)
    df['quarter'] = df['date'].dt.quarter
    
    # Holiday check (simplified version)
    holidays = ['01-01', '02-14', '07-04', '11-11', '11-23', '11-27', '11-28', '12-25', '12-31']
    df['Month_Day'] = df['date'].dt.strftime('%m-%d')
    df['is_holiday'] = np.where(df['Month_Day'].isin(holidays), 1, 0)
    
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year
    
    # Add constant features
    df['minimum_nights'] = min_nights
    df['maximum_nights'] = max_nights
    
    # Add price_lag as the 10th feature, using the average price
    df['price_lag'] = avg_price
    
    # Select and order features exactly as during training
    features = ['minimum_nights', 'maximum_nights', 
                'price_lag',  # Add this back
                'day_of_week', 'Month', 'Season', 
                'quarter', 'is_holiday', 'day', 'year']
    
    return df[features].values

def predict_prices(start_date, avg_price, min_nights, max_nights):
    # Generate features
    features = generate_feature_engineering(start_date, avg_price, min_nights, max_nights)
    
    # Ensure correct scaling
    # Option 1: Scale each time step individually
    features_scaled = np.array([scaler_X.transform(step.reshape(1, -1)).flatten() for step in features])
    
    # Reshape for LSTM input
    features_scaled = features_scaled.reshape(1, 14, 10)
    
    # Predict
    predictions_scaled = model.predict(features_scaled)
    
    # Inverse transform predictions
    predictions = scaler_y.inverse_transform(predictions_scaled)

    predictions = predictions / 2000
    
    return predictions.flatten()

# Streamlit App
def main():
    st.title('Airbnb Price Forecaster')
    
    # Sidebar inputs
    st.sidebar.header('User Inputs')
    avg_price = st.sidebar.number_input('Average Price', min_value=0.0, value=100.0, step=10.0)
    min_nights = st.sidebar.number_input('Minimum Nights', min_value=1, value=1)
    max_nights = st.sidebar.number_input('Maximum Nights', min_value=1, value=7)
    start_date = st.sidebar.date_input('Start Date', value=datetime.now())
    
    # Convert start_date to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    
    # Predict button
    if st.sidebar.button('Predict Prices'):
        try:
            # Get predictions
            predictions = predict_prices(start_date, avg_price, min_nights, max_nights)
            
            # Create dates for predictions
            dates = [start_date + timedelta(days=i+1) for i in range(7)]
            
            # Display predictions in a table
            pred_df = pd.DataFrame(data={
                '' : ['Day #1', 'Day #2', 'Day #3', 'Day #4', 'Day #5', 'Day #6', 'Day #7'],
                'Date': [date.strftime('%B %d, %Y') for date in dates],  # Format date as desired
                'Predicted Price': [f"${price:.2f}" for price in predictions],
                'Day of the Week': [date.strftime('%A') for date in dates]  # Add day of the week
            }, index=None)

            styled_pred_df = pred_df.style.hide(axis='index').set_table_styles(
                [{'selector': 'thead th', 'props': [('font-weight', 'bold'),('background-color', '#4CAF50'), ('color', 'white'), ('border', '1px solid lightblue')]},
                {'selector': 'td', 'props': [('background-color', 'white'),('color', 'black'), ('border', '1px solid lightblue')]}]
            )
            st.table(styled_pred_df)
            
            # Plot predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dates, predictions, marker='o')
            ax.set_title('Predicted Airbnb Prices')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
    st.markdown("""
<style>
    /* General body styling */
    body {
        background-color: #DEEFF5; /* Lightest blue background */
        font-family: 'Roboto', sans-serif;
    }
    
    /* Streamlit App Background */
    .stApp {
        background-color: #DEEAF5; /* Even less dark background */
    }
    
    /* Title Styling */
    .stTitle {
        color: #3187A2; /* Darkest blue for titles */
        margin-top: -20px; /* Remove extra spacing above the title */
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #4BAAC8; /* Less dark blue sidebar background */
        color: white;
    }
    
    /* Main Headers (h1, h2) */
    h1 {
        color: #3187A2; /* Darkest blue for main headers - text */
        margin-top: -10px; /* Adjust spacing above */
    }
    h2 {
        color: black; /* Less dark blue for subheaders */
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .stDataFrame th {
        background-color: #7CC1D7; /* Lesser dark blue for table headers */
        color: black;
        font-weight: bold;
    }
    .stDataFrame td {
        color: #3187A2; /* Text in darkest blue */
        font-weight:'bold';
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #FF5722; /* Bright orange for buttons */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #E64A19; /* Darker orange on hover */
    }
    
    /* Sidebar Inputs (Sliders, Selects, etc.) */
    .stSidebar input, .stSidebar select, .stSidebar textarea {
        background-color: white;
        color: #333;
        border: 1px solid #7CC1D7; /* Subtle border for inputs */
    }
    
    /* Plus and Minus Buttons (Darker Blue) */
    .stSlider > div > button {
        background-color: #215D6E; /* Darker blue, not black */
        color: white; /* White text */
        border-radius: 50%; /* Circular buttons */
        border: none;
    }
    .stSlider > div > button:hover {
        background-color: #4BAAC8; /* Lighter blue on hover */
    }
    
    /* Remove Black Rectangle (Replace with Blue) */
    div[data-testid="stDecoration"] {
        background-color: #4BAAC8 !important; /* Replace black with a blue that matches the theme */
    }
    
    /* Footer Styling */
    footer {
        color: #3187A2;
        font-size: 12px;
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)



# Run the app
if __name__ == '__main__':
    main()
