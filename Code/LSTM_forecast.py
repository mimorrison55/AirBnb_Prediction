# import statements
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import timeseries_dataset_from_array
import pickle

# pre-processing imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# model imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Exporting Model Imports
from joblib import dump, load

'''
Reading in Data
'''
print("Importing File")

# Path to File
timeSeriesLink = "socal_calendar.csv"
# Import Files
timeSeries = pd.read_csv(timeSeriesLink)
# rename listing_id to id for consistency
timeSeries = timeSeries.rename(columns={'listing_id': 'id'})
# extract only time series data for LA homes
timeSeries = timeSeries[(timeSeries['city'] == 'la')]

# print the first rows of the time series dataset
print(timeSeries.head())

'''
Exploratory Data Analysis
'''
# General Overview
print("Dataset Shape:", timeSeries.shape)
print("\nColumn Information:")
print(timeSeries.info())

# Summary Statistics for Numerical Columns
print("\nSummary Statistics:")
print(timeSeries.describe())

# Checking Missing Values
print("\nMissing Values:")
print(timeSeries.isnull().sum())

# drop all rows w/ missing values
timeSeries = timeSeries.dropna()

# Checking Unique Values for Categorical Columns
categorical_cols = timeSeries.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(timeSeries[col].value_counts())

# Check for NaN Values in the Dataset--specifically ID and Price
print("NaN Values in TimeSeries:")
print(timeSeries[['id', 'price']].isna().sum())

# checking the number of missing values in each feature
print(f"Time Series Missing Values: {timeSeries.isna().sum()}")

# Drop NaNs in crucial columns
timeSeries = timeSeries.dropna(subset=['id', 'price'])

print(timeSeries.head())

'''
Data Cleaning & Pre-Processing
'''
print("Data Cleaning")
# Parse date columns as datetime type
timeSeries['date'] = pd.to_datetime(timeSeries['date'])
# Step 1: Convert categorical type to string
timeSeries['price'] = timeSeries['price'].astype(str)
# Step 2: Clean the price columns (remove dollar sign and commas)
timeSeries['price'] = timeSeries['price'].replace({'\$': '', ',': ''}, regex=True)
# Step 3: Convert to numeric, coercing any errors to NaN
timeSeries['price'] = pd.to_numeric(timeSeries['price'], errors='coerce')

print(timeSeries.head())

'''
Feature Engineering for Time Series
'''
print("Feature Engineering")
# Sort the time series by date (important for LSTM)
timeSeries = timeSeries.sort_values(by=['id', 'date'])

# Create a lookback feature for time series data (previous day's price)
timeSeries['price_lag'] = timeSeries.groupby('id')['price'].shift(1)

# Create a feature that extracts the day of the week
# 0 = Monday, 6 = Sunday
timeSeries['day_of_week'] = timeSeries['date'].dt.dayofweek

# Create a feature that extracts the month & assign a specific season
# Extract month
timeSeries['Month'] = timeSeries['date'].dt.month

# Assign season based on month
def assign_season(month):
    if month in [12, 1, 2]:
        return 3
    elif month in [3, 4, 5]:
        return 0
    elif month in [6, 7, 8]:
        return 1
    else:
        return 2
# Convert 'available' column to numeric (0 for 'f', 1 for 't')
timeSeries['available'] = timeSeries['available'].map({'f': 0, 't': 1})
# get the season of the month
timeSeries['Season'] = timeSeries['Month'].apply(assign_season)
# Create a feature that determines if date is a weekend
timeSeries['is_weekend'] = (timeSeries['day_of_week'] >= 5).astype(int)
# Extract the day of the year
timeSeries['day_of_year'] = timeSeries['date'].dt.dayofyear
# Extract quarter (Q1 = 1, Q2 = 2, etc.)
timeSeries['quarter'] = timeSeries['date'].dt.quarter
# List of public holidays, focusing on just month-day
holidays = ['01-01', '02-14', '07-04', '11-11', '11-23', '11-27', '11-28', '12-25', '12-31']
# Extract month-day from the actual dates in the dataset
timeSeries['Month_Day'] = timeSeries['date'].dt.strftime('%m-%d')
# Check if the month-day is in the holiday list
timeSeries['is_holiday'] = np.where(timeSeries['Month_Day'].isin(holidays), 1, 0)
# Drop missing values resulting from lag or rolling window
timeSeries = timeSeries.dropna(subset=['price_lag'])

timeSeries['day'] = timeSeries['date'].dt.day
timeSeries['year'] = timeSeries['date'].dt.year

# timeSeries = timeSeries.drop('date', axis=1)
timeSeries = timeSeries.drop('city', axis=1)
timeSeries = timeSeries.drop('Month_Day', axis=1)

print(timeSeries.head())

print("TIME SERIES FEATURES")
print(timeSeries.columns)

print("Dataset Sizes")
print(f"Time Series Dataset: {timeSeries.shape}")

timeSeries.to_csv("FINAL_timeSeries.csv", index = True)

'''
CHECKPOINT
'''
print("Reading in Finalized Dataset:")
# Path to File
timeSeriesLink = "FINAL_timeSeries.csv"
# Import Files
timeSeries = pd.read_csv(timeSeriesLink)

print(timeSeries.head())

'''
EXPLORATORY DATA ANALYSIS
'''

def show_heatmap(data, string):
    plt.figure(figsize=(20, 10))  # Set a wide figure size for better fit
    corr_matrix = data.corr()  # Compute the correlation matrix
    plt.matshow(corr_matrix, fignum=1)  # Plot the heatmap on the same figure
    plt.xticks(range(data.shape[1]), data.columns, fontsize=12, rotation=90)  # Rotate x-ticks
    plt.gca().xaxis.tick_bottom()  # Place x-axis ticks at the bottom
    plt.yticks(range(data.shape[1]), data.columns, fontsize=12)  # Format y-ticks
    cb = plt.colorbar()  # Add a colorbar
    cb.ax.tick_params(labelsize=12)  # Adjust colorbar label size
    plt.title("Feature Correlation Heatmap", fontsize=16)  # Add title
    plt.tight_layout()  # Adjust layout to fit everything
    plt.savefig(string, bbox_inches='tight')  # Save the figure with tight bounding box
    plt.show()  # Display the plot

show_heatmap(timeSeries, "NEW_my_plot.png")

'''
LSTM MODEL FOR TIME SERIES FORECASTING
'''
# Parameters
look_back = 14         # past 14 days
forecast_horizon = 7   # next 7 days

features = ['minimum_nights', 'maximum_nights', 
            'price_lag',
            'day_of_week', 'Month', 'Season', 
            'quarter', 'is_holiday', 'day', 'year']
target = 'price'

# Sliding window function for multivariate data
def create_sequences_multivariate(X, y, look_back, forecast_horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - look_back - forecast_horizon + 1):
        X_seq.append(X[i:i + look_back])  # Include all features
        y_seq.append(y[i + look_back:i + look_back + forecast_horizon])  # Only target
    return np.array(X_seq), np.array(y_seq)

def prepare_time_series_data(timeSeries):
    # Initialize lists to store processed data
    all_X_list = []
    all_y_list = []

    # Track number of listings processed and skipped
    processed_listings = 0
    skipped_listings = 0

    # Global scalers to be returned
    global_scaler_X = MinMaxScaler()
    global_scaler_y = MinMaxScaler()

    # Process each listing with groupby
    for listing_id, group in timeSeries.groupby('id'):
        # Preprocess the group
        group = group.dropna(subset=features + [target])
        
        # Skip if insufficient data
        if len(group) < look_back + forecast_horizon:
            skipped_listings += 1
            continue

        # Extract features and target
        X_data = group[features].values
        y_data = group[target].values

        # Create sequences for the current listing
        X_seq, y_seq = create_sequences_multivariate(X_data, y_data, look_back, forecast_horizon)

        # Only add if sequences are valid
        if X_seq.shape[0] > 0:
            # Scale features and target
            X_seq_scaled = global_scaler_X.fit_transform(X_seq.reshape(-1, len(features))).reshape(X_seq.shape)
            y_seq_scaled = global_scaler_y.fit_transform(y_seq.reshape(-1, 1)).reshape(y_seq.shape)
            
            all_X_list.append(X_seq_scaled)
            all_y_list.append(y_seq_scaled)
            processed_listings += 1

    # Concatenate processed data
    all_X = np.concatenate(all_X_list, axis=0)
    all_y = np.concatenate(all_y_list, axis=0)

    # Logging and verification
    print(f"Total listings processed: {processed_listings}")
    print(f"Total listings skipped: {skipped_listings}")
    print(f"Processed all_X shape: {all_X.shape}")
    print(f"Processed all_y shape: {all_y.shape}")

    # Perform train-test split
    train_size = int(0.8 * len(all_X))
    X_train, X_test = all_X[:train_size], all_X[train_size:]
    y_train, y_test = all_y[:train_size], all_y[train_size:]

    # Further split training data into training and validation sets
    train_size = int(0.8 * len(X_train))
    X_train_seq, X_val_seq = X_train[:train_size], X_train[train_size:]
    y_train_seq, y_val_seq = y_train[:train_size], y_train[train_size:]

    print(f"Training Data Shape: {X_train_seq.shape}, {y_train_seq.shape}")
    print(f"Validation Data Shape: {X_val_seq.shape}, {y_val_seq.shape}")
    print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")

    return {
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        'X_test': X_test,
        'y_test': y_test,
        'scaler_X': global_scaler_X,
        'scaler_y': global_scaler_y
    }

# Usage
prepared_data = prepare_time_series_data(timeSeries)

# Access prepared data
X_train = prepared_data['X_train']
y_train = prepared_data['y_train']
X_val = prepared_data['X_val']
y_val = prepared_data['y_val']
X_test = prepared_data['X_test']
y_test = prepared_data['y_test']

# Access scalers if needed for inverse transformation
scaler_X = prepared_data['scaler_X']
scaler_y = prepared_data['scaler_y']

print(f"Columns for Predicting: {X_train.columns}")

def create_lstm_model(input_shape, output_shape):
    model = Sequential([
        # First LSTM layer with more units and return sequences
        LSTM(128, input_shape=input_shape, return_sequences=True, 
             activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        # Dropout to prevent overfitting
        Dropout(0.3),
        # Second LSTM layer
        LSTM(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        # Dropout layer
        Dropout(0.3),
        # Dense layers for output
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(output_shape, activation='linear')  # Linear activation for regression
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])
    
    return model

def train_lstm_model(X_train, y_train, X_val, y_val):
    # Prepare callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,
        patience=5, 
        min_lr=0.00001
    )
    
    # Create the model
    model = create_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_shape=y_train.shape[1]
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

# Evaluation function
def evaluate_model(model, X_test, y_test, scaler_y):
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    print("Model Evaluation Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")
    
    return y_pred_original, y_test_original

# Plotting function for results
def plot_predictions(y_test_original, y_pred_original, string):
    plt.figure(figsize=(12,6))
    plt.plot(y_test_original[0], label='Actual Price', color='blue')
    plt.plot(y_pred_original[0], label='Predicted Price', color='red')
    plt.title('Price Prediction: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.savefig(string)

# Train the model
model, history = train_lstm_model(X_train, y_train, X_val, y_val)

# print('Loading Model')
# new_model = tf.keras.models.load_model('model.h5')

# # Show the model architecture
# new_model.summary()

print("Predicting")
# Evaluate the model
y_pred_original, y_test_original = evaluate_model(new_model, X_test, y_test, scaler_y)

print("Plotting Results")
# Plot results
plot_predictions(y_test_original, y_pred_original, "prediction_plot.png")

# model.save("model.h5") -- UNCOMMENT IF NEEDED

print("One Sample Prediction Starting...")
prediction = new_model.predict(X_test)  # Predict for the entire test set

# Print the prediction and actual values for the first sample in the test set
sample_index = 0  # Change this to inspect different samples
print(f"Prediction for Sample {sample_index + 1}: {prediction[sample_index]}")
print(f"Actual Value for Sample {sample_index + 1}: {y_test[sample_index]}")

'''
Saving the model -- UNCOMMENT IF NEEDED
'''
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# # Loading the model in Shiny
# with open('model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# # Saving the model
# dump(model, 'model.joblib')

# # Saving the model
# model.save('saved_model_directory')
