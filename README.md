# AirBnb_Prediction
This repository includes a machine learning model that predicts the prices of AirBnb properties given the houses' features and availability over a year.

**Link to the Slides:** https://docs.google.com/presentation/d/1ZJtg86SDy4PtLH7bZK5IrIjNsI3L7cVFLtcx3Yuq9MI/edit?usp=sharing
**Link to the Application:** https://airbnbprediction-lstm.streamlit.app/
**Link to the Time Series Data:** https://drive.google.com/file/d/18pi_HUZRThSuAd9AU8HT2-BJEskdgfjx/view?usp=sharing

## IDENTIFICATION
* NAME: Tiffany Le, Michael Morrison, Samina Aziz
* EMAIL: tifle@chapman.edu, mimorrison@chapman.edu, saziz@chapman.edu

## PROJECT DESCRIPTION
Our project focuses on predicting Airbnb prices for the next seven (7) days using data from the previous 14 days. The dataset contains a diverse collection of dates, prices, availabilities, etc. By creating a model that can forecast prices, we can accomplish tasks involving time series analysis, investment analysis, and other time-series-related applications. Additionally, this project's sub goal is the classification of a host's response time based on various factors. This subgoal was not incorporated in the application, but the model is trained and ready for deployment.

## PREQUISITES
* Install the packages listed in requirements.txt through `pip install`
* Having a Streamlit account or similar service such AWS or Azure works

## SOURCE FILES
1. `Data\LSTM` - contains the split-up calendar dataset (for GitHub storage purposes)
2. `Data\XGBoost` - Contains the Listing Dataset with Distance features
3. `Exploratory Data Analysis` - Contains plots and diagrams for the LSTM and XGBoost model
4. `model_files` - Includes the scaler files utilized for training the model and deploying the application
5. `LSTM_forecast.py` - Contains cleaning, building, training, and evaluation for LSTM
6. `LSTM_model.h5` - Contains the trained LSTM model
7. `XGBoost_prediction.py` - Contains cleaning, building, training, and evaluation for XGBoost
8. `app.py` - Contains the Streamlit Application
9. `change.python-version` - Contains the specified Python Version to run the Streamlit Application (3.11.10)
10. `pyproject.toml` - Contains the specified Python Version to run the Streamlit Application (3.11.10)
11. `requirements.txt` - Specifies the necessary packages for the application
12. `runtime.txt` - Contains the specified Python Version to run the Streamlit Application (3.11.10)
13. `scaler_X.joblib` - Contains the predictor scalers for the XGBoost
14. `scaler_y.joblib` - Contains the target variable scalers for the XGBoost
15. `xgboost_model.joblib` - Contains the trained XGBoost model
16. `xgboost_model.pkl` - Contains the trained XGBoost model

## CONTRIBUTIONS:
1. Tiffany oversaw the development of the LSTM and XGBoost models, Streamlit application and deployment, and model details & Chapman Server in the white paper.
2. Michael handled the content of the presentation, appendix, and conclusion of the white paper.
3. Samina reviewed the white paper and handled the content (introduction, Executive Summary, Sagemaker, and Results) of the white paper.

## REFERENCES & GRATITUDE
Thank you to Prof. Frenzel and Prof. Abby Bechtel-Mar for their guidance on this project.
