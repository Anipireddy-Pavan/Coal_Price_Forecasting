# Coal_Price_Forcasting
Forcasting Project

#Project Work Flow
✔ Business Understanding
✔ Data Understanding
✔ Data Collection
✔ Data Preprocessing
✔ EDA
✔ Graphical Representation
✔ Model Building
✔ Evaluation
✔ Deployment
✔ Monitoring & Maintenance

Project Workflow (End-to-End) – CRISP-ML(Q) Based
1️⃣ Business Understanding

Coal prices fluctuate due to global demand, supply disruptions, economic conditions, and seasonal variations.
Industries such as power plants, steel, cement, and manufacturing heavily depend on coal pricing.

Business Goals

Predict future coal prices accurately.

Enable companies to plan procurement and negotiations.

Reduce business risk caused by price volatility.

Support long-term budgeting and forecasting decisions.

Machine Learning Objective

Build a reliable forecasting model using ML/Deep Learning techniques that can predict future coal prices for the next few months.

2️⃣ Data Understanding

The dataset contains historical coal prices from 2020–2024 with the following key features:

Key Columns

Date

Price (main coal price index)

Price1–Price4 (additional price sources)

Year, Month, Week, Quarter

Rolling Average

Lag Features

Initial Observations

Non-stationary time series

Missing values in certain periods

Outliers and price jumps

Clear monthly seasonality

Upward trends due to market conditions

3️⃣ Data Collection
Data was collected from:

Public coal price API / Excel datasets

Historical coal index websites

Internal files downloaded as Excel or CSV

Data Loading Steps

Read using pandas.read_excel()

Stored raw data in /data/raw_data.xlsx

Loaded into MySQL database for data integrity

Retrieved clean data for machine learning

4️⃣ Data Preprocessing
Steps Performed

✔ Handling Missing Values

Forward fill / backward fill

Mean/median imputation (where required)

✔ Outlier Treatment

Winsorization using IQR

Capped extreme values without losing data structure

✔ Date Column Cleanup

Converted to datetime format

Sorted chronologically

✔ Feature Scaling

MinMaxScaler for LSTM/GRU

StandardScaler for ML models

✔ Train–Test Split (Time-Series Aware)

No shuffling

Last portion reserved as test data

✔ Database Storage

Cleaned data exported back into MySQL

Ensures reproducibility

5️⃣ Exploratory Data Analysis (EDA)

Performed in-depth analysis to understand structure and patterns:

Univariate Analysis

Distribution of coal prices

Summary statistics

Histogram, KDE plots

Outlier detection

Bivariate Analysis

Correlation heatmap

Time vs Price

Price vs rolling averages

Multivariate Analysis

Price trends across different columns (Price1–Price4)

Feature interactions

6️⃣ Graphical Representation

Visualizations created using Matplotlib and Seaborn:

-- Time Series Plots

Price over time

Moving averages

Daily/weekly trends

Seasonal decomposition

-- Rolling Analysis

Rolling mean and rolling standard deviation

-- Advanced Plots

ACF/PACF plots

Heatmaps

Boxplots

Pair plots

These visualizations help understand:

Trend

Seasonality

Volatility

Stationarity

7️⃣ Model Building

Multiple forecasting models were implemented:

Machine Learning Models

Linear Regression

Random Forest

Gradient Boosting

XGBoost

Support Vector Regression (SVR)

Time Series Models

ARIMA

SARIMA

Holt-Winters Exponential Smoothing

Deep Learning Models

LSTM

GRU

Bidirectional LSTM

CNN-GRU

Transformer (optional)

Feature Engineering for models

Lag features

Rolling features

Difference features

Momentum and percentage change

Seasonal indicators

8️⃣ Model Evaluation
Evaluation Metrics

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Percentage Error)

R² Score

Residual Error Analysis

Visualization for Evaluation

Actual vs Predicted plots

Error distribution

Forecast horizon graph

Best Model Example (Your Results)

(You can update with your exact values)

Model	R² Score	RMSE	MAE	MAPE
GRU	0.932	0.0685	0.0262	11.95%
LSTM	0.924	0.0722	0.032	10.30%
XGBoost	0.91	2.23	1.22	—
9️⃣ Deployment

Deployment-ready structure is created using:

✔ Stored Models:

/models/lstm_model.h5

/models/gru_model.h5

/models/arima_model.pkl

✔ Deployment Options:

Flask API

FastAPI microservice

Streamlit dashboard

Power BI integration

Dockerized application

✔ Pipeline Includes:

Data preprocessing script

Prediction function

Scheduled forecasting (batch mode)

🔟 Monitoring and Maintenance

Once deployed, the model must be monitored:

Monitoring KPIs

Forecast accuracy drift

Data drift (new patterns in coal prices)

Feature drift

Model prediction stability

Maintenance Plan

Retrain model monthly or quarterly

Auto-trigger retraining when error exceeds threshold

Update API or dashboard

Re-run EDA for new trends

Add new data sources

Logging & Alerts

Store predictions & errors daily

Notify if model deviates beyond acceptable limits.
