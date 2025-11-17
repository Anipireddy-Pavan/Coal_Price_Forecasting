# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:26:08 2025

@author: anipi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.plotting import lag_plot
# If you're using urllib.request to open URLs:
import urllib.request
# If you're using urllib.parse for URL parsing or query strings:
import urllib.parse
# If you're using urllib.error to handle exceptions:
import urllib.error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from pmdarima import auto_arima
import joblib
import tensorflow as tf
import warnings

#Data set:
    
df = pd.read_excel(r"C:\Users\anipi\OneDrive\Desktop\CPF- DOC's\CPF--Historical_Prices --2020-2024.xlsx")    

df.columns = ['Date', 'Price', 'Price1', 'Price2', 'Price3', 'Price4']
df = df.drop(index=0).reset_index(drop=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')


#--------------------------
#Sql-alchemy
#--------------------------
from sqlalchemy import create_engine
import urllib.parse

# Original password
password = "Pavan@123"

# Encode the password to make it URL-safe
encoded_password = urllib.parse.quote_plus(password)
# Database connection string
connection_string = f"mysql+pymysql://root:{encoded_password}@localhost/cpf_code"
# Create SQLAlchemy engine for connection
engine = create_engine(connection_string)
# Save DataFrame to SQL
df.to_sql('coal_price', con=engine, if_exists="replace", index=False)
# Read data from SQL
df = pd.read_sql_query("SELECT * FROM coal_price", engine)

#--------------------------------------
#Features
#--------------------------------------

# Time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
#df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Quarter'] = df['Date'].dt.quarter  # Quarter feature added

#--------------------
#EDA
#--------------------

import sweetviz as sv
eda_report = sv.analyze(df)
eda_report.show_html("sweetviz_report.html")

#--------------------------
#Graphical analysis
#--------------------------
# Extract numerical columns (excluding 'Date')
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Univariate Analysis - Histograms and Boxplots
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# Boxplots for outlier detection
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'{col} Boxplot')
    plt.xlabel(col)
    plt.show()

#Bivariate analysis.
# Pairplot (scatter plot matrix)
sns.pairplot(df[numerical_cols])
plt.suptitle("Pairwise Scatter Plots", y=1.02)
plt.show()

# Correlation heatmap
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Multivariate Analysis - Pairwise Relationships (using pairplot)
sns.pairplot(df[numerical_cols])
plt.suptitle("Multivariate Pairwise Analysis", y=1.02)
plt.show()


# Time Series Analysis - Line plots for numerical columns over time
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    plt.plot(df['Date'], df[col], label=f'{col} Over Time')
    plt.title(f'{col} Over Time')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.legend()
    plt.show()


# Line Plots for all numerical columns
plt.figure(figsize=(8, 6))
for col in numerical_cols:
    plt.plot(df['Date'], df[col], label=col)
plt.title("Line Plot of All Numerical Columns Over Time")
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()


# Bubble Plot - Price vs Price1 vs Price2
plt.figure(figsize=(8, 6))
plt.scatter(df['Price'], df['Price1'], s=df['Price2']*10, alpha=0.5)
plt.title("Bubble Plot: Price vs Price1 vs Price2")
plt.xlabel("Price")
plt.ylabel("Price1")
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

# Make sure 'Date' is the index
if df.index.name != 'Date':
    df.set_index('Date', inplace=True)

# Loop over each numerical column and perform decomposition
for col in df.select_dtypes(include=np.number).columns:
    try:
        print(f"Decomposing time series for: {col}")
        result = seasonal_decompose(df[col].dropna(), model='multiplicative', period=365)

        # Plot the decomposition
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(result.observed)
        plt.title(f'{col} - Observed')

        plt.subplot(4, 1, 2)
        plt.plot(result.trend)
        plt.title('Trend')

        plt.subplot(4, 1, 3)
        plt.plot(result.seasonal)
        plt.title('Seasonality')

        plt.subplot(4, 1, 4)
        plt.plot(result.resid)
        plt.title('Residual')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not decompose {col}: {e}")


from pandas.plotting import lag_plot

# Loop through all numeric columns
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(6, 6))
    lag_plot(df[col].dropna())
    plt.title(f'Lag Plot for {col}')
    plt.xlabel(f'{col} (t)')
    plt.ylabel(f'{col} (t+1)')
    plt.grid(True)
    plt.show()

#plot_acf, plot_pacf

# Loop through all numerical columns
for col in df.select_dtypes(include=np.number).columns:
    series = df[col].dropna()

    print(f"ACF and PACF for: {col}")
    
    # ACF plot
    plt.figure(figsize=(10, 4))
    plot_acf(series, lags=40, title=f"ACF for {col}")
    plt.tight_layout()
    plt.show()

    # PACF plot
    plt.figure(figsize=(10, 4))
    plot_pacf(series, lags=40, method='ywm', title=f"PACF for {col}")
    plt.tight_layout()
    plt.show()

#-----------------------
#Data preprocessing:-
#-------------------------
duplicates = df.duplicated().sum()
print(duplicates)

drop = df.drop_duplicates(inplace = True)
print(drop)

missing = df.isnull().sum()
print(missing)

f_missing = df.fillna(df.mean(), inplace = True)
df.isnull().sum()

#--------------------------
#Outlier - Winsorization.
#----------------------------
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#df.drop(columns=['IsWeekend'], inplace=True)

# Initialize Winsorizer for numerical columns using Gaussian method
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=numerical_cols
)

df_winsorized = winsor.fit_transform(df)


# Compare boxplots before and after Winsorization
for column in numerical_cols:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column], color='blue')
    plt.title(f'Original Boxplot - {column}')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_winsorized[column], color='red')
    plt.title(f'Winsorized Boxplot - {column}')

    plt.tight_layout()
    plt.show()

# Summary Statistics after Winsorization
print("\nSummary Statistics After Winsorization:")
print(df_winsorized.describe())

# Save the cleaned and winsorized DataFrame to Excel
df_winsorized.to_excel('cleaned_coal_price_data333.xlsx', index=False)

#Scaler
scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
# Apply StandardScaler
df_scaled = df_winsorized.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_winsorized[numerical_cols])
# Summary Statistics After Standardization
print("\nSummary Statistics After Standardization:")
print(df_scaled.describe())

#---------------
#PIPELINE
#---------------
# Define numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'year' and col != 'month' and col != 'day' and col != 'day_of_week' and col != 'quarter']  # Optionally exclude time features

# Define the pipeline for numerical features
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),            
    ('winsorizer', Winsorizer(capping_method='iqr',         
                              tail='both', 
                              fold=1.5,
                              variables=None)),             
    ('scaler', MinMaxScaler())                              
])

# Create ColumnTransformer to apply pipeline only to numerical columns
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_cols)
], remainder='passthrough')  

# Apply the transformation
df_processed = preprocessor.fit_transform(df)

# Convert transformed output back to DataFrame
processed_column_names = numerical_cols + [col for col in df.columns if col not in numerical_cols]
df_processed = pd.DataFrame(df_processed, columns=processed_column_names)

# Ensure Date column is in datetime format again (if needed)
if 'Date' in df_processed.columns:
    df_processed['Date'] = pd.to_datetime(df_processed['Date'])

# Display
print("\n Data After Preprocessing Pipeline:")
print(df_processed.head())

joblib.dump(preprocessor, "preprocessing_pipeline1111.pkl")



#-----------------------------
#Target - Columns
#-------------------------------
target_column = 'Price'
target_column = 'Price1'
target_column = 'Price2'
target_column = 'Price3'
target_column = 'Price4'



import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the list to store metrics for comparison
metrics_list = []

# Assuming you have defined a function to create lag features
def create_lag_features(df, target_column, n_lags=7):
    df_lstm = df.copy()
    
    for lag in range(1, n_lags + 1):
        df_lstm[f'{target_column}_lag{lag}'] = df_lstm[target_column].shift(lag)
    
    df_lagged = df_lstm.dropna().copy()
    
    return df_lagged


# Iterate over the target columns
#for target_column in ['Price', 'Price1', 'Price2', 'Price3', 'Price4']:
#    print(f"\nProcessing target column: {target_column}")
    
    # Create lag features
    df_lagged = create_lag_features(df, target_column)
    
    # Prepare X and y
    X_cols = [col for col in df_lagged.columns if col != target_column]
    X = df_lagged[X_cols].values
    y = df_lagged[target_column].values
    # Scale the features
    scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)
    # Convert the NumPy array (X_scaled) to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled)

    # Save to Excel
    X_scaled_df.to_excel('X_scaled_forecast4.xlsx', index=False)
    # Split the data into training and testing sets
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define model evaluation function
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return mse, rmse, mae, r2, mape, model

    # 1. Linear Regression Model
    print("Training Linear Regression...")
    lin_reg_model = LinearRegression()
    mse, rmse, mae, r2, mape, lin_reg_model = evaluate_model(lin_reg_model, X_train, X_test, y_train, y_test)
    metrics_list.append({'Model': 'Linear Regression', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

    # 2. Decision Tree Model
    print("Training Decision Tree...")
    tree_model = DecisionTreeRegressor()
    mse, rmse, mae, r2, mape, tree_model = evaluate_model(tree_model, X_train, X_test, y_train, y_test)
    metrics_list.append({'Model': 'Decision Tree', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

    # 3. Lasso Regression Model
    print("Training Lasso Regression...")
    lasso_model = Lasso()
    mse, rmse, mae, r2, mape, lasso_model = evaluate_model(lasso_model, X_train, X_test, y_train, y_test)
    metrics_list.append({'Model': 'Lasso Regression', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

    # 4. Ridge Regression Model
    print("Training Ridge Regression...")
    ridge_model = Ridge()
    mse, rmse, mae, r2, mape, ridge_model = evaluate_model(ridge_model, X_train, X_test, y_train, y_test)
    metrics_list.append({'Model': 'Ridge Regression', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

    # 5. Gradient Boosting Model
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor()
    mse, rmse, mae, r2, mape, gb_model = evaluate_model(gb_model, X_train, X_test, y_train, y_test)
    metrics_list.append({'Model': 'Gradient Boosting', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})
    
    #Random Forest Regressor.
    from sklearn.ensemble import RandomForestRegressor
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred_rf)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_rf)
    r2 = r2_score(y_test, y_pred_rf)
    mape = np.mean(np.abs((y_test - y_pred_rf) / y_test))  # MAPE as decimal

    # Store the model and metrics
    random_forest_regressor_model = rf_model
    metrics_list.append({
      'Model': 'Random Forest Regressor',
      'MSE': mse,
      'RMSE': rmse,
      'MAE': mae,
      'R²': r2,
      'MAPE': mape
       })

    # XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

   # Evaluate
    mse = mean_squared_error(y_test, y_pred_xgb)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_xgb)
    r2 = r2_score(y_test, y_pred_xgb)
    mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test))  # MAPE as decimal

    # Store the model and metrics
    xgboost_regressor_model = xgb_model
    metrics_list.append({
      'Model': 'XGBoost Regressor',
      'MSE': mse,
      'RMSE': rmse,
      'MAE': mae,
      'R²': r2,
      'MAPE': mape
       })

   #  K-Nearest Neighbors Regressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    import numpy as np

    def evaluate_model(model, X_train, X_test, y_train, y_test, y_pred):
       mse = mean_squared_error(y_test, y_pred)
       rmse = np.sqrt(mse)
       mae = mean_absolute_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)
       mape = mean_absolute_percentage_error(y_test, y_pred)
       return mse, rmse, mae, r2, mape

    print("Training K-Nearest Neighbors Regressor...")
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

   # Evaluate the model using the evaluate_model function
    mse, rmse, mae, r2, mape = evaluate_model(knn_model, X_train, X_test, y_train, y_test, y_pred_knn)

   # Store the metrics
    metrics_list.append({'Model': 'KNN Regressor', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

   #  Support Vector Regressor (SVR)  
    print("Training Support Vector Regressor...")
    svr_model = SVR(kernel='rbf')  # You can also try 'linear' or 'poly' kernels
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)

    # Evaluate the model using the evaluate_model function
    mse, rmse, mae, r2, mape = evaluate_model(svr_model, X_train, X_test, y_train, y_test, y_pred_svr)

    # Store the metrics
    metrics_list.append({'Model': 'SVR', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

   # 10. AdaBoost Regressor
    print("Training AdaBoost Regressor...")
    ada_boost_model = AdaBoostRegressor(n_estimators=50)  # You can adjust the number of estimators
    ada_boost_model.fit(X_train, y_train)
    y_pred_ada_boost = ada_boost_model.predict(X_test)

    # Evaluate the model using the evaluate_model function
    mse, rmse, mae, r2, mape = evaluate_model(ada_boost_model, X_train, X_test, y_train, y_test, y_pred_ada_boost)

   # Store the metrics
    metrics_list.append({'Model': 'AdaBoost Regressor', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})


    #  ARIMA Model
    print("Training ARIMA...")

    # Fit ARIMA model on the training data
    arima_model = ARIMA(y_train, order=(5, 1, 0))
    arima_model_fit = arima_model.fit()

    # Forecast for the length of the test data
    y_pred_arima = arima_model_fit.forecast(steps=len(y_test))

    # Evaluate manually (MAPE not in %)
    mse = mean_squared_error(y_test, y_pred_arima)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_arima)
    r2 = r2_score(y_test, y_pred_arima)
    mape = np.mean(np.abs((y_test - y_pred_arima) / y_test))  # <-- Not multiplied by 100

    # Store metrics
    metrics_list.append({
        'Model': 'ARIMA',
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
         })
  
    # 7. Exponential Smoothing Model
    
    print("Training Exponential Smoothing...")

    # Fit the model
    exp_smoothing_model = ExponentialSmoothing(
    y_train, trend='add', seasonal='add', seasonal_periods=12
     )
    exp_smoothing_model_fit = exp_smoothing_model.fit()

    # Forecast for the test period
    y_pred_exp_smooth = exp_smoothing_model_fit.forecast(len(y_test))

    # Manual evaluation
    mse = mean_squared_error(y_test, y_pred_exp_smooth)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_exp_smooth)
    r2 = r2_score(y_test, y_pred_exp_smooth)
    mape = np.mean(np.abs((y_test - y_pred_exp_smooth) / y_test))  # MAPE as decimal

    # Save the metrics
    metrics_list.append({
    'Model': 'Exponential Smoothing',
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R²': r2,
    'MAPE': mape
      })

   # 8. LSTM Model
    print("Training LSTM...")

   # Reshape input for LSTM [samples, timesteps, features]
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(128))
    lstm_model.add(Dense(1))

   # Compile and fit the model
    lstm_model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=16, validation_data=(X_test_lstm, y_test), callbacks=[early_stop])

    # Make predictions
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()  # flatten to match shape with y_test

   # Evaluate
    mse, rmse, mae, r2, mape = evaluate_model(lstm_model, X_train_lstm, X_test_lstm, y_train, y_test, y_pred_lstm)

   # Store metrics
    metrics_list.append({'Model': 'LSTM', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})

   # 9. GRU Model
    print("Training GRU...")

   # Build the GRU model
    gru_model = Sequential() 
    gru_model.add(GRU(128, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    gru_model.add(Dropout(0.2))
    gru_model.add(GRU(128))
    gru_model.add(Dense(1))

   # Compile and train the model
    gru_model.compile(optimizer='adam', loss='mse')
    gru_model.fit(X_train_lstm, y_train, epochs=100, batch_size=16, validation_data=(X_test_lstm, y_test), callbacks=[early_stop])

   # Predict
    y_pred_gru = gru_model.predict(X_test_lstm).flatten()  # Flatten to match shape of y_test

    # Evaluate the model
    mse, rmse, mae, r2, mape = evaluate_model(gru_model, X_train_lstm, X_test_lstm, y_train, y_test, y_pred_gru)

   # Store the metrics
    metrics_list.append({'Model': 'GRU', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape})


# Convert to DataFrame and sort by R²
eval_df = pd.DataFrame(metrics_list)
eval_df = eval_df.sort_values(by='R²', ascending=False).reset_index(drop=True)

# Display the comparison table
print("\n  Updated Model Evaluation Comparison:")
print(eval_df)

# Show best model
best_model = eval_df.loc[0]
best_model_name = best_model['Model']
print(f"\n  Best Model Based on R²: {best_model['Model']} (R² = {best_model['R²']:.4f})")

# Dynamically generate model variable name
model_var_name = best_model_name.lower().replace(' ', '_').replace('+', '').replace("’", '').replace("'", '').replace('-', '') + '_model'

# Get the best model object dynamically
best_model_object = globals().get(model_var_name)

# Save the model based on type

if 'lstm' in model_var_name or 'gru' in model_var_name:  # lowercase check
    filename = f"{model_var_name}_{target_column}_best_model.keras"
    best_model_object.save(filename)  # Saves in Keras recommended format
    print(f"✅ Saved Keras model: {filename}")
else:
    filename = f"{model_var_name}_{target_column}_best_model.pkl"
    import joblib
    joblib.dump(best_model_object, filename)
    print(f"✅ Saved model: {filename}")




# Save results to Excel
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_excel("model_comparison_results4.xlsx", index=False)


'''
#PRICE - Traget


	Model	MSE	RMSE	MAE	R²	MAPE
0	GRU	3.663004662879488	1.9138977670919333	1.4702387235310095	0.9365347597926242	0.0188695859907375
1	Random Forest Regressor	3.822519876332519	1.9551265627402536	1.489253035518744	0.9337709982716194	0.02009477997584855
2	AdaBoost Regressor	3.9737426035705505	1.9934248427193213	1.585465236445134	0.9311509124152635	0.020651005709408546
3	XGBoost Regressor	4.639984921249411	2.154062422783846	1.6901333240707341	0.9196075941235067	0.02249025456726914
4	Gradient Boosting	5.106465980466631	2.2597490967951797	1.7163999346030023	0.9115253405639004	2.3057616303810664
5	LSTM	5.5054449444865705	2.346368458807476	1.93196117241438	0.9046126287003806	0.026792746815881924
6	Ridge Regression	6.6886344284789505	2.5862394375770683	2.282253165040125	0.8841126807823815	2.9657230457257167
7	Lasso Regression	7.376902235029272	2.716045330076299	2.281857362808285	0.8721877487416502	2.9714336796994734
8	Decision Tree	15.347653023255823	3.917608074227924	2.6455813953488385	0.7340864739782522	3.538774605068249
9	Linear Regression	17.02334390812215	4.125935519142556	3.7228974447033982	0.7050534439089563	5.0986574003115575
10	SVR	50.11796289781857	7.079404134375899	6.1876914680158155	0.13165588178257537	0.08671961464281411
11	Exponential Smoothing	53.756410086072215	7.331876300516274	5.934090932792302	0.06861612452415955	0.07665758097486862
12	ARIMA	71.78894438741939	8.472835675700278	7.509908004220168	-0.2438156702208405	0.10209653656067962
13	KNN Regressor	223.5217766426743	14.950644689867868	11.022226198922608	-2.8727396090878075	0.15391522351527437

#GRU

'''

'''
#Price1 - Target

	Model	MSE	RMSE	MAE	R²	MAPE
0	Lasso Regression	3.6826245404941855	1.9190165555550023	1.5597698759327245	0.9524231549214282	1.602161223319642
1	Gradient Boosting	4.21233132342653	2.0523964829989674	1.750814061947821	0.9455797264720922	1.8659215470534074
2	XGBoost Regressor	5.40792752098011	2.3254951130845467	1.8464881063242524	0.9301334884855548	0.019549376176011816
3	Random Forest Regressor	6.097575610867251	2.469326955035977	2.045151746156894	0.9212237340507758	0.021497127343324665
4	GRU	8.43197200828542	2.9037858061994553	2.220294492872672	0.8910650212820187	0.023676568630338752
5	Linear Regression	9.589643726183807	3.0967149894983566	2.9560883169723997	0.8761087401383254	3.1743124919680223
6	Decision Tree	11.113566005792254	3.333701547198287	2.6618435159637377	0.8564207666804107	2.829714252188899
7	Ridge Regression	11.718126172550676	3.423174867363728	3.2328941161941245	0.8486102866604515	3.482651071009658
8	LSTM	14.087093980750733	3.753277764934369	2.8373541057678384	0.818004936273107	0.0302167002523993
9	AdaBoost Regressor	26.42895526626381	5.1409099648081575	4.116409124762501	0.6585570164796648	0.043648042574467555
10	SVR	75.72091891181573	8.70177676752373	7.330478885352322	0.0217405036378987	0.08277851589075931
11	ARIMA	86.4021015217659	9.29527307408265	7.682255523646045	-0.11625264899050913	0.08418908522580601
12	Exponential Smoothing	216.95735062973625	14.72947217756754	13.346499823725145	-1.8029320247193001	0.14815491247369011
13	KNN Regressor	268.8632082871334	16.39704876760246	12.99960260149783	-2.4735181573216183	0.1459729659862726

#Lasso
'''
'''
#Price2 -  Target

	Model	MSE	RMSE	MAE	R²	MAPE
0	Random Forest Regressor	6.7428262769108125	2.5966952606940255	2.1260320903954644	0.9403764663821659	0.020223880988230993
1	XGBoost Regressor	8.288387780650915	2.8789560227017907	2.4833861879903343	0.9267098176369312	0.023553121144518577
2	Decision Tree	10.65062807588557	3.263530002295914	2.608285507817633	0.9058216755030286	2.4577842282839644
3	Gradient Boosting	10.688996462312014	3.269403074310663	2.785964507757604	0.905482402521046	2.661170293773189
4	Ridge Regression	10.986483517851509	3.314586477654718	2.6617096454002964	0.9028518691618274	2.5093033515176932
5	Lasso Regression	12.381760798282858	3.5187726266814767	3.1824260135662508	0.8905141107175875	3.038356878822299
6	GRU	16.191144615704374	4.023822140167775	2.9427936586934593	0.8568295821870195	0.027888481884314205
7	Linear Regression	16.962496595704426	4.118555158754636	3.8310524328592903	0.850008891749213	3.611680250349214
8	LSTM	20.908373938786514	4.572567543381564	3.455088938262864	0.815117417350307	0.03345822990711487
9	AdaBoost Regressor	42.79220074062521	6.541574790570326	5.453369385425144	0.6216093794116412	0.05096667023875789
10	ARIMA	139.01071174499506	11.790280392976033	10.013905151784734	-0.22920412073318475	0.09809770548452454
11	SVR	189.41543204242004	13.76282790862474	12.101510979358208	-0.674908549667087	0.12033705622855205
12	KNN Regressor	384.32733065981137	19.604268174553503	15.288867691499142	-2.39841968023327	0.15248827089592962
13	Exponential Smoothing	522.1442543457395	22.850476020112566	18.671644256018556	-3.6170677137191705	0.1732107436583397


#Random Forest Regressor
'''

'''
Price3 - Target

	Model	MSE	RMSE	MAE	R²	MAPE
0	Ridge Regression	10.763652163300547	3.280800536957489	1.9104004420438159	0.9322350766891049	1.7750258110499728
1	XGBoost Regressor	14.30911024866077	3.7827384589290296	1.6635282302384409	0.9099138708742603	0.014974931598503211
2	Gradient Boosting	14.74717152568094	3.840204620287953	1.5810001414303994	0.9071559604185547	1.4313201796814379
3	Linear Regression	15.10904337750284	3.887035294090194	2.207833368304453	0.904877717131328	2.080826089092283
4	Random Forest Regressor	15.361238152939418	3.9193415458389715	2.0738302653492937	0.9032899698353742	0.019058821516335984
5	GRU	22.41325031734631	4.734263439791485	3.5148850460201437	0.8588924868752786	0.03189988186140559
6	AdaBoost Regressor	23.262745770894853	4.823146874281858	3.1676424315003713	0.8535443026912015	0.02828543349327352
7	Decision Tree	23.49021734112324	4.846670748165511	2.722259567674849	0.8521122057339539	2.523600706063233
8	Lasso Regression	27.19314922543713	5.214705094771624	4.109290314489626	0.8287995892206147	3.913695726826997
9	LSTM	34.696371550307774	5.890362599221526	4.573304953700922	0.7815614141369688	0.042741362599263026
10	ARIMA	226.1657705352138	15.038808813706417	13.258345714105348	-0.4238760100520278	0.12842020184505049
11	SVR	313.42052894710986	17.70368687440867	15.601244398057062	-0.9732074007906635	0.1531921673549945
12	KNN Regressor	524.082678089244	22.892852117838967	17.660631660698105	-2.299476975888829	0.17469616050753367
13	Exponential Smoothing	620.5410077009581	24.91066052317678	22.363714726746853	-2.9067514594624884	0.21888247375278808


#Ridge
'''
'''
Price4 - Target

	Model	MSE	RMSE	MAE	R²	MAPE
0	Lasso Regression	3.7402697119533235	1.933977691689675	1.3877571087794576	0.9284004575218374	1.247002929198785
1	Ridge Regression	4.033926178904813	2.008463636440753	1.584089512763877	0.922779026368815	1.4795047457280348
2	Linear Regression	4.304311858341628	2.0746835561939627	1.6530346928516384	0.9176030651598004	1.5423465512206984
3	LSTM	7.015306175131733	2.648642326765117	2.0387595632463964	0.8657068203187563	0.01896035787911464
4	GRU	8.90286236484299	2.983766472906851	2.382577513827792	0.829573554540292	0.021775353871590124
5	Random Forest Regressor	10.330828621943729	3.2141606403451166	2.4009707502299262	0.802238164700378	0.021774525497068997
6	Decision Tree	14.359235768976118	3.7893582265307297	2.708164586344324	0.7251228412461642	2.4605244720964836
7	Gradient Boosting	23.8881066976681	4.887546081385637	4.161064817723072	0.542712787594845	3.805000711863565
8	XGBoost Regressor	29.5547740920986	5.436430271060101	4.524403894357861	0.4342364412178731	0.041396072846324405
9	SVR	39.32287861621498	6.270795692431302	4.6951840180669775	0.2472467670319568	0.04546044065546871
10	AdaBoost Regressor	45.728049629193144	6.762251816458268	5.964534048563753	0.124633332883615	0.05438294434825186
11	ARIMA	75.50616782182772	8.689428509506694	6.9446110271389	-0.44540567570430145	0.06191176323963964
12	Exponential Smoothing	221.84065204786114	14.89431609869554	13.049464540917684	-3.2466694684937583	0.11706868612260019
13	KNN Regressor	318.0257907728696	17.833277622828327	14.19571062935225	-5.087930248137541	0.13470468727272955

#Lasso
'''







###################################################################
####################################################################

