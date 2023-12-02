import numpy as np 
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


plt.ioff()

filepath = 'C:/Local Drive D/Heyt/Chicago College Docs/Sem 1/Subject CS584 - Machine Learning/Final Project/Global Economy Indicators/Global Economy Indicators.csv'
df = pd.read_csv(filepath)
pd.set_option('display.max_columns', None)  # Display all columns
print(df.head(10))

#Fill all empty values 
df.fillna(0, inplace=True)

# Select non-numeric columns and apply label encoding
non_numeric_cols = df.select_dtypes(include=['object']).columns

# Initialize label encoder for each column
label_encoders = {}
for col in non_numeric_cols:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    label_encoders[col] = label_encoder

# Correlation matrix
correlation_matrix = df.corr()

print(df[non_numeric_cols])

#Correlation Matrix Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8},cbar_kws={"shrink": 0.8})
plt.show()

for col, label_encoder in label_encoders.items():
    df[col] = label_encoder.inverse_transform(df[col])


df3 = df.groupby([' Year ']).sum()
df3_gdp = df3[' Gross Domestic Product (GDP) ']
df3_gdp.plot();

plt.ylabel('Global GDP (Tens of Trillions)')
plt.xlabel('Year')
plt.title('Global GDP since 1970')
plt.show()

#List of top 10 countries in GDP for year 2021
df3_2021 = df[df[' Year '] == 2021] 
df3_country = df3_2021.groupby([' Country ']).sum()
df3_gdp_sorted = df3_country[' Gross Domestic Product (GDP) '].sort_values().tail(10)
df3_gdp_sorted.plot(kind = 'barh');
plt.ylabel('GDP in 2021 (Tens of Trillions)')
plt.xlabel('Countries')
plt.title('List of top 10 countries by GDP in Year 2021');
plt.show()

#List of bottom 10 countries in GDP for year 2021
#df['gdp per capita'] = df[' Gross Domestic Product (GDP) '] / df[' Population ']
df3_gdp_sorted = df3_country[' Gross Domestic Product (GDP) '].sort_values().head(10)
df3_gdp_sorted.plot(kind = 'barh');

plt.ylabel('GDP in 2021 (Tens of Trillions)')
plt.xlabel('Countries')
plt.title('List of bottom 10 countries by GDP in Year 2021');
plt.show()


#Identifying Cross Entropy Strategy
# Select top and bottom 10 countries based on GDP in 2021
df_top = df[df[' Year '] == 2021].nlargest(10, ' Gross Domestic Product (GDP) ')
df_bottom = df[df[' Year '] == 2021].nsmallest(10, ' Gross Domestic Product (GDP) ')

# Create a DataFrame for top and bottom countries over the years
df_top_over_time = df[df[' Country '].isin(df_top[' Country '])]
df_bottom_over_time = df[df[' Country '].isin(df_bottom[' Country '])]

#Implementing Cross Entropy Strategies For Top 10 Countries 
# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=10)

# Initialize Linear Regression model
model = LinearRegression()
# Training and cross-validation for top countries
for train_index, val_index in tscv.split(df_top_over_time[' Year ']):
    X_train, X_val = df_top_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val = df_top_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train, y_val = df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_top_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the model
    model.fit(X_train, y_train)
    # Predict on validation set
    y_pred = model.predict(X_val)
    # Evaluate performance (e.g., using mean squared error)
    mse = mean_squared_error(y_val, y_pred)
    print(f'Mean Squared Error (Linear Regression - Top Countries): {mse}')
    r2 = r2_score(y_val, y_pred)
    print(f'R squared (Linear Regression - Top Countries): {r2}')
# Visualization of the Linear Regression Model
plt.figure(figsize=(10, 6))
# Scatter plot of actual GDP values for top countries
sns.scatterplot(x=df_top_over_time[' Year '], y=df_top_over_time[' Gross Domestic Product (GDP) '], label='Actual (Top Countries)')
# Scatter plot of predicted values for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred, label='Predicted (Top Countries)')
# Set labels and title
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression Model for Top Countries')
# Show the plot
plt.legend()
plt.show()


# Initialize Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
# Training and cross-validation for top countries using Random Forest
for train_index, val_index in tscv.split(df_top_over_time[' Year ']):
    X_train, X_val = df_top_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val = df_top_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train, y_val = df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_top_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the Random Forest model
    model_rf.fit(X_train, y_train)
    # Predict on validation set
    y_pred_rf = model_rf.predict(X_val)
    # Evaluate performance
    mse_rf = mean_squared_error(y_val, y_pred_rf)
    r2_rf = r2_score(y_val, y_pred_rf)
    print(f'Mean Squared Error (Random Forest - Top Countries): {mse_rf}') 
    print(f'R Squared (Random Forest - Top Countries): {r2_rf}')
# Visualization of the Models (Random Forest and Linear Regression) for top countries
plt.figure(figsize=(12, 8))
# Scatter plot of actual GDP values for top countries
sns.scatterplot(x=df_top_over_time[' Year '], y=df_top_over_time[' Gross Domestic Product (GDP) '], label='Actual (Top Countries)')
# Scatter plot of predicted values using Linear Regression for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred, label='Linear Regression - Predicted (Top Countries)')
# Scatter plot of predicted values using Random Forest for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred_rf, label='Random Forest - Predicted (Top Countries)')
# Set labels and title for top countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression vs. Random Forest for Top Countries')
# Show the plot for top countries
plt.legend()
plt.show()


# Initialize Gradient Boosting model
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
# Training and cross-validation for top countries using Gradient Boosting
for train_index, val_index in tscv.split(df_top_over_time[' Year ']):
    X_train, X_val = df_top_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val = df_top_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train, y_val = df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_top_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the Gradient Boosting model
    model_gb.fit(X_train, y_train)
    # Predict on validation set
    y_pred_gb = model_gb.predict(X_val)
    # Evaluate performance
    mse_gb = mean_squared_error(y_val, y_pred_gb)
    r2_gb=r2_score(y_val,y_pred_gb)
    print(f'Mean Squared Error (Gradient Boosting - Top Countries): {mse_gb}')
    print(f'R Squared (Gradient Boosting - Top Countries): {r2_gb}')
# Visualization of the Models (Gradient Boosting,Random Forest and Linear Regression) for top countries
plt.figure(figsize=(12, 8))
# Scatter plot of actual GDP values for top countries
sns.scatterplot(x=df_top_over_time[' Year '], y=df_top_over_time[' Gross Domestic Product (GDP) '], label='Actual (Top Countries)')
# Scatter plot of predicted values using Linear Regression for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred, label='Linear Regression - Predicted (Top Countries)')
# Scatter plot of predicted values using Random Forest for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred_rf, label='Random Forest - Predicted (Top Countries)')
# Scatter plot of predicted values using Gradient Boosting for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred_gb, label='Gradient Boosting - Predicted (Top Countries)')
# Set labels and title for top countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression vs. Random Forest vs. Gradient Boosting for Top Countries')
# Show the plot for top countries
plt.legend()
plt.show()    

# Initialize XGBoost model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Training and cross-validation for top countries using XGBoost
for train_index, val_index in tscv.split(df_top_over_time[' Year ']):
    X_train, X_val = df_top_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val = df_top_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train, y_val = df_top_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_top_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]    
    # Fit the XGBoost model
    model_xgb.fit(X_train, y_train) 
    # Predict on validation set
    y_pred_xgb = model_xgb.predict(X_val) 
    # Evaluate performance
    mse_xgb = mean_squared_error(y_val, y_pred_xgb)
    r2_xgb = r2_score(y_val, y_pred_xgb)   
    print(f'Mean Squared Error (XGBoost - Top Countries): {mse_xgb}')
    print(f'R Squared (XGBoost - Top Countries): {r2_xgb}')

# Visualization of the Models (XGBoost,Gradient Boosting, Random Forest, and Linear Regression) for top countries
plt.figure(figsize=(12, 8))
# Scatter plot of actual GDP values for top countries
sns.scatterplot(x=df_top_over_time[' Year '], y=df_top_over_time[' Gross Domestic Product (GDP) '], label='Actual (Top Countries)')
# Scatter plot of predicted values using Linear Regression for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred, label='Linear Regression - Predicted (Top Countries)')
# Scatter plot of predicted values using Random Forest for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred_rf, label='Random Forest - Predicted (Top Countries)')
# Scatter plot of predicted values using Gradient Boosting for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred_gb, label='Gradient Boosting - Predicted (Top Countries)')
# Scatter plot of predicted values using XGBoost for top countries
sns.scatterplot(x=df_top_over_time[' Year '].iloc[val_index], y=y_pred_xgb, label='XGBoost - Predicted (Top Countries)')
# Set labels and title for top countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression vs. Random Forest vs. Gradient Boosting vs. XGBoost for Top Countries')
# Show the plot for top countries
plt.legend()
plt.show()

    
#Implementing Cross Entropy Strategies For Bottom 10 Countries 
# Initialize Linear Regression model for bottom countries
model_bottom = LinearRegression()
# Training and cross-validation for bottom countries
for train_index, val_index in tscv.split(df_bottom_over_time[' Year ']):
    X_train_bottom, X_val_bottom = df_bottom_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val_bottom = df_bottom_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train_bottom, y_val_bottom = df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the model for bottom countries
    model_bottom.fit(X_train_bottom, y_train_bottom)
    # Predict on validation set for bottom countries
    y_pred_bottom = model_bottom.predict(X_val_bottom)
    # Evaluate performance for bottom countries (e.g., using mean squared error)
    mse_bottom = mean_squared_error(y_val_bottom, y_pred_bottom)
    r2_bottom = r2_score(y_val_bottom, y_pred_bottom)
    print(f'Mean Squared Error (Linear Regression - Bottom Countries): {mse_bottom}')
    print(f'R squared (Linear Regression - Bottom Countries): {r2_bottom}')
# Visualization of the Linear Regression Model for bottom countries
plt.figure(figsize=(10, 6))
# Scatter plot of actual GDP values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '], y=df_bottom_over_time[' Gross Domestic Product (GDP) '], label='Actual (Bottom Countries)')
# Scatter plot of predicted values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_bottom, label='Predicted (Bottom Countries)')
# Set labels and title for bottom countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression Model for Bottom Countries')
# Show the plot for bottom countries
plt.legend()
plt.show()


# Initialize Random Forest model for bottom countries
model_rf_bottom = RandomForestRegressor(n_estimators=100, random_state=42)
# Training and cross-validation for bottom countries using Random Forest
for train_index, val_index in tscv.split(df_bottom_over_time[' Year ']):
    X_train_bottom, X_val_bottom = df_bottom_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val_bottom = df_bottom_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train_bottom, y_val_bottom = df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the Random Forest model for bottom countries
    model_rf_bottom.fit(X_train_bottom, y_train_bottom)
    # Predict on validation set for bottom countries
    y_pred_rf_bottom = model_rf_bottom.predict(X_val_bottom)
    # Evaluate performance for bottom countries
    mse_rf_bottom = mean_squared_error(y_val_bottom, y_pred_rf_bottom)
    r2_rf_bottom = r2_score(y_val_bottom, y_pred_rf_bottom)
    print(f'Mean Squared Error (Random Forest - Bottom Countries): {mse_rf_bottom}')
    print(f'R Squared (Random Forest - Bottom Countries): {r2_rf_bottom}')
# Visualization of the Models (Random Forest, Gradient Boosting, and XGBoost) for bottom countries
plt.figure(figsize=(12, 8))
# Scatter plot of actual GDP values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '], y=df_bottom_over_time[' Gross Domestic Product (GDP) '], label='Actual (Bottom Countries)')
# Scatter plot of predicted values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_bottom, label='Linear Regression - Predicted (Bottom Countries)')
# Scatter plot of predicted values using Random Forest for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_rf_bottom, label='Random Forest - Predicted (Bottom Countries)')
# Set labels and title for bottom countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression vs. Random Forest for Bottom Countries')
# Show the plot for bottom countries
plt.legend()
plt.show()



# Initialize Gradient Boosting model for bottom countries
model_gb_bottom = GradientBoostingRegressor(n_estimators=100, random_state=42)
# Training and cross-validation for bottom countries using Gradient Boosting
for train_index, val_index in tscv.split(df_bottom_over_time[' Year ']):
    X_train_bottom, X_val_bottom = df_bottom_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val_bottom = df_bottom_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train_bottom, y_val_bottom = df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the Gradient Boosting model for bottom countries
    model_gb_bottom.fit(X_train_bottom, y_train_bottom)
    # Predict on validation set for bottom countries
    y_pred_gb_bottom = model_gb_bottom.predict(X_val_bottom)
    # Evaluate performance for bottom countries
    mse_gb_bottom = mean_squared_error(y_val_bottom, y_pred_gb_bottom)
    r2_gb_bottom = r2_score(y_val_bottom, y_pred_gb_bottom)
    print(f'Mean Squared Error (Gradient Boosting - Bottom Countries): {mse_gb_bottom}')
    print(f'R Squared (Gradient Boosting - Bottom Countries): {r2_gb_bottom}')
# Visualization of the Models (Random Forest, Gradient Boosting, and XGBoost) for bottom countries
plt.figure(figsize=(12, 8))
# Scatter plot of actual GDP values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '], y=df_bottom_over_time[' Gross Domestic Product (GDP) '], label='Actual (Bottom Countries)')
# Scatter plot of predicted values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_bottom, label='Linear Regression - Predicted (Bottom Countries)')
# Scatter plot of predicted values using Random Forest for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_rf_bottom, label='Random Forest - Predicted (Bottom Countries)')
# Scatter plot of predicted values using Gradient Boosting for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_gb_bottom, label='Gradient Boosting - Predicted (Bottom Countries)')
# Set labels and title for bottom countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression vs. Random Forest vs. Gradient Boosting for Bottom Countries')
# Show the plot for bottom countries
plt.legend()
plt.show()



# Initialize XGBoost model for bottom countries
model_xgb_bottom = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# Training and cross-validation for bottom countries using XGBoost
for train_index, val_index in tscv.split(df_bottom_over_time[' Year ']):
    X_train_bottom, X_val_bottom = df_bottom_over_time[' Year '].iloc[train_index].values.reshape(-1, 1), df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index]
    X_val_bottom = df_bottom_over_time[' Year '].iloc[val_index].values.reshape(-1, 1)
    y_train_bottom, y_val_bottom = df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[train_index], df_bottom_over_time[' Gross Domestic Product (GDP) '].iloc[val_index]
    # Fit the XGBoost model for bottom countries
    model_xgb_bottom.fit(X_train_bottom, y_train_bottom)
    # Predict on validation set for bottom countries
    y_pred_xgb_bottom = model_xgb_bottom.predict(X_val_bottom)
    # Evaluate performance for bottom countries
    mse_xgb_bottom = mean_squared_error(y_val_bottom, y_pred_xgb_bottom)
    r2_xgb_bottom = r2_score(y_val_bottom, y_pred_xgb_bottom)
    print(f'Mean Squared Error (XGBoost - Bottom Countries): {mse_xgb_bottom}')
    print(f'R Squared (XGBoost - Bottom Countries): {r2_xgb_bottom}')
# Visualization of the Models (Random Forest, Gradient Boosting, and XGBoost) for bottom countries
plt.figure(figsize=(12, 8))
# Scatter plot of actual GDP values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '], y=df_bottom_over_time[' Gross Domestic Product (GDP) '], label='Actual (Bottom Countries)')
# Scatter plot of predicted values for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_bottom, label='Linear Regression - Predicted (Bottom Countries)')
# Scatter plot of predicted values using Random Forest for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_rf_bottom, label='Random Forest - Predicted (Bottom Countries)')
# Scatter plot of predicted values using Gradient Boosting for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_gb_bottom, label='Gradient Boosting - Predicted (Bottom Countries)')
# Scatter plot of predicted values using XGBoost for bottom countries
sns.scatterplot(x=df_bottom_over_time[' Year '].iloc[val_index], y=y_pred_xgb_bottom, label='XGBoost - Predicted (Bottom Countries)')
# Set labels and title for bottom countries
plt.xlabel('Year')
plt.ylabel('Gross Domestic Product (GDP)')
plt.title('Linear Regression vs. Random Forest vs. Gradient Boosting vs. XGBoost for Bottom Countries')
# Show the plot for bottom countries
plt.legend()
plt.show()









