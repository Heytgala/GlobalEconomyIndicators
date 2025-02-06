# Global Economy Indicators

This project explores the "Global Economy Indicators" dataset from Kaggle, which consists of economic indicators across multiple countries and years. The goal is to analyze and predict global economic trends, with a focus on GDP dynamics. Various machine learning models, including <b>Linear Regression, Random Forest, Gradient Boosting, and XGBoost,</b> are applied to predict GDP trends based on economic indicators.

<b>Key Features:</b>

✅ Data Loading & Preprocessing: Loaded the dataset into a Pandas DataFrame, handled missing values by imputing zeros, and applied label encoding to categorical variables. <br/>
✅ Exploratory Data Analysis (EDA): Analyzed global GDP trends over time using line plots; computed and visualized the correlation matrix using a heatmap. <br/>
✅ Machine Learning Models for GDP Prediction: Implemented Linear Regression, Random Forest, Gradient Boosting, and XGBoost to predict GDP; Evaluated models using Mean Squared Error (MSE) and R-squared (R²) scores; Visualized model predictions with scatter plots. <br/>
✅ Comparison of Cross-Entropy Strategies: Conducted a comparative study of regression models to determine the most effective approach for GDP forecasting.

<b>Steps:</b>
1) Make sure you install the zip properly

2) Make sure every file is in one main folder

3) Change filepath of Csv file in GlobalEconomyIndicators.py

4) Install XgBoost library in env folder since xgboost.dll is more than 100MB file and cant be pushed in git

5) Once everything is placed and installed properly code will be executed
