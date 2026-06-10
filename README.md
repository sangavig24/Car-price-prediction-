📌 Project Overview
This project is a Machine Learning regression model that predicts the selling price of a car based on various features such as year, present price, kilometers driven, fuel type, seller type, transmission, and ownership history.
It helps understand how different factors affect car pricing in the real world.
🧠 Objective
To build a predictive model that can estimate the resale value of a car using supervised machine learning techniques.
📂 Dataset
The dataset includes car details such as:
Features:
•Car Name
•Year of Manufacture
•Present Price
•Kilometers Drive
Seller Type (Dealer/Individual)
Transmission (Manual/Automatic)
Owner Type
Target:
Selling Price
⚙️ Technologies Used
Python 🐍
Google Colab / Jupyter Notebook
Pandas
NumPy
Scikit-learn
Matplotlib / Seaborn (optional for visualization)
🚀 Steps Performed
Import required libraries
Load dataset (CSV file)
Data cleaning and preprocessing
Encode categorical variables
Split dataset into training and testing sets
Train model using Linear Regression / Random Forest Regressor
Predict car prices
Evaluate model performance
🧾 Sample Code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("car data.csv")

# Select features and target
X = df[['Year', 'Present_Price', 'Kms_Driven']]
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy (R2 Score)
print("R2 Score:", r2_score(y_test, y_pred))
Python
📊 Output
The model predicts car prices with a reasonable R² score (accuracy) depending on preprocessing and algorithm used.
