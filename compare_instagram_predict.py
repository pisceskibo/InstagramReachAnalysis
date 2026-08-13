# Libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor, LinearRegression


# Import dataset
instagram_data = pd.read_csv("datasets/instagram_new_data.csv", encoding = "utf-8-sig")
x = np.array(instagram_data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(instagram_data["Impressions"])

def predict_passive_aggressive_regressor_model(x, y):
    print("==============================")
    print("Passive Aggressive Regressor")
    print("==============================")
    # Split dataset (80% train + 20% test)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    model = PassiveAggressiveRegressor(C=1.0, epsilon=0.1)
    model.fit(xtrain, ytrain)

    # Predict trên tập test
    y_pred = model.predict(xtest)
    score = model.score(xtest, ytest)
    mae = mean_absolute_error(ytest, y_pred)

    print("R² Score PAR =", score)
    print("MAE PAR =", mae)

    joblib.dump(model, "datasets/instagram_passive_aggressive_regressor_model.pkl")

def predict_linear_regressor_model(x, y):
    print("==============================")
    print("Linear Regression")
    print("==============================")
    # Split dataset (80% train + 20% test)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    # Predict trên tập test
    y_pred = model.predict(xtest)
    score = model.score(xtest, ytest)
    mae = mean_absolute_error(ytest, y_pred)

    print("R² Score LR =", score)
    print("MAE LR =", mae)

    joblib.dump(model, "datasets/instagram_linear_regressor_model.pkl")

def creat_model_path_file(x, y):
    predict_passive_aggressive_regressor_model(x, y)
    print()
    predict_linear_regressor_model(x, y)


if __name__ == "__main__":
    creat_model_path_file(x, y)

    # passive_aggressive_model = joblib.load("datasets/instagram_passive_aggressive_regressor_model.pkl")
    # linear_model = joblib.load("datasets/instagram_linear_regressor_model.pkl")

    # # Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
    # test_features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])

    # passive_prediction = passive_aggressive_model.predict(test_features)
    # print(f"Passive Aggressive Predicted Impressions: {round(passive_prediction[0])}")

    # linear_prediction = linear_model.predict(test_features)
    # print(f"Linear Regression Predicted Impressions: {round(linear_prediction[0])}")
