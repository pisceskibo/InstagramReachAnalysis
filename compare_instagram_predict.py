# Libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor, LinearRegression


# Import dataset
instagram_data = pd.read_csv("datasets/instagram_new_data.csv", encoding = "utf-8-sig")
x = np.array(instagram_data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(instagram_data["Impressions"])


def fit_manual_par(x, y, C=1.0, epsilon=0.1, max_iter=1000, random_state=42):
    """Huấn luyện PA-I bằng NumPy, không gọi thuật toán PAR của thư viện."""
    weights = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    rng = np.random.default_rng(random_state)

    for _ in range(max_iter):
        for index in rng.permutation(x.shape[0]):
            prediction = np.dot(weights, x[index]) + intercept
            error = y[index] - prediction
            loss = abs(error) - epsilon
            if loss <= 0.0:
                continue

            squared_norm = np.dot(x[index], x[index]) + 1.0
            tau = min(C, loss / squared_norm)
            direction = 1.0 if error >= 0.0 else -1.0
            weights += tau * direction * x[index]
            intercept += tau * direction

    return weights, intercept


def manual_par_predict(x, weights, intercept):
    return np.dot(x, weights) + intercept


def evaluate_predictions(y_true, y_pred):
    residuals = y_true - y_pred
    r2 = 1.0 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)
    return float(r2), float(mean_absolute_error(y_true, y_pred))


def compare_manual_and_library_par(x, y):
    print("==============================")
    print("PAR thủ công và PAR thư viện")
    print("==============================")
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain).astype(np.float64)
    xtest = scaler.transform(xtest).astype(np.float64)
    parameters = {"C": 1.0, "epsilon": 0.1, "max_iter": 1000}

    start = time.perf_counter()
    manual_weights, manual_intercept = fit_manual_par(
        xtrain, ytrain, random_state=42, **parameters
    )
    manual_time = time.perf_counter() - start
    manual_pred = manual_par_predict(xtest, manual_weights, manual_intercept)
    manual_r2, manual_mae = evaluate_predictions(ytest, manual_pred)

    start = time.perf_counter()
    library_model = PassiveAggressiveRegressor(
        random_state=42, shuffle=True, tol=None, **parameters
    )
    library_model.fit(xtrain, ytrain)
    library_time = time.perf_counter() - start
    library_pred = library_model.predict(xtest)
    library_r2, library_mae = evaluate_predictions(ytest, library_pred)

    print(f"{'Mô hình':<24} {'R²':>10} {'MAE':>12} {'Thời gian (s)':>16}")
    print(f"{'PAR thủ công':<24} {manual_r2:>10.4f} {manual_mae:>12.2f} {manual_time:>16.4f}")
    print(f"{'PAR scikit-learn':<24} {library_r2:>10.4f} {library_mae:>12.2f} {library_time:>16.4f}")

    return {
        "manual": (manual_weights, manual_intercept),
        "library": library_model,
        "scaler": scaler,
    }

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
    comparison = compare_manual_and_library_par(x, y)
    joblib.dump(
        comparison["library"],
        "datasets/instagram_passive_aggressive_regressor_model.pkl",
    )
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
