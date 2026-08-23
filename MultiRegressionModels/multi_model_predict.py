# Libraries
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    PassiveAggressiveRegressor,
    Ridge,
    Lasso,
    SGDRegressor,
    LogisticRegression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


# Danh sách các mô hình cần đánh giá
models = {
    "Passive Aggressive (PAR)": PassiveAggressiveRegressor(max_iter=1000, random_state=42),
    "Linear Regression (OLS)": LinearRegression(),
    "Logistic Regression": LogisticRegression(),
    "SGD Regressor": SGDRegressor(max_iter=1000, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
}

# Import dataset
instagram_data = pd.read_csv("datasets/instagram_new_data.csv", encoding = "utf-8-sig")
x = np.array(instagram_data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(instagram_data["Impressions"])


# Compare models
def compare_models(x, y):
    # Split dataset (80% train + 20% test)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Pipeline chuẩn hóa dữ liệu & chạy thử nghiệm
    results = []
    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Đo thời gian huấn luyện (Training Time)
        start_train = time.perf_counter()
        pipe.fit(xtrain, ytrain)
        end_train = time.perf_counter()
        train_time_ms = (end_train - start_train) * 1000    # Đổi sang ms

        # Đo thời gian dự đoán (Testing Time)
        start_pred = time.perf_counter()
        y_pred = pipe.predict(xtest)
        end_pred = time.perf_counter()
        pred_time_ms = (end_pred - start_pred) * 1000       # Đổi sang ms
        
        results.append({
            "Model": name,
            "R² Score": r2_score(ytest, y_pred),
            "MAE": mean_absolute_error(ytest, y_pred),
            "RMSE": root_mean_squared_error(ytest, y_pred),
            "Training Time (ms)": train_time_ms,
            "Predict Time (ms)": pred_time_ms
        })

    # Tổng hợp bảng kết quả
    df_results = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    compare_models(x, y)
    