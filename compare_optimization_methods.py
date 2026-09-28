"""
So sanh PAR, GD va Nesterov theo epoch va thoi gian.

Chay:
    python compare_optimization_methods.py

Ket qua luu trong plots_optimizers/.

Luu y: PAR toi uu epsilon-insensitive loss dang goc. GD va Nesterov dung
cung loss nhung tinh dao ham tren mot phien ban lam tron.
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from compare_par_ridge import _par_epoch, fit_par


RANDOM_STATE = 42
MAX_ITER = 100
LONG_MAX_ITER = 1000
STREAM_CHUNK_SIZE = 800
STREAM_RETRAIN_EPOCHS = 3
EPSILON = 0.1
SMOOTHING = 1.0
RIDGE_ALPHA = 1e-4
PLOT_DIR = "plots_optimizers"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Objective, gradient and Hessian
# ---------------------------------------------------------------------------


def epsilon_loss(X, y, theta, epsilon=EPSILON):
    residual = y - X @ theta
    return float(np.mean(np.maximum(0.0, np.abs(residual) - epsilon)))


def smooth_epsilon_objective(X, y, theta, epsilon=EPSILON, smoothing=SMOOTHING):
    residual = y - X @ theta
    smooth_abs = np.sqrt(residual * residual + smoothing * smoothing)
    z = (smooth_abs - epsilon) / smoothing
    loss = smoothing * np.logaddexp(0.0, z)
    return float(np.mean(loss) + 0.5 * RIDGE_ALPHA * np.dot(theta[1:], theta[1:]))


def smooth_gradient_hessian(X, y, theta, epsilon=EPSILON, smoothing=SMOOTHING):
    residual = y - X @ theta
    smooth_abs = np.sqrt(residual * residual + smoothing * smoothing)
    z = (smooth_abs - epsilon) / smoothing
    probability = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    first_abs = residual / smooth_abs
    loss_gradient = probability * first_abs
    gradient = -(X.T @ loss_gradient) / X.shape[0]
    gradient[1:] += RIDGE_ALPHA * theta[1:]

    second_abs = smoothing * smoothing / (smooth_abs ** 3)
    probability_gradient = probability * (1.0 - probability) / smoothing
    loss_hessian = probability_gradient * first_abs ** 2 + probability * second_abs
    hessian = (X.T * loss_hessian) @ X / X.shape[0]
    hessian[1:, 1:] += RIDGE_ALPHA * np.eye(X.shape[1] - 1)
    return gradient, hessian


# ---------------------------------------------------------------------------
# Optimizer runners
# ---------------------------------------------------------------------------


def make_record(
    name,
    iteration,
    elapsed,
    X_train,
    y_train,
    X_test,
    y_test,
    theta,
    target_mean=0.0,
    target_scale=1.0,
    samples_processed=None,
):
    train_loss = epsilon_loss(X_train, y_train, theta)
    train_pred = X_train @ theta
    train_target_original = y_train * target_scale + target_mean
    train_pred_original = train_pred * target_scale + target_mean
    train_mse = float(mean_squared_error(train_target_original, train_pred_original))
    train_r2 = float(r2_score(train_target_original, train_pred_original))

    test_start = time.perf_counter()
    test_pred = X_test @ theta
    test_pred_original = test_pred * target_scale + target_mean
    y_test_original = y_test * target_scale + target_mean
    test_mse = float(mean_squared_error(y_test_original, test_pred_original))
    test_r2 = float(r2_score(y_test_original, test_pred_original))
    test_time_ms = (time.perf_counter() - test_start) * 1000.0
    return {
        "algorithm": name,
        "iteration": iteration,
        "samples_processed": (
            iteration if samples_processed is None else samples_processed
        ),
        "time_ms": elapsed * 1000.0,
        "time_per_sample_ms": elapsed * 1000.0 / (
            iteration if samples_processed is None else samples_processed
        ),
        "train_epsilon_loss": train_loss,
        "train_time_ms": elapsed * 1000.0,
        "train_mse": train_mse,
        "train_r2": train_r2,
        "test_time_ms": test_time_ms,
        "test_mse": test_mse,
        "test_r2": test_r2,
    }


def fit_batch_method(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    target_mean,
    target_scale,
    step_size=None,
    max_iter=MAX_ITER,
    algorithm_name=None,
):
    theta = np.zeros(X_train.shape[1], dtype=np.float64)
    learning_rate = 0.01 if step_size is None else step_size
    velocity = theta.copy()
    records = []
    start = time.perf_counter()

    algorithm_name = name if algorithm_name is None else algorithm_name
    for iteration in range(1, max_iter + 1):
        gradient, hessian = smooth_gradient_hessian(X_train, y_train, theta)
        if name == "GD":
            theta -= learning_rate * gradient
        elif name == "Nesterov":
            lookahead = theta - 0.9 * velocity
            gradient, _ = smooth_gradient_hessian(X_train, y_train, lookahead)
            velocity = 0.9 * velocity + learning_rate * gradient
            theta -= velocity
        elif name == "Proximal Gradient":
            theta -= learning_rate * gradient
            theta[1:] /= 1.0 + learning_rate * RIDGE_ALPHA
        elif name == "Newton":
            direction = np.linalg.solve(
                hessian + 1e-6 * np.eye(hessian.shape[0]), gradient
            )
            current_objective = smooth_epsilon_objective(X_train, y_train, theta)
            step = 1.0
            candidate = theta - step * direction
            while (
                smooth_epsilon_objective(X_train, y_train, candidate) > current_objective
                and step > 1e-4
            ):
                step *= 0.5
                candidate = theta - step * direction
            theta = candidate
        else:
            raise ValueError(f"Unknown optimizer: {name}")

        records.append(
            make_record(algorithm_name, iteration, time.perf_counter() - start,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        theta,
                        target_mean,
                        target_scale,
                        iteration * X_train.shape[0],
                    )
        )

    return pd.DataFrame(records), theta


def fit_par_history(
    X_train,
    y_train,
    X_test,
    y_test,
    target_mean,
    target_scale,
    max_iter=MAX_ITER,
):
    X_train_design = np.column_stack((np.ones(X_train.shape[0]), X_train))
    X_test_design = np.column_stack((np.ones(X_test.shape[0]), X_test))
    checkpoints = list(range(1, max_iter + 1))
    # Warm up Numba outside the measured training run.
    warmup_w = np.zeros(X_train.shape[1], dtype=np.float64)
    _par_epoch(
        X_train[:1],
        y_train[:1],
        warmup_w,
        0.0,
        1.0,
        EPSILON,
        np.array([0], dtype=np.int64),
    )
    result = fit_par(
        X_train,
        y_train,
        C=1.0,
        epsilon=EPSILON,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        track=True,
        checkpoint_iters=checkpoints,
        variant="pa2",
    )
    records = []
    for iteration in checkpoints:
        checkpoint = result["checkpoints"][iteration]
        theta = np.concatenate(([checkpoint["b"]], checkpoint["w"]))
        records.append(
            make_record(
                "PAR",
                iteration,
                checkpoint["elapsed"],
                X_train_design,
                y_train,
                X_test_design,
                y_test,
                theta,
                target_mean,
                target_scale,
                iteration * X_train.shape[0],
            )
        )
    return pd.DataFrame(records), np.concatenate(([result["b"]], result["w"]))


def fit_streaming_history(X_train, y_train, X_test, y_test, target_mean, target_scale):
    """So sánh PAR online với GD/Newton phải train lại khi có batch mới."""
    rng = np.random.default_rng(RANDOM_STATE)
    par_w = np.zeros(X_train.shape[1], dtype=np.float64)
    par_b = 0.0
    par_elapsed = 0.0
    batch_elapsed = {"GD": 0.0, "Nesterov": 0.0}
    batch_samples = {"GD": 0, "Nesterov": 0}
    records = []
    seen_features = []
    seen_targets = []
    stream_indices = np.array_split(
        rng.permutation(X_train.shape[0]),
        max(1, int(np.ceil(X_train.shape[0] / STREAM_CHUNK_SIZE))),
    )

    _par_epoch(
        X_train[:1],
        y_train[:1],
        par_w.copy(),
        par_b,
        1.0,
        EPSILON,
        np.array([0], dtype=np.int64),
    )

    for batch_number, indices in enumerate(stream_indices, start=1):
        batch_X = X_train[indices]
        batch_y = y_train[indices]
        seen_features.append(batch_X)
        seen_targets.append(batch_y)
        current_X = np.concatenate(seen_features)
        current_y = np.concatenate(seen_targets)

        start = time.perf_counter()
        par_w, par_b = _par_epoch(
            batch_X,
            batch_y,
            par_w,
            par_b,
            1.0,
            EPSILON,
            np.arange(batch_X.shape[0], dtype=np.int64),
        )
        par_elapsed += time.perf_counter() - start
        par_theta = np.concatenate(([par_b], par_w))
        records.append(
            make_record(
                "PAR streaming",
                batch_number,
                par_elapsed,
                np.column_stack((np.ones(current_X.shape[0]), current_X)),
                current_y,
                np.column_stack((np.ones(X_test.shape[0]), X_test)),
                y_test,
                par_theta,
                target_mean,
                target_scale,
                current_X.shape[0],
            )
        )

        current_design = np.column_stack((np.ones(current_X.shape[0]), current_X))
        test_design = np.column_stack((np.ones(X_test.shape[0]), X_test))
        for name, step_size in [("GD", 0.01), ("Nesterov", 0.01)]:
            local_history, theta = fit_batch_method(
                name,
                current_design,
                current_y,
                test_design,
                y_test,
                target_mean,
                target_scale,
                step_size=step_size,
                max_iter=STREAM_RETRAIN_EPOCHS,
                algorithm_name=f"{name} retrain",
            )
            local_final = local_history.iloc[-1]
            batch_elapsed[name] += float(local_final["time_ms"]) / 1000.0
            batch_samples[name] += current_X.shape[0] * STREAM_RETRAIN_EPOCHS
            records.append(
                make_record(
                    f"{name} retrain",
                    batch_number,
                    batch_elapsed[name],
                    current_design,
                    current_y,
                    test_design,
                    y_test,
                    theta,
                    target_mean,
                    target_scale,
                    batch_samples[name],
                )
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots and experiment
# ---------------------------------------------------------------------------


def save_plot(name):
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def save_comparison_table(summary, suffix=""):
    columns = [
        "algorithm",
        "train_time_ms",
        "train_mse",
        "train_r2",
        "test_time_ms",
        "test_mse",
        "test_r2",
    ]
    table = summary[columns].copy()
    table_path = os.path.join(PLOT_DIR, f"comparison_table{suffix}.csv")
    table.to_csv(table_path, index=False)

    display_table = table.copy()
    display_table.columns = [
        "Algorithm",
        "Train time (ms)",
        "Train MSE",
        "Train R2",
        "Test time (ms)",
        "Test MSE",
        "Test R2",
    ]
    display_table["Train time (ms)"] = display_table["Train time (ms)"].map(lambda value: f"{value:.3f}")
    display_table["Test time (ms)"] = display_table["Test time (ms)"].map(lambda value: f"{value:.3f}")
    for column in ["Train MSE", "Test MSE"]:
        display_table[column] = display_table[column].map(lambda value: f"{value:,.0f}")
    for column in ["Train R2", "Test R2"]:
        display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")

    figure, axis = plt.subplots(figsize=(14, 2.6))
    axis.axis("off")
    axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="center",
        loc="center",
    )
    axis.set_title("So sanh PAR, GD va Nesterov: train va test")
    table_image_path = os.path.join(PLOT_DIR, f"comparison_table{suffix}.png")
    figure.savefig(table_image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {table_path}")
    print(f"saved {table_image_path}")


def plot_history(history, suffix=""):
    for y_column, ylabel, filename, title in [
        ("train_epsilon_loss", "Epsilon-insensitive loss", f"objective_vs_iteration{suffix}.png", "Objective theo buoc lap"),
        ("test_mse", "MSE", f"mse_vs_iteration{suffix}.png", "MSE test theo buoc lap"),
        ("test_r2", "R2", f"r2_vs_iteration{suffix}.png", "R2 test theo buoc lap"),
    ]:
        plt.figure(figsize=(10, 6))
        for algorithm, frame in history.groupby("algorithm"):
            plt.plot(frame["iteration"], frame[y_column], label=algorithm)
        plt.xlabel("Buoc lap / epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        save_plot(filename)

    plt.figure(figsize=(10, 6))
    for algorithm, frame in history.groupby("algorithm"):
        time_values = frame["time_ms"].to_numpy()
        time_values = time_values - time_values[0]
        plt.plot(time_values, frame["test_mse"], label=algorithm)
    plt.xlabel("Thoi gian (ms)")
    plt.ylabel("MSE")
    plt.title("MSE test theo thoi gian")
    plt.legend()
    save_plot(f"objective_vs_time{suffix}.png")

    final_rows = history.sort_values("iteration").groupby("algorithm").tail(1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(final_rows["algorithm"], final_rows["test_mse"])
    axes[0].set_title("MSE cuoi")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(final_rows["algorithm"], final_rows["test_r2"])
    axes[1].set_title("R2 cuoi")
    axes[1].tick_params(axis="x", rotation=35)
    axes[2].bar(final_rows["algorithm"], final_rows["time_ms"])
    axes[2].set_title("Thoi gian huan luyen")
    axes[2].set_ylabel("Mili giay")
    axes[2].tick_params(axis="x", rotation=35)
    save_plot(f"final_metrics{suffix}.png")


def plot_streaming_history(history):
    for x_column, xlabel, filename in [
        ("samples_processed", "So mau da xu ly", "streaming_objective_vs_samples.png"),
        ("time_ms", "Thoi gian (ms)", "streaming_objective_vs_time.png"),
    ]:
        plt.figure(figsize=(10, 6))
        for algorithm, frame in history.groupby("algorithm"):
            x_values = frame[x_column].to_numpy()
            objective_values = frame["test_mse"].to_numpy()
            if x_column == "time_ms":
                x_values = x_values - x_values[0]
            plt.plot(x_values, objective_values, marker="o", label=algorithm)
        plt.xlabel(xlabel)
        plt.ylabel("MSE")
        plt.title("Streaming: MSE test theo " + xlabel.lower())
        plt.legend()
        save_plot(filename)

    final_rows = history.groupby("algorithm").tail(1)
    plt.figure(figsize=(10, 6))
    plt.bar(final_rows["algorithm"], final_rows["time_per_sample_ms"])
    plt.ylabel("Mili giay / mau")
    plt.title("Streaming: thoi gian xu ly trung binh moi mau")
    plt.xticks(rotation=25)
    save_plot("streaming_time_per_sample.png")


def main():
    data = pd.read_csv("datasets/instagram_new_data.csv", encoding="utf-8-sig")
    feature_cols = ["Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows"]
    X = data[feature_cols].to_numpy(dtype=np.float64)
    y = data["Impressions"].to_numpy(dtype=np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float64)
    X_test = scaler.transform(X_test).astype(np.float64)
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    y_train = (y_train - target_mean) / target_scale
    y_test = (y_test - target_mean) / target_scale
    X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
    X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

    history_frames = []
    par_history, _ = fit_par_history(
        X_train[:, 1:],
        y_train,
        X_test[:, 1:],
        y_test,
        target_mean,
        target_scale,
    )
    history_frames.append(par_history)
    for name, step_size in [("GD", 0.01), ("Nesterov", 0.01)]:
        frame, _ = fit_batch_method(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            target_mean,
            target_scale,
            step_size=step_size,
        )
        history_frames.append(frame)
    history = pd.concat(history_frames, ignore_index=True)
    history_path = os.path.join(PLOT_DIR, "optimizer_history.csv")
    summary_path = os.path.join(PLOT_DIR, "optimizer_summary.csv")
    history.to_csv(history_path, index=False)
    summary = history.sort_values("iteration").groupby("algorithm").tail(1)
    summary.to_csv(summary_path, index=False)
    plot_history(history)
    save_comparison_table(summary)
    print(summary[["algorithm", "iteration", "time_ms", "train_epsilon_loss", "test_mse", "test_r2"]].to_string(index=False))
    print(f"saved {history_path}")
    print(f"saved {summary_path}")

    long_history_frames = []
    long_par_history, _ = fit_par_history(
        X_train[:, 1:],
        y_train,
        X_test[:, 1:],
        y_test,
        target_mean,
        target_scale,
        max_iter=LONG_MAX_ITER,
    )
    long_history_frames.append(long_par_history)
    for name, step_size in [("GD", 0.01), ("Nesterov", 0.01)]:
        frame, _ = fit_batch_method(
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            target_mean,
            target_scale,
            step_size=step_size,
            max_iter=LONG_MAX_ITER,
        )
        long_history_frames.append(frame)

    long_history = pd.concat(long_history_frames, ignore_index=True)
    long_summary = long_history.sort_values("iteration").groupby("algorithm").tail(1)
    long_history_path = os.path.join(PLOT_DIR, "optimizer_history_1000.csv")
    long_summary_path = os.path.join(PLOT_DIR, "optimizer_summary_1000.csv")
    long_history.to_csv(long_history_path, index=False)
    long_summary.to_csv(long_summary_path, index=False)
    plot_history(long_history, suffix="_1000")
    save_comparison_table(long_summary, suffix="_1000")
    print("\n1000-iteration summary:")
    print(
        long_summary[
            ["algorithm", "iteration", "time_ms", "train_epsilon_loss", "test_mse", "test_r2"]
        ].to_string(index=False)
    )
    print(f"saved {long_history_path}")
    print(f"saved {long_summary_path}")

    streaming_history = fit_streaming_history(
        X_train[:, 1:],
        y_train,
        X_test[:, 1:],
        y_test,
        target_mean,
        target_scale,
    )
    streaming_summary = streaming_history.groupby("algorithm").tail(1)
    streaming_history_path = os.path.join(PLOT_DIR, "streaming_history.csv")
    streaming_summary_path = os.path.join(PLOT_DIR, "streaming_summary.csv")
    streaming_history.to_csv(streaming_history_path, index=False)
    streaming_summary.to_csv(streaming_summary_path, index=False)
    plot_streaming_history(streaming_history)
    print("\nStreaming summary:")
    print(
        streaming_summary[
            [
                "algorithm",
                "iteration",
                "samples_processed",
                "time_ms",
                "time_per_sample_ms",
                "test_mse",
                "test_r2",
            ]
        ].to_string(index=False)
    )
    print(f"saved {streaming_history_path}")
    print(f"saved {streaming_summary_path}")


if __name__ == "__main__":
    main()
