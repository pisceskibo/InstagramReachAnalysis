"""
So sánh Passive Aggressive Regressor (PAR) và PAR + Ridge.

Chạy: python compare_par_ridge.py
Kết quả/biểu đồ lưu vào thư mục plots_par_ridge/
Nội dung tương ứng notebook compare_par_ridge.ipynb
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PLOT_DIR = "plots_par_ridge"
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "font.size": 12,
    }
)


# ---------------------------------------------------------------------------
# Objective / metrics
# ---------------------------------------------------------------------------

def epsilon_insensitive_loss(X, y, w, b, epsilon):
    residual = np.abs(y - (X @ w + b))
    return float(np.mean(np.maximum(0.0, residual - epsilon)))


def ridge_objective(X, y, w, b, epsilon, alpha):
    return epsilon_insensitive_loss(X, y, w, b, epsilon) + 0.5 * alpha * float(np.dot(w, w))


def evaluate_model(X, y, w, b):
    y_pred = X @ w + b
    return {
        "r2": float(r2_score(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
    }


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _par_epoch(X, y, w, b, C, epsilon, indices):
    """PA-II: τ = L / (||x||² + 1/(2C)) — khớp công thức README."""
    for k in range(indices.shape[0]):
        i = indices[k]
        pred = b
        for j in range(w.shape[0]):
            pred += X[i, j] * w[j]
        err = y[i] - pred
        loss = abs(err) - epsilon
        if loss > 0.0:
            xnorm = 0.0
            for j in range(w.shape[0]):
                xnorm += X[i, j] * X[i, j]
            tau = loss / (xnorm + 1.0 / (2.0 * C))
            sign = 1.0 if err >= 0.0 else -1.0
            for j in range(w.shape[0]):
                w[j] = w[j] + tau * sign * X[i, j]
            b = b + tau * sign
    return w, b


@njit(cache=True)
def _par_epoch_pa1(X, y, w, b, C, epsilon, indices):
    """PA-I: τ = min(C, L / ||x||²) — khớp sklearn PassiveAggressiveRegressor default."""
    for k in range(indices.shape[0]):
        i = indices[k]
        pred = b
        for j in range(w.shape[0]):
            pred += X[i, j] * w[j]
        err = y[i] - pred
        loss = abs(err) - epsilon
        if loss > 0.0:
            xnorm = 0.0
            for j in range(w.shape[0]):
                xnorm += X[i, j] * X[i, j]
            if xnorm < 1e-12:
                continue
            tau = loss / xnorm
            if tau > C:
                tau = C
            sign = 1.0 if err >= 0.0 else -1.0
            for j in range(w.shape[0]):
                w[j] = w[j] + tau * sign * X[i, j]
            b = b + tau * sign
    return w, b


@njit(cache=True)
def _par_ridge_epoch(X, y, w, b, C, epsilon, alpha, indices):
    """PAR + Ridge với cập nhật ổn định: w <- (w + τ sgn x) / (1+α)."""
    for k in range(indices.shape[0]):
        i = indices[k]
        pred = b
        for j in range(w.shape[0]):
            pred += X[i, j] * w[j]
        err = y[i] - pred
        loss = abs(err) - epsilon
        if loss > 0.0:
            xnorm = 0.0
            for j in range(w.shape[0]):
                xnorm += X[i, j] * X[i, j]
            # Nghiệm xấp xỉ của: 1/2||w-w_t||^2 + C L + (α/2)||w||^2
            tau = loss / (xnorm / (1.0 + alpha) + 1.0 / (2.0 * C))
            sign = 1.0 if err >= 0.0 else -1.0
            inv = 1.0 / (1.0 + alpha)
            for j in range(w.shape[0]):
                w[j] = (w[j] + tau * sign * X[i, j]) * inv
            b = b + tau * sign
    return w, b


def fit_par(
    X,
    y,
    C=1.0,
    epsilon=0.1,
    max_iter=200,
    shuffle=True,
    random_state=RANDOM_STATE,
    track=True,
    checkpoint_iters=None,
    variant="pa2",
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    rng = np.random.default_rng(random_state)
    objectives, times = [], []
    checkpoints = {}
    checkpoint_set = set(checkpoint_iters or [])
    t0 = time.perf_counter()
    epoch_fn = _par_epoch_pa1 if variant == "pa1" else _par_epoch

    _ = epoch_fn(X[:1], y[:1], w.copy(), b, C, epsilon, np.array([0], dtype=np.int64))

    for epoch in range(1, max_iter + 1):
        indices = (
            rng.permutation(n_samples).astype(np.int64)
            if shuffle
            else np.arange(n_samples, dtype=np.int64)
        )
        w, b = epoch_fn(X, y, w, b, C, epsilon, indices)

        if track:
            objectives.append(epsilon_insensitive_loss(X, y, w, b, epsilon))
            times.append(time.perf_counter() - t0)

        if epoch in checkpoint_set:
            checkpoints[epoch] = {
                "w": w.copy(),
                "b": float(b),
                "objective": epsilon_insensitive_loss(X, y, w, b, epsilon),
                "elapsed": time.perf_counter() - t0,
            }

    return {
        "w": w,
        "b": float(b),
        "objectives": np.array(objectives, dtype=np.float64),
        "times": np.array(times, dtype=np.float64),
        "elapsed": time.perf_counter() - t0,
        "final_objective": epsilon_insensitive_loss(X, y, w, b, epsilon),
        "checkpoints": checkpoints,
    }


def fit_par_ridge(
    X,
    y,
    C=1.0,
    epsilon=0.1,
    alpha=0.01,
    max_iter=200,
    shuffle=True,
    random_state=RANDOM_STATE,
    track=True,
    checkpoint_iters=None,
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    rng = np.random.default_rng(random_state)
    objectives, times = [], []
    checkpoints = {}
    checkpoint_set = set(checkpoint_iters or [])
    t0 = time.perf_counter()

    _ = _par_ridge_epoch(
        X[:1], y[:1], w.copy(), b, C, epsilon, alpha, np.array([0], dtype=np.int64)
    )

    for epoch in range(1, max_iter + 1):
        indices = (
            rng.permutation(n_samples).astype(np.int64)
            if shuffle
            else np.arange(n_samples, dtype=np.int64)
        )
        w, b = _par_ridge_epoch(X, y, w, b, C, epsilon, alpha, indices)

        if track:
            objectives.append(ridge_objective(X, y, w, b, epsilon, alpha))
            times.append(time.perf_counter() - t0)

        if epoch in checkpoint_set:
            checkpoints[epoch] = {
                "w": w.copy(),
                "b": float(b),
                "objective": ridge_objective(X, y, w, b, epsilon, alpha),
                "elapsed": time.perf_counter() - t0,
            }

    return {
        "w": w,
        "b": float(b),
        "objectives": np.array(objectives, dtype=np.float64),
        "times": np.array(times, dtype=np.float64),
        "elapsed": time.perf_counter() - t0,
        "final_objective": ridge_objective(X, y, w, b, epsilon, alpha),
        "final_loss_only": epsilon_insensitive_loss(X, y, w, b, epsilon),
        "checkpoints": checkpoints,
    }


def sklearn_par_objective(model, X, y, epsilon):
    y_pred = model.predict(X)
    return float(np.mean(np.maximum(0.0, np.abs(y - y_pred) - epsilon)))


def sklearn_ridge_objective(model, X, y, epsilon, alpha):
    y_pred = model.predict(X)
    loss = float(np.mean(np.maximum(0.0, np.abs(y - y_pred) - epsilon)))
    w = model.coef_.ravel()
    return loss + 0.5 * alpha * float(np.dot(w, w))


def savefig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def main():
    print("=" * 60)
    print("A. Load data")
    print("=" * 60)
    data = pd.read_csv("datasets/instagram_new_data.csv", encoding="utf-8-sig")
    feature_cols = ["Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows"]
    X = np.array(data[feature_cols], dtype=np.float64)
    y = np.array(data["Impressions"], dtype=np.float64)

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain).astype(np.float64)
    xtest = scaler.transform(xtest).astype(np.float64)
    ytrain = ytrain.astype(np.float64)
    ytest = ytest.astype(np.float64)
    print(f"Train={xtrain.shape[0]} | Test={xtest.shape[0]} | d={xtrain.shape[1]}")

    C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    EPS_VALUES = [0.01, 0.1, 1.0]
    ALPHA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0]
    MAX_ITER_CHECKPOINTS = [50, 100, 200, 500, 1000]
    TUNE_EPOCHS = 200

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("D. Tune PAR")
    print("=" * 60)
    par_C_rows = []
    for C in C_VALUES:
        res = fit_par(xtrain, ytrain, C=C, epsilon=0.1, max_iter=TUNE_EPOCHS, track=True)
        metrics = evaluate_model(xtest, ytest, res["w"], res["b"])
        par_C_rows.append(
            {
                "C": C,
                "objective": res["final_objective"],
                "time_s": res["elapsed"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
            }
        )
        print(
            f"PAR | C={C:<7} | J={res['final_objective']:.4f} | "
            f"time={res['elapsed']:.2f}s | R²={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}"
        )
    df_par_C = pd.DataFrame(par_C_rows)
    best_par_C = float(df_par_C.loc[df_par_C["objective"].idxmin(), "C"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_par_C["C"], df_par_C["objective"], marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("Objective J(w)")
    axes[0].set_title("PAR: Objective vs C")
    axes[1].plot(df_par_C["C"], df_par_C["mae"], marker="o", color="tab:orange")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("MAE (test)")
    axes[1].set_title("PAR: MAE vs C")
    savefig("par_tune_C.png")

    par_eps_rows = []
    for eps in EPS_VALUES:
        res = fit_par(
            xtrain, ytrain, C=best_par_C, epsilon=eps, max_iter=TUNE_EPOCHS, track=True
        )
        metrics = evaluate_model(xtest, ytest, res["w"], res["b"])
        par_eps_rows.append(
            {
                "epsilon": eps,
                "objective": res["final_objective"],
                "time_s": res["elapsed"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
            }
        )
        print(
            f"PAR | ε={eps:<5} | J={res['final_objective']:.4f} | "
            f"time={res['elapsed']:.2f}s | R²={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}"
        )
    df_par_eps = pd.DataFrame(par_eps_rows)
    best_par_eps = float(df_par_eps.loc[df_par_eps["objective"].idxmin(), "epsilon"])

    par_long = fit_par(
        xtrain,
        ytrain,
        C=best_par_C,
        epsilon=best_par_eps,
        max_iter=max(MAX_ITER_CHECKPOINTS),
        track=True,
        checkpoint_iters=MAX_ITER_CHECKPOINTS,
    )
    par_iter_rows = []
    for k in MAX_ITER_CHECKPOINTS:
        ck = par_long["checkpoints"][k]
        metrics = evaluate_model(xtest, ytest, ck["w"], ck["b"])
        par_iter_rows.append(
            {
                "max_iter": k,
                "objective": ck["objective"],
                "time_s": ck["elapsed"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
            }
        )
        print(
            f"PAR | epochs={k:<4} | J={ck['objective']:.4f} | "
            f"time={ck['elapsed']:.2f}s | R²={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}"
        )
    df_par_iter = pd.DataFrame(par_iter_rows)
    best_par_iter = int(df_par_iter.loc[df_par_iter["objective"].idxmin(), "max_iter"])
    BEST_PAR = {"C": best_par_C, "epsilon": best_par_eps, "max_iter": best_par_iter}
    print("Best PAR:", BEST_PAR)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("E. Tune PAR+Ridge")
    print("=" * 60)
    ridge_alpha_rows = []
    for alpha in ALPHA_VALUES:
        res = fit_par_ridge(
            xtrain,
            ytrain,
            C=BEST_PAR["C"],
            epsilon=BEST_PAR["epsilon"],
            alpha=alpha,
            max_iter=TUNE_EPOCHS,
            track=True,
        )
        metrics = evaluate_model(xtest, ytest, res["w"], res["b"])
        ridge_alpha_rows.append(
            {
                "alpha": alpha,
                "objective": res["final_objective"],
                "loss_only": res["final_loss_only"],
                "time_s": res["elapsed"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
            }
        )
        print(
            f"PAR+Ridge | α={alpha:<7} | J={res['final_objective']:.4f} | "
            f"loss={res['final_loss_only']:.4f} | time={res['elapsed']:.2f}s | "
            f"R²={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}"
        )
    df_ridge_alpha = pd.DataFrame(ridge_alpha_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_ridge_alpha["alpha"], df_ridge_alpha["objective"], marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("PAR+Ridge: Objective vs alpha")
    axes[1].plot(df_ridge_alpha["alpha"], df_ridge_alpha["mae"], marker="o", color="tab:orange")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("PAR+Ridge: MAE vs alpha")
    savefig("par_ridge_tune_alpha.png")

    ridge_grid_rows = []
    for C in C_VALUES:
        for alpha in ALPHA_VALUES:
            res = fit_par_ridge(
                xtrain,
                ytrain,
                C=C,
                epsilon=BEST_PAR["epsilon"],
                alpha=alpha,
                max_iter=TUNE_EPOCHS,
                track=False,
            )
            obj = ridge_objective(
                xtrain, ytrain, res["w"], res["b"], BEST_PAR["epsilon"], alpha
            )
            metrics = evaluate_model(xtest, ytest, res["w"], res["b"])
            ridge_grid_rows.append(
                {
                    "C": C,
                    "alpha": alpha,
                    "objective": obj,
                    "time_s": res["elapsed"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                }
            )
    df_ridge_grid = pd.DataFrame(ridge_grid_rows)
    print("Top 5 PAR+Ridge by objective:")
    print(df_ridge_grid.sort_values("objective").head(5).to_string(index=False))
    best_ridge_row = df_ridge_grid.loc[df_ridge_grid["objective"].idxmin()]
    best_ridge_C = float(best_ridge_row["C"])
    best_ridge_alpha = float(best_ridge_row["alpha"])

    pivot = df_ridge_grid.pivot(index="alpha", columns="C", values="objective")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("C")
    ax.set_ylabel("alpha")
    ax.set_title("PAR+Ridge: Objective heatmap (C x alpha)")
    fig.colorbar(im, ax=ax, label="Objective")
    savefig("par_ridge_heatmap.png")

    ridge_long = fit_par_ridge(
        xtrain,
        ytrain,
        C=best_ridge_C,
        epsilon=BEST_PAR["epsilon"],
        alpha=best_ridge_alpha,
        max_iter=max(MAX_ITER_CHECKPOINTS),
        track=True,
        checkpoint_iters=MAX_ITER_CHECKPOINTS,
    )
    ridge_iter_rows = []
    for k in MAX_ITER_CHECKPOINTS:
        ck = ridge_long["checkpoints"][k]
        metrics = evaluate_model(xtest, ytest, ck["w"], ck["b"])
        ridge_iter_rows.append(
            {
                "max_iter": k,
                "objective": ck["objective"],
                "time_s": ck["elapsed"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
            }
        )
        print(
            f"PAR+Ridge | epochs={k:<4} | J={ck['objective']:.4f} | "
            f"time={ck['elapsed']:.2f}s | R²={metrics['r2']:.4f} | MAE={metrics['mae']:.2f}"
        )
    df_ridge_iter = pd.DataFrame(ridge_iter_rows)
    best_ridge_iter = int(df_ridge_iter.loc[df_ridge_iter["objective"].idxmin(), "max_iter"])
    BEST_PAR_RIDGE = {
        "C": best_ridge_C,
        "epsilon": BEST_PAR["epsilon"],
        "alpha": best_ridge_alpha,
        "max_iter": best_ridge_iter,
    }
    print("Best PAR+Ridge:", BEST_PAR_RIDGE)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("F. Compare best setups")
    print("=" * 60)
    # Đường cong hội tụ: cùng số epoch để so sánh trực quan
    COMPARE_EPOCHS = max(BEST_PAR["max_iter"], BEST_PAR_RIDGE["max_iter"], 500)

    par_best_hist = fit_par(
        xtrain,
        ytrain,
        C=BEST_PAR["C"],
        epsilon=BEST_PAR["epsilon"],
        max_iter=COMPARE_EPOCHS,
        track=True,
        checkpoint_iters=[BEST_PAR["max_iter"]],
    )
    ridge_best_hist = fit_par_ridge(
        xtrain,
        ytrain,
        C=BEST_PAR_RIDGE["C"],
        epsilon=BEST_PAR_RIDGE["epsilon"],
        alpha=BEST_PAR_RIDGE["alpha"],
        max_iter=COMPARE_EPOCHS,
        track=True,
        checkpoint_iters=[BEST_PAR_RIDGE["max_iter"]],
    )

    # Metric tại đúng best max_iter (không lấy epoch cuối nếu đã overshoot)
    par_ck = par_best_hist["checkpoints"][BEST_PAR["max_iter"]]
    ridge_ck = ridge_best_hist["checkpoints"][BEST_PAR_RIDGE["max_iter"]]
    par_test = evaluate_model(xtest, ytest, par_ck["w"], par_ck["b"])
    ridge_test = evaluate_model(xtest, ytest, ridge_ck["w"], ridge_ck["b"])
    par_obj_best = par_ck["objective"]
    ridge_obj_best = ridge_ck["objective"]
    par_time_best = par_ck["elapsed"]
    ridge_time_best = ridge_ck["elapsed"]

    print(
        f"PAR best @ epoch {BEST_PAR['max_iter']} | J={par_obj_best:.6f} | "
        f"time={par_time_best:.3f}s | R²={par_test['r2']:.4f} | MAE={par_test['mae']:.2f}"
    )
    print(
        f"PAR+Ridge best @ epoch {BEST_PAR_RIDGE['max_iter']} | J={ridge_obj_best:.6f} | "
        f"time={ridge_time_best:.3f}s | R²={ridge_test['r2']:.4f} | MAE={ridge_test['mae']:.2f}"
    )

    epochs = np.arange(1, COMPARE_EPOCHS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        par_best_hist["objectives"],
        label=f"PAR (C={BEST_PAR['C']}, ε={BEST_PAR['epsilon']})",
        linewidth=2,
    )
    plt.plot(
        epochs,
        ridge_best_hist["objectives"],
        label=(
            f"PAR+Ridge (C={BEST_PAR_RIDGE['C']}, ε={BEST_PAR_RIDGE['epsilon']}, "
            f"α={BEST_PAR_RIDGE['alpha']})"
        ),
        linewidth=2,
        linestyle="--",
    )
    plt.axvline(BEST_PAR["max_iter"], color="C0", linestyle=":", alpha=0.6)
    plt.axvline(BEST_PAR_RIDGE["max_iter"], color="C1", linestyle=":", alpha=0.6)
    plt.xlabel("Iterations (Epochs)")
    plt.ylabel("Objective Function Value")
    plt.title("Comparison of PAR vs PAR+Ridge\nTraining Objective vs Iterations")
    plt.legend()
    savefig("compare_objective_vs_iterations.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        par_best_hist["times"],
        par_best_hist["objectives"],
        label=f"PAR (C={BEST_PAR['C']}, ε={BEST_PAR['epsilon']})",
        linewidth=2,
    )
    plt.plot(
        ridge_best_hist["times"],
        ridge_best_hist["objectives"],
        label=(
            f"PAR+Ridge (C={BEST_PAR_RIDGE['C']}, ε={BEST_PAR_RIDGE['epsilon']}, "
            f"α={BEST_PAR_RIDGE['alpha']})"
        ),
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Objective Function Value")
    plt.title("Comparison of PAR vs PAR+Ridge\nTraining Objective vs Time")
    plt.legend()
    savefig("compare_objective_vs_time.png")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("G. Custom vs sklearn")
    print("=" * 60)
    t0 = time.perf_counter()
    sk_par = PassiveAggressiveRegressor(random_state=RANDOM_STATE)
    sk_par.fit(xtrain, ytrain)
    sk_par_time = time.perf_counter() - t0
    sk_par_eps = float(getattr(sk_par, "epsilon", 0.1))
    sk_par_obj = sklearn_par_objective(sk_par, xtrain, ytrain, sk_par_eps)
    sk_par_test = {
        "r2": float(r2_score(ytest, sk_par.predict(xtest))),
        "mae": float(mean_absolute_error(ytest, sk_par.predict(xtest))),
    }

    # Custom PAR với PA-I (khớp sklearn default loss='epsilon_insensitive')
    # max_iter=1000 nhưng sklearn thường early-stop sớm nhờ tol
    custom_par_default = fit_par(
        xtrain, ytrain, C=1.0, epsilon=0.1, max_iter=1000, track=True, variant="pa1"
    )
    custom_par_default_test = evaluate_model(
        xtest, ytest, custom_par_default["w"], custom_par_default["b"]
    )

    sk_alpha = 0.0001
    t0 = time.perf_counter()
    sk_ridge = SGDRegressor(
        loss="epsilon_insensitive",
        penalty="l2",
        alpha=sk_alpha,
        epsilon=0.1,
        learning_rate="pa1",
        eta0=1.0,
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )
    sk_ridge.fit(xtrain, ytrain)
    sk_ridge_time = time.perf_counter() - t0
    sk_ridge_obj = sklearn_ridge_objective(sk_ridge, xtrain, ytrain, 0.1, sk_alpha)
    sk_ridge_test = {
        "r2": float(r2_score(ytest, sk_ridge.predict(xtest))),
        "mae": float(mean_absolute_error(ytest, sk_ridge.predict(xtest))),
    }

    custom_ridge_default = fit_par_ridge(
        xtrain, ytrain, C=1.0, epsilon=0.1, alpha=sk_alpha, max_iter=1000, track=True
    )
    custom_ridge_default_test = evaluate_model(
        xtest, ytest, custom_ridge_default["w"], custom_ridge_default["b"]
    )

    df_sklearn_compare = pd.DataFrame(
        [
            {
                "Model": "Custom PAR",
                "Objective": custom_par_default["final_objective"],
                "Time (s)": custom_par_default["elapsed"],
                "R2": custom_par_default_test["r2"],
                "MAE": custom_par_default_test["mae"],
            },
            {
                "Model": "sklearn PAR",
                "Objective": sk_par_obj,
                "Time (s)": sk_par_time,
                "R2": sk_par_test["r2"],
                "MAE": sk_par_test["mae"],
            },
            {
                "Model": "Custom PAR+Ridge",
                "Objective": custom_ridge_default["final_objective"],
                "Time (s)": custom_ridge_default["elapsed"],
                "R2": custom_ridge_default_test["r2"],
                "MAE": custom_ridge_default_test["mae"],
            },
            {
                "Model": "sklearn PA+L2",
                "Objective": sk_ridge_obj,
                "Time (s)": sk_ridge_time,
                "R2": sk_ridge_test["r2"],
                "MAE": sk_ridge_test["mae"],
            },
        ]
    )
    print(df_sklearn_compare.to_string(index=False))

    labels = df_sklearn_compare["Model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        labels,
        df_sklearn_compare["Objective"],
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
    )
    axes[0].set_ylabel("Objective Function Value")
    axes[0].set_title("Custom vs sklearn: Objective")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(
        labels,
        df_sklearn_compare["Time (s)"],
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
    )
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title("Custom vs sklearn: Runtime")
    axes[1].tick_params(axis="x", rotation=20)
    savefig("custom_vs_sklearn.png")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("H. Summary")
    print("=" * 60)
    df_summary = pd.DataFrame(
        [
            {
                "Algorithm": "PAR (best)",
                "C": BEST_PAR["C"],
                "epsilon": BEST_PAR["epsilon"],
                "alpha": None,
                "max_iter": BEST_PAR["max_iter"],
                "Objective": par_obj_best,
                "Time (s)": par_time_best,
                "R2": par_test["r2"],
                "MAE": par_test["mae"],
            },
            {
                "Algorithm": "PAR+Ridge (best)",
                "C": BEST_PAR_RIDGE["C"],
                "epsilon": BEST_PAR_RIDGE["epsilon"],
                "alpha": BEST_PAR_RIDGE["alpha"],
                "max_iter": BEST_PAR_RIDGE["max_iter"],
                "Objective": ridge_obj_best,
                "Time (s)": ridge_time_best,
                "R2": ridge_test["r2"],
                "MAE": ridge_test["mae"],
            },
            {
                "Algorithm": "sklearn PAR default",
                "C": 1.0,
                "epsilon": sk_par_eps,
                "alpha": None,
                "max_iter": getattr(sk_par, "n_iter_", None),
                "Objective": sk_par_obj,
                "Time (s)": sk_par_time,
                "R2": sk_par_test["r2"],
                "MAE": sk_par_test["mae"],
            },
            {
                "Algorithm": "sklearn PA+L2 default-like",
                "C": 1.0,
                "epsilon": 0.1,
                "alpha": sk_alpha,
                "max_iter": getattr(sk_ridge, "n_iter_", None),
                "Objective": sk_ridge_obj,
                "Time (s)": sk_ridge_time,
                "R2": sk_ridge_test["r2"],
                "MAE": sk_ridge_test["mae"],
            },
        ]
    )
    print(df_summary.to_string(index=False))
    summary_path = os.path.join(PLOT_DIR, "summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
