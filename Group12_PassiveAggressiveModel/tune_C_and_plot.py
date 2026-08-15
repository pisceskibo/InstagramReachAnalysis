"""Tune and compare Passive Aggressive variants.

This script:
- Sweeps `C` and `epochs` for each variant (standard, PA-I, PA-II)
- Uses the implementations in `optimization_algorithms.py`
- Records test MSE at checkpoints (from returned history weights)
- Selects the best configuration (by final test MSE) per variant
- Plots comparison of best runs: MSE vs iterations and MSE vs elapsed time

Usage: python tune_C_and_plot.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimization_algorithms import (
    passive_aggressive_standard,
    passive_aggressive_pa1,
    passive_aggressive_pa2,
)


def load_data(path: Path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    features = ['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']
    X_raw = df[features].values.astype(float)
    y = df['Impressions'].values.astype(float)
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0] = 1.0
    X = (X_raw - mean) / std
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X, y


# fixed epsilon for all PA runs
EPSILON = 0.01


def compute_metrics(y_true, y_pred):
    res = y_true - y_pred
    mse = float(np.mean(res ** 2))
    mae = float(np.mean(np.abs(res)))
    ss = float(np.sum(res ** 2))
    st = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss / st if st != 0 else 1.0
    return mse, mae, r2


def run_sweep_for_variant(X, y, variant_name, variant_func, C_values, epochs_list, epsilon=0.01, record_every=10):
    """Run sweeps over C and epochs for a given variant.
    Returns per-run histories and the best run summary.
    """
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_train = int(0.8 * len(X))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    all_runs = []

    for epochs in epochs_list:
        for C in C_values:
            # call variant implementation; it returns final w and a history dict with stored checkpoints
            w_final, history = variant_func(X_train, y_train, C=C, epsilon=epsilon, epochs=epochs, record_every=record_every)

            # history contains 'weights', 'times', 'iters'
            weights = history.get('weights', [])
            times = history.get('times', [])
            iters = history.get('iters', [])

            # compute test metrics for each stored weight
            mses = []
            maes = []
            r2s = []
            for w in weights:
                y_pred = X_test.dot(w)
                mse, mae, r2 = compute_metrics(y_test, y_pred)
                mses.append(mse)
                maes.append(mae)
                r2s.append(r2)

            # if no checkpoints were recorded, evaluate final weight
            if len(mses) == 0:
                y_pred = X_test.dot(w_final)
                mse, mae, r2 = compute_metrics(y_test, y_pred)
                mses = [mse]
                maes = [mae]
                r2s = [r2]
                times = [0.0]
                iters = [epochs * max(1, X_train.shape[0])]

            run = {
                'variant': variant_name,
                'C': C,
                'epochs': epochs,
                'iters': np.array(iters),
                'times': np.array(times),
                'mses': np.array(mses),
                'maes': np.array(maes),
                'r2s': np.array(r2s),
            }
            all_runs.append(run)

    # choose best run by final mse
    def final_mse(r):
        return float(r['mses'][-1])

    best_run = min(all_runs, key=final_mse)
    return all_runs, best_run


def plot_compare_best(best_runs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for r in best_runs:
        plt.plot(r['iters'], r['mses'], label=f"{r['variant']} (C={r['C']}, ep={r['epochs']})")
    plt.xlabel('Iterations')
    plt.ylabel('Test MSE')
    plt.title('Test MSE vs Iterations (best config per variant)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'best_variants_mse_iters.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for r in best_runs:
        plt.plot(r['times'], r['mses'], label=f"{r['variant']} (C={r['C']}, ep={r['epochs']})")
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Test MSE')
    plt.title('Test MSE vs Time (best config per variant)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'best_variants_mse_time.png')
    plt.close()


def plot_sweep_heatmap(runs, out_dir: Path, metric='mse'):
    # produce simple table/heatmap of final metric for combinations of (C, epochs)
    import matplotlib
    Cs = sorted(list({r['C'] for r in runs}))
    Es = sorted(list({r['epochs'] for r in runs}))
    grid = np.zeros((len(Es), len(Cs)))
    for i, e in enumerate(Es):
        for j, c in enumerate(Cs):
            matches = [r for r in runs if r['C']==c and r['epochs']==e]
            if matches:
                val = float(matches[0]['mses'][-1]) if metric=='mse' else float(matches[0]['maes'][-1])
            else:
                val = np.nan
            grid[i, j] = val

    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, label=metric)
    plt.xticks(np.arange(len(Cs)), Cs)
    plt.yticks(np.arange(len(Es)), Es)
    plt.xlabel('C')
    plt.ylabel('epochs')
    plt.title(f'Final {metric.upper()} heatmap (rows=epochs, cols=C)')
    plt.tight_layout()
    # try to get variant name from runs
    variant = runs[0]['variant'] if runs else 'variant'
    plt.savefig(out_dir / f'heatmap_{variant}_{metric}.png')
    plt.close()


def plot_C_vs_mse_for_variant(runs, out_dir: Path):
    # For a given variant's runs, plot final MSE vs C for each epoch value
    Cs = sorted(list({r['C'] for r in runs}))
    Es = sorted(list({r['epochs'] for r in runs}))
    variant = runs[0]['variant'] if runs else 'variant'

    plt.figure(figsize=(8, 6))
    for e in Es:
        vals = []
        for c in Cs:
            matches = [r for r in runs if r['C']==c and r['epochs']==e]
            if matches:
                vals.append(float(matches[0]['mses'][-1]))
            else:
                vals.append(np.nan)
        plt.plot(Cs, vals, label=f'epochs={e}')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Final Test MSE')
    plt.title(f'Final MSE vs C — {variant}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f'C_vs_mse_{variant}.png')
    plt.close()


def plot_C_sweep_iters(runs, out_dir: Path):
    """Plot MSE vs iterations for different C values (choose run with max epochs per C)."""
    Cs = sorted(list({r['C'] for r in runs}))
    variant = runs[0]['variant'] if runs else 'variant'
    plt.figure(figsize=(10, 6))
    for c in Cs:
        matches = [r for r in runs if r['C'] == c]
        if not matches:
            continue
        # pick run with max epochs (most training)
        run = max(matches, key=lambda x: x['epochs'])
        plt.plot(run['iters'], run['mses'], label=f'C={c} (ep={run["epochs"]})')
    plt.xlabel('Iterations')
    plt.ylabel('Test MSE')
    plt.title(f'C sweep: MSE vs Iterations — {variant}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f'C_sweep_iterations_{variant}.png')
    plt.close()


def plot_C_sweep_time(runs, out_dir: Path):
    """Plot MSE vs elapsed time for different C values (pick run with max epochs per C)."""
    Cs = sorted(list({r['C'] for r in runs}))
    variant = runs[0]['variant'] if runs else 'variant'
    plt.figure(figsize=(10, 6))
    for c in Cs:
        matches = [r for r in runs if r['C'] == c]
        if not matches:
            continue
        run = max(matches, key=lambda x: x['epochs'])
        plt.plot(run['times'], run['mses'], label=f'C={c} (ep={run["epochs"]})')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Test MSE')
    plt.title(f'C sweep: MSE vs Time — {variant}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f'C_sweep_time_{variant}.png')
    plt.close()


def main():
    repo = Path(__file__).resolve().parent
    data_path = repo / 'datasets' / 'instagram_new_data.csv'
    if not data_path.exists():
        data_path = repo / 'datasets' / 'instagram_data.csv'
    X, y = load_data(data_path)

    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    epochs_list = [3]
    record_every = max(1, int(0.1 * 0.8 * len(X)))

    variants = {
        'standard': passive_aggressive_standard,
        'pa1': passive_aggressive_pa1,
        'pa2': passive_aggressive_pa2,
    }

    all_best = []
    all_runs_by_variant = {}
    for name, func in variants.items():
        print(f"Running sweep for {name}...")
        runs, best = run_sweep_for_variant(X, y, name, func, C_values, epochs_list, epsilon=EPSILON, record_every=record_every)
        all_best.append(best)
        all_runs_by_variant[name] = runs
        print(f"Best for {name}: C={best['C']}, epochs={best['epochs']}, final MSE={best['mses'][-1]:.6f}")

    out_dir = repo / 'plots'
    plot_compare_best(all_best, out_dir)

    # also save heatmaps per variant
    for name, runs in all_runs_by_variant.items():
        plot_sweep_heatmap(runs, out_dir, metric='mse')
        # also produce C vs MSE line plots for pa1 and pa2
        if name in {'pa1', 'pa2'}:
            plot_C_vs_mse_for_variant(runs, out_dir)
            # produce C sweep curves over iterations and time
            plot_C_sweep_iters(runs, out_dir)
            plot_C_sweep_time(runs, out_dir)

    print('\nDone. Plots saved to', out_dir)


if __name__ == '__main__':
    main()
