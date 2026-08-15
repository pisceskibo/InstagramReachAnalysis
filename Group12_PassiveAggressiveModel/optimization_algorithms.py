# Libraries
import time
import numpy as np
from typing import Optional


def passive_aggressive_regression(X: np.ndarray, y: np.ndarray, C: float = 1.0, epsilon: float = 0.01,
                                 variant: str = 'standard', epochs: Optional[int] = None,
                                 record_every: int = 1):
    """Online Passive Aggressive regression for the three variants.

    variant in {'standard', 'pa1', 'pa2'}
      - standard: tau = loss / ||x||^2
      - pa1:      tau = min(C, loss / ||x||^2)
      - pa2:      tau = loss / (||x||^2 + 1/(2C))
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have matching number of rows.')

    variant = variant.lower()
    if variant not in {'standard', 'pa1', 'pa2'}:
        raise ValueError("variant must be one of {'standard', 'pa1', 'pa2'}")
    if C <= 0:
        raise ValueError('C must be positive.')

    n, d = X.shape
    w = np.zeros(d, dtype=float)
    history = {'weights': [], 'loss': [], 'times': [], 'iters': []}
    t0 = time.time()
    it = 0

    if epochs is None:
        epochs = max(1, n)

    for _ in range(epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            pred = float(xi.dot(w))
            diff = yi - pred
            loss = max(0.0, abs(diff) - epsilon)

            if loss > 0.0:
                norm_sq = float(xi.dot(xi))
                if norm_sq <= 0.0:
                    tau = 0.0
                elif variant == 'standard':
                    tau = loss / norm_sq
                elif variant == 'pa1':
                    tau = min(C, loss / norm_sq)
                else:
                    tau = loss / (norm_sq + 1.0 / (2.0 * C))
                sign = 1.0 if diff >= 0 else -1.0
                w = w + tau * sign * xi

            it += 1
            if it % record_every == 0:
                full_loss = float(np.mean((X.dot(w) - y) ** 2))
                history['weights'].append(w.copy())
                history['loss'].append(full_loss)
                history['times'].append(time.time() - t0)
                history['iters'].append(it)

    if not history['loss']:
        full_loss = float(np.mean((X.dot(w) - y) ** 2))
        history['weights'].append(w.copy())
        history['loss'].append(full_loss)
        history['times'].append(time.time() - t0)
        history['iters'].append(it)

    return w, history


def passive_aggressive_standard(X: np.ndarray, y: np.ndarray, C: float = 1.0, epsilon: float = 0.01,
                               epochs: Optional[int] = None, record_every: int = 1):
    return passive_aggressive_regression(X, y, C=C, epsilon=epsilon, variant='standard', epochs=epochs,
                                       record_every=record_every)


def passive_aggressive_pa1(X: np.ndarray, y: np.ndarray, C: float = 1.0, epsilon: float = 0.01,
                          epochs: Optional[int] = None, record_every: int = 1):
    return passive_aggressive_regression(X, y, C=C, epsilon=epsilon, variant='pa1', epochs=epochs,
                                       record_every=record_every)


def passive_aggressive_pa2(X: np.ndarray, y: np.ndarray, C: float = 1.0, epsilon: float = 0.01,
                          epochs: Optional[int] = None, record_every: int = 1):
    return passive_aggressive_regression(X, y, C=C, epsilon=epsilon, variant='pa2', epochs=epochs,
                                       record_every=record_every)


if __name__ == '__main__':
    print('This module contains only the Passive Aggressive variants: standard, PA-I, PA-II.')
