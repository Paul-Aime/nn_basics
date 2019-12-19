# kohonen.py

import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat


def som(X, d1, d2, epoch_max=100, lr0=1, sigma0=.3):

    n_samples, n_features = X.shape[0], X.shape[1]

    W = np.random.randn(n_features, d1*d2)

    W_save = np.expand_dims(W, axis=0).copy()

    # Normalization
    W /= norm(W, ord=1, axis=0)
    X /= norm(X, ord=2, axis=1)[:, None]
    X = (X - X.mean(axis=0))/X.var(axis=0)

    # TODO see why using generator in the for statement cause GeneratorExit
    lr_list = list(lr_it(epoch_max, lr0))
    radius_decay_list = list(radius_decay_it(epoch_max, sigma0))

    # Learning
    for lr, rad_coeff in zip(lr_list, radius_decay_list):

        # Randomly select samples
        for x in (X[i] for i in np.random.choice(n_samples, n_samples, replace=False)):

            # Compute activation of neurons
            y = np.matmul(W.T, x)

            # Compute distance between neurons and input
            dist = norm(W.T - repmat(x, d1*d2, 1), ord=2, axis=1)

            # Find the idx of the Best Matching Unit
            BMU_idx_d = np.argmin(dist)
            BMU_idx_a = np.argmax(y)
            # assert BMU_idx_d == BMU_idx_a # TODO Why not true ?
            BMU_idx = BMU_idx_a
            BMU_grid_idx = lin2grid(BMU_idx, d2)

            # Compute its distance with the input sample
            BMU_dist = x - W[:, BMU_idx]

            # Update weights
            for i in range(d1*d2):

                # Get index on the grid
                ci = lin2grid(i, d2)

                # Compute the neighborhood coefficient
                nei_coeff = neighbor_coeff(ci, BMU_grid_idx, rad_coeff)

                # Update weight
                W[:, i] += nei_coeff * lr * BMU_dist
                W[:, i] /= sum(W[:, i])

        W_save = np.append(W_save,
                           np.expand_dims(W, axis=0).copy(),
                           axis=0)

    return W, W_save


def map_input_samples(X, W, d1, d2):

    n_samples = X.shape[0]

    # Normalization
    X /= norm(X, ord=2, axis=1)[:, None]
    X = (X - X.mean(axis=0))/X.var(axis=0)

    # Get neurons grid coordinates
    neurons_grid_idxs = np.array([lin2grid(i, d2) for i in range(d1*d2)])

    # Prepare output
    X_grid_idxs = np.zeros((n_samples, 2))

    for i, x in enumerate(X):

        # Compute activation of neurons for each sample
        y = np.matmul(W.T, x)  # shape (d1*d2, )

        # Compute x position on the grid, by linear combination
        x_grid_idx = np.sum(np.multiply(neurons_grid_idxs, repmat(y, 2, 1).T))
        x_grid_idx /= y.sum()

        # Register output
        X_grid_idxs[i] = x_grid_idx

    return X_grid_idxs


def lr_it(tmax, lr0=5):
    for t in range(tmax):
        yield lr0 * np.exp(-t/tmax)


def radius_decay_it(tmax, sigma0=1):
    for t in range(tmax):
        yield 1 / (2 * (sigma0 * np.exp(-t / tmax)) ** 2)


def neighbor_coeff(ci, ck, h):
    return np.exp(-h * norm(np.array(ci) - np.array(ck), ord=2))


def lin2grid(i, d2):
    return np.array([i//d2, i % d2])
