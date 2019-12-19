import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import norm
from sklearn.datasets import load_wine

from kohonen import lin2grid, lr_it, map_input_samples, radius_decay_it, som, count
from radar_plot import get_cmap, kohonen_radar_plot


def main():

    # Load data
    data = load_wine(return_X_y=False)

    # Extract useful data / info
    X = data.data
    Y = data.target
    n_classes = data.target_names.size
    n_samples = X.shape[0]
    Y_clrs = [get_cmap(n_classes, name='jet')(c) for c in Y]

    # Parameters
    d1, d2 = 2, 3
    epoch_max = 60
    lr0 = 1e-1
    sigma0 = 1

    # Learn
    W, W_save = som(X, d1, d2,
                    epoch_max=epoch_max, lr0=lr0, sigma0=sigma0)

    # Map input samples onto the grid
    X_grid_idxs = map_input_samples(X, W, d1, d2)

    # Save results
    np.save("./data/W", W_save)
    np.save("./data/X_map", X_grid_idxs)

    ####################################################################
    # Plots

    # Show lattice final values
    kohonen_radar_plot(W, d1, d2, mode='area', same_scale=True,
                       features_names=data.feature_names,
                       cmap_name='jet')

    # Plot mapping
    # fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    # ax1.scatter(X_grid_idxs[:, 0], X_grid_idxs[:, 1], c=Y_clrs, alpha=.5)
    # ax1.grid()
    K = count(X_grid_idxs, Y, d1, d2)
    kohonen_radar_plot(K, d1, d2, mode='radius', same_scale=True,
                       features_names=data.target_names, cmap_name='jet',
                       suptitle="Nombre de d'exemples assignés à chaque neurone")

    # --- Plot coefficients evolution
    lr_list = list(lr_it(epoch_max, lr0))
    radius_decay_list = list(radius_decay_it(epoch_max, sigma0))

    fig2 = plt.figure(constrained_layout=False)
    gs = fig2.add_gridspec(2, 2)

    # Plot lr evolution
    # TODO Do it for various lr0
    ax21 = fig2.add_subplot(gs[0, 0])
    ax21.plot(lr_list, label='lr(t)')
    ax21.set_xlabel('t')
    ax21.legend()
    ax21.grid()

    # Plot h evolution
    # TODO do it fo various sigma0
    ax22 = fig2.add_subplot(gs[0, 1])
    ax22.plot(radius_decay_list, label='h(t)')
    ax22.set_xlabel('t')
    ax22.legend()
    ax22.grid()

    # Plot neighboorhood function evolution
    ax23 = fig2.add_subplot(gs[1, :])
    dlinspace = np.linspace(0, norm(np.array([d1-1, d2-1]), ord=2))
    for t in range(0, epoch_max, epoch_max//10):
        h = radius_decay_list[t]
        nei_curve = np.exp(-h * dlinspace)
        ax23.plot(nei_curve, label='t={}'.format(t))
        plt.legend()

    plt.show()

    # Make a gif from W evolution
    # W_gif = np.array([cv2.resize(img, dsize=(250, 250),
    #                              interpolation=cv2.INTER_CUBIC)
    #                   for img in W_save])
    # imageio.mimsave('./data/W.gif', W_gif.astype(np.uint8), fps=2)


if __name__ == "__main__":
    main()
