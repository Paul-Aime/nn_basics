from colorsys import hsv_to_rgb, rgb_to_hsv

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.text import Text


def main():

    d1, d2 = 4, 4
    n_features = 13
    feature_names = ["feature number #{}".format(i) for i in range(n_features)]

    W = np.random.uniform(-5, 5, (n_features, d1*d2))

    kohonen_radar_plot(W, d1, d2, mode='area', same_scale=True,
                       features_names=feature_names, cmap_name='winter',
                       suptitle="mode=area, same_scale=True")

    kohonen_radar_plot(W, d1, d2, mode='area', same_scale=False,
                       features_names=feature_names, cmap_name='autumn',
                       suptitle="mode=area, same_scale=False")

    kohonen_radar_plot(W, d1, d2, mode='radius', same_scale=True,
                       features_names=feature_names, cmap_name='hsv',
                       suptitle="mode=radius, same_scale=True")

    kohonen_radar_plot(W, d1, d2, mode='radius', same_scale=False,
                       features_names=feature_names, cmap_name='jet',
                       suptitle="mode=radius, same_scale=False")

    plt.show()


def kohonen_radar_plot(W, d1, d2, mode='radius', same_scale=True,
                       features_names=None, cmap_name=None,
                       suptitle=None):

    n_features, n_neurons = W.shape

    assert n_neurons == d1 * d2

    colors = get_cmap(n_features, cmap_name)

    # Legend will be put in its on line big axis
    nrows, ncols = d1 if features_names is None else d1+1, d2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             subplot_kw=dict(projection='polar'))

    # Remove last line for legend
    """ See :
    Make a big axis for legend :
    https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    Put legend inside it :
    https://stackoverflow.com/questions/41561469/put-legend-on-a-place-of-a-subplot
    More on legend :
    https://stackoverflow.com/a/43439132 """
    if features_names is not None:
        gs = axes[-1, -1].get_gridspec()
        # remove the underlying axes
        for ax in axes[-1, :]:
            ax.remove()
        ax_lgd = fig.add_subplot(gs[-1, :])
        ax_lgd.set_axis_off()
        axes = axes[:-1, :]

    # Construct radar plot
    max_value = None if not same_scale else np.abs(W).max()
    for w, ax in zip(W.T, axes.flat):
        polygons = radar_plot(w, ax=ax, mode=mode,
                              max_value=max_value, colors=colors)

    # Add legend
    if features_names is not None:
        ax_lgd.legend(polygons, features_names, loc='center', ncol=ncols)

    # Add title
    if suptitle is not None:
        fig.suptitle(suptitle)

    # fig.tight_layout()


def radar_plot(w, ax=None, mode='radius', max_value=None,
               colors=None, cmap_name=None):

    is_area = True if mode != 'radius' else False

    n_features = w.shape[0]

    if ax is None:
        _, ax = plt.subplot(subplot_kw=dict(projection='polar'))

    if colors is None:
        colors = get_cmap(n_features, cmap_name)

    # Check for negative values
    is_neg = [r < 0 for r in w]
    w = np.abs(w)

    # Split the circle
    xticks = np.linspace(0, 2*np.pi, n_features+1)[:-1]
    xticks_closed = np.append(xticks, 2*np.pi)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_xticklabels([])

    # Max radius value
    if not max_value:
        max_value = w.max()
    if is_area:
        max_value = np.sqrt(max_value)
    ax.set_ylim(top=max_value)

    # Let only max value as ytick
    ax.set_yticks([max_value])

    # Display actual maximum value outside the circle
    yticklabel = Text(0, 0, text='{:.2f}'.format(np.abs(w).max()))
    ax.set_yticklabels([yticklabel])

    # Hide grid
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # Draw radar plot
    polygons = []
    for i, r in enumerate(w):

        # Radius or area
        if is_area:
            r = np.sqrt(r)

        # Create inteval
        theta_start = xticks_closed[i]
        theta_end = xticks_closed[i+1]
        theta = np.linspace(theta_start, theta_end, 100)

        # Create radius
        r_lin = np.ones_like(theta) * r

        # Make sure to close the polygon
        theta = np.append(theta, theta[-1])
        r_lin = np.append(r_lin, 0)

        # Draw the polygon
        polygons.append(ax.fill(theta, r_lin, fill=True, color=colors(i))[0])

        # Annotate for negative value
        if is_neg[i]:
            color = list(complementary(*colors(i)[:3]))
            color.append(1.0)
            theta_mid = theta_start + ((theta_end - theta_start)/2)
            ax.plot([theta_mid], [0.7*r], '.',
                    ms=3, mfc=color, mec=color)

    return polygons


def get_cmap(n, name=None):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a
    distinct RGB color; the keyword argument name must be a standard
    matplotlib colormap name.

    From https://stackoverflow.com/a/25628397'''
    if name is None:
        name = 'jet'

    return plt.cm.get_cmap(name, n)


def complementary(r, g, b):
    """returns RGB components of complementary color

    Adapted from https://stackoverflow.com/a/40234511"""
    if r == g == b:
        c = 1 - np.round(r).item(0)
        return (c, c, c)

    hsv = rgb_to_hsv(r, g, b)
    return hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])


if __name__ == "__main__":
    main()
