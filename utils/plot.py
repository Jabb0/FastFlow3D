import matplotlib.pyplot as plt


def plot_pillars_center(pillars, grid_max, grid_min):
    """ pillars must be in shape (n_pillars, 2) """
    x, y = [], []
    for pillar in pillars:
        x.append(pillar.x_c)
        y.append(pillar.y_c)

    plt.plot(x, y, marker='.', color='k', linestyle='none')
    plt.ylim([grid_min, grid_max])
    plt.xlim([grid_min, grid_max])
    plt.show()


def plot_pillars(pillars):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x_pos, y_pos, z_pos = [], [], []
    x_size, y_size, z_size = [], [], []
    for p in pillars:
        x_pos.append(p.x_c)
        y_pos.append(p.y_c)
        z_pos.append(0)

        x_size.append(1)
        y_size.append(1)
        z_size.append(len(p))

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size)
    plt.show()
