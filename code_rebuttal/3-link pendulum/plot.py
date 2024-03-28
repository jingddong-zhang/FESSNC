import numpy as np
import torch
import matplotlib.pyplot as plt
import math

colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
    [31/255,145/255,158/255],
    [127/255,172/255,204/255],
    [233/255,108/255,102/255],
]
colors = np.array(colors)


def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass


def success_(data):
    vmin = np.zeros([3, 5])
    for i in range(1, 4):
        v = 2 * np.pi - (data[i, :, :, 0] + data[i, :, :, 2])  # theta1+theta2
        vmin[i - 1] = np.min(v, axis=1)
    return vmin.T



def plot(case):
    import matplotlib
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction']='in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 18

    data = np.load('./data/trajectory_data_v1.npy',allow_pickle=True).item()
    data = data['state']
    m = data.shape[1]
    N = data.shape[2]
    dt = 0.01
    t = np.linspace(0,30,N)

    def plot1(data, color, label):
        v = np.sin(data)
        mean_v, std_v = np.mean(v, axis=0), np.std(v, axis=0)
        plt.fill_between(np.arange(N) * dt, mean_v - std_v, mean_v + std_v, color=color, alpha=0.2)
        plt.plot(np.arange(N) * dt, mean_v, color=color, label=label)
        plt.xticks([0, 15, 30], [0, '', 30], fontsize=fontsize)


    if case==0:
        fig = plt.figure(figsize=(8, 3))
        plt.subplots_adjust(left=0.06, bottom=0.1, right=0.97, top=0.88, hspace=0.25, wspace=0.22)
        # plt.subplots_adjust(left=0.04, bottom=0.1, right=0.97, top=0.95, hspace=0.25, wspace=0.32)
        plt.subplot(131)
        for i in range(m):
            plt.plot(t,data[0,i,:,0], color=colors[i],zorder=1)
        plt.axhline(math.pi*7/6,ls='--',lw=2,label='Safe Line')
        # plt.axhline(math.pi, ls='solid', lw=2,color='k',zorder=0)
        plt.axhline(-math.pi/6, ls='--', lw=2)
        plt.xlabel('Time', fontsize=fontsize,labelpad=-15)
        plt.ylabel(r'$\theta_1$', fontsize=fontsize,labelpad=-20)
        plt.ylim(-math.pi/2, math.pi*4/2+0.2)
        plt.xticks([0, 15, 30], [0, '', 30], fontsize=fontsize)
        plt.yticks([-math.pi/6, math.pi*7/6], [r'$-\frac{\pi}{6}$', r'$\frac{7\pi}{6}$'], fontsize=fontsize)
        plt.legend(fontsize=fontsize,frameon=False)
        plot_grid()
        plt.title('FESSNC',fontsize=fontsize)

        plt.subplot(132)
        for i in range(m):
            plt.plot(t, data[1, i, :, 0], color=colors[i])
        plt.axhline(math.pi * 7 / 6, ls='--', lw=2, label='Safe Line')
        plt.axhline(-math.pi / 6, ls='--', lw=2)
        plt.xlabel('Time', fontsize=fontsize, labelpad=-15)
        plt.ylabel(r'$\theta_1$', fontsize=fontsize, labelpad=-20)
        plt.ylim(-math.pi/2, math.pi*4/2+0.2)
        plt.xticks([0, 15, 30], [0, '', 30], fontsize=fontsize)
        plt.yticks([-math.pi / 6, math.pi * 7 / 6], [r'$-\frac{\pi}{6}$', r'$\frac{7\pi}{6}$'], fontsize=fontsize)
        plot_grid()
        plt.title('Kernel', fontsize=fontsize)


        plt.subplot(133)
        for i in range(m):
            plt.plot(t, data[2, i, :, 0], color=colors[i])
        plt.axhline(math.pi * 7 / 6, ls='--', lw=2, label='Safe Line')
        plt.axhline(-math.pi / 6, ls='--', lw=2)
        plt.xlabel('Time', fontsize=fontsize, labelpad=-15)
        plt.ylabel(r'$\theta_1$', fontsize=fontsize, labelpad=-20)
        plt.ylim(-math.pi / 2, math.pi * 4 / 2+0.2)
        plt.xticks([0, 15, 30], [0, '', 30], fontsize=fontsize)
        plt.yticks([-math.pi / 6, math.pi * 7 / 6], [r'$-\frac{\pi}{6}$', r'$\frac{7\pi}{6}$'], fontsize=fontsize)
        plot_grid()
        plt.title('FESS+Kernel', fontsize=fontsize)


plot(0)
plt.show()