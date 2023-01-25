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
    [0.4, 1.0, 0.0], # brightgreen
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
    fontsize = 20
    labelsize = 13
    data = np.load('./data/trajectory_data.npy')
    N = data.shape[2]
    dt = 0.01

    if case==0:
        fig = plt.figure(figsize=(8, 3))
        plt.subplots_adjust(left=0.04, bottom=0.1, right=0.97, top=0.95, hspace=0.25, wspace=0.32)
        plt.subplot(131)
        def plot1(data,color,label):
            v = np.sqrt(data)
            mean_v,std_v=np.mean(v, axis=0), np.std(v, axis=0)
            bottom = mean_v - std_v
            bottom[np.where(bottom<=0.)[0]]=0.
            plt.fill_between(np.arange(N) * dt, bottom, mean_v + std_v, color=color, alpha=0.2)
            plt.plot(np.arange(N) * dt, mean_v, color=color,label=label)
            plt.xticks([0,10,20],[0,'',20],fontsize=fontsize)

        plot1(np.sum(data[1,:,:,0:2]**2,axis=-1), colors[0], 'FESSNC')
        plot1(np.sum(data[2,:,:,0:2]**2,axis=-1), colors[1],'GP-MPC')
        plot1(np.sum(data[3,:,:,0:2]**2,axis=-1), colors[-1], 'BALSA')
        plt.axhline(2, ls='--',color=colors[-3],label='Safe Line')
        plt.legend(loc=5,fontsize=labelsize, frameon=False)
        plt.xlabel('Time', fontsize=fontsize,labelpad=-15)
        plt.ylabel('$r$', fontsize=fontsize,labelpad=-8)
        plt.yticks([0,2],  fontsize=fontsize)
        plt.ylim(-0.5, 2.5)
        plot_grid()

        plt.subplot(132)
        def plot2(data,color,label):
            x,y = data[:,:,0],data[:,:,1]
            mean_x,std_x,mean_y,std_y = np.mean(x,axis=0),np.std(x,axis=0),np.mean(y,axis=0),np.std(y,axis=0)
            plt.plot(mean_x,mean_y,color=color,label=label)
            plt.ylim(-0.6,0.6)
            plt.xlim(-0.7,0.6)
            plt.yticks([-0.5,0.5],fontsize=fontsize)
            plt.xticks([-0.5, 0.5],fontsize=fontsize)
        plot2(data[1], colors[0], 'FESSNC')
        plot2(data[2], colors[1], 'GP-MPC')
        plot2(data[3], colors[-1], 'BALSA')
        plt.legend(loc=4, fontsize=labelsize, frameon=False)
        plt.xlabel(r'$x$', fontsize=fontsize, labelpad=-15)
        plt.ylabel(r'$y$', fontsize=fontsize, labelpad=-35)
        plot_grid()

        plt.subplot(133)
        n = 200  # per trial : n time steps
        L = int(2000 / n)
        def success_rate(data):
            v = data[1:4, :, :, 0]**2 + data[1:4, :, :, 1]**2  # delta_theta
            v_slice = np.zeros([3, 5, L])
            v_num = np.zeros([3, L])
            for k in range(3):
                for i in range(L):
                    value = np.max(v[k, :, n * i:n * (i + 1)], axis=1)
                    v_slice[k, :, i] = value
                    v_num[k, i] = len(np.where(value < 0.1)[0])
            return v_num / 5 * 100
        v = success_rate(data)
        plt.plot(np.arange(L),v[0,:],'bo--',label='FESSNC',alpha=0.5)
        plt.plot(np.arange(L), v[1, :],'mo--', label='GP-MPC',alpha=0.5)
        plt.plot(np.arange(L), v[2, :],'ro--', label='BALSA',alpha=0.5)
        plt.legend(loc=4, fontsize=labelsize, frameon=False)
        plt.xlabel('Trial', fontsize=fontsize, labelpad=-15)
        plt.ylabel('Success $\%$', fontsize=fontsize, labelpad=-8)
        plt.xticks([0,10],fontsize=fontsize)
        plt.yticks([0,20,40,60,80,100],fontsize=fontsize)
        plot_grid()

def com_time():
    data = np.load('./data/time_cost.npy')
    for i in range(1,4):
        time = np.mean(data[i])
        print(time)
# com_time()
plot(0)

plt.show()