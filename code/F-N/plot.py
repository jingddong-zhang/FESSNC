import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd

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
    xlabelpad = -15
    num = 50
    rangeT = 100
    uncon = np.load('./data/uncontrol_trajectory.npy')
    con = np.load('./data/control_trajectory.npy')
    N = con.shape[1]
    dt = 0.01

    if case==0:
        fig = plt.figure(figsize=(8, 3))
        plt.subplots_adjust(left=0.08, bottom=0.15, right=0.9, top=0.9, hspace=0.25, wspace=0.2)
        plt.subplot(131)
        def plot1(data,color,label):
            v = np.zeros(data.shape[0:2])
            for i in range(5):
                v[i] = np.max(np.abs(data[i,:,0:100:2]-np.tile(data[i,:,0:1],(1,50))),axis=1)
            mean_v,std_v=np.mean(v, axis=0), np.std(v, axis=0)
            plt.fill_between(np.arange(N) * dt, mean_v - std_v, mean_v + std_v, color=color, alpha=0.6)
            plt.plot(np.arange(N) * dt, mean_v, color=color,label=label)
            plt.xticks([0,100],fontsize=fontsize)
            plt.yticks([0, 5],fontsize=fontsize)
        plot1(uncon,colors[-1],'Original')
        plot1(con, colors[-2],'Controlled')
        plt.axhline(5, ls='--',color=colors[-3],label='Safe Line')
        plt.legend(loc=4,fontsize=fontsize-7, frameon=False)
        plt.xlabel('Time', fontsize=fontsize,labelpad=xlabelpad)
        plt.ylabel(r'$|\tilde{v}|_{\max}$', fontsize=fontsize,labelpad=-10)
        plot_grid()

        plt.subplot(132)
        plt.imshow(uncon[0,:,0:100:2].T,extent=[0,rangeT,0,num],cmap='RdBu',aspect='auto')
        plt.clim(-2,2)
        plt.yticks([0,50],fontsize=fontsize)
        plt.xticks([0,100],fontsize=fontsize)
        plt.xlabel('Time',fontsize=fontsize,labelpad=xlabelpad)
        plt.ylabel('Oscillator Index',fontsize=fontsize,labelpad=-22)

        ax = plt.subplot(133)
        h=plt.imshow(con[0,:,0:100:2].T,extent=[0,rangeT,0,num],cmap='RdBu',aspect='auto')
        plt.yticks([])
        # plt.yticks([0, 50], fontsize=fontsize)
        plt.xticks([0,100],fontsize=fontsize)
        plt.xlabel('Time',fontsize=fontsize,labelpad=xlabelpad)
        cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
        cb=plt.colorbar(h,cax=cax)
        # cb.set_label(r'$v_i$', fontdict={'size':fontsize})
        cb.ax.set_title(r'$v_i$',fontsize=fontsize)
        cb.set_ticks([-2, 0, 2])
        cb.ax.tick_params(labelsize=fontsize)
        plt.clim(-2,2)




        # plt.subplot(235)
        # A = np.load('./data/laplace_matrix.npy')
        # A = A - np.diag(np.diagonal(A))
        # G = nx.Graph(A)
        # pos = nx.circular_layout(G)
        # nodecolor = G.degree()  # 度数越大，节点越大，连接边颜色越深
        # nodecolor2 = pd.DataFrame(nodecolor)  # 转化称矩阵形式
        # nodecolor3 = nodecolor2.iloc[:, 1]  # 索引第二列
        # edgecolor = range(G.number_of_edges())  # 设置边权颜色
        # print(edgecolor)
        # nx.draw(G, pos, with_labels=False, node_size=nodecolor3 * 12, node_color=nodecolor3 * 15, edge_color=edgecolor,
        #         cmap=plt.cm.jet)
        #
        # plt.subplot(236)
        # # plt.plot(uncon[0,:,0],uncon[0,:,1],color=colors[-1])
        # plt.plot(con[0, :, 0], con[0, :, 1], color=colors[-2])

# ws = nx.Graph(A)
# data = np.load('./data/control_trajectory.npy')
# a = np.zeros(data.shape[0:2])
# print(data.shape,a.shape)
# sol = data[0]
# plt.plot(np.arange(len(sol)),sol[:,0])
# print(sol[4900:4990,0])
# ws = nx.watts_strogatz_graph(50, 4, 0.0) #小世界网络
# nx.draw(ws)
# syn_matrix = np.array(nx.to_numpy_matrix(ws))
# print(syn_matrix)

plot(0)
plt.show()