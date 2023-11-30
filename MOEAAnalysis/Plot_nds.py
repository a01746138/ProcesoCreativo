from ExperimentsMATLAB.ProcessingData import decode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dominate(p, q):
    flag = True

    # Dominates if every objective value of p is less than the one of q
    for i in range(len(p)):
        if p[i] >= q[i]:
            flag = False
            return flag
    return flag


def non_dominated_samples(front):
    indexes = []
    for i in range(len(front)):
        p = front[i][6:]
        n = 0
        for j in range(len(front)):
            q = front[j][6:]
            if dominate(q, p):
                n += 1
        if n == 0:
            indexes.append(i)
    return indexes


def save_nds(algorithm_list: list):
    for algorithm in algorithm_list:
        for lam in range(10):
            data = []
            for k in range(1, 31):
                if k < 10:
                    exp = f'0{k}'
                else:
                    exp = f'{k}'
                for x in np.loadtxt(fname=f'../MOEARuns/{algorithm}_nds_lambda{lam}_exp{exp}.txt', delimiter=','):
                    data.append(list(x))
            nds_index = non_dominated_samples(data)

            nds = [data[ind] for ind in nds_index]
            df_nds = pd.DataFrame(nds, columns=['Vbatt', 'Qbatt', 'Ndiff',
                                                'Rwheel', 'MaxPmot', 'Mass',
                                                'Hcons', 'Pmech']).sort_values(by=['Hcons', 'Pmech'])
            if algorithm == 'pimia':
                df_nds['Mass'] = df_nds['Mass'] / 9
            df_nds.to_csv(path_or_buf=f'NDS/{algorithm}_nds_lambda{lam}.csv', index=False)


def plot3d(algorithm: str):
    ax = plt.figure(dpi=300).add_subplot(projection='3d')
    for lam in range(10):
        file = pd.read_csv(filepath_or_buffer=f'NDS/{algorithm}_nds_lambda{lam}.csv')
        file['Pmech'] = -file['Pmech']
        d_file = decode(file)
        x = d_file['Hcons']
        y = -d_file['Pmech']
        z = d_file['Mass'].iloc[0]
        ax.plot(x, y, zs=z, zdir='z', label=fr'$\lambda_{lam}$')
    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(-100, -10)
    ax.set_zlim(1500, 2600)
    ax.set_xlabel(r'$f_{1,\lambda}(\mathbf{x})$ [kg]', fontsize=15)
    ax.set_ylabel(r'$f_{2,\lambda}(\mathbf{x})$ [kW]', fontsize=15)
    ax.set_zlabel(r'$\lambda$ [kg]', fontsize=15)

    ax.view_init(elev=20., azim=-35)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
    )

    plt.savefig(fname=f'../Images/{algorithm}_3dfamily.png')


def plot2d(algorithm: str):
    ax = plt.figure(dpi=300).add_subplot()
    for lam in range(10):
        file = pd.read_csv(filepath_or_buffer=f'NDS/{algorithm}_nds_lambda{lam}.csv')
        file['Pmech'] = -file['Pmech']
        d_file = decode(file)
        x = d_file['Hcons']
        y = -d_file['Pmech']
        ax.plot(x, y, label=fr'$\lambda_{lam}$')
    ax.legend()
    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(-100, -10)
    ax.set_xlabel('Hydrogen consumption [kg]', fontsize=15)
    ax.set_ylabel('Total mechanical power of the motor [kW]', fontsize=15)

    plt.savefig(fname=f'../Images/{algorithm}_2dfamily.png')


al = ['pimia', 'imia', 'sms', 'nsga3', 'moead']

for a in al:
    plot3d(a)
    plot2d(a)
