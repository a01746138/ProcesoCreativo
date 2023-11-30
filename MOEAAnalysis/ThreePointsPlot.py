import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def plot(alg, mirai, lbl, lam):
    ax = plt.figure(dpi=300).add_subplot()
    ll = 3 * lam
    ul = 3 * (lam + 1)
    if lbl == 'sms':
        ax.scatter(alg[ll:ul, 8], -alg[ll:ul, 9], label='SMS-EMOA', marker=',')
    elif lbl == 'imia':
        ax.scatter(alg[ll:ul, 8], -alg[ll:ul, 9], label='IMIA', marker=',')
    elif lbl == 'pimia':
        ax.scatter(alg[ll:ul, 8], -alg[ll:ul, 9], label='P-IMIA', marker=',')
    ax.scatter(mirai[lam, 6], -mirai[lam, 7], label='Toyota Mirai 2023', marker='v')
    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(-100, -10)
    ax.set_xlabel('Hydrogen consumption [kg]', fontsize=15)
    ax.set_ylabel('Total mechanical power of the motor [kW]', fontsize=15)
    ax.set_title(f'Total mass of the vehicle {mirai[lam, 5]} kg')

    plt.savefig(fname=f'../Images/Mirai_{lbl}_lam{lam}.png')


sms = np.loadtxt(fname='MATLAB/MATLAB_sms_3points.txt', delimiter=',')
imia = np.loadtxt(fname='MATLAB/MATLAB_imia_3points.txt', delimiter=',')
pimia = np.loadtxt(fname='MATLAB/MATLAB_pimia_3points.txt', delimiter=',')
mirai = np.loadtxt(fname='MATLAB/MATLAB_Mirai_3points.txt', delimiter=',')

alg_dict = {'sms': sms, 'imia': imia, 'pimia': pimia}

for alg_lbl in ['sms', 'imia', 'pimia']:
    for lam in range(10):
        plot(alg_dict[alg_lbl], mirai, alg_lbl, lam)

    frames = [Image.open(f'../Images/Mirai_{alg_lbl}_lam{lam}.png') for lam in range(10)]
    frame_one = frames[0]
    frame_one.save(f'../Images/Mirai_{alg_lbl}.gif', format="GIF", append_images=frames,
                   save_all=True, duration=500, loop=0)
