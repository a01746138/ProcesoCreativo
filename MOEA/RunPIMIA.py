# Run the P-IMIA algorithm
# ==========================================================
import numpy as np
from PMOP_PIMIA import MyPMOP
from PIMIA import IMIA
import time

n_gen = 10
pop_size = 100
algorithm = 'pimia'
nuc = 1


def save_data(pop_lam, nds_lam, hv_lam):

    for lam in range(len(pop)):

        # Save the nds in a file
        x = [list(np.append(np.array(j), lam)) for j in nds_lam[f'lam{lam}']['X']]
        nds['X'] = x
        txt_nds = [list(np.append(nds['X'][i], nds_lam[f'lam{lam}']['F'][i]))
                   for i in range(len(nds[f'lam{lam}']['X']))]
        np.savetxt(fname=f'../MOEARuns/{algorithm}_nds_lambda{lam}_exp{experiment}.txt',
                   X=txt_nds, delimiter=',')

        # Save the pop in a file
        y = [list(np.append(np.array(j), lam)) for j in pop_lam[f'lam{lam}']['X']]
        pop['X'] = y
        txt_pop = [list(np.append(pop['X'][i], pop_lam[f'lam{lam}']['F'][i]))
                   for i in range(len(pop[f'lam{lam}']['X']))]
        np.savetxt(fname=f'../MOEARuns/{algorithm}_pop_lambda{lam}_exp{experiment}.txt',
                   X=txt_pop, delimiter=',')

        # Save hypervolume history
        hv_lambda = []
        for j in range(len(hv_lam)):
            hv_lambda.append([hv_lam[j][0]/10, hv_lam[j][lam + 1]])
        np.savetxt(fname=f'../MOEARuns/{algorithm}_hv_lambda{lam}_exp{experiment}.txt',
                   X=hv_lambda, delimiter=',')


for i in range(1, 7):
    ex = (nuc - 1) * 6 + i
    if ex < 10:
        experiment = f'0{ex}'
    else:
        experiment = f'{ex}'

    start_time = time.time()
    run = IMIA(pop_size=pop_size, n_gen=n_gen,
               problem=MyPMOP(), verbose=True,
               indicators=['HV', 'R2', 'EpsPlus', 'DeltaP', 'IGDPlus'])
    pop, nds, hv = run()
    simulation_time = time.time() - start_time
    save_data(pop, nds, hv)
    np.savetxt(X=[simulation_time], fname=f'../MOEATimes/{algorithm}_exp{experiment}.txt')
