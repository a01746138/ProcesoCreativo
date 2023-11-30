import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

ref_point = np.array([1, 0])

indicator = HV(ref_point=ref_point)

algorithm = ['sms', 'moead', 'nsga3', 'imia', 'pimia']

columns = []
data = []
for lam in range(10):
    for a in algorithm:
        exp_data = []
        for i in range(1, 31):
            if i < 10:
                exp = f'0{i}'
            else:
                exp = f'{i}'

            file = np.loadtxt(fname=f'../MOEARuns/{a}_nds_lambda{lam}_exp{exp}.txt', delimiter=',')
            objectives = file[:, [6, 7]]
            hv = indicator(objectives)
            exp_data.append(file[-1, 1])
        data.append(exp_data)
        columns.append(f'lambda{lam}_{a}')

df = pd.DataFrame(np.array(data).T, columns=columns)
# df.to_csv(path_or_buf='HV_lambda.csv', index=False)

stats = df.describe()

# stats.to_csv(path_or_buf='HVStats.csv')

median = df.median()
# median.to_csv(path_or_buf='HVMedian.csv')

alg = ['sms', 'imia', 'pimia']
for lam in range(10):
    av = []
    std = []
    med = []
    for a in alg:
        av.append(np.round(stats[f'lambda{lam}_{a}'].loc['mean'], decimals=4))
        std.append(np.round(stats[f'lambda{lam}_{a}'].loc['std'], decimals=4))
        med.append(np.round(median[f'lambda{lam}_{a}'], decimals=4))

    string = f'$\lambda_{lam}$ & {av[0]} & {std[0]} & {med[0]} & {av[1]} & {std[1]} & {med[1]} & {av[2]} & {std[2]} & {med[2]} \\\\ \\hline'
    # string = f'$\lambda_{lam}$ & {av[0]} & {std[0]} & {med[0]} & {av[1]} & {std[1]} & {med[1]} \\\\ \\hline'
    print(string)