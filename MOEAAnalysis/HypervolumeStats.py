# Obtain the stats of the HV values for all the algorithms
# ================================================================

import numpy as np
import pandas as pd

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

            file = np.loadtxt(fname=f'../MOEARuns/{a}_hv_lambda{lam}_exp{exp}.txt', delimiter=',')
            exp_data.append(file[-1, 1])
        data.append(exp_data)
        columns.append(f'lambda{lam}_{a}')

df = pd.DataFrame(np.array(data).T, columns=columns)
# df.to_csv(path_or_buf='HV_lambda.csv', index=False)

stats = df.describe()

# stats.to_csv(path_or_buf='HVStats.csv')


