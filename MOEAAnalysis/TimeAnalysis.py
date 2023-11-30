import numpy as np
import pandas as pd

algorithms = ['sms', 'imia', 'pimia']

for a in algorithms:
    times_list = []
    for exp in range(1, 31):
        if exp < 10:
            e = f'0{exp}'
        else:
            e = f'{exp}'
        file = np.loadtxt(fname=f'../MOEATimes/{a}_exp{e}.txt')
        times_list.append(file.max())

    times = pd.DataFrame(times_list)
    print(f'{a} & {times[0].mean()} & {times[0].std()} & {times[0].median()} \\\\ \\hline')

