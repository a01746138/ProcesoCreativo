import numpy as np
from scipy.stats import ranksums


algorithm = 'pimia'
file = np.loadtxt(fname=f'MATLAB/MATLAB_{algorithm}_3points.txt', delimiter=',')


p = ranksums(file[:, 6], file[:, 8]).pvalue
print(fr'The p-value of {algorithm} for f1 is: {p}')

p = ranksums(file[:, 7], file[:, 9]).pvalue
print(fr'The p-value of {algorithm} for f2 is: {p}')