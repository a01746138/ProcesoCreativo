import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm


def discrepancy():
    n_ind = [1e2, 2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2,
             1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
             1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]

    disc = []

    for i in tqdm(n_ind):
        file = np.loadtxt(f'lhs\\lhs{int(i)}.txt', delimiter=',')
        disc.append([int(i), qmc.discrepancy(file, workers=-1)])

    np.savetxt(fname='disc_output.txt', X=np.array(disc, dtype='float'))


def plot_discrepancy():
    results = np.loadtxt('disc_output.txt')
    plt.plot(results[9:, 0], results[9:, 1])
    plt.xscale('log')
    plt.xlabel('Number of individuals')
    plt.ylabel('Discrepancy value')
    plt.show()


plot_discrepancy()
