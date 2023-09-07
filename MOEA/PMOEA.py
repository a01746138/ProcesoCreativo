# PMOEA implementation in python
# ==========================================================

from SMSEMOA import SMSEMOA
from MOEAD import MOEAD
from NSGA3 import NSGA3
import numpy as np


class PMOEA:

    def __init__(self,
                 n_gen=100,
                 pop_size=100,
                 problem=None,
                 lambda_partitions=10,
                 algorithm='sms',
                 verbose=False):
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.problem = problem
        self.lambda_partitions = lambda_partitions
        self.verbose = verbose

        if algorithm not in ['sms', 'moead', 'nsga3']:
            print('Defined algorithm not found.')
            quit()
        else:
            moea = {'sms': SMSEMOA, 'moead': MOEAD, 'nsga3': NSGA3}
            self.algorithm = moea[algorithm]
            self.algorithm_ref = algorithm

    def _lambda_vector(self):
        v = []
        for i in range(self.lambda_partitions):
            val = i / (self.lambda_partitions - 1)
            v.append(val)

        return v

    def _save_data(self, pop, nds, hv, lam, index, experiment):

        # Save the nds in a file
        x = [list(np.append(np.array(j), lam)) for j in nds['X']]
        nds['X'] = x
        txt_nds = [list(np.append(nds['X'][i], nds['F'][i])) for i in range(len(nds['X']))]
        np.savetxt(fname=f'../MOEARuns/{self.algorithm_ref}_nds_lambda{index}_exp{experiment}.txt',
                   X=txt_nds, delimiter=',')

        # Save the pop in a file
        y = [list(np.append(np.array(j), lam)) for j in pop['X']]
        pop['X'] = y
        txt_pop = [list(np.append(pop['X'][i], pop['F'][i])) for i in range(len(pop['X']))]
        np.savetxt(fname=f'../MOEARuns/{self.algorithm_ref}_pop_lambda{index}_exp{experiment}.txt',
                   X=txt_pop, delimiter=',')

        # Save hypervolume history
        np.savetxt(fname=f'../MOEARuns/{self.algorithm_ref}_hv_lambda{index}_exp{experiment}.txt',
                   X=hv, delimiter=',')

    def _do(self, experiment):
        lambda_vec = self._lambda_vector()

        for index in range(self.lambda_partitions):
            lam = lambda_vec[index]
            pop, nds, hv = self.algorithm(n_gen=self.n_gen, problem=self.problem(lambda_mass=lam),
                                          pop_size=self.pop_size, verbose=self.verbose)()

            # Save the pop, nds, and hypervolume convergence data
            self._save_data(pop, nds, hv, lam, index, experiment)

            if self.verbose:
                print(63 * '=')
                print('')

    def __call__(self, experiment):
        return self._do(experiment)
