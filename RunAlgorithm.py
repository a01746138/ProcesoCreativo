import numpy as np
import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling  # Sampling
from pymoo.operators.selection.tournament import TournamentSelection  # Selection
from pymoo.operators.crossover.sbx import SBX  # Crossover
from pymoo.operators.mutation.pm import PM  # Mutation
from pymoo.algorithms.moo.sms import cv_and_dom_tournament
from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival  # Sorting
from PMOP import MyPMOP


class RunAlgorithm:
    def __init__(self,
                 lam=None,  # Partition values of lambda
                 n_exp=1,  # Number of experiments per lambda
                 n_pop=300,  # Number of individuals per generation
                 n_gen=None,  # Number of generations to be run
                 rand=True  # Random seed for the algorithm to initialize
                 ):

        self.algorithm = None
        self.n_exp = n_exp
        self.n_pop = n_pop
        self.rand = rand

        if lam is None:
            self.lam = [0.0, 1.0]
        else:
            self.lam = lam

        if n_gen is None:
            self.n_gen = [1000]
        else:
            self.n_gen = n_gen

    def sms_evaluate(self):
        self.algorithm = SMSEMOA(pop_size=self.n_pop,
                                 sampling=FloatRandomSampling(),
                                 selection=TournamentSelection(func_comp=cv_and_dom_tournament),
                                 crossover=SBX(prob_exch=0.5),
                                 mutation=PM(),
                                 survival=LeastHypervolumeContributionSurvival()
                                 )
        self.txt_output()

    def run_algorithm(self, lamb, gen):
        if self.rand:
            seed = np.random.randint(0, 1e5)
        else:
            seed = 1827

        res = minimize(problem=MyPMOP(lambda_mass=lamb, elementwise_runner=self.runner),
                       algorithm=self.algorithm,
                       termination=('n_gen', gen),
                       seed=seed,
                       verbose=True)
        return res

    def txt_output(self):
        for i in self.lam:
            for j in self.n_gen:
                for k in range(self.n_exp):
                    res = self.run_algorithm(lamb=i, gen=j)

                    f = []
                    for m in range(len(res.F)):
                        ind = []
                        for n in range(5):
                            ind.append(res.X[m][n])
                        ind.append(i)
                        for n in range(2):
                            ind.append(res.F[m][n])
                        f.append(ind)

                    np.savetxt(fname=f'runs\\sms_lam{i}_ngen{j}_exp{k}.txt',
                               X=np.array(f),
                               delimiter=',')
