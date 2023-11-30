# SMS-EMOA implementation in python
# ==========================================================

import numpy as np
from GeneticOperators import TournamentSelection as ts
from GeneticOperators import SimulatedBinaryCrossover as sbx
from GeneticOperators import PolynomialMutation as pm
from PerformanceIndicators import Hypervolume as hv
from PerformanceIndicators import IndividualContribution as ic
from pymoo.indicators.hv import HV

class SMSEMOA:

    def __init__(self,
                 problem_data,
                 pop_size=100,
                 n_gen=1000,
                 problem=None,
                 verbose=False
                 ):
        self.problem = problem  # Problem to be optimized
        self.pop_size = pop_size  # Population size
        self.n_gen = n_gen  # Number of generations
        self.verbose = verbose  # Print the results so far
        self.lam, self.exp = problem_data

        ref_point = np.array([1, 0])
        self.total_hv = HV(ref_point=ref_point)

    @staticmethod
    def _dominate(p, q):
        flag = True

        # Dominates if every objective value of p is less than the one of q
        for i in range(len(p)):
            if p[i] >= q[i]:
                flag = False
                return flag
        return flag

    @staticmethod
    def _min_contribution(f, n_pop):
        # Sort every solution
        front = []
        for index in f:
            front.append(n_pop[index])
        front = np.array(front)
        front = np.c_[front, f]
        front = front[front[:, 0].argsort()]

        # Create reference point
        nadir_point = front.max(axis=0)[:2]
        hypervolume = hv(nadir_point)  # Create the reference space

        # Obtain total Hypervolume value
        total_hv = hypervolume(front)

        # Obtain the contribution to the HV of every individual
        min_hv = np.inf
        r = None
        for ind in range(1, len(front) - 1):
            ind_hv = ic()(hypervolume, total_hv, front, ind)
            if ind_hv < min_hv:
                min_hv = ind_hv
                r = ind

        return int(front[r][2])

    @staticmethod
    def _nds_hv(nds):
        # Sort every solution
        front = np.array(nds['F'])
        front = front[front[:, 0].argsort()]

        # Create reference point
        nadir_point = np.array([1, 0])
        hypervolume = hv(nadir_point)  # Create the reference space

        # Obtain total Hypervolume value
        total_hv = hypervolume(front)

        return total_hv

    def _non_dominated_samples(self, front):
        indexes = []
        for i in range(len(front['F'])):
            p = front['F'][i]
            n = 0
            for j in range(len(front['F'])):
                q = front['F'][j]
                if self._dominate(q, p):
                    n += 1
            if n == 0:
                indexes.append(i)

        return indexes

    def _fast_non_dominated_sorting(self, pop):
        f1 = []
        fronts = {}
        sp = {}
        nq = {}
        for i in range(len(pop['F'])):
            p = pop['F'][i]
            s = []
            n = 0
            for j in range(len(pop['F'])):
                q = pop['F'][j]
                if self._dominate(p, q):  # if p dominates q
                    # Add q to the set of dominated solutions by p
                    s.append(j)
                elif self._dominate(q, p):  # if q dominates p
                    # Increment the domination counter of p
                    n += 1
            sp[f'p_{i}'] = s
            nq[f'q_{i}'] = n
            if n == 0:  # p belongs to the first front
                f1.append(i)
        k = 1
        fronts[f'F{k}'] = f1
        while fronts[f'F{k}'] != []:
            next_front = []  # next front indexes
            for i in fronts[f'F{k}']:
                for j in sp[f'p_{i}']:
                    nq[f'q_{j}'] -= 1
                    if nq[f'q_{j}'] == 0:  # q belongs to the next front
                        next_front.append(j)
            k += 1
            fronts[f'F{k}'] = next_front

        return fronts

    def _new_individual(self, pop):

        # Tournament Selection
        parents = ts(n_parents=2)(pop=pop)

        # Simulated Binary Crossover
        offspring = sbx(eta=20, problem=self.problem)(parents, pop)

        # Polynomial Mutation
        x = pm(eta=20, problem=self.problem, prob=1 / self.problem.n_var)(offspring)

        # Evaluate the new individual
        x_f = self.problem.evaluate(x)

        return {'X': [list(x)], 'F': [x_f]}

    def _initialize_pop(self):
        x, x_f = [], []
        n_eval = 0

        # Generates the initial population and evaluations
        for _ in range(self.pop_size):
            ind = []
            for i in range(self.problem.n_var):
                value = np.random.rand()
                while value < self.problem.xl[i] or value > self.problem.xu[i]:
                    value = np.random.rand()
                ind.append(value)
            x.append(ind)
            x_f.append(self.problem.evaluate(ind))
            n_eval += 1

        return {'X': x, 'F': x_f}, n_eval

    def _reduce(self, pop, q):
        n_pop = {}

        # Join the population and the new individual
        for key in q.keys():
            n_pop[key] = pop[key] + q[key]

        # Fast-non-dominated sorting
        fronts = self._fast_non_dominated_sorting(n_pop)

        # Obtain the index of the element with the least contribution
        if len(fronts) - 1 > 1:
            f = fronts[f'F{len(fronts) - 1}']
            if len(f) < 4:
                r = np.random.choice(f)
            else:
                r = self._min_contribution(f, n_pop['F'])
        else:
            r = self._min_contribution(fronts['F1'], n_pop['F'])

        # Eliminates the r element of the population
        for key in n_pop.keys():
            n_pop[key].pop(r)

        return n_pop

    def _do(self):
        c = 0
        nds = {}
        hv_history = []

        # Initialize population
        pop, n_eval = self._initialize_pop()

        txt_pop = [list(np.append(pop['X'][i], pop['F'][i])) for i in range(len(pop['X']))]
        np.savetxt(fname=f'../LastOne/SMS/Experiment{self.exp}/pop_lambda{self.lam}_c{n_eval}.txt',
                   X=txt_pop, delimiter=',')

        # Run until the termination condition is fulfilled
        while c < self.n_gen:
            # Generate a new individual
            q = self._new_individual(pop)
            # Update the population
            pop = self._reduce(pop, q)

            # Obtain the nds list
            nds_index = self._non_dominated_samples(pop)

            # Determine the nds per population
            for key in pop.keys():
                nds[key] = [pop[key][index] for index in nds_index]

            # Obtain the total hv of the nds
            nds_hv_value = np.around(self.total_hv(np.array(nds['F'])), 7)

            if (c + 1) % 100 == 0:
                txt_pop = [list(np.append(pop['X'][i], pop['F'][i])) for i in range(len(pop['X']))]
                np.savetxt(fname=f'../LastOne/SMS/Experiment{self.exp}/pop_lambda{self.lam}_c{n_eval + 1}.txt',
                           X=txt_pop, delimiter=',')

            n_eval += 1
            c += 1

            if n_eval % 500 == 0:
                hv_history.append([n_eval, nds_hv_value])

            if self.verbose:
                if c == 1:
                    print('     n_gen     | n_evaluations |      nds      |    hv_value   ')
                    print('---------------------------------------------------------------')
                if c % 100 == 0:
                    s1 = (9 - len(str(c))) * ' ' + str(c) + 6 * ' '
                    s2 = (9 - len(str(n_eval))) * ' ' + str(n_eval) + 6 * ' '
                    s3 = (9 - len(str(len(nds_index)))) * ' ' + str(len(nds_index)) + 6 * ' '
                    s4 = (12 - len(str(nds_hv_value))) * ' ' + str(nds_hv_value) + 3 * ' '
                    print(s1 + '|' + s2 + '|' + s3 + '|' + s4)

        return pop, nds, hv_history

    def __call__(self):
        return self._do()
