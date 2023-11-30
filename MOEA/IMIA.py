# IMIA implementation in python
# ==========================================================

import numpy as np
from GeneticOperators import TournamentSelection as ts
from GeneticOperators import SimulatedBinaryCrossover as sbx
from GeneticOperators import PolynomialMutation as pm
from PerformanceIndicators import Hypervolume, R2, EpsPlus, DeltaP, IGDPlus2, RieszEnergy
from PerformanceIndicators import IndividualContribution as ic
from pymoo.indicators.hv import HV


class IMIA:

    def __init__(self,
                 problem_data,
                 indicators=None,
                 pop_size=100,
                 n_gen=1000,
                 n_mig=1,
                 problem=None,
                 verbose=False,
                 ):
        if indicators is None:
            indicators = ['HV', 'R2', 'EpsPlus', 'DeltaP', 'IGDPlus']

        self.problem = problem  # Problem to be optimized
        self.pop_size = pop_size  # Population size
        self.n_gen = n_gen  # Number of generations
        self.verbose = verbose  # Print the results so far
        self.n_mig = n_mig  # Number of individuals to migrate
        self.mig_freq = int(pop_size / len(indicators))  # Migration frequency
        self.indicators = indicators
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
    def _nds_hv(nds):
        # Sort every solution
        front = np.array(nds['F'])
        front = front[front[:, 0].argsort()]

        # Create reference point
        nadir_point = np.array([1, 0])
        hypervolume = Hypervolume(nadir_point)  # Create the reference space

        # Obtain total Hypervolume value
        total_hv = hypervolume(front)

        return total_hv

    @staticmethod
    def _min_contribution(f, n_pop, indicator, a):

        # Sort every solution
        front = np.array([n_pop[index] for index in f])
        front = np.c_[front, f]
        front = front[front[:, 0].argsort()]
        r = None

        # Hypervolume indicator
        if indicator == 'HV':
            # Create reference point
            nadir_point = front.max(axis=0)[:2]
            hypervolume = Hypervolume(nadir_point)  # Create the reference space

            # Obtain total contribution value
            total_contribution = hypervolume(front)

            # Obtain the contribution of every individual
            min_contribution = np.inf
            for ind in range(1, len(front) - 1):
                ind_contribution = ic()(hypervolume, total_contribution, front, ind)
                if ind_contribution < min_contribution:
                    min_contribution = ind_contribution
                    r = ind

        # R2 indicator
        elif indicator == 'R2':
            utopian_point = front.min(axis=0)[:2]
            r2 = R2(utopian_point=utopian_point)

            # Obtain total contribution value
            total_contribution = r2(front[:, :2])

            # Obtain the contribution of every individual
            min_contribution = np.inf
            for ind in range(len(front)):
                ind_contribution = ic()(r2, total_contribution, front[:, :2], ind)
                if ind_contribution < min_contribution:
                    min_contribution = ind_contribution
                    r = ind

        # Epsilon Plus indicator
        elif indicator == 'EpsPlus':
            eps = EpsPlus(nds=np.array(a['F']))

            # Obtain total contribution value
            total_contribution = eps(front[:, :2])

            # Obtain the contribution of every individual
            min_contribution = np.inf
            for ind in range(len(front)):
                ind_contribution = ic()(eps, total_contribution, front[:, :2], ind)
                if ind_contribution < min_contribution:
                    min_contribution = ind_contribution
                    r = ind

        # Delta P indicator
        elif indicator == 'DeltaP':
            delta_p = DeltaP(nds=np.array(a['F']))

            # Obtain total contribution value
            total_contribution = delta_p(front[:, :2])

            # Obtain the contribution of every individual
            min_contribution = np.inf
            for ind in range(len(front)):
                ind_contribution = ic()(delta_p, total_contribution, front[:, :2], ind)
                if ind_contribution < min_contribution:
                    min_contribution = ind_contribution
                    r = ind

        # IGD Plus indicator
        elif indicator == 'IGDPlus':
            igd = IGDPlus2(nds=np.array(a['F']))

            # Obtain total contribution value
            total_contribution = igd(front[:, :2])

            # Obtain the contribution of every individual
            min_contribution = np.inf
            for ind in range(len(front)):
                ind_contribution = ic()(igd, total_contribution, front[:, :2], ind)
                if ind_contribution < min_contribution:
                    min_contribution = ind_contribution
                    r = ind

        return int(front[r][2])

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

    def _insert(self, a, q):
        delete_list = []
        for i in range(len(a['F'])):
            p = a['F'][i]
            if self._dominate(q['F'][0], p):
                delete_list.append(i)
            else:
                return a

        for key in a.keys():
            a[key] += q[key]
            a[key] = np.delete(arr=np.array(a[key]), obj=delete_list, axis=0)
            a[key] = [list(a[key][index, :])for index in range(len(a[key]))]

        while len(a['F']) > self.pop_size:
            r = None
            r_energy = RieszEnergy()

            # Obtain total contribution value
            total_contribution = r_energy(np.array(a['F']))

            # Obtain the contribution of every individual
            min_contribution = np.inf
            for ind in range(len(a['F'])):
                ind_contribution = ic()(r_energy, total_contribution, a['F'], ind, half=True)
                if ind_contribution < min_contribution:
                    min_contribution = ind_contribution
                    r = ind

            for key in a.keys():
                a[key].pop(r)

        return a

    def _ib_moea(self, pop, a, indicator):

        g = 0
        while g < self.mig_freq:
            # Generate a new individual
            q = self._new_individual(pop)

            for key in pop.keys():
                pop[key] = pop[key] + q[key]

            # Fast-non-dominated sorting
            fronts = self._fast_non_dominated_sorting(pop)

            # Obtain the index of the element with the least contribution
            if len(fronts) - 1 > 1:
                f = fronts[f'F{len(fronts) - 1}']
                if len(f) < 4:
                    r = np.random.choice(f)
                else:
                    r = self._min_contribution(f, pop['F'], indicator, a)
            else:
                r = self._min_contribution(fronts['F1'], pop['F'], indicator, a)

            # Insert the new individual if not the worst
            if r != self.mig_freq:
                a = self._insert(a, q)

            # Eliminates the r element of the population
            for key in pop.keys():
                pop[key].pop(r)
            g += 1

        return pop, a

    def _migration(self, pop, a):
        migrants = {}
        for i in range(len(self.indicators)):
            p = pop[f'{i}']
            for j in range(len(self.indicators)):
                if i != j:
                    mig = {'X': [], 'F': []}
                    for _ in range(self.n_mig):
                        index = np.random.randint(0, self.mig_freq)
                        for key in p.keys():
                            mig[key] = mig[key] + [p[key][index]]
                    migrants[f'{i}_{j}'] = mig

        for i in range(len(self.indicators)):
            t = 0
            while t < (len(self.indicators) - 1) * self.n_mig:
                # Fast-non-dominated sorting
                fronts = self._fast_non_dominated_sorting(pop[f'{i}'])

                # Obtain the index of the element with the least contribution
                if len(fronts) - 1 > 1:
                    f = fronts[f'F{len(fronts) - 1}']
                    if len(f) < 4:
                        r = np.random.choice(f)
                    else:
                        r = self._min_contribution(f, pop[f'{i}']['F'], self.indicators[i], a[f'{i}'])
                else:
                    r = self._min_contribution(fronts['F1'], pop[f'{i}']['F'], self.indicators[i], a[f'{i}'])

                # Eliminates the r element of the population
                for key in pop[f'{i}'].keys():
                    pop[f'{i}'][key].pop(r)

                t += 1

            # Insert migrants to the population
            for j in range(len(self.indicators)):
                if i != j:
                    q = migrants[f'{j}_{i}']

                    for key in pop[f'{i}'].keys():
                        pop[f'{i}'][key] = pop[f'{i}'][key] + q[key]

                    a[f'{i}'] = self._insert(a[f'{i}'], q)

        return pop, a

    def _do(self):
        c = 0
        total_pop = {}
        nds = {}
        hv_history = []

        # Initialize population
        initial_pop, n_eval = self._initialize_pop()

        txt_pop = [list(np.append(initial_pop['X'][i], initial_pop['F'][i])) for i in range(len(initial_pop['X']))]
        np.savetxt(fname=f'../LastOne/IMIA/Experiment{self.exp}/pop_lambda{self.lam}_c{n_eval}.txt',
                   X=txt_pop, delimiter=',')

        # Initialize indicator-island populations and nds
        a, pop = {}, {}
        for i in range(len(self.indicators)):
            a[f'{i}'] = {}
            pop[f'{i}'] = {}

        # Insert the individuals to the dictionaries
        for i in range(len(self.indicators)):
            li = i * self.mig_freq
            ui = (i + 1) * self.mig_freq
            for key in initial_pop.keys():
                pop[f'{i}'][key] = initial_pop[key][li:ui]

            # Obtain the nds of each population
            a_index = self._non_dominated_samples(pop[f'{i}'])
            for key in pop[f'{i}'].keys():
                a[f'{i}'][key] = [pop[f'{i}'][key][index] for index in a_index]

        # Run until the termination condition is fulfilled
        while c < self.n_gen:
            for index in range(len(self.indicators)):
                indicator = self.indicators[index]
                pop[f'{index}'], a[f'{index}'] = self._ib_moea(pop[f'{index}'], a[f'{index}'], indicator)

            # Migration of random individuals between islands
            pop, a = self._migration(pop, a)

            # Join island populations to identify the nds
            total_pop = {'X': [], 'F': []}
            for key in pop.keys():
                for key2 in pop[key].keys():
                    total_pop[key2] += pop[key][key2]

            # Obtain the nds list
            nds_index = self._non_dominated_samples(total_pop)

            # Determine the nds per population
            for key in total_pop.keys():
                nds[key] = [total_pop[key][index] for index in nds_index]

            # Obtain the total hv of the nds
            nds_hv_value = np.around(self.total_hv(np.array(nds['F'])), 7)

            txt_pop = [list(np.append(total_pop['X'][i], total_pop['F'][i])) for i in range(len(total_pop['X']))]
            np.savetxt(fname=f'../LastOne/IMIA/Experiment{self.exp}/pop_lambda{self.lam}_c{n_eval}.txt',
                       X=txt_pop, delimiter=',')

            n_eval += self.pop_size
            c += 1

            if n_eval % 500 == 0:
                hv_history.append([n_eval, nds_hv_value])

            if self.verbose:
                if c == 1:
                    print('     n_gen     | n_evaluations |      nds      |    hv_value   ')
                    print('---------------------------------------------------------------')
                if c % 1 == 0:
                    s1 = (9 - len(str(c))) * ' ' + str(c) + 6 * ' '
                    s2 = (9 - len(str(n_eval))) * ' ' + str(n_eval) + 6 * ' '
                    s3 = (9 - len(str(len(nds_index)))) * ' ' + str(len(nds_index)) + 6 * ' '
                    s4 = (12 - len(str(nds_hv_value))) * ' ' + str(nds_hv_value) + 3 * ' '
                    print(s1 + '|' + s2 + '|' + s3 + '|' + s4)

        return total_pop, nds, hv_history

    def __call__(self):
        return self._do()
