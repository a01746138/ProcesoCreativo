# Parameter-dependent IMIA implementation in python
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
                 exp,
                 indicators=None,
                 pop_size=100,
                 n_gen=1000,
                 n_mig=1,
                 n_lam=10,
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
        self.n_lam = n_lam
        self.exp = exp

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
        n_eval = 0
        initial_pop = {}

        # Generates the initial population and evaluations
        for lam in range(self.n_lam):
            x, x_f = [], []
            for _ in range(self.pop_size):
                ind = []
                for i in range(self.problem.n_var):
                    value = np.random.rand()
                    while value < self.problem.xl[i] or value > self.problem.xu[i]:
                        value = np.random.rand()
                    ind.append(value)
                x.append(ind)
                x_f.append(self.problem.evaluate(ind, lam/(self.n_lam - 1)))
                n_eval += 1
            initial_pop[f'lam{lam}'] = {'X': x, 'F': x_f}

        return initial_pop, n_eval

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

    def _new_individual(self, pop, lam):

        # Tournament Selection
        parents = ts(n_parents=2)(pop=pop)

        # Simulated Binary Crossover
        offspring = sbx(eta=20, problem=self.problem)(parents, pop)

        # Polynomial Mutation
        x = pm(eta=20, problem=self.problem, prob=1 / self.problem.n_var)(offspring)

        # Evaluate the new individual
        x_f = self.problem.evaluate(x, lam/(self.n_lam - 1))

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
            a[key] = [list(a[key][index, :]) for index in range(len(a[key]))]

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

    def _ib_moea(self, pop, a, indicator, lam):

        g = 0
        while g < self.mig_freq:
            # Generate a new individual
            q = self._new_individual(pop, lam)

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
        for lam in range(self.n_lam):
            for i in range(len(self.indicators)):
                for j in range(len(self.indicators)):
                    if i != j:
                        mig = {'X': [], 'F': []}
                        for k in range(self.n_mig):
                            index = np.random.randint(0, self.mig_freq)
                            lamb = np.random.randint(0, self.n_lam)
                            mig['X'] = mig['X'] + [pop[f'lam{lamb}_i{i}']['X'][index]]
                            mig['F'] = mig['F'] + [self.problem.evaluate(mig['X'][k], lam/(self.n_lam - 1))]
                        migrants[f'{i}_{lam}_{j}'] = mig

        for lam in range(self.n_lam):
            for i in range(len(self.indicators)):
                t = 0
                while t < (len(self.indicators) - 1) * self.n_mig:
                    # Fast-non-dominated sorting
                    fronts = self._fast_non_dominated_sorting(pop[f'lam{lam}_i{i}'])

                    # Obtain the index of the element with the least contribution
                    if len(fronts) - 1 > 1:
                        f = fronts[f'F{len(fronts) - 1}']
                        if len(f) < 4:
                            r = np.random.choice(f)
                        else:
                            r = self._min_contribution(f, pop[f'lam{lam}_i{i}']['F'],
                                                       self.indicators[i], a[f'lam{lam}_i{i}'])
                    else:
                        r = self._min_contribution(fronts['F1'], pop[f'lam{lam}_i{i}']['F'],
                                                   self.indicators[i], a[f'lam{lam}_i{i}'])

                    # Eliminates the r element of the population
                    for key in pop[f'lam{lam}_i{i}'].keys():
                        pop[f'lam{lam}_i{i}'][key].pop(r)

                    t += 1

                # Insert migrants to the population
                for j in range(len(self.indicators)):
                    if i != j:
                        q = migrants[f'{j}_{lam}_{i}']

                        for key in pop[f'lam{lam}_i{i}'].keys():
                            pop[f'lam{lam}_i{i}'][key] = pop[f'lam{lam}_i{i}'][key] + q[key]

                        a[f'lam{lam}_i{i}'] = self._insert(a[f'lam{lam}_i{i}'], q)

        return pop, a

    def _do(self):
        c = 0
        total_pop_lam = {}
        nds_lam = {}
        hv_history = []

        # Initialize population
        initial_pop, n_eval = self._initialize_pop()

        for lam in range(self.n_lam):
            txt_pop = [list(np.append(initial_pop[f'lam{lam}']['X'][i], initial_pop[f'lam{lam}']['F'][i])) for i in range(len(initial_pop[f'lam{lam}']['X']))]
            np.savetxt(fname=f'../LastOne/PIMIA/Experiment{self.exp}/pop_lambda{lam}_c{int(n_eval / self.n_lam)}.txt',
                       X=txt_pop, delimiter=',')

        # Initialize indicator-island populations and nds
        a, pop = {}, {}
        for lam in range(self.n_lam):
            for i in range(len(self.indicators)):
                a[f'lam{lam}_i{i}'] = {}
                pop[f'lam{lam}_i{i}'] = {}

        # Insert the individuals to the dictionaries
        for lam in range(self.n_lam):
            for i in range(len(self.indicators)):
                li = i * self.mig_freq
                ui = (i + 1) * self.mig_freq
                for key in initial_pop[f'lam{lam}'].keys():
                    pop[f'lam{lam}_i{i}'][key] = initial_pop[f'lam{lam}'][key][li:ui]

                # Obtain the nds of each population
                a_index = self._non_dominated_samples(pop[f'lam{lam}_i{i}'])
                for key in pop[f'lam{lam}_i{i}'].keys():
                    a[f'lam{lam}_i{i}'][key] = [pop[f'lam{lam}_i{i}'][key][index] for index in a_index]

        # Run until the termination condition is fulfilled
        while c < self.n_gen:
            for lam in range(self.n_lam):
                for index in range(len(self.indicators)):
                    indicator = self.indicators[index]
                    pop[f'lam{lam}_i{index}'], \
                        a[f'lam{lam}_i{index}'] = self._ib_moea(pop[f'lam{lam}_i{index}'],
                                                                a[f'lam{lam}_i{index}'],
                                                                indicator, lam)

            # Migration of random individuals between islands
            pop, a = self._migration(pop, a)

            # Join island populations to identify the nds
            total_pop_lam = {}
            for lam in range(self.n_lam):
                total_pop = {'X': [], 'F': []}
                for i in range(len(self.indicators)):
                    for key in pop[f'lam{lam}_i{i}'].keys():
                        total_pop[key] += pop[f'lam{lam}_i{i}'][key]
                total_pop_lam[f'lam{lam}'] = total_pop

            nds_hv_value = []
            nds_number = 0
            nds_hv_sum = 0
            nds_lam = {}
            for lam in range(self.n_lam):
                # Obtain the nds list
                nds_index = self._non_dominated_samples(total_pop_lam[f'lam{lam}'])
                nds_number += len(nds_index)

                # Determine the nds per population
                nds = {}
                for key in total_pop_lam[f'lam{lam}'].keys():
                    nds[key] = [total_pop_lam[f'lam{lam}'][key][index] for index in nds_index]
                nds_lam[f'lam{lam}'] = nds

                # Obtain the total hv of the nds
                hv_value = np.around(self.total_hv(np.array(nds['F'])), 7)
                nds_hv_value.append(hv_value)
                nds_hv_sum += hv_value

            n_eval += self.pop_size * self.n_lam + self.n_lam * len(self.indicators) * (len(self.indicators) - 1)
            c += 1

            for lam in range(self.n_lam):
                txt_pop = [list(np.append(total_pop_lam[f'lam{lam}']['X'][i], total_pop_lam[f'lam{lam}']['F'][i])) for i in
                           range(len(total_pop_lam[f'lam{lam}']['X']))]
                np.savetxt(fname=f'../LastOne/PIMIA/Experiment{self.exp}/pop_lambda{lam}_c{int(n_eval / self.n_lam)}.txt',
                           X=txt_pop, delimiter=',')

            if n_eval % (5800 + (c // 5) * 6000) == 0:
                hv_history.append([n_eval] + nds_hv_value)
            if self.verbose:
                if c == 1:
                    print('     n_gen     | n_evaluations |      nds      |    hv_value   ')
                    print('---------------------------------------------------------------')
                if c % 1 == 0:
                    s1 = (9 - len(str(c))) * ' ' + str(c) + 6 * ' '
                    s2 = (9 - len(str(n_eval))) * ' ' + str(n_eval) + 6 * ' '
                    s3 = (9 - len(str(nds_number))) * ' ' + str(nds_number) + 6 * ' '
                    s4 = (12 - len(str(np.around(nds_hv_sum, 7)))) * ' ' + str(np.around(nds_hv_sum, 7)) + 3 * ' '
                    print(s1 + '|' + s2 + '|' + s3 + '|' + s4)

        return total_pop_lam, nds_lam, hv_history

    def __call__(self):
        return self._do()
