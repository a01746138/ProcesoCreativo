# NSGA-III implementation in python
# ==========================================================

import numpy as np
from GeneticOperators import TournamentSelection as ts
from GeneticOperators import SimulatedBinaryCrossover as sbx
from GeneticOperators import PolynomialMutation as pm
from PerformanceIndicators import Hypervolume as hv


class NSGA3:

    def __init__(self,
                 pop_size=100,
                 n_gen=1000,
                 problem=None,
                 verbose=False
                 ):
        self.problem = problem  # Problem to be optimized
        self.pop_size = pop_size  # Population size
        self.n_gen = n_gen  # Number of generations
        self.verbose = verbose  # Print the results so far

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
        hypervolume = hv(nadir_point)  # Create the reference space

        # Obtain total Hypervolume value
        total_hv = hypervolume(front)

        return total_hv

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

    def _generate_q(self, pop):
        x = []
        x_f = []
        for _ in range(self.pop_size):
            q = self._new_individual(pop)
            x.append(q['X'][0])
            x_f.append(q['F'][0])

        return {'X': x, 'F': x_f}

    def _weights_vector(self):
        w_v = {}

        # Generates the weights vector
        for i in range(self.pop_size):
            w1 = (self.pop_size - 1 - i) / (self.pop_size - 1)
            w2 = i / (self.pop_size - 1)
            w_v[f'w{i}'] = [w1, w2]

        return w_v

    def _normalize(self, s, pop):

        # Obtain the min-max values of each objective value in s
        min_obj = [np.inf, np.inf]
        max_obj = [-np.inf, -np.inf]
        for index in s:
            for obj in range(self.problem.n_obj):
                if pop[index][obj] < min_obj[obj]:
                    min_obj[obj] = pop[index][obj]
                if pop[index][obj] > max_obj[obj]:
                    max_obj[obj] = pop[index][obj]

        # Normalize the values of the population
        normalized = []
        for index in s:
            ind = []
            for obj in range(self.problem.n_obj):
                ind.append((pop[index][obj] - min_obj[obj]) / (max_obj[obj] - min_obj[obj]))
            ind.append(index)
            normalized.append(ind)

        return normalized

    def _associate(self, norm):
        a = {}
        for index in range(len(norm)):
            ind = norm[index][:self.problem.n_obj]
            d_min = np.inf
            w_min = []
            for w_index in range(len(self.ref_points)):
                w = self.ref_points[f'w{w_index}']
                d = abs((ind[0] * w[1]) - (w[0] * ind[1])) / np.sqrt(w[0] ** 2 + w[1] ** 2)
                if d < d_min:
                    d_min = d
                    w_min = f'w{w_index}'
            a[f'{norm[index][2]}'] = [d_min, w_min]

        return a

    def _niching(self, l_front, niche, a, pop_index):
        k = self.pop_size - len(pop_index)
        count = 0
        flag = True
        while flag:
            l_ref = list(np.unique([a[f'{index}'][1] for index in l_front]))
            empty_ref = [key for key in niche.keys() if niche[key] == 0]
            if len(empty_ref) > 0:
                for w in empty_ref:
                    if w in l_ref:
                        d_min = np.inf
                        closer = None
                        for i in l_front:
                            if a[f'{i}'][1] == w:
                                if a[f'{i}'][0] < d_min:
                                    d_min = a[f'{i}'][0]
                                    closer = i
                        pop_index.append(closer)
                        l_front.remove(closer)
                        niche[w] += 1
                        count += 1
                    else:
                        niche.pop(w)
                    empty_ref.remove(w)

                    if count >= k:
                        flag = False
                        break
            else:
                for w in l_ref:
                    d_min = np.inf
                    closer = None
                    for i in l_front:
                        if a[f'{i}'][1] == w:
                            if a[f'{i}'][0] < d_min:
                                d_min = a[f'{i}'][0]
                                closer = i
                    pop_index.append(closer)
                    l_front.remove(closer)
                    niche[w] += 1
                    count += 1

                    if count >= k:
                        flag = False
                        break

        return pop_index

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

    def _do(self):
        c = 0
        nds = {}
        hv_history = []
        self.ref_points = self._weights_vector()

        # Initialize population
        pop, n_eval = self._initialize_pop()

        while c < self.n_gen:

            # Generate Q
            q_pop = self._generate_q(pop)

            # Union between P and Q
            r_pop = {}
            for key in pop.keys():
                r_pop[key] = pop[key] + q_pop[key]

            # Fast-non-dominated sorting
            fronts = self._fast_non_dominated_sorting(r_pop)

            # Select the first l fronts until the size of s is equal or bigger than the pop_size
            s, f = [], 0
            while len(s) < self.pop_size:
                f += 1
                for x in fronts[f'F{f}']:
                    s.append(x)

            # Review the size of s to determine the next population
            if len(s) == self.pop_size:
                for key in r_pop.keys():
                    pop[key] = [r_pop[key][index] for index in s]
            else:
                pop_index = [item for j in range(1, f) for item in fronts[f'F{j}']]
                last_front = fronts[f'F{f}']

                # Normalize the s individuals objective values
                normal = self._normalize(s, r_pop['F'])

                # Associate each member of s with a reference point
                a = self._associate(normal)

                # Compute niche count of reference point
                niche_c = {}
                for key in self.ref_points:
                    niche_c[key] = 0
                for index in pop_index:
                    niche_c[a[f'{index}'][1]] += 1

                # Determine the pop_size elements for the new population
                pop_index = self._niching(last_front, niche_c, a, pop_index)

                for key in pop.keys():
                    pop[key] = [r_pop[key][i] for i in pop_index]

            # Obtain the nds list
            nds_index = self._non_dominated_samples(pop)

            # Determine the nds per population
            for key in pop.keys():
                nds[key] = [pop[key][index] for index in nds_index]

            # Obtain the total hv of the nds
            nds_hv_value = np.around(self._nds_hv(nds), 7)

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

        return pop, nds, hv_history

    def __call__(self):
        return self._do()
