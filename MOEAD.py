# MOEA/D implementation in python
# ==========================================================
import numpy as np
from GeneticOperators import TournamentSelection as ts
from GeneticOperators import SimulatedBinaryCrossover as sbx
from GeneticOperators import PolynomialMutation as pm


class MOEAD:
    def __init__(self,
                 pop_size=100,
                 n_gen=1000,
                 n_neighbors=5,
                 problem=None,
                 verbose=False
                 ):
        self.problem = problem  # Problem to be optimized
        self.pop_size = pop_size  # Population size
        self.n_gen = n_gen  # Number of generations
        self.n_neighbors = n_neighbors  # Number of elements in the neighborhood
        self.verbose = verbose  # Print the results so far

    @staticmethod
    def dominate(p, q):
        flag = True

        # Dominates if every objective value of p is less than the one of q
        for i in range(len(p)):
            if p[i] >= q[i]:
                flag = False
                return flag
        return flag

    def _new_individual(self, index, pop):

        # Generate a new population based on the neighborhood of each index
        n_pop = {}
        for key in pop.keys():
            t = []
            for i in self.b[index]:
                t.append(pop[key][i])
            n_pop[key] = t

        # Tournament Selection
        parents = ts(n_parents=2)(pop=n_pop)

        # Simulated Binary Crossover
        offspring = sbx(eta=20, problem=self.problem)(parents, n_pop)

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

    def _weights_vector(self):
        w_v = {}

        # Generates the weights vector
        for i in range(self.pop_size):
            w1 = (self.pop_size - 1 - i) / (self.pop_size - 1)
            w2 = i / (self.pop_size - 1)
            w_v[f'w{i}'] = [w1, w2]

        return w_v

    def _neighborhoods(self):
        index = None
        b = {}
        for key in self.weights.keys():
            neg, pos = 0, 0
            w = int(key[1:])
            neighbors = []
            for i in range(1, self.n_neighbors + 1):
                index = int(np.floor((i + 1) / 2) * (-1) ** i)
                if 0 <= w + index <= self.pop_size - 1:
                    neighbors.append(w + index)
                elif w + index > self.pop_size - 1:
                    pos += 1
                elif w + index < 0:
                    neg += 1
            if neg > 0:
                if index > 0:  # if n_neighbors is pair
                    for j in range(index + 1, index + 1 + neg):
                        neighbors.append(w + j)
                else:
                    for j in range(abs(index), abs(index) + neg):
                        neighbors.append(w + j)
            elif pos > 0:
                for j in range(abs(index) + 1, abs(index) + 1 + pos):
                    neighbors.append(w - j)
            b[w] = neighbors

        return b

    def tchebycheff(self, x_f, z, w):
        max_value = -np.inf
        for i in range(self.problem.n_obj):
            value = w[i] * np.abs(x_f[i] - z[i])
            if value > max_value:
                max_value = value

        return max_value

    def _update(self, utopian_point, pop):

        for i in range(self.pop_size):

            # Create a new individual based on the neighborhoods
            q = self._new_individual(i, pop)

            # Update the utopian point in case of finding a better value for each objective
            for j in range(self.problem.n_obj):
                if q['F'][0][j] < utopian_point[j]:
                    utopian_point[j] = q['F'][0][j]

            # Substitute the individual in case the condition between Tchebycheff methods is fulfilled
            for j in self.b[i]:
                a = self.tchebycheff(q['F'][0], utopian_point, self.weights[f'w{j}'])
                b = self.tchebycheff(pop['F'][j], utopian_point, self.weights[f'w{j}'])
                if a < b:
                    for key in pop.keys():
                        pop[key][j] = q[key][0]

        # Count the non-dominated solutions
        front = {}
        nds = []
        for key in pop.keys():
            front[key] = list(np.unique(pop[key], axis=0))

        for i in range(len(front['F'])):
            p = front['F'][i]
            n = 0
            for j in range(len(front['F'])):
                q = front['F'][j]
                if self.dominate(q, p):
                    n += 1
            if n == 0:
                nds.append(i)

        return pop, nds, utopian_point

    def _do(self):
        c = 0
        nds_index = []

        # Initialize weights, neighborhoods, and population
        self.weights = self._weights_vector()
        self.b = self._neighborhoods()
        pop, n_eval = self._initialize_pop()

        utopian_point = np.min(pop['F'], axis=0)

        # Run until the termination condition is fulfilled
        while c < self.n_gen:
            pop, nds_index, utopian_point = self._update(utopian_point, pop)

            n_eval += self.pop_size
            c += 1

            if self.verbose:
                if c == 1:
                    print('     n_gen     | n_evaluations |      nds      ')
                    print('===============================================')
                if c % 1 == 0:
                    s1 = (9 - len(str(c))) * ' ' + str(c) + 6 * ' '
                    s2 = (9 - len(str(n_eval))) * ' ' + str(n_eval) + 6 * ' '
                    s3 = (9 - len(str(len(nds_index)))) * ' ' + str(len(nds_index)) + 6 * ' '
                    print(s1 + '|' + s2 + '|' + s3)

        nds = {}
        for key in pop.keys():
            nds[key] = [pop[key][index] for index in nds_index]

        return pop, nds

    def __call__(self):
        return self._do()
