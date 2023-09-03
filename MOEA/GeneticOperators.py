import math
import numpy as np
from PerformanceIndicators import Dominance


class TournamentSelection:
    def __init__(self,
                 n_select=1,
                 n_parents=1,
                 pressure=2):
        self.n_select = n_select  # number of tournaments
        self.n_parents = n_parents  # number of parents
        self.pressure = pressure  # rate of convergence
        self.n_random = n_select * n_parents * pressure  # number of random individuals needed

    def random_permutations(self, length, concat=True):
        p = []
        for i in range(self.n_perms):
            p.append(np.random.permutation(length))
        if concat:
            p = np.concatenate(p)  # from matrix to vector
        return p

    def _do(self, pop):

        # number of permutations needed
        self.n_perms = math.ceil(self.n_random / len(pop['X']))

        # get random permutations and reshape them
        p = self.random_permutations(len(pop['X']))[:self.n_random]
        p = np.reshape(p, (self.n_select * self.n_parents, self.pressure))

        # compare using tournament function
        n_tournaments, _ = p.shape
        s = np.full(n_tournaments, np.nan)

        for i in range(n_tournaments):
            a, b = p[i, 0], p[i, 1]
            a_f, b_f = pop['F'][a], pop['F'][b]

            # if one dominates another choose the nds one
            rel = Dominance.get_relation(a_f, b_f)
            if rel == 1:
                s[i] = a
            elif rel == -1:
                s[i] = b

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(s[i]):
                s[i] = np.random.choice([a, b])

        return s[:, None].astype(int, copy=False)

    def __call__(self, pop):
        return self._do(pop)


class SimulatedBinaryCrossover:

    def __init__(self,
                 problem,
                 eta=2,
                 prob=0.9):
        self.problem = problem
        self.eta = eta
        self.prob = prob

    def _beta(self):
        u = np.random.uniform(0, 1, size=(self.problem.n_var,))
        betas = []
        for auc in u:
            if auc > 0.5:
                auc = auc - 0.5
                b = (1 / (1 - 2 * auc)) ** (1 / (self.eta + 1))
                betas.append(b)
            elif auc < 0.5:
                b = (2 * auc) ** (1 / (self.eta + 1))
                betas.append(b)
            else:
                betas.append(1.0)
        return betas

    def _repair(self, x):
        for i in range(len(x)):
            if x[i] < self.problem.xl[i]:
                x[i] = self.problem.xl[i]
            elif x[i] > self.problem.xu[i]:
                x[i] = self.problem.xu[i]
        return x

    def _do(self, i_par, pop):
        p = np.random.uniform(0, 1, size=(self.problem.n_var,))
        beta = self._beta()
        parent1 = pop['X'][i_par[0][0]]
        parent2 = pop['X'][i_par[1][0]]
        off = np.full(self.problem.n_var, np.nan)

        for i in range(self.problem.n_var):
            if p[i] <= self.prob:
                b = beta[i]
                mean = (parent1[i] + parent2[i]) / 2
                c1 = mean - (b * (parent1[i] - parent2[i])) / 2
                c2 = mean + (b * (parent1[i] - parent2[i])) / 2
            else:
                c1 = parent1[i]
                c2 = parent2[i]

            off[i] = np.random.choice([c1, c2])

        off = self._repair(off)

        return off

    def __call__(self, parents, pop):
        return self._do(i_par=parents, pop=pop)


class PolynomialMutation:

    def __init__(self,
                 problem=None,
                 prob=0.1,
                 eta=2):
        self.problem = problem
        self.eta = eta
        self.prob = prob

    def _delta(self):
        u = np.random.uniform(0, 1, size=(self.problem.n_var,))
        deltas = []
        for auc in u:
            if auc > 0.5:
                auc = auc - 0.5
                d = 1 - (1 - 2 * auc) ** (1 / (self.eta + 1))
                deltas.append(d)
            elif auc < 0.5:
                d = (2 * auc) ** (1 / (self.eta + 1)) - 1
                deltas.append(d)
            else:
                deltas.append(0.0)
        return deltas

    def _repair(self, x):
        for i in range(len(x)):
            if x[i] < self.problem.xl[i]:
                x[i] = self.problem.xl[i]
            elif x[i] > self.problem.xu[i]:
                x[i] = self.problem.xu[i]
        return x

    def _do(self, parent):
        p = np.random.uniform(0, 1, size=(self.problem.n_var,))
        delta = self._delta()

        off = np.full(self.problem.n_var, np.nan)

        for i in range(self.problem.n_var):
            if p[i] <= self.prob:
                delta_max = self.problem.xu[i] - self.problem.xl[i]
                off[i] = parent[i] + delta[i] * delta_max
            else:
                off[i] = parent[i]

        off = self._repair(off)

        return off

    def __call__(self, parent):
        return self._do(parent)
