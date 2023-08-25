from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.vendor.hv import HyperVolume as hv
import numpy as np


class Dominance:

    @staticmethod
    def get_relation(a, b):

        val = 0
        for i in range(len(a)):
            if a[i] < b[i]:
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1
        return val


class Hypervolume:
    def __init__(self,
                 nadir_point):
        self.nadir_point = nadir_point

    def _do(self, front):
        hv_value = 0
        for i in range(1, len(front) - 1):
            base = front[i + 1][0] - front[i][0]
            height = self.nadir_point[1] - front[i][1]
            hv_value += base * height

        return hv_value

    def __call__(self, front):
        return self._do(front)


class IndividualContribution:

    @staticmethod
    def _do(indicator, total_con, f, a, half=False):
        # Eliminate index a from the front
        n_f = np.delete(arr=f, obj=a, axis=0)
        res = total_con - indicator(n_f)

        # Divide by half the individual contribution
        if half:
            return res / 2
        else:
            return res

    def __call__(self, indicator, total_con, f, a):
        return self._do(indicator, total_con, f, a)


class EpsPlus:
    def __init__(self, pf):
        self.pf = pf

    def _do(self, F):

        val2 = -np.inf
        for z in self.pf:
            val1 = np.inf
            for a in F:
                val0 = a[0] - z[0]
                for i in range(1, len(a)):
                    dif = a[i] - z[i]
                    if dif > val0:
                        val0 = dif
                if val0 < val1:
                    val1 = val0
            if val1 > val2:
                val2 = val1

        return val2

    def __call__(self, F):
        return self._do(F)


class R2:
    def __init__(self, weights, unary_func):

        self.weights = weights
        self.unary_func = unary_func

    def __call__(self, F):
        return self._do(F)

    def _do(self, F):

        res = 0
        for w in self.weights:
            val0 = -np.inf
            for a in F:
                val = self.unary_func(a, w)
                if val > val0:
                    val0 = val
            res += val0

        return res[0][0] / len(self.weights)


class RieszEnergy:

    def __call__(self, F):
        return self._do(F)

    def _do(self, F):

        res = 0
        for i in range(len(F)):
            a = F[i]
            for b in np.delete(F, i, 0):
                euc_dist = 1 / (np.sqrt(np.sum((a - b) ** 2)) ** (len(a) ** 2))
                res += euc_dist

        return res


class DeltaP:
    def __init__(self, pf):
        self.pf = pf

    def __call__(self, F):
        return self._do(F)

    def _do(self, F):
        return max(GD(self.pf)(F), IGD(self.pf)(F))
