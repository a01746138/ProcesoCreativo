from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
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


class IndividualContribution:

    @staticmethod
    def _do(indicator, total_con, f, a, half):
        # Eliminate index a from the front
        n_f = np.delete(arr=f, obj=a, axis=0)
        res = abs(total_con - indicator(n_f))

        # Divide by half the individual contribution
        if half:
            return res / 2
        else:
            return res

    def __call__(self, indicator, total_con, f, a, half=False):
        return self._do(indicator, total_con, f, a, half)


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


class EpsPlus:
    def __init__(self, nds):
        self.pf = nds

    def _do(self, front):

        val2 = -np.inf
        for z in self.pf:
            val1 = np.inf
            for a in front:
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

    def __call__(self, front):
        return self._do(front)


class R2:
    def __init__(self, utopian_point, theta=5):
        self.theta = theta
        self.utopian_point = utopian_point

    @staticmethod
    def _weights_vector(size):
        w_v = []

        # Generates the weights vector
        for i in range(size):
            w1 = (size - 1 - i) / (size - 1)
            w2 = i / (size - 1)
            w_v.append([w1, w2])

        return w_v

    def _do(self, front):
        weights = np.array(self._weights_vector(len(front)))
        norm_w = np.linalg.norm(weights)
        d1 = np.linalg.norm(np.matmul((front - self.utopian_point).T, weights)) / norm_w
        d2 = np.linalg.norm(front - (self.utopian_point + d1 * weights))

        return d1 + self.theta * d2

    def __call__(self, front):
        return self._do(front)


class DeltaP:
    def __init__(self, nds):
        self.pf = nds

    def _do(self, front):
        return max(GD(self.pf)(front), IGD(self.pf)(front))

    def __call__(self, front):
        return self._do(front)


class IGDPlus2:
    def __init__(self, nds):
        self.pf = nds

    def _do(self, front):
        return IGDPlus(self.pf)(front)

    def __call__(self, front):
        return self._do(front)


class RieszEnergy:

    @staticmethod
    def _do(front, constant):

        res = 0
        for i in range(len(front)):
            a = front[i]
            for b in np.delete(front, i, 0):
                euc_dist = 1 / (np.sqrt(np.sum((a - b) ** 2)) ** (len(a) ** 2))
                res += euc_dist * constant

        return res

    def __call__(self, front, constant=1e-10):
        return self._do(front, constant)
