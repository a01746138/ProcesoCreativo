from joblib import load
import numpy as np


class MyPMOP:

    def __init__(self,
                 n_var=5,
                 n_obj=2,
                 xl=None,
                 xu=None):

        self.n_var = n_var
        self.n_obj = n_obj

        if xu is None:
            self.xu = [1, 1, 1, 1, 1]
        if xl is None:
            self.xl = [0.4879, 0, 0, 0, 0]

    @staticmethod
    def evaluate(x, lam):

        h_model = load(filename='../Surrogate/model_h_ann.joblib',
                       mmap_mode='r')
        mech_model = load(filename='../Surrogate/model_mech_dtr.joblib',
                          mmap_mode='r')

        xx = np.reshape([x[0], x[1], x[2], x[3], x[4], lam], (1, -1))

        f1 = h_model.predict(xx)[0]
        f2 = -1 * mech_model.predict(xx)[0]

        return [f1, f2]
