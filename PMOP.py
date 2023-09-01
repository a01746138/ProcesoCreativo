import joblib
import numpy as np


class MyPMOP:

    def __init__(self,
                 n_var=5,
                 n_obj=2,
                 lambda_mass=None,
                 xl=None,
                 xu=None):

        self.n_var = n_var
        self.n_obj = n_obj

        if lambda_mass is None:
            self.lm = 0.0
        else:
            self.lm = lambda_mass

        if xu is None:
            self.xu = [1, 1, 1, 1, 1]
        if xl is None:
            self.xl = [0.4879, 0, 0, 0, 0]

    def evaluate(self, x):
        path = 'surrogate\\'

        h_model = joblib.load(filename=path + 'ann_h_model.joblib',
                              mmap_mode='r')
        mech_model = joblib.load(filename=path + 'svr_mech_model.joblib',
                                 mmap_mode='r')

        xx = np.reshape([x[0], x[1], x[2], x[3], x[4], self.lm], (1, -1))

        f1 = h_model.predict(xx)[0]
        f2 = -1 * mech_model.predict(xx)[0]

        return [f1, f2]
