import numpy as np
import joblib
from pymoo.core.problem import ElementwiseProblem


class MyPMOP(ElementwiseProblem):

    def __init__(self, lambda_mass):
        super().__init__(n_var=5,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=np.array([0.4879, 0, 0, 0, 0]),
                         xu=np.array([1, 1, 1, 1, 1]))
        self.lm = lambda_mass

    def _evaluate(self, x, out, *args, **kwargs):
        path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\surrogate\\'

        h_model = joblib.load(filename=path + 'ann_h_model.joblib',
                              mmap_mode='r')
        mech_model = joblib.load(filename=path + 'ann_mech_model.joblib',
                                 mmap_mode='r')

        xx = np.reshape([x[0], x[1], x[2], x[3], x[4], self.lm], (1, -1))

        f1 = h_model.predict(xx)
        f2 = -1 * mech_model.predict(xx)

        out["F"] = [f1, f2]
        out["G"] = []
