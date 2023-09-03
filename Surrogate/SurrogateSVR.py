import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVR
from ExperimentsMATLAB.ProcessingData import normalization


data = pd.read_csv(filepath_or_buffer='../ExperimentsMATLAB/Data.csv')
df = normalization(data)


def create_model(param_grid, x, y):
    base_estimator = SVR(tol=1e-5)
    sh = HalvingGridSearchCV(estimator=base_estimator, param_grid=param_grid,
                             cv=5, factor=3, n_jobs=-1,
                             scoring='neg_mean_squared_error').fit(x, y)

    return sh.best_estimator_, sh.best_params_


def grid_search(x, y):
    param_grid = {'C': [float(x) for x in range(1, 12, 2)],
                  'epsilon': [x * 1e-3 for x in range(10, 101, 10)]}
    model, params = create_model(param_grid=param_grid, x=x, y=y)

    return model, params


# model_h, params_h = grid_search(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
# print(model_h)
# print(params_h)
# joblib.dump(filename='model_h_svr.joblib', value=model_h)

model_mech, params_mech = grid_search(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))
print(model_mech)
print(params_mech)
joblib.dump(filename='model_mech_svr.joblib', value=model_mech)
