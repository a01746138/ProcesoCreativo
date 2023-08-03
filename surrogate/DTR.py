import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import DecisionTreeRegressor
from ProcessingData import normalization
from ReadExperiments import separate

seeD = 1827
path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\surrogate\\'
a, b, c = separate()
df = normalization(a)


def create_model_dtr(param_grid, x, y):
    base_estimator = DecisionTreeRegressor()
    sh = HalvingGridSearchCV(estimator=base_estimator, param_grid=param_grid,
                             cv=10, factor=2, max_resources=100, n_jobs=-1,
                             scoring='neg_mean_squared_error').fit(x, y)
    params = sh.best_params_

    return sh.best_estimator_, params


def grid_search_dtr(x, y):
    np.random.seed(seeD)
    param_grid_dt = {'criterion': ['squared_error', 'friedman_mse',
                                   'absolute_error', 'poisson'],
                     'splitter': ['best', 'random'],
                     'max_depth': np.random.randint(1, 1000, 20),
                     'ccp_alpha': np.random.uniform(0, 0.035, 20),
                     'min_samples_split': [2, 3]}
    dt_model, params_dt = create_model_dtr(param_grid=param_grid_dt, x=x, y=y)

    return dt_model, params_dt


dtr_h_model, params_dt_h = grid_search_dtr(x=df.drop(['Hcons', 'Pmech'], axis=1), y=df['Hcons'])
dtr_mech_model, params_dt_mech = grid_search_dtr(x=df.drop(['Hcons', 'Pmech'], axis=1), y=df['Pmech'])

joblib.dump(filename=path + 'dtr_h_model.joblib', value=dtr_h_model)
joblib.dump(filename=path + 'dtr_mech_model.joblib', value=dtr_mech_model)
