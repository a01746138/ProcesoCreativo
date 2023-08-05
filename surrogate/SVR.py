import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVR
from ProcessingData import normalization
from ReadExperiments import separate

seeD = 1827
path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\surrogate\\'
a, b, c = separate()
df = normalization(a)


def create_model_svr(param_grid, x, y):
    base_estimator = SVR()
    sh = HalvingGridSearchCV(estimator=base_estimator, param_grid=param_grid,
                             cv=10, max_resources=30, n_jobs=-1,
                             scoring='neg_mean_squared_error').fit(x, y)
    params = sh.best_params_

    return sh.best_estimator_, params


def grid_search_svr(x, y):
    np.random.seed(seeD)
    param_grid_svr = {'kernel': ['linear', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto'],
                      'tol': np.random.uniform(1e-4, 1e-2, 20),
                      'epsilon': np.random.uniform(1e-3, 1e-1, 20),
                      'C': np.random.uniform(0, 1, 10)}
    svr_model, params_svr = create_model_svr(param_grid=param_grid_svr, x=x, y=y)

    return svr_model, params_svr


svr_h_model, params_svr_h = grid_search_svr(x=df.drop(['Hcons', 'Pmech'], axis=1), y=df['Hcons'])
svr_mech_model, params_svr_mech = grid_search_svr(x=df.drop(['Hcons', 'Pmech'], axis=1), y=df['Pmech'])

joblib.dump(filename=path + 'svr_h_model.joblib', value=svr_h_model)
joblib.dump(filename=path + 'svr_mech_model.joblib', value=svr_mech_model)
