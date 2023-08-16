import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from ProcessingData import normalization
from ReadExperiments import separate

seeD = 1827
path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\surrogate\\'
a, b, c = separate()
df = normalization(a)


def create_model_rfr(param_grid, x, y):
    base_estimator = RandomForestRegressor()
    sh = HalvingGridSearchCV(estimator=base_estimator,
                             param_grid=param_grid,
                             cv=5, factor=2, n_jobs=-1,
                             max_resources=250,
                             min_resources='exhaust',
                             aggressive_elimination=True,
                             scoring='neg_mean_squared_error').fit(x, y)
    params = sh.best_params_

    return sh.best_estimator_, params


def grid_search_rfr(x, y):
    np.random.seed(seeD)
    param_grid_rfr = {'criterion': ['squared_error', 'absolute_error', 'poisson'],
                      'n_estimators': np.random.randint(1, 10, 20),
                      'max_depth': np.random.randint(800, 1500, 20),
                      'min_samples_split': [2, 3, 4]}
    rfr_model, params_rfr = create_model_rfr(param_grid=param_grid_rfr, x=x, y=y)

    return rfr_model, params_rfr


rfr_h_model, params_rfr_h = grid_search_rfr(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
rfr_mech_model, params_rfr_mech = grid_search_rfr(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))

joblib.dump(filename=path + 'rfr_h_model.joblib', value=rfr_h_model)
joblib.dump(filename=path + 'rfr_h_params.joblib', value=params_rfr_h)
joblib.dump(filename=path + 'rfr_mech_model.joblib', value=rfr_mech_model)
joblib.dump(filename=path + 'rfr_mech_params.joblib', value=params_rfr_mech)
