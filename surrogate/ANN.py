import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from ProcessingData import normalization
from ReadExperiments import separate

seeD = 1827
path = 'C:\\Users\\luiz4\\PycharmProjects\\ProcesoCreativo\\surrogate\\'
a, b, c = separate()
df = normalization(a)


def hidden_layer():
    np.random.seed(seeD)
    layers = np.random.randint(4, 8, 40)
    lista = []
    for lay in layers:
        tup_n = []
        for i in range(lay + 1):
            node = np.random.randint(16, 32)
            tup_n.append(node)
        lista.append(tuple(tup_n))
    return lista


def create_model_ann(param_grid, x, y):
    base_estimator = MLPRegressor(max_iter=10000,
                                  activation='logistic',
                                  tol=1e-6,
                                  alpha=0.0,
                                  n_iter_no_change=50,
                                  learning_rate='adaptive')
    sh = HalvingGridSearchCV(estimator=base_estimator,
                             param_grid=param_grid,
                             cv=5, factor=2, n_jobs=-1,
                             min_resources='exhaust',
                             aggressive_elimination=True,
                             scoring='neg_mean_squared_error').fit(x, y)

    return sh.best_estimator_, sh.best_params_


def grid_search_ann(x, y):
    param_grid_ann = {'hidden_layer_sizes': hidden_layer()}
    ann_model, params_ann = create_model_ann(param_grid=param_grid_ann, x=x, y=y)

    return ann_model, params_ann




ann_h_model, params_ann_h = grid_search_ann(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
# ann_mech_model, params_ann_mech = grid_search_ann(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))


joblib.dump(filename=path + 'ann_h_model.joblib', value=ann_h_model)
joblib.dump(filename=path + 'ann_h_params.joblib', value=params_ann_h)
# joblib.dump(filename=path + 'ann_mech_model.joblib', value=ann_mech_model)
# joblib.dump(filename=path + 'ann_mech_params.joblib', value=params_ann_mech)
