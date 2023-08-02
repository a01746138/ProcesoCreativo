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
    layers = np.random.randint(1, 10, 10)
    list = []
    for l in layers:
        tup_n = []
        for i in range(l + 1):
            node = np.random.randint(4, 12)
            tup_n.append(node)
        list.append(tuple(tup_n))

    return list


def create_model_ann(param_grid, x, y):
    base_estimator = MLPRegressor(max_iter=1000)
    sh = HalvingGridSearchCV(estimator=base_estimator, param_grid=param_grid,
                             cv=10, factor=2, max_resources=100, n_jobs=-1,
                             scoring='neg_mean_squared_error').fit(x, y)
    params = sh.best_params_

    return sh.best_estimator_, params


def grid_search_ann(x, y):
    np.random.seed(seeD)
    param_grid_ann = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'learning_rate': ['constant', 'invscaling'],
                      'hidden_layer_sizes': hidden_layer(),
                      'learning_rate_init': np.random.uniform(0, 1, 10),
                      'alpha': np.random.uniform(0, 0.001, 10)}
    ann_model, params_ann = create_model_ann(param_grid=param_grid_ann, x=x, y=y)

    return ann_model, params_ann


ann_h_model, params_ann_h = grid_search_ann(x=df.drop([6, 7], axis=1), y=df[6])
ann_mech_model, params_ann_mech = grid_search_ann(x=df.drop([6, 7], axis=1), y=df[7])

joblib.dump(filename=path + 'ann_h_model.joblib', value=ann_h_model)
joblib.dump(filename=path + 'ann_mech_model.joblib', value=ann_mech_model)
