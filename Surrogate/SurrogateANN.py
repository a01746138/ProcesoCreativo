import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from ExperimentsMATLAB.ProcessingData import normalization


data = pd.read_csv(filepath_or_buffer='../ExperimentsMATLAB/Data.csv')
df = normalization(data)


def hl(q):
    main_list = []
    for i in range(1, 4):
        for _ in range(q):
            c = 0
            sublist = []
            while c < i:
                n = np.random.choice([64, 128, 256])
                sublist.append(n)
                c += 1
            main_list.append(sublist)

    new_list = []
    for elem in main_list:
        if elem not in new_list:
            new_list.append(elem)

    return new_list


def create_model(param_grid, x, y):
    base_estimator = MLPRegressor(tol=1e-5, n_iter_no_change=50, max_iter=10000,
                                  learning_rate='adaptive', activation='logistic')
    sh = HalvingGridSearchCV(estimator=base_estimator, param_grid=param_grid,
                             cv=5, factor=3, n_jobs=-1,
                             scoring='neg_mean_squared_error').fit(x, y)

    return sh.best_estimator_, sh.best_params_


def grid_search(x, y):
    param_grid = {'alpha': [x * 1e-5 for x in range(1, 1100, 100)],
                  'learning_rate_init': [x * 1e-4 for x in range(1, 1100, 100)],
                  'hidden_layer_sizes': hl(3)}
    model, params = create_model(param_grid=param_grid, x=x, y=y)

    return model, params


model_h, params_h = grid_search(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
print(model_h)
print(params_h)
joblib.dump(filename='model_h_ann.joblib', value=model_h)

model_mech, params_mech = grid_search(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))
print(model_mech)
print(params_mech)
joblib.dump(filename='model_mech_ann.joblib', value=model_mech)
