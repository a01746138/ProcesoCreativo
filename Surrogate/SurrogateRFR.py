import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from ExperimentsMATLAB.ProcessingData import normalization


data = pd.read_csv(filepath_or_buffer='../ExperimentsMATLAB/Data.csv')
df = normalization(data)


def create_model(param_grid, x, y):
    base_estimator = RandomForestRegressor()
    sh = HalvingGridSearchCV(estimator=base_estimator, param_grid=param_grid,
                             cv=5, factor=3, n_jobs=-1,
                             scoring='neg_mean_squared_error').fit(x, y)

    return sh.best_estimator_, sh.best_params_


def grid_search(x, y):
    param_grid = {'n_estimators': range(2, 11),
                  'max_features': range(1, 11),
                  'max_depth': range(100, 1600, 100),
                  'min_samples_split': range(2, 11),
                  'min_samples_leaf': range(1, 11)}
    model, params = create_model(param_grid=param_grid, x=x, y=y)

    return model, params


model_h, params_h = grid_search(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
print(model_h)
print(params_h)
joblib.dump(filename='model_h_rfr.joblib', value=model_h)

model_mech, params_mech = grid_search(x=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))
print(model_mech)
print(params_mech)

joblib.dump(filename='model_mech_rfr.joblib', value=model_mech)
