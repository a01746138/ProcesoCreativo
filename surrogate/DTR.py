import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from ProcessingData import normalization
from ReadExperiments import separate

a, b, c = separate()
df = normalization(a)


dtr_model = DecisionTreeRegressor(min_samples_split=3)

dtr_h_model = dtr_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
dtr_mech_model = dtr_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))


joblib.dump(filename='dtr_h_model.joblib', value=dtr_h_model)
joblib.dump(filename='dtr_mech_model.joblib', value=dtr_mech_model)
