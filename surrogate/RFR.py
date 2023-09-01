import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ProcessingData import normalization
from ReadExperiments import separate

a, b, c = separate()
df = normalization(a)


rfr_model = RandomForestRegressor(n_estimators=20, min_samples_split=3, n_jobs=-1)

rfr_h_model = rfr_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Hcons']))
rfr_mech_model = rfr_model.fit(X=np.array(df.drop(['Hcons', 'Pmech'], axis=1)), y=np.array(df['Pmech']))


joblib.dump(filename='rfr_h_model.joblib', value=rfr_h_model)
joblib.dump(filename='rfr_mech_model.joblib', value=rfr_mech_model)
