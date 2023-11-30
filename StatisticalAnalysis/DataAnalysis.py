import pandas as pd

df = pd.read_csv(filepath_or_buffer='CV_Pmech.csv')

print(df.describe())