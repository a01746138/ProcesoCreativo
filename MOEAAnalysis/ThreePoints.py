import copy
import numpy as np
import pandas as pd
from ExperimentsMATLAB.ProcessingData import decode

a = ['pimia']
dec = 4


def latex():
    for algorithm in a:
        for lam in range(10):
            file = pd.read_csv(f'NDS/{algorithm}_nds_lambda{lam}.csv')
            decode_file = copy.deepcopy(file)
            decode_file['Pmech'] = -decode_file['Pmech']
            decode_file = decode(decode_file)

            min_f = np.inf
            min_index = None
            for index in range(file.shape[0]):
                f = file['Hcons'].iloc[index] * 0.5 + file['Pmech'].iloc[index] * 0.5
                if f < min_f:
                    min_f = f
                    min_index = index

            index_list = [0, min_index, file.shape[0] - 1]
            for i in index_list:
                x1 = np.trunc(decode_file['Vbatt'].iloc[i] * 10 ** dec) / (10 ** dec)
                x2 = np.trunc(decode_file['Qbatt'].iloc[i] * 10 ** dec) / (10 ** dec)
                x3 = np.trunc(decode_file['Ndiff'].iloc[i] * 10 ** dec) / (10 ** dec)
                x4 = np.trunc(decode_file['Rwheel'].iloc[i] * 10 ** dec) / (10 ** dec)
                x5 = np.trunc(decode_file['MaxPmot'].iloc[i] * 10 ** dec) / (10 ** dec)
                el = np.trunc(decode_file['Mass'].iloc[i] * 10 ** dec) / (10 ** dec)
                f1 = np.trunc(decode_file['Hcons'].iloc[i] * 10 ** dec) / (10 ** dec)
                f2 = np.trunc(decode_file['Pmech'].iloc[i] * 10 ** dec) / (10 ** dec)
                if i == 0:
                    print(f'{el} & {x1} & {x2} & {x3} & {x4} & {x5} & {f1} & {f2} \\\\ \\hline')
                elif i == file.shape[0] - 1:
                    print('\\rowcolor[gray]{0.6}' + str(el) + ' & ' + str(x1) + ' & ' + str(x2) + ' & ' + str(x3) +
                          ' & ' + str(x4) + ' & ' + str(x5) + ' & ' + str(f1) + ' & ' + str(f2) + '\\\\ \\hline')
                else:
                    print('\\rowcolor[gray]{0.8}' + str(el) + ' & ' + str(x1) + ' & ' + str(x2) + ' & ' + str(x3) +
                          ' & ' + str(x4) + ' & ' + str(x5) + ' & ' + str(f1) + ' & ' + str(f2) + '\\\\ \\hline')


def data(al):
    for algorithm in al:
        d = []
        for lam in range(10):
            file = pd.read_csv(f'NDS/{algorithm}_nds_lambda{lam}.csv')
            decode_file = copy.deepcopy(file)
            decode_file['Pmech'] = -decode_file['Pmech']
            decode_file = decode(decode_file)

            min_f = np.inf
            min_index = None
            for index in range(file.shape[0]):
                f = file['Hcons'].iloc[index] * 0.5 + file['Pmech'].iloc[index] * 0.5
                if f < min_f:
                    min_f = f
                    min_index = index

            index_list = [0, min_index, file.shape[0] - 1]
            for i in index_list:
                x1 = np.trunc(decode_file['Vbatt'].iloc[i] * 10 ** dec) / (10 ** dec)
                x2 = np.trunc(decode_file['Qbatt'].iloc[i] * 10 ** dec) / (10 ** dec)
                x3 = np.trunc(decode_file['Ndiff'].iloc[i] * 10 ** dec) / (10 ** dec)
                x4 = np.trunc(decode_file['Rwheel'].iloc[i] * 10 ** dec) / (10 ** dec)
                x5 = np.trunc(decode_file['MaxPmot'].iloc[i] * 10 ** dec) / (10 ** dec)
                el = np.trunc(decode_file['Mass'].iloc[i] * 10 ** dec) / (10 ** dec)
                f1 = np.trunc(decode_file['Hcons'].iloc[i] * 10 ** dec) / (10 ** dec)
                f2 = np.trunc(decode_file['Pmech'].iloc[i] * 10 ** dec) / (10 ** dec)

                d.append([x1, x2, x3, x4, x5, el, f1, f2])

        np.savetxt(fname=f'MATLAB\{algorithm}_3points.txt', X=np.array(d), delimiter=',')


def latex2(a):
    for algorithm in a:
        for lam in range(10):
            d = np.loadtxt(fname=f'MATLAB\MATLAB_{algorithm}_3points.txt', delimiter=',')

            df = pd.DataFrame(data=d, columns=['Vbatt', 'Qbatt', 'Ndiff',
                                               'Rwheel', 'MaxPmot', 'Mass', 'f1', 'f2',
                                               'Hcons', 'Pmech'])
            for i in range(3):
                index = 3 * lam + i
                el = np.trunc(df['Mass'].iloc[index] * 10 ** dec) / (10 ** dec)
                sf1 = np.trunc(df['f1'].iloc[index] * 10 ** dec) / (10 ** dec)
                sf2 = np.trunc(df['f2'].iloc[index] * 10 ** dec) / (10 ** dec)
                f1 = np.trunc(df['Hcons'].iloc[index] * 10 ** dec) / (10 ** dec)
                f2 = np.trunc(df['Pmech'].iloc[index] * 10 ** dec) / (10 ** dec)

                if i == 0:
                    print(f'{el} & {sf1} & {sf2} & {f1} & {f2} \\\\ \\hline')
                elif i == 1:
                    print('\\rowcolor[gray]{0.8}' + str(el) + ' & ' + str(sf1) + ' & ' + str(sf2) +
                          ' & ' + str(f1) + ' & ' + str(f2) + '\\\\ \\hline')
                elif i == 2:
                    print('\\rowcolor[gray]{0.6}' + str(el) + ' & ' + str(sf1) + ' & ' + str(sf2) +
                          ' & ' + str(f1) + ' & ' + str(f2) + '\\\\ \\hline')

latex2(['pimia'])
