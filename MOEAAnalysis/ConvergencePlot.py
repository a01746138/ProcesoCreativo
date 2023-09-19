import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def save():
    for lam in range(10):
        for a in ['sms', 'moead', 'nsga3']:
            hv_list = []
            for i in range(1, 31):
                if i < 10:
                    exp = f'0{i}'
                else:
                    exp = f'{i}'
                history = np.loadtxt(fname=f'../MOEARuns/{a}_hv_lambda{lam}_exp{exp}.txt', delimiter=',')[:, 1]
                hv_list.append(list(history))
            columns = []
            for j in range(120):
                columns.append((j + 1) * 500)
            df = pd.DataFrame(np.array(hv_list), columns=columns)
            # df.to_csv(path_or_buf=f'HV/{a}_hv_lambda{lam}.csv', index=False)


def plot():
    for lam in range(10):
        ax = plt.figure(dpi=300).add_subplot()
        data_sms = pd.read_csv(filepath_or_buffer=f'HV/sms_hv_lambda{lam}.csv')
        data_moead = pd.read_csv(filepath_or_buffer=f'HV/moead_hv_lambda{lam}.csv')
        data_nsga3 = pd.read_csv(filepath_or_buffer=f'HV/nsga3_hv_lambda{lam}.csv')
        n_gen = [int(x) for x in data_sms.columns]
        data_sms_m = data_sms.median()
        data_moead_m = data_moead.median()
        data_nsga3_m = data_nsga3.median()
        sb.lineplot(y=data_sms_m, x=n_gen, ax=ax, label='SMS-EMOA')
        sb.lineplot(y=data_moead_m, x=n_gen, ax=ax, label='MOEA/D')
        sb.lineplot(y=data_nsga3_m, x=n_gen, ax=ax, label='NSGA-III')
        plt.legend()
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Hypervolume value')
        plt.title(fr'Convergence plot when $\lambda_{lam}$')
        plt.tight_layout()
        plt.savefig(fname=f'../Images/convergence_lambda{lam}.png')
