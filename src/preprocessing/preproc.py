import pandas as pd
import numpy as np


def map_cywilny(value: str):
    value = value.lower()
    if 'wolny' in value:
        return 0.0
    elif 'partnerski' in value:
        return 1.0
    elif 'malzenski' in value:
        return 2.0
    else:
        return np.nan


def read_dfs():
    df_phen = pd.read_csv('ucn2aghphen.txt', sep='  |\t', header=0, engine='python', index_col=False)
    df_phen = df_phen.drop(['"elem"', '"numer"', '"X"'], axis=1)
    df_phen['"cywilny"'] = df_phen['"cywilny"'].map(map_cywilny)
    df_xp = pd.read_csv('ucn2aghxp.txt', sep='  |\t', engine='python').transpose()

    return df_phen, df_xp
