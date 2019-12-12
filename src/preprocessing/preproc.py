import numpy as np
import pandas as pd


def map_fl(value: str):
    trimmed_str = value.replace(',', '.').replace('"', '').replace(' ', '')
    if not trimmed_str or 'innewyniki' in trimmed_str:
        return np.nan
    else:
        return float(trimmed_str)


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


def map_praca(value: str):
    value = value.lower()
    if 'renta' in value:
        return 0.0
    elif 'emeryt' in value:
        return 0.0
    elif 'aktywny' in value:
        return 1.0
    else:
        return np.nan


def map_tryb_pracy(value: str):
    value = value.lower()
    if 'siedzaca' in value:
        return 0.0
    elif 'ruch/siedzaca' in value:
        return 1.0
    elif 'ruch' in value:
        return 2.0
    else:
        return np.nan


def map_akt_fiz(value: str):
    value = value.lower()
    if 'nigdy' in value:
        return 0.0
    elif 'sporadycznie' in value:
        return 1.0
    elif 'regularnie' in value:
        return 2.0
    else:
        return np.nan


def map_lie_min(value: str):
    if '<40' in value:
        return 0.0
    elif '>40' in value:
        return 1.0
    else:
        return np.nan


def map_sol(value: str):
    value = value.lower()
    if 'nigdy' in value:
        return 0.0
    elif 'rzadko' in value:
        return 1.0
    elif 'czasem' in value:
        return 2.0
    elif 'często' in value:
        return 3.0
    elif 'zawsze' in value:
        return 4.0
    else:
        return np.nan


def read_dfs(nan_fill: int = None):
    df_phen = pd.read_csv('ucn2aghphen.txt', sep='  |\t', header=0, engine='python', index_col=False)
    df_phen = df_phen.drop(['"elem"', '"numer"', '"X"', '"wyksztalcenie"', '"lata"', '"zawod"', '"przychod"',
                            '"dyscyplina"', '"tryb_pracy"', '"BADANIA_LAB"', '"ABPM"', '"Complior"',
                            '"HRV"', '"ECHO.serca"', '"Sphigmocor"', '"czestosc"', '"czestosc"', '"nie_pali"',
                            '"Paczkolata"', '"ciaze"', '"porody"', '"wiek_meno"', '"ile_min"'], axis=1)

    df_phen['"cywilny"'] = df_phen['"cywilny"'].map(map_cywilny)
    df_phen['"praca"'] = df_phen['"praca"'].map(map_praca)
    df_phen['"sol"'] = df_phen['"sol"'].map(map_sol)
    # df_phen['"ile_min"'] = df_phen['"ile_min"'].map(map_lie_min)
    df_phen['"akt_fiz"'] = df_phen['"akt_fiz"'].map(map_akt_fiz)
    # df_phen['"tryb_pracy"'] = df_phen['"tryb_pracy"'].map(map_sol)
    df_phen.loc[:, df_phen.dtypes == object] = df_phen.loc[:, df_phen.dtypes == object].applymap(map_fl)

    df_xp = pd.read_csv('ucn2aghxp.txt', sep='  |\t', engine='python').transpose()

    if nan_fill is not None:
        df_phen = df_phen.fillna(nan_fill)

    return df_phen, df_xp
