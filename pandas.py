import pandas as pd


def split(df, column):
    """
    Splits df by groupby on column, yielding pairs of (value, dataframe) per unique column entry.
    """
    for _, df_sub in df.groupby(column):
        value = df_sub[column].iloc[0]
        yield value, df_sub.drop(columns=[column])


def flatten_column_indexes(df):
    """
    https://stackoverflow.com/a/45214611/142712
    """
    df.columns = ['_'.join(tuple(map(str, t))).rstrip('_') for t in df.columns.values]
