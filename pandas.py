import pandas as pd


def split(df, column):
    """
    Splits df by groupby on column, yielding pairs of (value, dataframe) per unique column entry.
    """
    for _, df_sub in df.groupby(column):
        value = df_sub[column].iloc[0]
        yield value, df_sub.drop(columns=[column])


def flatten_column_indexes(df, delim='_'):
    """
    https://stackoverflow.com/a/45214611/142712
    """
    df.columns = [delim.join(tuple(map(str, t))).rstrip(delim) for t in df.columns.values]


def options_for_show(max_columns=None, expand_frame_repr=False, max_rows=60):
    """
    Every time I use pandas I have to look this up, so created a convencience function here.
    """
    pd.options.display.max_columns = max_columns
    pd.options.display.expand_frame_repr = expand_frame_repr
    pd.options.display.max_rows = max_rows
