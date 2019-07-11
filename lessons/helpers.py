import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype
from typing import Optional, List, Dict
import re
from toolz import compose


def display_all(df: pd.DataFrame):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        

def to_snake_case(string: str) -> str:
    fns = [
        lambda x: re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x),
        lambda x: re.sub('([a-z0-9])([A-Z])', r'\1_\2', x),
        lambda x: x.lower(),
        lambda x: x.split("_"),
        lambda x: filter(lambda s: s != "", x),
        lambda x: "_".join(x)        
    ]
    
    return compose(*reversed(fns))(string)



def add_date_parts_to_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    attrs = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear', 'is_month_end',
            'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']

    df = df.copy()
    for attr in attrs:
        df[f"{column}_{attr}"] = getattr(df[column].dt, attr)
    
    return df


def add_date_parts(df: pd.DataFrame, columns: Optional[List[str]] = None, drop=False) -> pd.DataFrame:
    df = df.copy()
    
    if not columns:
        columns = [c for c in list(df.columns) if is_datetime64_any_dtype(df[c])]
    
    for column in columns:
        df = add_date_parts_to_column(df, column)
    
    if drop:
        df = df.drop(columns, axis=1)
    
    return df


def tranform_columns_to_categorical(df: pd.DataFrame, ordered: Dict = dict()) -> pd.DataFrame:
    df = df.copy()
    
    for n,c in df.items():
        if is_string_dtype(c):
            df[n] = c.astype('category').cat.as_ordered()
            
            if n in ordered:
                df[n] = df[n].cat.set_categories(ordered[n], ordered=True)

    return df


def get_mapper(df: pd.DataFrame, ignored_columns: Optional[List[śtr]], )