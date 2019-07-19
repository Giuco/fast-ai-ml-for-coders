import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype
from typing import Optional, List, Dict, Tuple, Callable
import re
from toolz import compose
from unidecode import unidecode
from sklearn.tree import export_graphviz, DecisionTreeRegressor
import IPython
import graphviz


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


def separate_features_by_dtype(df: pd.DataFrame) -> Dict[str, List[str]]:
    dtypes_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={0: "dtype"})
    dtypes_df["dtype"] = dtypes_df["dtype"].map(str)
    return dtypes_df.groupby("dtype")["index"].apply(list).to_dict()


def onehot_categorizer(df: pd.DataFrame, columns_to_categorize: List[str], drop_original: bool = True, drop_first_columns: bool = True) -> Tuple[Callable, pd.DataFrame, Dict]:
    categ_getter = lambda col: list(np.sort(df[col].dropna(axis=0, how='any').unique()))
    vec = {column: categ_getter(column) for column in sorted(columns_to_categorize)}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        col_name_creator = lambda r: f"fklearn_feat__{r}"
        dummies_df = pd.get_dummies(
            new_df,
            prefix=list(map(col_name_creator, columns_to_categorize)),
            prefix_sep="==",
            columns=columns_to_categorize
        )
        
        categories_missing = list()
        
        for column, categories in vec.items():
            for category in categories:
                new_column_name = f"fklean_feat__{column}=={category}"
                
                if new_column_name not in dummies_df.columns:
                    categories_missing.append(new_column_name)
        
        missing_df = pd.DataFrame(np.zeros((len(new_df), len(categories_missing))), columns=categories_missing)
        
        if drop_original:
            new_df = new_df.drop(columns_to_categorize, axis=1)
            
        new_df = pd.concat([new_df, dummies_df, missing_df], axis=1)
        
        if drop_first_columns:
            first_columns = list()
            for column, categories in vec.items():
                category = categories[0]
                new_column_name = f"fklean_feat__{column}=={category}"
                first_columns.append(new_column_name)
            
    
            new_df = new_df.drop(first_columns, axis=1)
        
        return new_df
    
    return p, p(df), None


def to_normalized_string(original_name: str) -> str:
    """
    >>> to_normalized_string('São Paulo/Moema-1')
    'sao_paulo_moema_1'
    """
    ascii_name = to_lowercase_ascii(original_name)
    return re.sub(r'[^\w]+', '_', ascii_name)


def to_lowercase_ascii(unicode_string: str) -> str:
    """
    >>> to_lowercase_ascii(u'André')
    'andre'
    """
    return to_ascii(unicode_string).lower()


def to_ascii(unicode_string: str) -> str:
    return unidecode(unicode_string)


def draw_tree(t: DecisionTreeRegressor, df: pd.DataFrame, size: int = 10, ratio: float = 0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s)))