import pandas as pd
from pandas.errors import ParserError
from pandas import DataFrame

def load_csv(uploaded_file) -> DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except (ParserError, ValueError):
        return pd.read_csv(uploaded_file, sep=";")