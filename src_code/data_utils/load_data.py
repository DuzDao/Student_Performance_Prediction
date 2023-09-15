import pandas as pd

def load_data(df):
    return {
        "in": df.iloc[:,:-1].values,
        "out": df.iloc[:,-1].values
    }
    