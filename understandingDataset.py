from pathlib import Path

import pandas as pd

def readDataAndLoad():
    Base = Path(__file__).parent
    Data = Base / "data" / "Heart_failure_data.csv"

    # put Data into a dataframe
    df = pd.read_csv(Data)

    return df

def countRowsAndColumns(df):
    # Number of Rows
    print(f"Number of Rows of patient data: {len(df)}")

    # Number of Columns
    print(f"Number of Columns of patient data: {len(df.columns)}")

    # Header row
    headers = list(df.columns)
    print(f"Column headers: {headers}")

def checkDataTypes(df):
    numericOrNot = df.select_dtypes(exclude="number")
    columns = list(numericOrNot.columns)
    if columns:
        print(f"Warning: {columns} Non-Numeric columns found in the dataset")
    else:
        print("All data is numeric")

def checkIfDiscreteOrContinuousData(df):
    discreet = []
    continuous = []

    for col in df.columns:
        # count the unique values from 
        unique = df[col].nunique()
        if unique == 2:


if __name__ == "__main__":
    df = readDataAndLoad()
    countRowsAndColumns(df)
    checkDataTypes(df)