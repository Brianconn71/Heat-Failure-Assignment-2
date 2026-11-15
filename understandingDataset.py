from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

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
            discreet.append(col)
        else:
            continuous.append(col)

    print(discreet)
    print(continuous)
    return discreet , continuous

def summaryStatistics(df, discreet, continuous):
    print(f"----------Describing values in the continuous column---------\n")
    print(round(df[continuous].describe(), 2))
    print(f"----------% of patients that have binary value of 1(True)---------\n")
    print(round(df[discreet].mean()*100, 1))
    print(f"----------How many values are missing from each column---------\n")
    print(df.isna().sum())
    print(f"----------Amount of patients with no missing values---------\n")
    print(len(df.dropna()))

def dfHistogram(df, continuous):
    df[continuous].hist(bins=30, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "figures" / "continuous_histogram.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    df = readDataAndLoad()
    countRowsAndColumns(df)
    df = df.drop(columns=["patient_id","time"])
    checkDataTypes(df)
    discreet, continuous = checkIfDiscreteOrContinuousData(df)
    summaryStatistics(df, discreet, continuous)
    dfHistogram(df, continuous)