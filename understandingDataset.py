"""
    I want to highlight the usage of Claude (Anthropic)in doing this assignment. It was usead primarily as a learning tool to support the development of this python script.
    I used it to explain and sow examples of how particular methods of scikit learn could be used such as Ward linkage, K-Means, Hierarchial Clustering, Preprocessing decisions made and figuring out a way
    to get the best K value. I used it to help debug my code when stuck, review if what I wrote made sense and again to provide examples of methods usage. I adapted what i learnt from this into my own code
    to help with this assignment. I checked any references and websites with carious methods independently to ensure I understood what was happening. Examples of code were provided by claude which I adapted and made my own by
    ensuring I fully understood everything. The final decisions made on analysis of dataset, code and final written report are my own choices and decisions to which I take full responsibility for accuracy of results and how the results have been perceived.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# setting display settings to ensure columns are not cut off when dispalyed
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
    print(f"Column names: {headers}")

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

    print(f"Discreet / Binary Columns: {discreet}")
    print(f"Continuous Columns: {continuous}")
    return discreet , continuous

def summaryStatistics(df, discreet, continuous):
    print(f"\n----------Describing values in the continuous column---------\n")
    print(round(df[continuous].describe(), 2))
    print(f"\n----------% of patients that have binary value of 1(True)---------\n")
    print(round(df[discreet].mean()*100, 1))
    print(f"\n----------How many values are missing from each column---------\n")
    print(df.isna().sum())
    print(f"\n----------Amount of patients with no missing values---------\n")
    print(f"The Amount of patients with no missing values: {len(df.dropna())} when the total amount of participants in the study is: {len(df)}")

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