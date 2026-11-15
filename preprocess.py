"""
This file is used to pre process the csv file.
This means that this file willload the csv file. It will Check for missing values and decide what to do with them. It will also be used to validate the data from the csv, It will impute any cells which may be missing values in the columns of the csv file.
It will transform the data and scale features which have been chosen to be scaled.
It will return a single clean processed DataFrame ready for use with the Machine Learning Algorithms. Also, it will store the processed csv in a new csv file.
"""
from pathlib import Path
import pandas as pd

# Build the path to the datafile
Data = Path(__file__).parent / "data" / "Heart_failure_data.csv"

# Function to load the file.
def loadData():
    # Using Pandas to read the data file from the path
    df = pd.read_csv(Data)
    # return the dataframe
    return df

# Check the data shape - columns and amount of row values before we do any validating
def checkDataShapeBeforeValidating(df):
    print(f"Shape of data before removing unused columns {df.shape}")

def removeunUsedColumns(df):
    # removing the patient id and time column as they have no bearing on outcome and will not be used for training the ML algorithm
    df = df.drop(columns=["patient_id", "time"])
    print(f"Shape of data after removing unused columns {df.shape}")
    return df

def checkForNoEmptyValues(df):
    # empty values is assigned to the sum of all columns that have a missing values totaled together
    # hence the .sum().sum() sum per column and then sum per dataframe
    emptyValues = df.isnull().sum().sum()
    # if there still are empty values then print a warning
    # Otherwise let the user know there are no empty values
    if emptyValues:
        print(f"Warning: {emptyValues} empty values found in the dataset")
    else:
        print("No more empty values in Dataset")

def checkNoDuplicates(df):
    # Check the dataset to ensure that no two rows of values are exactly the same
    # follows similar logic as checkForNoEmptyValues
    duplicates = df.duplicated().sum().sum()
    # Given the Time (Sec) column it would be very difficult for their to be any duplicates and I dont want to remove this column
    # This is just to confirm there are no two rows the exact same.
    if duplicates:
        print(f"Warning: {duplicates} duplicate values found in the dataset")
    else:
        print("No duplicated values detected in the dataset")

def checkDataTypes(df):
    # Checking that the data in the columns are consistent.
    # mixed or text value data types would make pandas store it as a non numeric type
    # So, we need a check to make sure they are numeric.

    # assigns any columns which are non numeric to the variable nonNumeric
    nonNumeric = df.select_dtypes(exclude="number")
    # checks the lenght of the columns in the non numeric dataframe.
    # If there is a column thats non-numeric then len will be > 0
    # use this to print a warning message to the user.
    if len(nonNumeric.columns):
        columns = list(nonNumeric.columns)
        print(f"Warning: {columns} Non-Numeric columns found in the dataset")
    else:
        print("No Non-Numeric columns detected in the dataset")


def checkRangeOfValues(df):
    # below checks the columns in the dataframe, loops through values and if all values are 1 or 0 
    # then its is binary and we append to binary columnlist otherwise its not binary so we append to continuous
    binaryColumns = []
    continuousColumn = []
    for column in df.columns:
        binaryValue = True
        for idx, value in df[column].items():
            # ignore missing values
            if pd.isna(value):
                continue
            if value != 0.0 and value != 1.0:
                binaryValue = False
                break
        # decide if binary or not once per column rather than once per iteration
        if binaryValue:
            binaryColumns.append(column)
        else:
            continuousColumn.append(column)

    print(f"Binary columns: {binaryColumns}")
    print(f"continuous columns: {continuousColumn}")
    return binaryColumns, continuousColumn

def checkNoNegatives(df):
    # Check that theres no negative values as this would not be possible given dataset
    negativeValues = (df < 0).sum().sum()
    if negativeValues:
        print(f"{negativeValues} have been found")
    else:
        print(f"No Negative Values found!")

def handle_missing_values(df, binaryVlues, continuousValues):
    # We cannot delete columns with missing values from the data set
    # There are currently 319 empty values in the dataset if we removed all that data from the csv we would be losing over half our data so we need another way to deal with missing values so we are going to impute rather than delete
    dfCopy = df.copy()
    for column in dfCopy.columns:
        if column in continuousValues:
            value = dfCopy[column].median()
            dfCopy[column] = dfCopy[column].fillna(value)
            print(f"Continuous column {column} filled with {value:.1f}")
        elif column in binaryVlues:
            value = dfCopy[column].mode()[0]
            dfCopy[column] = dfCopy[column].fillna(value)
            print(f"Binary column {column} filled with {value:.1f}")
    checkForNoEmptyValues(dfCopy)
    
    return dfCopy

def saveProcessedDfasCsv(df):
    # setting output path for nely processed df which I will create a csv from for comparison
    output = Path(__file__).parent / "data" / "processed_Heart_Failure.csv"

    # Changing the dataframe to a csv and using index=false as without it then pandas will write an unnamed first column to the csv
    # will lways contain newest data as it runs and overwrites file on each run
    df.to_csv(output, index=False)

    print(f"Processed data file saved to {output}")


def returnCleanData():
    # This function acts as the conductor of the file, I will run this function from __main__ block
    # below which will run the necessary functions and clean the data. Making it ready to be worked 
    # with the ML algorithms provided.

    # Load the data from the file
    df = loadData()

    # Check dhape of data loaded from file
    checkDataShapeBeforeValidating(df)
    checkForNoEmptyValues(df)

    df = removeunUsedColumns(df)

    binary, continuous = checkRangeOfValues(df)
    checkNoNegatives(df)

    # CLean the Dataframe we have loaded above to remove missing vales
    df_copy = handle_missing_values(df, binary, continuous)

    saveProcessedDfasCsv(df_copy)

    # Now, check the Dataframe and ensure we are getting valid data such as,
    # The data has no more empty values, It has no duplicate values and the datatypes are consistent
    checkForNoEmptyValues(df)
    checkNoDuplicates(df)
    checkDataTypes(df)

    # Now we can return the cleaned data
    return df_copy




if __name__ == "__main__":
    returnCleanData()