from pathlib import Path

import Pandas as pd

def readDataAndLoad():
    Base = Path(__file__).parent
    Data = Base / "data" / "Heart_failure_data.csv"

    # put Data into a dataframe
    df = pd.read_csv(Data)

    return df

if __name__ == "__main__":
    readDataAndLoad()