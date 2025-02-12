import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.replace({False: 0, True: 1})
    dataframe = dataframe.drop_duplicates()

    # URL column is not needed for model training
    dataframe = dataframe.drop(columns=['url'])

    # normalize every column except 'label', 'has_ip', 'has_port'
    cols_to_normalize = dataframe.columns.difference(['label', 'has_ip', 'has_port'])
    scaler = MinMaxScaler()
    dataframe[cols_to_normalize] = scaler.fit_transform(dataframe[cols_to_normalize])

    return dataframe

def main():
    pass


if __name__ == "__main__":
    main()

