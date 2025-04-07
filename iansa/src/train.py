
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import click


def _train_model(df: pd.DataFrame, model: str):
    pass


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:

    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.drop_duplicates()
    df = df.drop(columns=["url"])

    cols_to_normalize = df.columns.difference(["label", "has_ip", "has_port"])
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df


@click.command()
@click.option("--model", required=True)
@click.option("--path" , required=True, type=click.Path(exists=True, dir_okay=False))
def train(model: str, path: str) -> None:

    df = pd.read_csv(path)
    df = _preprocess(df) 
    if model == "all":
        for key in _models.keys():
            _train_model(df, key)
    else:
        _train_model(df, model)

