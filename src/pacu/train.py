
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import click

from pacu.models.model import Model, _MODELS


def _train_model(df: pd.DataFrame, model_name: str) -> None:

    positives = df[df["label"] == 1]
    unlabeled = df[df["label"] != 1].copy()
    pseudo_negatives = unlabeled.sample(frac=0.8, random_state=42)
    pseudo_negatives["label"] = 0
    pu_df = pd.concat([positives, pseudo_negatives], ignore_index=True)

    X = pu_df.drop(columns=["label"])
    y = pu_df["label"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = x_train.shape[1]

    model = Model(
        model_name,
        input_dim,
        x_train,
        x_test,
        y_train,
        y_test
    )

    model.train(epochs=10)
    model.accuracy()


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
        for key in _MODELS.keys():
            _train_model(df, key)
    else:
        _train_model(df, model)

