
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import click

from pacu.models.model import Model, _MODELS

def _train_model(df: pd.DataFrame, model_name: str, options: dict) -> None:

    positives = df[df["label"] == 1]
    unlabeled = df[df["label"] != 1].copy()
    pseudo_negatives = unlabeled.sample(frac=0.8, random_state=42)
    pseudo_negatives["label"] = 0
    pu_df = pd.concat([positives, pseudo_negatives], ignore_index=True)

    X = pu_df.drop(columns=(["label"]))
    y = pu_df["label"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = x_train.shape[1]

    model = Model(
        model_name,
        input_dim,
        x_train,
        x_test,
        y_train,
        y_test,
        options
    )

    model.train(epochs=10)
    model.accuracy()


def _preprocess(df: pd.DataFrame, drop: str) -> pd.DataFrame:

    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.drop_duplicates()
    df = df.drop(columns=["url"]) 
    cols_to_normalize = df.columns.difference(["label", "has_ip", "has_port"])
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    df = df.drop(columns=drop.split(","))

    return df

def parse_layer_list(ctx, param, value):

    if value == None:
        return None

    layer_list = []
    for x in value.split(","):
        layer_list.append(int(x.strip()))

    return layer_list

@click.command()
@click.option("--model", required=True)
@click.option("--path" , required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--options", is_flag=True)
@click.option("--layers", callback=parse_layer_list)
@click.option("--kernel_size", type=int)
@click.option("--out-channels", type=int)
@click.option("--padding", type=int)
@click.option("--hidden-dim", type=int)
@click.option("--drop-features")
def train(model: str, path: str, options: bool, layers: int, kernel_size: int, out_channels: int, padding: int, hidden_dim: int, drop_features: str) -> None:

    df = pd.read_csv(path)
    df = _preprocess(df, drop_features) 
    if model == "all" and options:
        pritn("--model all cannot be used together with --options") 
        exit(1)

    options_dict = {}
    if options:
        if layers != () and layers != None:
            options_dict["layers"] = layers;  
        if kernel_size != None: 
            options_dict["kernel_size"] = kernel_size; 
        if out_channels != None: 
            options_dict["out_channels"] = out_channels; 
        if padding != None: 
            options_dict["padding"] = padding; 
        if hidden_dim != None: 
            options_dict["hidden_dim"] = hidden_dim; 

    if model == "all":
        for key in _MODELS.keys():
            _train_model(df, key, {})
    else:
        _train_model(df, model, options_dict)

