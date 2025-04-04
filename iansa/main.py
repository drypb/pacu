
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import feature_extractor as fext
import pandas as pd
import click


DEF_OUT_PATH = "out.csv"


pd.set_option('future.no_silent_downcasting', True)


models = {
    "mlp"    : MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), max_iter=300, random_state=42),
    "rf"     : RandomForestClassifier(n_estimators=100, random_state=42),
    "logreg" : LogisticRegression(max_iter=1000, random_state=42),
    "svm"    : SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    "gb"     : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "nb"     : GaussianNB(),
    "lgm"    : LGBMClassifier(random_state=42),
    "xgb"    : XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "knn"    : KNeighborsClassifier(n_neighbors=5),
    "ada"    : AdaBoostClassifier(n_estimators=50, random_state=42)
}


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out" , required=False)
def extract(path: str, out: str) -> None:
    fe = fext.FeatureExtractor(path) 
    fe.extract()
    out = out if out else DEF_OUT_PATH
    fe.export(out)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:

    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.drop_duplicates()
    df = df.drop(columns=["url"])

    cols_to_normalize = df.columns.difference(["label", "has_ip", "has_port"])
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df


def _train_model(df: pd.DataFrame, model: str):
    # Separar positivos confirmados
    positives = df[df["label"] == 1]

    # Separar não rotulados (label != 1)
    unlabeled = df[df["label"] != 1].copy()

    # Estratégia simples: amostrar 80% dos não rotulados como negativos provisórios
    pseudo_negatives = unlabeled.sample(frac=0.8, random_state=42)
    pseudo_negatives["label"] = 0  # atribui rótulo negativo falso

    # Conjunto de treino com positivos + pseudo-negativos
    pu_df = pd.concat([positives, pseudo_negatives], ignore_index=True)

    X = pu_df.drop(columns=["label"])
    y = pu_df["label"]

    # Divisão treino/teste interna no PU (opcional — poderia treinar com tudo)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _model = models[model]
    print(f"Training PU model: {model.upper()}...")
    _model.fit(x_train, y_train)

    print("Testing...")
    y_pred = _model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))


@cli.command()
@click.option("--model", required=True)
@click.option("--path" , required=True, type=click.Path(exists=True, dir_okay=False))
def train(model: str, path: str) -> None:

    df = pd.read_csv(path)
    df = _preprocess(df) 

    if model == "all":
        for key in models.keys():
            _train_model(df, key)
    else:
        _train_model(df, model)


cli()
