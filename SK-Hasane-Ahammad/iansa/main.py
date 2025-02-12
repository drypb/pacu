import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('future.no_silent_downcasting', True)

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
    df = pd.read_csv('out.csv')

    print(df.head())
    print(df.info())

    print('preprocessing...')
    df = preprocess(df)
    print(df.head())
    print(df.info())

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), max_iter=300, random_state=42)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'MLP accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
