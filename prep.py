import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99


def load_data():
    kdd = fetch_kddcup99(subset=None, as_frame=True)
    data = kdd.data

    text_cols = ['protocol_type', 'service', 'flag']
    other_cols = []
    for c in data.columns:
        if c not in text_cols:
            other_cols.append(c)

    # (i) one hot encoding for text columns, float for the rest
    text_part = data[text_cols].apply(lambda col: col.str.decode('utf-8'))
    text_part = pd.get_dummies(text_part, dtype=float)
    num_part = data[other_cols].astype(float)
    df = pd.concat([num_part, text_part], axis=1)

    # (ii) remove columns which have the same value in all rows
    col_min = df.min()
    col_max = df.max()
    same_cols = df.columns[col_min == col_max]
    df = df.drop(columns=same_cols)
    col_min = col_min.drop(same_cols)
    col_max = col_max.drop(same_cols)

    # (iii) scale each column to [0,1]
    df = (df - col_min) / (col_max - col_min)

    D = df.to_numpy()
    y = (kdd.target != b'normal.').to_numpy(dtype=float)
    return D, y, list(df.columns), list(same_cols)


if __name__ == "__main__":
    D, y, cols, dropped = load_data()
    print("n, d =", D.shape)
    print("dropped columns:", dropped)
    print("min and max of D:", D.min(), D.max())
    print("fraction of attacks:", y.mean())
    print("rank of D:", np.linalg.matrix_rank(D))
    print("unique rows:", len(np.unique(D, axis=0)))
