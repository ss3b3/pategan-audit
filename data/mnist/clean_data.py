import gzip
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


image_len = 28
image_size = image_len * image_len
n_images = 60_000


# data path, headers and dtypes
train_path = "train-images-idx3-ubyte.gz"
train_lables_path = "train-labels-idx1-ubyte.gz"

headers = [f"p_{i}" for i in range(image_size)]
dtype = "float"


# load data
def load_data(path, n_images):
    """TODO."""
    with gzip.open(path, 'r') as f:
        f.read(16)
        buffer = f.read(n_images * image_size)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(n_images, image_size)

    return data


# load labels
def load_labels(path, n_images):
    """TODO."""
    with gzip.open(path, 'r') as f:
        f.read(8)
        buffer = f.read(n_images)
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)

    return labels


data = load_data(train_path, n_images)
labels = load_labels(train_lables_path, n_images)


# create dfs
def create_df(data, headers, labels):
    """TODO."""
    df = pd.DataFrame(data, columns=headers, dtype=dtype) / 255
    df['label'] = labels
    df['label'] = df['label'].astype(int)

    return df


df = create_df(data, headers, labels)
# train test split
train_df, test_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=13)


# save data
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_pickle("train_mnist.pkl.gz", compression="gzip")
test_df.to_pickle("test_mnist.pkl.gz", compression="gzip")
