import pandas as pd
from pathlib import Path
import torch

base_dir = Path(__file__).resolve().parent.parent

train_path = base_dir/"data"/"train.csv"
test_path = base_dir/"data"/"test.csv"

train_read = pd.read_csv(train_path)
test_read = pd.read_csv(test_path)

labels_pd = train_read["label"].values.astype("float32")
feature_pd = train_read.drop(columns=["label"]).values.astype("float32")

img = feature_pd.reshape(-1,1,28,28)
labels = labels_pd.reshape(feature_pd.shape[0],1)




