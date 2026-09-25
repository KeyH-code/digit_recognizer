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

train_img = feature_pd.reshape(-1,1,28,28)
labels_img = labels_pd.reshape(feature_pd.shape[0],1)

train_tensor = torch.tensor(feature_pd,device=torch.device("cuda"),dtype=torch.float32)

TRAIN_MEAN = train_tensor.mean().item()
TRAIN_STD = train_tensor.std().item()


