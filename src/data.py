from sklearn.model_selection import train_test_split
import torch
from process_data import img,labels
from torch.utils.data import Dataset,DataLoader
import numpy as np

SEED = 20260915
DEVICE = torch.device("cuda")
BATCH_SIZE = 64

# 划分数据
def make_split(train_img,test_size,seed,shuffle,stratify):
    full_indices = np.arange(len(train_img))
    train_indices,val_indices = train_test_split(
        full_indices,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle,
        stratify=stratify
    )
    train_set = train_img[train_indices]
    val_set = train_img[val_indices]
    train_label = stratify[train_indices]
    val_label = stratify[val_indices]
    return train_set,val_set,train_label,val_label

# 创建dataset
class Digit_dataset(Dataset):
    def __init__(self,dataset,label,transform):
        self.dataset = dataset
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = torch.from_numpy(np.asarray(self.dataset[index])).float()
        label = self.label[index]

        return image,label

# 创建数据集
def make_dataset(images,labels,trainsform):
    dataset = Digit_dataset(images,labels,trainsform)
    return dataset

# 创建数据加载器
def make_loader(dataset,batch_size,shuffle,generator=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator
    )
    return loader

# 创建generator
def make_generator(seed):
    generator = torch.Generator().manual_seed(seed)

test_size = 1/4
train_set,val_set,train_label,val_label = make_split(img,test_size,SEED,True,labels)

train_tensor = torch.tensor(train_set,device=DEVICE,dtype=torch.float32)


TRAIN_STD = train_tensor.std(dim=(0,2,3)).tolist()
TRAIN_MEAN = train_tensor.mean(dim=(0,2,3)).tolist()

