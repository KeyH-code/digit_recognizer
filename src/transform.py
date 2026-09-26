from torchvision.transforms import v2
import torch
def make_train_transform(mean,std):
    train_transform = v2.Compose(
        [
            v2.Normalize(std=std,mean=mean)
        ]
    )
    return train_transform

def make_val_transform(mean,std):
    val_transform = v2.Compose(
        [
            v2.Normalize(std=std,mean=mean)
        ]
    )
    return val_transform