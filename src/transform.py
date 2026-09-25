from torchvision.transforms import v2

def make_train_transform(std,mean):
    train_transform = v2.Compose(
        [
            v2.Normalize(std=std,mean=mean)
        ]
    )
    return train_transform

def make_val_transform(std,mean):
    val_transform = v2.Compose(
        [
            v2.Normalize(std=std,mean=mean)
        ]
    )
    return val_transform