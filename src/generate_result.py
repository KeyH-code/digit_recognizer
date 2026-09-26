import numpy as np
import pandas as pd
import torch
from train import load_model_parameter_only, Digit_CNN
from process_data import base_dir, test_img
from transform import make_val_transform
from data import TRAIN_MEAN, TRAIN_STD, DEVICE
from pathlib import Path

if __name__ == "__main__":
    model = Digit_CNN(1)
    model.to(DEVICE)

    checkpoint_path = base_dir / Path("checkpoint") / Path("best.pt")   
    load_model_parameter_only(model, checkpoint_path)

    val_transform = make_val_transform(TRAIN_MEAN, TRAIN_STD)           
    model.eval()

    preds = []
    with torch.inference_mode():
        for i in range(0, len(test_img), 512):                          
            batch = torch.from_numpy(np.asarray(test_img[i:i+512])).float().to(DEVICE)
            batch = val_transform(batch)
            preds.extend(model(batch).argmax(dim=1).cpu().tolist())     

    assert len(preds) == len(test_img), \
        "预测数量(%d)与测试集(%d)不一致" % (len(preds), len(test_img))

    result_dir = base_dir / Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(preds) + 1),
            "Label": preds
        }
    )
    submission_path = result_dir / Path("submission.csv")
    submission.to_csv(submission_path, index=False)

    print("已写入:", submission_path)
    print("行数:", len(submission))
    print(submission.head())
    