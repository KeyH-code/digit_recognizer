import json
from typing import Any

import torch
from torch import nn
from optimizer import make_optimizer
from data import make_split,make_dataset,make_generator,make_loader,SEED,train_set,val_set,train_label,val_label,DEVICE,TRAIN_MEAN,TRAIN_STD,BATCH_SIZE
from process_data import img,labels,base_dir
from transform import make_train_transform,make_val_transform
from pathlib import Path
from checkpoint import load_checkpoint,make_checkpoint,save_checkpoint
import numpy as np



# 创建一个模型
class Digit_CNN(nn.Module):
    def __init__(self,in_fea):
        super().__init__()
        self.network = nn.Sequential(
            self.basic_block(in_fea,32),
            nn.MaxPool2d(2),
            self.basic_block(32,64),
            nn.MaxPool2d(2),
            self.basic_block(64,128),
            nn.MaxPool2d(2),
            self.basic_block(128,256),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,10)
        )

    def basic_block(self,in_c,out_c):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_c
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )
        return block

    def forward(self,x):
        output = self.network(x)
        result = self.classification(output)
        return result

def train_one_epoch(model,loader,optimizer,critirion,scaler=None):
    model.train()

    all_loss = []
    all_pred = []
    all_error = {
        "false_pred":[],
        "true_label":[],
        "confidence":[]
    }
    samples_count = 0
    loss_sum = 0
    correct = 0
    best_acc = 0.0

    for x,y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=DEVICE.type,dtype=torch.float16):
                logits = model(x)
                loss = critirion(logits,y)
            all_loss.append(loss.item())            
            pred = logits.argmax(dim=1)

            mask = pred != y
            all_error["false_pred"].extend(pred[mask].detach().cpu().tolist())
            all_error["true_label"].extend(y[mask].detach().cpu().tolist())
            all_error["confidence"].extend(logits.max(dim=1).values[mask].detach().float().cpu().tolist())


            correct += (pred == y).sum().item()
            all_pred.extend(pred.detach().cpu().tolist())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = critirion(logits,y)
            pred = logits.argmax(dim=1)

            mask = pred != y
            all_error["false_pred"].extend(pred[mask])
            all_error["true_label"].extend(y[mask])
            all_error["confidence"].extend(logits.max(dim=1))

            correct += (pred == y).sum().item()
            all_loss.append(loss.item())
            all_pred.extend(pred.detach().cpu().tolist())
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()*x.size(0)
        samples_count += x.size(0)
    avg_loss = loss_sum/samples_count
    avg_accu = correct/samples_count
    if avg_accu > best_acc:
        best_acc = avg_accu

    return avg_loss,avg_accu,all_loss,all_pred,all_error,best_acc

def valuate(model,loader,optimizer,critirion,scaler=None):
    model.eval()

    all_loss = []
    all_pred = []
    all_error = {
        "false_pred":[],
        "true_label":[],
        "confidence":[]
    }
    samples_count = 0
    loss_sum = 0
    correct = 0
    best_acc = 0.0
    with torch.inference_mode():
        for x,y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast(device_type=DEVICE.type,dtype=torch.float16):
                    logits = model(x)
                    loss = critirion(logits,y)
                all_loss.append(loss.item())            
                pred = logits.argmax(dim=1)

                mask = pred != y
                all_error["false_pred"].extend(pred[mask])
                all_error["true_label"].extend(y[mask])
                all_error["confidence"].extend(logits.max(dim=1))


                correct += (pred == y).sum().item()
                all_pred.extend(pred.detach().cpu().tolist())
            else:
                logits = model(x)
                loss = critirion(logits,y)
                pred = logits.argmax(dim=1)

                mask = pred != y
                all_error["false_pred"].extend(pred[mask])
                all_error["true_label"].extend(y[mask])
                all_error["confidence"].extend(logits.max(dim=1))

                correct += (pred == y).sum().item()
                all_loss.append(loss.item())
                all_pred.extend(pred.detach().cpu().tolist())

            loss_sum += loss.item()*x.size(0)
            samples_count += x.size(0)
    avg_loss = loss_sum/samples_count
    avg_accu = correct/samples_count
    if avg_accu > best_acc:
        best_acc = avg_accu

    return avg_loss,avg_accu,all_loss,all_pred,all_error,best_acc

def load_config(name):
    config_path = base_dir/"config"/name
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return config

def main():
    model = Digit_CNN(1)
    model.to(DEVICE)
    config_name = "config01"
    config = load_config(config_name)
    torch.manual_seed(config["SEED"])
    np.random.seed(config["SEED"])
    optimizer = make_optimizer(config["optimizer"],config["lr"],model.parameters())

    scaler = torch.GradScaler()

    train_transform = make_train_transform(TRAIN_MEAN,TRAIN_STD)
    val_transform = make_val_transform(TRAIN_MEAN,TRAIN_STD)

    train_dataset = make_dataset(train_set,train_label,train_transform)
    val_dataset = make_dataset(val_set,val_label,val_transform)

    train_genertor = torch.Generator().manual_seed(config["SEED"])
    train_loader = make_loader(train_dataset,config["batch_size"],True,train_genertor)

    val_loader = make_loader(val_dataset,config["batch_size"],False)

    criterion = nn.CrossEntropyLoss()

    last_checkpoint_path = base_dir/Path("checkpoint")/Path("last.pt")
    best_checkpoint_path = base_dir/Path("checkpoint")/Path("best.pt")

    last_checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    best_accuracy = 0.0
    for epoch in range(config["epoch"]):
        train_avg_loss,train_avg_accu,_,_,_,_ = train_one_epoch(model,train_loader,optimizer,criterion,scaler)
        val_avg_loss,val_avg_accu,all_loss,all_pred,all_error,best_acc = valuate(model,val_loader,optimizer,criterion,scaler)
        if best_acc > best_accuracy:
            best_accuracy = best_acc
            best_checkpoint = make_checkpoint(model,epoch,best_accuracy,optimizer,scaler)
            save_checkpoint(best_checkpoint_path,best_checkpoint)
        last_checkpoint = make_checkpoint(model,epoch,best_accuracy,optimizer,scaler)
        save_checkpoint(last_checkpoint_path,last_checkpoint)
            
if __name__ == "__main__":
    main()