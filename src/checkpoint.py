import torch
import numpy

def make_checkpoint(model,epoch,best_acc,optimizer,scaler):
    return {
        "epoch":epoch,
        "best_acc":best_acc,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "scaler_state_dict":scaler.state_dict(),
        "rng_state":torch.get_rng_state(),
        "cuda_state":torch.cuda.get_rng_state_all()
    }

def save_checkpoint(path,checkpoint):
    torch.save(path,checkpoint)

def load_checkpoint(path,weights_only,map_location):
    return torch.load(path,weights_only=weights_only,map_location=map_location)