"""Metrics, schedules, checkpoint helpers, and reproducibility utilities."""
import random
import numpy as np
import torch

class AverageMeter:
    def __init__(self): self.sum=0.0; self.count=0
    def update(self,val,n=1): self.sum += val*n; self.count += n
    def average(self): return self.sum/max(1,self.count)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

def confusion_matrix(label,pred,num_class,ignore=255):
    seg_pred = pred.detach().cpu().numpy().argmax(1).astype(np.uint8)
    seg_gt = label.detach().cpu().numpy().astype(np.int64)
    idx = seg_gt != ignore
    index = (seg_gt[idx]*num_class + seg_pred[idx]).astype("int64")
    counts = np.bincount(index,minlength=num_class*num_class)
    return counts[:num_class*num_class].reshape(num_class,num_class).astype(np.float64)

def iou_from_confusion(cm):
    pos,res,tp = cm.sum(1),cm.sum(0),np.diag(cm)
    iou = tp/np.maximum(1.0,pos+res-tp)
    return iou, float(iou.mean()), float(tp.sum()/max(1.0,cm.sum()))

def poly_lr(optimizer,base_lr,max_iters,cur_iters,power=0.9):
    frac = max(0.0, 1-float(cur_iters)/max_iters)
    lr = base_lr*frac**power
    for g in optimizer.param_groups: g["lr"]=lr
    return lr

def unwrap_training_state_dict(sd):
    return {k[len("model."):]:v for k,v in sd.items() if k.startswith("model.")}
