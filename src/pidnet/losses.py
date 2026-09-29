"""Losses used in the 150-epoch Experiment A reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossConfig:
    BALANCE_WEIGHTS = [0.4, 1.0]
    SB_WEIGHTS = 1.0
    OHEM_THRES = 0.9
    OHEM_KEEP = 131072
    ALIGN_CORNERS = True

LOSSCFG = LossConfig()

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thres=0.9, min_kept=131072, weight=None):
        super().__init__()
        self.thresh, self.min_kept, self.ignore_label = thres, max(1,min_kept), ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="none")

    def _ce(self, score, target):
        return self.criterion(score,target).mean()

    def _ohem(self, score, target):
        pred = F.softmax(score, dim=1)
        pix = self.criterion(score,target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label
        tmp = target.clone()
        tmp[tmp == self.ignore_label] = 0
        pred = pred.gather(1,tmp.unsqueeze(1)).contiguous().view(-1)[mask].sort()
        probs, ind = pred
        if probs.numel() == 0:
            return pix.mean() * 0.0
        min_value = probs[min(self.min_kept, probs.numel()-1)]
        threshold = max(float(min_value), self.thresh)
        selected = pix[mask][ind][probs < threshold]
        return selected.mean() if selected.numel() else pix[mask].mean()

    def forward(self, score, target):
        if not isinstance(score,(list,tuple)): score=[score]
        if len(score) == len(LOSSCFG.BALANCE_WEIGHTS):
            funcs = [self._ce]*(len(score)-1)+[self._ohem]
            return sum(w*f(x,target) for w,x,f in zip(LOSSCFG.BALANCE_WEIGHTS,score,funcs))
        if len(score) == 1:
            return LOSSCFG.SB_WEIGHTS*self._ohem(score[0],target)
        raise ValueError("prediction/target length mismatch")

def weighted_bce(bd_pre, target):
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1,-1)
    target_t = target.view(1,-1)
    pos, neg = target_t == 1, target_t == 0
    w = torch.zeros_like(log_p)
    pos_num, neg_num = pos.sum(), neg.sum()
    total = pos_num + neg_num
    if total == 0: return log_p.mean()*0
    w[pos] = neg_num.float()/total
    w[neg] = pos_num.float()/total
    return F.binary_cross_entropy_with_logits(log_p,target_t,w,reduction="mean")

class BoundaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super().__init__(); self.coeff_bce=coeff_bce
    def forward(self, bd_pre, bd_gt):
        return self.coeff_bce*weighted_bce(bd_pre,bd_gt)

class FullModel(nn.Module):
    def __init__(self, model, sem_loss, bd_loss, ignore_label=255):
        super().__init__()
        self.model, self.sem_loss, self.bd_loss = model, sem_loss, bd_loss
        self.ignore_label = ignore_label

    def pixel_acc(self,pred,label):
        p = pred.argmax(1)
        valid = (label != self.ignore_label).long()
        return torch.sum(valid*(p==label).long()).float()/(torch.sum(valid).float()+1e-10)

    def forward(self,inputs,labels,bd_gt):
        outputs = self.model(inputs)
        h,w = labels.shape[-2:]
        if outputs[0].shape[-2:] != (h,w):
            outputs = [F.interpolate(o,size=(h,w),mode="bilinear",
                       align_corners=LOSSCFG.ALIGN_CORNERS) for o in outputs]
        acc = self.pixel_acc(outputs[-2],labels)
        loss_s = self.sem_loss(outputs[:-1],labels)
        loss_b = self.bd_loss(outputs[-1],bd_gt)
        filler = torch.ones_like(labels)*self.ignore_label
        bd_label = torch.where(torch.sigmoid(outputs[-1][:,0]) > 0.8, labels, filler)
        loss_sb = self.sem_loss(outputs[-2],bd_label)
        loss = loss_s + loss_b + loss_sb
        return loss.unsqueeze(0), outputs[:-1], acc, [loss_s,loss_b]
