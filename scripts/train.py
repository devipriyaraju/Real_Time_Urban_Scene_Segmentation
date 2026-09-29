#!/usr/bin/env python3
import argparse, os, sys, time, yaml, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pidnet.model import build_pidnet
from pidnet.dataset import CityscapesDataset, build_pairs, CLASS_WEIGHTS
from pidnet.losses import OhemCrossEntropy, BoundaryLoss, FullModel, LOSSCFG
from pidnet.utils import AverageMeter, seed_everything, confusion_matrix, iou_from_confusion, poly_lr

def validate(model,loader,device,nc,ignore):
    model.eval(); cm=np.zeros((nc,nc)); losses=AverageMeter()
    with torch.no_grad():
        for image,label,bd in tqdm(loader,desc="val",leave=False):
            image,label,bd=image.to(device),label.long().to(device),bd.float().to(device)
            loss,preds,_,_=model(image,label,bd)
            main=preds[-1]
            cm += confusion_matrix(label,main,nc,ignore)
            losses.update(loss.mean().item())
    iou,miou,_=iou_from_confusion(cm)
    return losses.average(),miou,iou

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/pidnet_s_cityscapes.yaml")
    ap.add_argument("--images-dir")
    ap.add_argument("--labels-dir")
    ap.add_argument("--imagenet")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    images=args.images_dir or cfg["data"]["images_dir"]
    labels=args.labels_dir or cfg["data"]["labels_dir"]
    out=Path(cfg["output"]["dir"]); out.mkdir(parents=True,exist_ok=True)
    seed=cfg["training"]["seed"]; seed_everything(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pairs,_=build_pairs(images,labels,"train"); val_pairs,_=build_pairs(images,labels,"val")
    if not train_pairs or not val_pairs: raise RuntimeError("Cityscapes train/val pairs not found.")
    d=cfg["data"]; t=cfg["training"]
    train_ds=CityscapesDataset(train_pairs,tuple(d["crop_size"]),d["base_size"],d["scale_factor"],True,True,d["ignore_label"])
    val_ds=CityscapesDataset(val_pairs,multi_scale=False,flip=False,ignore_label=d["ignore_label"])
    train_ld=torch.utils.data.DataLoader(train_ds,batch_size=t["batch_size"],shuffle=True,num_workers=t["workers"],drop_last=True)
    val_ld=torch.utils.data.DataLoader(val_ds,batch_size=t["val_batch_size"],shuffle=False,num_workers=t["workers"])
    net=build_pidnet(cfg["model"]["size"],cfg["model"]["num_classes"],augment=True)
    pre=args.imagenet or cfg.get("pretrained",{}).get("imagenet")
    if pre:
        ck=torch.load(pre,map_location="cpu",weights_only=False); sd=ck.get("state_dict",ck); md=net.state_dict()
        loaded={k:v for k,v in sd.items() if k in md and v.shape==md[k].shape}; md.update(loaded); net.load_state_dict(md,strict=False)
        print(f"ImageNet init: loaded {len(loaded)}/{len(md)} tensors")
    sem=OhemCrossEntropy(d["ignore_label"],LOSSCFG.OHEM_THRES,LOSSCFG.OHEM_KEEP,CLASS_WEIGHTS.to(device))
    model=FullModel(net,sem,BoundaryLoss(),d["ignore_label"]).to(device)
    opt=torch.optim.SGD(model.parameters(),lr=t["base_lr"],momentum=t["momentum"],weight_decay=t["weight_decay"])
    start,best=0,0.0; ckpt=out/"checkpoint.pth.tar"
    if args.resume and ckpt.exists():
        ck=torch.load(ckpt,map_location="cpu",weights_only=False); model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        start,best=ck["epoch"],float(ck["best_mIoU"])
    epoch_iters=len(train_ld); max_iters=t.get("schedule_epochs",t["epochs"])*epoch_iters
    for epoch in range(start,t["epochs"]):
        model.train(); meter=AverageMeter()
        for i,(x,y,b) in enumerate(tqdm(train_ld,desc=f"epoch {epoch}",leave=False)):
            x,y,b=x.to(device),y.long().to(device),b.float().to(device)
            opt.zero_grad(); loss=model(x,y,b)[0].mean(); loss.backward(); opt.step()
            poly_lr(opt,t["base_lr"],max_iters,epoch*epoch_iters+i,t.get("poly_power",0.9)); meter.update(loss.item())
        if epoch%t["val_every"]==0 or epoch>=t["epochs"]-100 or epoch==t["epochs"]-1:
            vl,miou,_=validate(model,val_ld,device,cfg["model"]["num_classes"],d["ignore_label"])
            if miou>best: best=miou; torch.save(model.state_dict(),out/"best.pt")
            print(f"[{epoch:03d}] loss={meter.average():.4f} val_loss={vl:.4f} mIoU={miou:.4f} best={best:.4f}")
        torch.save({"epoch":epoch+1,"best_mIoU":best,"state_dict":model.state_dict(),"optimizer":opt.state_dict()},ckpt)
    torch.save(model.state_dict(),out/"final_state.pt")

if __name__=="__main__": main()
