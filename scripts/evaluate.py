#!/usr/bin/env python3
import argparse, sys, csv, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pidnet.model import build_pidnet
from pidnet.dataset import CityscapesDataset, build_pairs, CLASS_NAMES
from pidnet.utils import confusion_matrix, iou_from_confusion, unwrap_training_state_dict

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images-dir",required=True); ap.add_argument("--labels-dir",required=True)
    ap.add_argument("--weights",default="weights/best.pt"); ap.add_argument("--output",default="results/metrics/evaluation.csv")
    args=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs,_=build_pairs(args.images_dir,args.labels_dir,"val")
    ds=CityscapesDataset(pairs,multi_scale=False,flip=False); ld=torch.utils.data.DataLoader(ds,batch_size=1)
    net=build_pidnet("s",19,augment=False).to(device).eval()
    sd=torch.load(args.weights,map_location=device,weights_only=False)
    clean=unwrap_training_state_dict(sd)
    if not clean: clean=sd
    net.load_state_dict(clean,strict=False)
    cm=np.zeros((19,19))
    with torch.no_grad():
        for x,y,_ in tqdm(ld):
            x,y=x.to(device),y.long().to(device); out=net(x)
            out=torch.nn.functional.interpolate(out,size=y.shape[-2:],mode="bilinear",align_corners=True)
            cm += confusion_matrix(y,out,19,255)
    iou,miou,pa=iou_from_confusion(cm)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["class","IoU"])
        for n,v in zip(CLASS_NAMES,iou): w.writerow([n,f"{v:.4f}"])
        w.writerow(["mIoU",f"{miou:.4f}"]); w.writerow(["pixel_acc",f"{pa:.4f}"])
    print(f"mIoU={miou:.4f} pixel_acc={pa:.4f}")

if __name__=="__main__": main()
