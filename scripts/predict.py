#!/usr/bin/env python3
import argparse, sys, cv2, torch, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pidnet.model import build_pidnet
from pidnet.dataset import MEAN, STD
from pidnet.utils import unwrap_training_state_dict
PALETTE=np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),
(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),
(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)],np.uint8)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--weights",default="weights/best.pt"); ap.add_argument("--output",default="prediction.png")
    a=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bgr=cv2.imread(a.image); x=((bgr[:,:,::-1].astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)[None]
    net=build_pidnet("s",19,False).to(device).eval(); sd=torch.load(a.weights,map_location=device,weights_only=False)
    clean=unwrap_training_state_dict(sd); net.load_state_dict(clean if clean else sd,strict=False)
    with torch.no_grad():
        out=net(torch.from_numpy(x).float().to(device)); out=torch.nn.functional.interpolate(out,size=bgr.shape[:2],mode="bilinear",align_corners=True)
    seg=PALETTE[out.argmax(1)[0].cpu().numpy()]; cv2.imwrite(a.output,cv2.cvtColor(seg,cv2.COLOR_RGB2BGR)); print(a.output)
if __name__=="__main__": main()
