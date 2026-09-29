import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from pidnet.model import build_pidnet

def test_pidnet_s_shape():
    model=build_pidnet("s",19,augment=False).eval()
    with torch.no_grad():
        y=model(torch.randn(1,3,256,512))
    assert y.shape[1] == 19
