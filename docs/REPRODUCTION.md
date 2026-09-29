# Experiment A reproduction notes

These notes describe the Experiment A setup used in `CITY_LANDSCAPE.ipynb`.

- Dataset: Cityscapes fine annotations, 2,975 train images and 500 validation images.
- Model: PIDNet-S, 19 classes.
- Training horizon used for Experiment A: 150 epochs.
- Crop: 1024 x 1024.
- Batch size: 6.
- Optimizer: SGD, LR 0.01, momentum 0.9, weight decay 5e-4.
- Poly LR power: 0.9.
- OHEM threshold: 0.9; minimum kept: 131072.
- Boundary BCE coefficient: 20.
- Seed: 304.
- The notebook contains multiple exploratory/restart cells. The reported Experiment A results correspond to the saved checkpoint and evaluation outputs included here.
- The notebook documents an ImageNet-initialized/resumed path. Supply the original
  PIDNet-S ImageNet checkpoint with `--imagenet` when reproducing that initialization.

The original PIDNet paper trained with a substantially longer recipe. This repository's
66.24% validation mIoU is a 150-epoch implementation result and must not be presented as
the paper's reported benchmark.
