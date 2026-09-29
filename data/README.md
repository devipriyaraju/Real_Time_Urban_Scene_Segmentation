# Data directory

Cityscapes is not redistributed with this repository.

Expected structure:

```text
data/
├── leftImg8bit/
│   ├── train/<city>/*_leftImg8bit.png
│   ├── val/<city>/*_leftImg8bit.png
│   └── test/<city>/*_leftImg8bit.png
└── gtFine/
    ├── train/<city>/*_gtFine_labelIds.png
    ├── val/<city>/*_gtFine_labelIds.png
    └── test/<city>/
```

Pass these roots with `--images-dir` and `--labels-dir`. The original Compute Canada
experiment used separately extracted `leftImg8bit_trainvaltest/leftImg8bit` and
`gtFine_trainvaltest/gtFine` directories.
