# HAM10000 Dataset Download Instructions

## Overview

The **HAM10000** (Human Against Machine 10000) dataset contains **10,015** dermatoscopic images
across 7 skin lesion classes. It is the standard benchmark for skin lesion classification.

**Citation:**
> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset. *Sci. Data* 5, 180161 (2018).
> https://doi.org/10.1038/sdata.2018.161

---

## Class Distribution

| Abbreviation | Full Name                                     | Samples |
|:------------:|:----------------------------------------------|:-------:|
| `nv`         | Melanocytic nevi                              | 6,705   |
| `mel`        | Melanoma                                      | 1,113   |
| `bkl`        | Benign keratosis-like lesions                 | 1,099   |
| `bcc`        | Basal cell carcinoma                          | 514     |
| `akiec`      | Actinic keratoses / intraepithelial carcinoma | 327     |
| `vasc`       | Vascular lesions                              | 142     |
| `df`         | Dermatofibroma                                | 115     |

---

## Option 1: Kaggle (Recommended)

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/HAM10000
```

## Option 2: ISIC Archive (Official)

```bash
pip install isic-cli
isic dataset download ISIC_2018_Task3_Training_Input --output data/HAM10000/images
```

---

## Expected Structure After Download

```
data/HAM10000/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0024306.jpg
    └── ...
```

## Verify Download

```bash
python -c "
from datasets import HAM10000Dataset
ds = HAM10000Dataset('data/HAM10000', split='train')
print(f'Train samples: {len(ds)} — OK')
"
```
