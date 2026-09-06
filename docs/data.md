# Data acquisition

This repo does not redistribute any dataset. `data/` is gitignored; populate it
yourself following the layout below.

## Kermany chest X-ray

Source: https://data.mendeley.com/datasets/rscbjbr9sj/2 (Kermany et al., 2018).
License requires attribution; do not redistribute raw images.

```
data/kermany/
  train/{NORMAL,PNEUMONIA}/*.jpeg
  val/{NORMAL,PNEUMONIA}/*.jpeg
  test/{NORMAL,PNEUMONIA}/*.jpeg
```

## APTOS 2019 (diabetic retinopathy)

Source: the pre-split Kaggle dataset mirror `mariaherrerot/aptos2019` (not the
raw competition dataset). This mirror ships its own train/val/test split, which
`src/datasets/aptos.py` reads directly. Requires `KAGGLE_USERNAME` /
`KAGGLE_KEY` env vars (a Kaggle account and API token); download via:

```bash
kaggle datasets download mariaherrerot/aptos2019 -p data/aptos --unzip
```

```
data/aptos/
  train_1.csv          # id_code/diagnosis columns (or similarly-named)
  valid.csv
  test.csv
  train_images/[train_images/]*.png   # folder may be double-nested in this mirror
  val_images/[val_images/]*.png
  test_images/[test_images/]*.png
```

Labels are binarized to referable-DR vs not (grades {2,3,4} = positive) by
`src/datasets/aptos.py`; see `POS_GRADES` there to change the grouping.

## ISIC 2019 (skin lesion)

Source: the Kaggle mirror `andrewmvd/isic-2019` (~9GB), 8-class ground truth
(MEL, NV, BCC, AK, BKL, DF, VASC, SCC) collapsed to malignant
({MEL,BCC,AK,SCC}) vs benign ({NV,BKL,DF,VASC}) by `src/datasets/isic.py`.
Unlike Kermany and APTOS there is no provided split, so a single seeded 70/10/20
train/val/test shuffle is used. Requires `KAGGLE_USERNAME` / `KAGGLE_KEY` env
vars; download via:

```bash
kaggle datasets download andrewmvd/isic-2019 -p data/isic --unzip
```

```
data/isic/
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Training_Input/[ISIC_2019_Training_Input/]*.jpg   # folder may be double-nested
```

## Splits

No split indices are checked into the repo. Every script re-derives the same
split deterministically from the dataset's config (`seed`, and for
Kermany and ISIC also `val_frac`/`test_frac`) via `build_splits()` in
`src/datasets/{kermany,aptos,isic}.py`. Given the same raw data directory
and config, the split is reproducible without shipping separate index files.
