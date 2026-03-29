# Example dataset metadata

This folder contains examples you can copy into `data/` after downloading the Stanford COCA dataset.

## 1) Create your real data folder

Create:

```
data/
  raw/
  processed/
  metadata.csv
```

Note: `data/` is ignored by git in this scaffold.

## 2) Fill `data/metadata.csv`

Use [metadata.csv](refer: data_example/metadata.csv) as a template.

Next steps:
- Ensure `label` matches your intended class definition.
- Ensure `image` points to a DICOM series folder (or a .npy file if you export volumes).