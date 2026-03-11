# 🏠 Multimodal ML — Housing Price Prediction

## Objective
Predict housing sale prices by combining **house exterior images** and
**structured tabular data** using a multimodal deep learning pipeline.

---

## Datasets
| Dataset | Description |
|---------|-------------|
| King County House Sales (Kaggle) | 21,613 sales records with 19 features |
| SoCal House Images (Kaggle) | Exterior house photos with sale prices |

> Download datasets from Kaggle and place in the project folder:
> - [King County House Sales](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) → save as `kc_house_data.csv`
> - [SoCal House Images](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal) → extract images to `socal_images/socal_pics/` and save CSV as `socal2.csv`

---

## Approach

1. **Image Branch** — MobileNetV2 (pretrained, frozen) extracts visual
   features from house exterior photos
2. **Tabular Branch** — MLP processes structured features
   (sqft, bedrooms, location, grade etc.)
3. **Feature Fusion** — Both branches concatenated into a combined
   feature vector fed into a regression head
4. **Baseline** — Tabular-only MLP trained for comparison

---

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Baseline (Tabular Only) | $441,262 | $628,533 | 84.12% |
| Multimodal (CNN + Tabular) | $585,618 | $1,557,415 | 74.00% |

---

## Key Observations
- Baseline achieved lower MAE and RMSE due to a **geographic mismatch**
  between datasets — images (SoCal) and tabular data (King County, WA)
  came from different markets, introducing noise into the fusion layer
- Multimodal model achieved **better MAPE (74% vs 84%)** showing images
  did contribute proportional pricing signal
- Under a properly aligned single dataset, multimodal fusion is
  expected to outperform tabular-only models

---

## Tech Stack
`Python` `TensorFlow/Keras` `MobileNetV2` `Scikit-learn` `Google Colab (T4 GPU)`