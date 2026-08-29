# LapValue AI — Laptop Price Predictor

A complete machine-learning regression project with a polished Streamlit interface. It transforms raw laptop specifications, compares ensemble models, evaluates on an untouched test set, saves the winning end-to-end pipeline, and serves instant estimates.

## Highlights

- Real dataset with 1,300+ historical laptop listings
- Reusable feature engineering for CPU, GPU, storage, screen, memory, OS, and weight
- Reproducible 80/20 train/test split (`random_state=42`)
- Random Forest vs Extra Trees model comparison
- Log-target transformation to handle the skewed price distribution
- Unknown-category-safe preprocessing and serialized inference pipeline
- Responsive dark Streamlit UI with EUR and adjustable INR presentation
- Unit test for critical feature parsing

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Project structure

```text
├── app.py                      # Streamlit application
├── train.py                    # Training and evaluation workflow
├── src/features.py             # Shared feature engineering
├── data/raw/laptop_price.csv   # Historical source dataset
├── artifacts/                  # Trained pipeline, metrics, predictions
├── tests/test_features.py      # Feature extraction test
└── requirements.txt            # Deployment dependencies
```

## Evaluation

The training script selects the candidate with the lowest mean absolute error on the held-out test set. Generated metrics are saved in `artifacts/metadata.json`; predictions are saved separately for audit and plotting.

## Dataset and limitations

The dataset is the widely circulated *Laptop Price* dataset containing historical European listings and prices in euros. A copy used here was obtained from [TMaiza/ECD_proyecto_g15](https://github.com/TMaiza/ECD_proyecto_g15/blob/460355698121dae98e596dc516cab111ca546bf2/laptop_price.csv).

This project is an educational estimator, not a live marketplace quote. The data is historical; product age, condition, region, tax, launch year, and real-time supply are not represented. The INR output is a display conversion controlled by the user.

## Responsible use

Do not use the prediction as the sole basis for a purchase or financial decision. Retrain with recent, region-specific transaction data before production use.
