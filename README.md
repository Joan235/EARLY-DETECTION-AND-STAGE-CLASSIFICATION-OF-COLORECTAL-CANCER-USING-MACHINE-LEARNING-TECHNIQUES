# CRC Analytics Flask App

Lightweight Flask app with Tailwind UI for CRC detection, stage classification, and a survival analysis dashboard using trained ML models.

## Features

- **Page 1 — CRC Risk Detection**: Form with patient inputs (age, BMI, lifestyle, nutrition). Returns risk prediction with SHAP feature importance.
- **Page 2 — Stage Classification**: Predicts cancer stage (I–IV) with confidence probabilities and urgency messaging.
- **Page 3 — Survival Dashboard**: Read-only analytics dashboard showing population-level KM curves, Cox regression forest, treatment outcomes, and subgroup summaries.

## Quick Start

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure your dataset is at `datasets/crc_dataset.csv`.

3. Trained models are expected in:
   - `saved_models for cancer dectection/` (detection model)
   - `saved_models for stage classification/` (stage model)
   - `saved_models for survival analysis/` (Cox model for dashboard)

4. Run the app:

```bash
python app.py
```

Open http://localhost:5000 in your browser.

If you don't want to commit model binaries to the repository (recommended), store trained models externally and download them before running the app. A helper script is provided:

```bash
python scripts/download_models.py
```

Edit `scripts/download_models.py` to set the URLs for the `MODELS` dictionary (replace the placeholder strings with your model file URLs). You can upload models to cloud storage (S3, Google Drive, or Hugging Face) and paste the public URLs into the script.

## Architecture

- `app.py` – Flask routes, model loading, and API endpoints (`/api/predict-risk`, `/api/predict-stage`)
- `analysis.py` – Kaplan-Meier, Cox regression, table generation for survival dashboard
- `templates/` – Jinja2 HTML templates with Tailwind CSS (CDN)
- `requirements.txt` – Python dependencies

## Model Loading

Models are loaded at startup from the saved_models directories:
- Detection: LightGBM with StandardScaler and label encoders
- Stage: LightGBM with StandardScaler and label encoders
- Survival: Cox PH model for forest plot

SHAP explanations are computed server-side for risk predictions.

## Notes

- This is a production-ready scaffold; adjust feature names and model paths to match your notebooks.
- The `/api/predict-*` endpoints accept JSON and return JSON responses.
- Static plots (KM curves, Cox forest) are cached in `static/plots/`.

