
# Spam Detector (TF-IDF + Naive Bayes)

End-to-end SMS/email spam detector built in Colab.

## What’s inside
- Cleaned dataset (`spam_clean.csv` used in Colab)
- Stratified train/val/test split (70/15/15)
- TF-IDF (1–2 grams) + Multinomial Naive Bayes
- Threshold tuning on validation (best F1 at `t=0.13`)
- Final test metrics (Accuracy 97.9%, ROC-AUC 0.982, PR-AUC 0.949)
- Saved pipeline (`spam_nb_tfidf.joblib`) + threshold (`threshold.json`)
- Gradio demo (`app.py`)
- Project report (`spam_detector_report_v1.html/.docx`)

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
