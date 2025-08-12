# Spam Detector (TF-IDF — Naive Bayes & Logistic Regression)

End-to-end SMS/email spam detector built in Colab.  
v1 uses **TF-IDF (1–2 grams) + Multinomial Naive Bayes**;  
v2 adds **Logistic Regression (class_weight="balanced")** and a **model toggle** in the Gradio app.

---

## Results (held-out test set)

| Model (threshold) | Accuracy | ROC-AUC | PR-AUC | Spam Precision | Spam Recall | Spam F1 | Confusion (rows = actual) |
|---|---:|---:|---:|---:|---:|---:|---|
| **Naive Bayes (t=0.13)** | **0.9793** | 0.9819 | 0.9489 | 0.9762 | 0.8542 | 0.9111 | [[676, 2], [14, 82]] |
| **Logistic Regression (t=0.57)** | **0.9884** | 0.9984 | 0.9911 | 0.9780 | 0.9271 | 0.9519 | [[676, 2], [7, 89]] |

**Takeaway:** LR halves false negatives (14→7) with the same false positives (2), boosting recall and F1. LR @ **t=0.57** is the default.

---

## What’s inside

- Cleaned dataset used in Colab: `spam_clean.csv` *(not committed if redistribution isn’t allowed)*  
- Stratified **train/val/test** split (70/15/15)
- **TF-IDF (1–2 grams, min_df=2)** features
- Models  
  - v1: **Multinomial Naive Bayes** (threshold tuned to **t=0.13**)  
  - v2: **Logistic Regression** `class_weight="balanced"` (threshold tuned to **t=0.57**)
- Threshold tuning on validation (maximize F1), final metrics on test
- **Saved artifacts** in `model/`:
  - `spam_nb_tfidf.joblib`, `threshold.json`
  - `spam_lr_tfidf.joblib`, `threshold_lr.json`
- **Gradio demo** `app.py` with **NB/LR radio toggle**
- **Reports** in `reports/`:
  - `spam_detector_report_v1.docx` (NB baseline)
  - `spam_detector_report_v2.html`, `spam_detector_report_v2.docx` (NB vs LR)

---

## Repo structure

**notebooks/**

spam_detector_v1.ipynb # full walkthrough (clean → train → tune → eval → save → demo)

**model/**

spam_nb_tfidf.joblib
threshold.json # {"threshold": 0.13}
spam_lr_tfidf.joblib
threshold_lr.json # {"threshold": 0.57}

**reports/**
spam_detector_report_v1.docx
spam_detector_report_v2.html
spam_detector_report_v2.docx

**app.py

requirements.txt

LICENSE

README.md**

---

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
---

## Reproduce / retrain (Colab)

1. Open notebooks/spam_detector_v1.ipynb in Colab.
2. Run cells through:
   
   cleaning + stratified split
   
   TF-IDF + model training (NB and/or LR)
   
   threshold tuning on validatioN
   
   final evaluation on test
   
   save pipeline + threshold to model/
   
   launch Gradio demo
   
4. (Optional) regenerate reports in reports/.

---

## Deploy options

Colab (ephemeral): demo.launch(share=True)

public link dies when runtime sleeps.

Local: python app.py

Hugging Face Spaces (recommended for a permanent link):

Create a Gradio Space and upload: app.py, requirements.txt, and files under model/.


## requirements.txt

gradio

scikit-learn

joblib

numpy

scipy

---

## Notes & limitations

1. Dataset is imbalanced (~12.5% spam) → use F1/PR-AUC + threshold tuning (not accuracy only).
2. Some obfuscated adult/promo spam can slip through. Next steps that typically help:
   
      Add character n-grams (3–5) to TF-IDF (captures “txt”, “150p”, weird encodings).
   
      Try probability calibration if you need well-calibrated P(spam).
   
      Light normalization (mask URLs/phones), then retrain/tune.

---

## License

MIT — see LICENSE.

---

## Changelog

v1: NB baseline + threshold tuning + report.

v2: Add Logistic Regression + app toggle; big recall/F1 boost.

---





