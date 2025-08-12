"""
Spam Detector (TF-IDF) — NB & LR toggle

Loads any available models from ./model:
- Naive Bayes:  model/spam_nb_tfidf.joblib  + model/threshold.json
- LogisticReg:  model/spam_lr_tfidf.joblib  + model/threshold_lr.json

Returns (label, P(spam)) via a simple Gradio UI.
"""


import os
import json
import joblib
import gradio as gr

# ---------- Load models + thresholds ----------
MODELS = {}
THRESH = {}

def load_pair(name: str, model_path: str, thr_path: str):
    if os.path.exists(model_path) and os.path.exists(thr_path):
        MODELS[name] = joblib.load(model_path)
        with open(thr_path) as f:
            THRESH[name] = float(json.load(f)["threshold"])

load_pair("Naive Bayes",        "model/spam_nb_tfidf.joblib", "model/threshold.json")
load_pair("Logistic Regression","model/spam_lr_tfidf.joblib", "model/threshold_lr.json")

if not MODELS:
    raise RuntimeError(
        "No models found. Place artifacts in ./model:\n"
        "- NB: spam_nb_tfidf.joblib + threshold.json\n"
        "- LR: spam_lr_tfidf.joblib + threshold_lr.json"
    )

DEFAULT_MODEL = "Logistic Regression" if "Logistic Regression" in MODELS else list(MODELS.keys())[0]

# ---------- Inference ----------
def predict_sms(msg: str, model_name: str):
    msg = (msg or "").strip()
    if not msg:
        return "—", 0.0
    pipe = MODELS[model_name]
    t = THRESH[model_name]
    p = float(pipe.predict_proba([msg])[0, 1])
    label = "SPAM" if p >= t else "HAM"
    return label, round(p, 4)

# ---------- UI ----------
desc = (
    "Choose a model. LR @ t=0.57 → higher recall/F1 (your v2 default). "
    "NB @ t=0.13 → solid baseline.\n"
    "Tip: prediction uses the saved threshold for the selected model."
)

demo = gr.Interface(
    fn=predict_sms,
    inputs=[
        gr.Textbox(lines=4, label="Message", placeholder="Paste an SMS/email..."),
        gr.Radio(choices=list(MODELS.keys()), value=DEFAULT_MODEL, label="Model"),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="P(spam)")
    ],
    title="Spam Detector (TF-IDF) — NB vs LR",
    description=desc,
    examples=[
        ["Congratulations! You’ve won a free vacation. Click here to claim now.", DEFAULT_MODEL],
        ["Are we still on for class at 3 pm?", DEFAULT_MODEL],
        ["Reply STOP to unsubscribe from alerts.", DEFAULT_MODEL],
        ["URGENT! Your account will be suspended. Verify at http://bit.ly/xyz", DEFAULT_MODEL],
    ],
)

if __name__ == "__main__":
    # In Colab you’ll get a public link; locally you’ll get http://127.0.0.1:7860
    demo.launch(share=True)


"""
****************************************************************************************************************************************************************
"DONOT COPY THIS"

How to use
Replace your repo’s app.py with this file.

Make sure these exist in the repo (or Colab working dir):

model/spam_lr_tfidf.joblib + model/threshold_lr.json

(optional) model/spam_nb_tfidf.joblib + model/threshold.json

Run it:

Colab: just run the cell with python app.py or demo.launch(...) as above.

Local:
pip install -r requirements.txt
python app.py

open the printed URL.

******************************************************************************************************************************************************************
"""
