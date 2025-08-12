import gradio as gr
import joblib, json

pipe = joblib.load("spam_nb_tfidf.joblib")
with open("threshold.json") as f:
    T = json.load(f)["threshold"]

def predict_sms(msg: str):
    msg = (msg or "").strip()
    if not msg:
        return "—", 0.0
    p = float(pipe.predict_proba([msg])[0, 1])
    label = "SPAM" if p >= T else "HAM"
    return label, round(p, 4)

demo = gr.Interface(
    fn=predict_sms,
    inputs=gr.Textbox(lines=4, label="Message"),
    outputs=[gr.Label(label="Prediction"), gr.Number(label="P(spam)")],
    title="Spam Detector (TF-IDF + Naive Bayes)",
    description=f"Decision threshold t = {T:.2f} | TF-IDF(1–2) + MultinomialNB",
    examples=[
        ["Congratulations! You’ve won a free vacation. Click here to claim now."],
        ["Are we still on for class at 3 pm?"],
        ["Reply STOP to unsubscribe from alerts."],
        ["URGENT! Your account will be suspended. Verify at http://bit.ly/xyz"],
    ],
)

if __name__ == '__main__':
    demo.launch()
