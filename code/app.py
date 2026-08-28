import os
from flask import Flask, render_template, request
import joblib

# Charger les modèles et le vectorizer
model_spam = joblib.load("model_spam.pkl")
model_category = joblib.load("model_category.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Récupérer le message unique
    message = request.form["message"]

    # Transformer et prédire spam/ham
    X = vectorizer.transform([message])
    prediction_spam = model_spam.predict(X)[0]

    category = None
    if prediction_spam == "spam":
        category = model_category.predict(X)[0]

    return render_template("results.html", prediction=prediction_spam, category=category)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
