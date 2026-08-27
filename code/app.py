from flask import Flask, request, render_template
import joblib

# Charger le modèle et le vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Récupérer les messages collés dans le textarea
        messages = request.form["messages"].split("\n")
        # Transformer et prédire
        X = vectorizer.transform(messages)
        predictions = model.predict(X)
        results = list(zip(messages, predictions))
        return render_template("results.html", results=results)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
