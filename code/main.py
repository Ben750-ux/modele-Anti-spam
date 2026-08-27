import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Charger le dataset
df = pd.read_csv("data/messages.csv")

# 1. Vectorisation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# 2. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, df["label"], test_size=0.2, random_state=42)

# 3. Entraînement
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Prédiction
y_pred = model.predict(X_test)

# 5. Évaluation
print(classification_report(y_test, y_pred))

# 6. Sauvegarde
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# 7. Rechargement
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Exemple d’utilisation
message = ["Vous avez gagné un iPhone"]
X = vectorizer.transform(message)
print(model.predict(X))

