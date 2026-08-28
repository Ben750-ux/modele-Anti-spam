import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Charger le dataset
df = pd.read_csv("data/messages.csv")

# Nettoyage : supprimer les doublons
df = df.drop_duplicates(subset=["text"])

# Modèle binaire spam vs ham
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model_spam = MultinomialNB()
model_spam.fit(X_train, y_train)

print("Évaluation spam vs ham :")
print(classification_report(y_test, model_spam.predict(X_test)))

# Sauvegarde
joblib.dump(model_spam, "model_spam.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Modèle multi-classe pour les catégories (seulement sur les spams)
df_spam = df[df["label"] == "spam"]
X_spam = vectorizer.transform(df_spam["text"])
y_spam = df_spam["spam_category"]

model_category = MultinomialNB()
model_category.fit(X_spam, y_spam)

print("Évaluation catégories spam :")
print(classification_report(y_spam, model_category.predict(X_spam)))

joblib.dump(model_category, "model_category.pkl")
