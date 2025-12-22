import re
import nltk
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords", quiet=True)

news_df = pd.read_csv("train.csv")
news_df = news_df.fillna("")
news_df["content"] = news_df["author"] + " " + news_df["title"]

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def stemming(content):
    content = re.sub("[^a-zA-Z]", " ", content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stop_words]
    return " ".join(content)

print("🔄 Preprocessing data...")
news_df["content"] = news_df["content"].apply(stemming)

X = news_df["content"].values
y = news_df["label"].values

vector = TfidfVectorizer()
X = vector.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print("Training Accuracy:", accuracy_score(model.predict(X_train), y_train))
print("Testing Accuracy:", accuracy_score(model.predict(X_test), y_test))

app = Flask(__name__, template_folder="templates")
CORS(app)

def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        processed = preprocess_text(text)
        vector_input = vector.transform([processed])

        prediction = model.predict(vector_input)[0]
        confidence = float(model.predict_proba(vector_input).max())

        result = "fake" if prediction == 1 else "real"

        return jsonify({
            "label": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    
    app.run(debug=False)
