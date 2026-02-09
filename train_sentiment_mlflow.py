import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment-Analysis-Experiment")

# Load data
df = pd.read_csv("data.csv")

X = df["Review text"]
y = df["Ratings"]

# Convert ratings to sentiment
y = y.apply(lambda x: "positive" if x >= 4 else "negative")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

with mlflow.start_run(run_name="Prefect_Sentiment_Training"):
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TFIDF")
    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    mlflow.log_artifact("sentiment_model.pkl")
    mlflow.log_artifact("tfidf_vectorizer.pkl")

print("Training completed and logged to MLflow")
