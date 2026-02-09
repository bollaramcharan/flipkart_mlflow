from prefect import flow, task
import subprocess
import sys

@task
def train_sentiment_model():
    print("Running Sentiment Analysis training script...")
    subprocess.run(
        [sys.executable, "train_sentiment_mlflow.py"],
        check=True
    )
    print("Training completed successfully.")

@flow(name="Sentiment-Analysis-MLflow-Pipeline")
def sentiment_pipeline():
    train_sentiment_model()

if __name__ == "__main__":
    sentiment_pipeline()
