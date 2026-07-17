# baselines/tfidf_lr.py
"""
TF-IDF + Logistic Regression baseline
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.framework.data import load_splits, extract_conversation_text
from src.framework.evaluate import evaluate_predictions, print_results, save_results
from src.framework.config import OUTPUT_DIR


# Extracts raw conversation text from each row in the dataframe
def build_texts(df):
    return [extract_conversation_text(row) for _, row in df.iterrows()]


def main():
    train_df, _, test_df = load_splits()

    X_train = build_texts(train_df)
    X_test = build_texts(test_df)
    y_train = train_df["label"].tolist()
    y_test = test_df["label"].tolist()

    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # Unigrams and bigrams, sublinear TF scaling, balanced class weights
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test).tolist()

    results = evaluate_predictions(y_test, preds)
    print_results(results)

    save_results(
        results, "tfidf_lr", "baseline",
        OUTPUT_DIR / "results",
        preds=preds, labels=y_test,
    )


if __name__ == "__main__":
    main()
