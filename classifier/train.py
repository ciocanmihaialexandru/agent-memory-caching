"""Antrenează un intent classifier: TF-IDF (word + char) + LogisticRegression.
Evaluează cu cross-validation (accuracy stabil), apoi salvează modelul pe tot setul.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

from classifier.training_data import TRAIN_DATA

MODEL_PATH = Path(__file__).parent / "intent_model.joblib"


def build_pipeline() -> Pipeline:
    # combinăm două reprezentări: cuvinte (1-2 grame) + caractere (2-4 grame)
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    texts = [t for t, _ in TRAIN_DATA]
    labels = [lbl for _, lbl in TRAIN_DATA]

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. accuracy stabil, mediat pe 5 felii (nu depinde de un singur split)
    scores = cross_val_score(pipe, texts, labels, cv=cv, scoring="accuracy")
    print(f"Accuracy CV (5-fold): {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"  per felie: {[round(s, 2) for s in scores]}\n")

    # 2. raport per-clasă din predicții out-of-fold (fiecare exemplu testat o dată)
    y_pred = cross_val_predict(pipe, texts, labels, cv=cv)
    print("=== Raport (out-of-fold) ===")
    print(classification_report(labels, y_pred, zero_division=0))

    # 3. antrenăm pe TOT setul și salvăm
    pipe.fit(texts, labels)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model salvat în: {MODEL_PATH}")


if __name__ == "__main__":
    main()