"""Predicție de intenție cu modelul antrenat (încărcat o singură dată, lazy)."""

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "intent_model.joblib"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)   # încărcat o singură dată per proces
    return _model


def predict_intent(query: str) -> str:
    """Întoarce intenția prezisă: 'search' / 'extract' / 'summarize'."""
    return _get_model().predict([query])[0]


def predict_with_confidence(query: str) -> tuple[str, float]:
    """Intenția + încrederea (probabilitatea celei mai probabile clase)."""
    model = _get_model()
    label = model.predict([query])[0]
    confidence = float(model.predict_proba([query]).max())
    return label, confidence