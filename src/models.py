from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class TrainedModel:
    model: Pipeline

    def predict_proba(self, X) -> np.ndarray:
        # Returns probability of class 1 (fraud)
        return self.model.predict_proba(X)[:, 1]

def train_logreg_baseline(X_train, y_train, seed: int) -> TrainedModel:
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=200,
            random_state=seed,
            class_weight="balanced",  # important for imbalance baseline
            n_jobs=None,
        ))
    ])
    pipe.fit(X_train, y_train)
    return TrainedModel(model=pipe)
