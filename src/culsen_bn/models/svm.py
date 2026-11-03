"""
Classical Machine Learning Baseline for CulSen-BN:
Character TF-IDF + Calibrated Linear Support Vector Machine (Linear SVM).
Runs efficiently on CPU with balanced class weights and fixed random seed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import pickle


class BaselineSVMClassifier:
    """
    Character TF-IDF + Linear Support Vector Classifier baseline.
    Vectorization is fit strictly on training data to prevent leakage.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 5),
        min_df: int = 2,
        max_features: int = 300000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = 42,
        max_iter: int = 3000,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter

        self.vectorizer = None
        self.classifier = None
        self.calibrated_model = None
        self.training_time_sec = 0.0

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "BaselineSVMClassifier":
        """
        Fit character TF-IDF vectorizer and LinearSVC strictly on training data.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        start_time = time.time()

        # 1. Fit Vectorizer on Training Data ONLY
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        X_train = self.vectorizer.fit_transform(train_texts)

        # 2. Fit Linear Support Vector Classifier
        self.classifier = LinearSVC(
            C=self.C,
            loss="squared_hinge",
            penalty="l2",
            dual="auto",
            tol=1e-4,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.classifier.fit(X_train, train_labels)

        # 3. Fit Platt calibration for well-calibrated probabilities
        try:
            from sklearn.frozen import FrozenEstimator
            self.calibrated_model = CalibratedClassifierCV(
                estimator=FrozenEstimator(self.classifier),
                method="sigmoid",
            )
        except ImportError:
            try:
                from sklearn.calibration import FrozenEstimator
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(self.classifier),
                    method="sigmoid",
                )
            except ImportError:
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=self.classifier,
                    method="sigmoid",
                    cv="prefit",
                )
        self.calibrated_model.fit(X_train, train_labels)

        self.training_time_sec = time.time() - start_time
        return self

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict binary sentiment labels (0 = negative, 1 = positive).
        """
        if self.classifier is None or self.vectorizer is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """
        Predict calibrated class probabilities.
        """
        if self.calibrated_model is None or self.vectorizer is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.calibrated_model.predict_proba(X).tolist()

    def decision_function(self, texts: List[str]) -> List[float]:
        """
        Raw signed distance from decision boundary.
        """
        if self.classifier is None or self.vectorizer is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.classifier.decision_function(X).tolist()

    def save(self, filepath: str) -> None:
        """
        Serialize fitted vectorizer and model.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BaselineSVMClassifier":
        """
        Deserialize fitted vectorizer and model.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
