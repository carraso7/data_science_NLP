import os
import re
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _tokenize(text: str):
    return TOKEN_PATTERN.findall(_safe_text(text).lower())


def get_data_splits(quora_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a_df, test_df = train_test_split(quora_df, test_size=0.05, random_state=123)
    train_df, val_df = train_test_split(a_df, test_size=0.05, random_state=123)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def jaccard_similarity(text_1: str, text_2: str) -> float:
    tokens_1 = set(_tokenize(text_1))
    tokens_2 = set(_tokenize(text_2))

    union_size = len(tokens_1 | tokens_2)
    if union_size == 0:
        return 0.0

    return len(tokens_1 & tokens_2) / union_size


def get_jaccard_features(df: pd.DataFrame) -> np.ndarray:
    values = [
        jaccard_similarity(q1, q2)
        for q1, q2 in zip(df["question1"].values, df["question2"].values)
    ]
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def get_length_features(df: pd.DataFrame) -> np.ndarray:
    rows = []

    for q1_raw, q2_raw in zip(df["question1"].values, df["question2"].values):
        q1 = _safe_text(q1_raw)
        q2 = _safe_text(q2_raw)

        q1_tokens = _tokenize(q1)
        q2_tokens = _tokenize(q2)

        q1_token_set = set(q1_tokens)
        q2_token_set = set(q2_tokens)

        len_1 = len(q1)
        len_2 = len(q2)

        word_count_1 = len(q1_tokens)
        word_count_2 = len(q2_tokens)

        max_word_count = max(word_count_1, word_count_2)
        if max_word_count == 0:
            common_word_ratio = 0.0
        else:
            common_word_ratio = len(q1_token_set & q2_token_set) / max_word_count

        max_len = max(len_1, len_2)
        if max_len == 0:
            len_ratio = 0.0
        else:
            len_ratio = min(len_1, len_2) / max_len

        rows.append(
            [
                abs(len_1 - len_2),
                abs(word_count_1 - word_count_2),
                common_word_ratio,
                len_ratio,
            ]
        )

    return np.asarray(rows, dtype=np.float32)


def fit_count_vectorizer(
    train_df: pd.DataFrame,
    ngram_range=(1, 1),
    max_features=None,
) -> CountVectorizer:
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    all_train_questions = pd.concat(
        [train_df["question1"].fillna(""), train_df["question2"].fillna("")],
        axis=0,
        ignore_index=True,
    )
    vectorizer.fit(all_train_questions.values)
    return vectorizer


def get_bow_features(df: pd.DataFrame, vectorizer: CountVectorizer):
    q1 = vectorizer.transform(df["question1"].fillna("").values)
    q2 = vectorizer.transform(df["question2"].fillna("").values)
    return sparse.hstack([q1, q2], format="csr")


def fit_tfidf_vectorizer(
    train_df: pd.DataFrame,
    ngram_range=(1, 2),
    max_features=50000,
) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    all_train_questions = pd.concat(
        [train_df["question1"].fillna(""), train_df["question2"].fillna("")],
        axis=0,
        ignore_index=True,
    )
    vectorizer.fit(all_train_questions.values)
    return vectorizer


def cosine_similarity_sparse(v1, v2) -> float:
    numerator = v1.dot(v2.T).toarray()[0, 0]
    norm_1 = np.sqrt(v1.dot(v1.T).toarray()[0, 0])
    norm_2 = np.sqrt(v2.dot(v2.T).toarray()[0, 0])

    denominator = norm_1 * norm_2
    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def get_tfidf_cosine_features(df: pd.DataFrame, tfidf_vectorizer: TfidfVectorizer) -> np.ndarray:
    q1_tfidf = tfidf_vectorizer.transform(df["question1"].fillna("").values)
    q2_tfidf = tfidf_vectorizer.transform(df["question2"].fillna("").values)

    dot_products = q1_tfidf.multiply(q2_tfidf).sum(axis=1).A1
    norms_1 = np.sqrt(q1_tfidf.multiply(q1_tfidf).sum(axis=1)).A1
    norms_2 = np.sqrt(q2_tfidf.multiply(q2_tfidf).sum(axis=1)).A1

    denominators = norms_1 * norms_2
    cosine_values = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=np.float32),
        where=denominators != 0,
    )
    return cosine_values.astype(np.float32).reshape(-1, 1)


def build_all_features(
    df: pd.DataFrame,
    count_vectorizer: CountVectorizer,
    tfidf_vectorizer: TfidfVectorizer,
):
    x_bow = get_bow_features(df, count_vectorizer)
    x_jaccard = sparse.csr_matrix(get_jaccard_features(df))
    x_tfidf_cosine = sparse.csr_matrix(get_tfidf_cosine_features(df, tfidf_vectorizer))
    x_length = sparse.csr_matrix(get_length_features(df))

    return sparse.hstack([x_bow, x_jaccard, x_tfidf_cosine, x_length], format="csr")


def evaluate_model(model, x, y_true: np.ndarray, name: str = "model") -> Dict[str, float]:
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(x)[:, 1]
    else:
        y_scores = model.decision_function(x)

    y_pred = (y_scores >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    print(
        f"{name}: "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}"
    )

    return metrics


def save_model(model, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)
