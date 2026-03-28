from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_std = float(np.std(y_true))
    y_pred_std = float(np.std(y_pred))
    if y_true_std == 0.0 or y_pred_std == 0.0:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_ranks = _average_ranks(y_true)
    pred_ranks = _average_ranks(y_pred)
    return _pcc(true_ranks, pred_ranks)


def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_clip = np.clip(y_pred, eps, 1 - eps)
    return float(-(y_true * np.log(y_clip) + (1 - y_true) * np.log(1 - y_clip)).mean())


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos_count = int((y_true == 1).sum())
    neg_count = int((y_true == 0).sum())
    if pos_count == 0 or neg_count == 0:
        raise ValueError("AUC requires both positive and negative labels.")

    ranks = _average_ranks(y_score)

    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


MetricFn = Callable[[np.ndarray, np.ndarray], float]

_METRICS: Dict[str, Tuple[MetricFn, bool]] = {
    "rmse": (_rmse, False),
    "mae": (_mae, False),
    "r2": (_r2, True),
    "spearman": (_spearman_corr, True),
    "pcc": (_pcc, True),
    "pearson": (_pcc, True),
    "logloss": (_logloss, False),
    "auc": (_auc, True),
}


@dataclass
class SelectionStep:
    iteration: int
    selected_model: str
    score: float


class CaruanaGreedyEnsembler:
    """
    Caruana-style greedy ensemble selection (with replacement).

    At each iteration:
      1) Try adding each model prediction once.
      2) Pick the best candidate by metric.
      3) Keep the selected model in the ensemble (duplicates allowed).
    """

    def __init__(
        self,
        metric: str = "rmse",
        maximize: Optional[bool] = None,
        n_iterations: int = 100,
        pred_columns: Optional[Sequence[str]] = None,
        pred_prefix: str = "pred_",
        label_col: str = "Label",
        early_stopping_rounds: Optional[int] = None,
        tol: float = 0.0,
        use_best_iteration: bool = True,
        verbose: bool = False,
    ) -> None:
        metric = metric.lower()
        if metric not in _METRICS:
            valid = ", ".join(sorted(_METRICS.keys()))
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {valid}")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be > 0")
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be > 0 when provided")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        metric_fn, metric_default_maximize = _METRICS[metric]
        self.metric_name = metric
        self.metric_fn = metric_fn
        self.maximize = metric_default_maximize if maximize is None else maximize
        self.n_iterations = n_iterations
        self.pred_columns = list(pred_columns) if pred_columns is not None else None
        self.pred_prefix = pred_prefix
        self.label_col = label_col
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.use_best_iteration = use_best_iteration
        self.verbose = verbose

        self.pred_columns_: List[str] = []
        self.selected_models_: List[str] = []
        self.model_counts_: Dict[str, int] = {}
        self.model_weights_: Dict[str, float] = {}
        self.history_: List[SelectionStep] = []
        self.best_score_: Optional[float] = None
        self.best_iteration_: Optional[int] = None

    def _resolve_pred_columns(self, df: pd.DataFrame) -> List[str]:
        if self.pred_columns is not None:
            missing = [c for c in self.pred_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Missing pred_columns in DataFrame: {missing}")
            return list(self.pred_columns)

        cols = [c for c in df.columns if c.startswith(self.pred_prefix)]
        if not cols:
            raise ValueError(
                f"No prediction columns found with prefix '{self.pred_prefix}'. "
                "Set pred_columns explicitly or adjust pred_prefix."
            )
        return cols

    def _is_better(self, score: float, best: float) -> bool:
        if self.maximize:
            return score > best + self.tol
        return score < best - self.tol

    def fit(self, df: pd.DataFrame) -> "CaruanaGreedyEnsembler":
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found.")

        pred_cols = self._resolve_pred_columns(df)
        y = df[self.label_col].to_numpy(dtype=float)
        pred_matrix = df[pred_cols].to_numpy(dtype=float)

        if np.isnan(y).any():
            raise ValueError("Label column contains NaN.")
        if np.isnan(pred_matrix).any():
            raise ValueError("Prediction columns contain NaN.")
        if self.metric_name in {"logloss", "auc"}:
            unique_labels = set(np.unique(y))
            if not unique_labels.issubset({0.0, 1.0}):
                raise ValueError(
                    f"Metric '{self.metric_name}' requires binary labels in {{0,1}}."
                )

        n_samples, n_models = pred_matrix.shape
        running_sum = np.zeros(n_samples, dtype=float)
        chosen_idx: List[int] = []
        history: List[SelectionStep] = []

        init_best = -np.inf if self.maximize else np.inf
        best_score = init_best
        best_iter = 0
        rounds_no_improve = 0

        for it in range(1, self.n_iterations + 1):
            candidate_best_score = init_best
            candidate_best_idx = -1

            denom = len(chosen_idx) + 1
            for m in range(n_models):
                candidate_pred = (running_sum + pred_matrix[:, m]) / denom
                score = self.metric_fn(y, candidate_pred)

                if candidate_best_idx < 0:
                    candidate_best_idx = m
                    candidate_best_score = score
                    continue

                if self.maximize:
                    if score > candidate_best_score:
                        candidate_best_score = score
                        candidate_best_idx = m
                else:
                    if score < candidate_best_score:
                        candidate_best_score = score
                        candidate_best_idx = m

            if candidate_best_idx < 0:
                raise RuntimeError("No candidate model selected. Check input predictions.")

            chosen_idx.append(candidate_best_idx)
            running_sum += pred_matrix[:, candidate_best_idx]
            selected_name = pred_cols[candidate_best_idx]
            history.append(
                SelectionStep(
                    iteration=it,
                    selected_model=selected_name,
                    score=float(candidate_best_score),
                )
            )

            if self.verbose:
                print(f"[{it:03d}] add={selected_name}, score={candidate_best_score:.8f}")

            if self._is_better(candidate_best_score, best_score):
                best_score = candidate_best_score
                best_iter = it
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1

            if (
                self.early_stopping_rounds is not None
                and rounds_no_improve >= self.early_stopping_rounds
            ):
                if self.verbose:
                    print(
                        f"Early stopping at iteration {it} "
                        f"(no improvement for {self.early_stopping_rounds} rounds)."
                    )
                break

        if not history:
            raise RuntimeError("Ensemble history is empty after fitting.")

        if best_iter == 0:
            best_iter = 1
            best_score = history[0].score

        final_steps = history[:best_iter] if self.use_best_iteration else history
        final_models = [s.selected_model for s in final_steps]
        counts = dict(Counter(final_models))
        total = sum(counts.values())
        weights = {k: v / total for k, v in counts.items()}

        self.pred_columns_ = pred_cols
        self.selected_models_ = final_models
        self.model_counts_ = counts
        self.model_weights_ = weights
        self.history_ = history
        self.best_score_ = float(best_score)
        self.best_iteration_ = int(best_iter)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.model_weights_:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        missing = [c for c in self.pred_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required prediction columns in DataFrame: {missing}")

        pred_matrix = df[self.pred_columns_].to_numpy(dtype=float)
        if np.isnan(pred_matrix).any():
            raise ValueError("Prediction columns contain NaN.")

        weights = np.array([self.model_weights_.get(c, 0.0) for c in self.pred_columns_])
        return pred_matrix @ weights

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.__dict__ for s in self.history_])


def greedy_ensemble(
    df: pd.DataFrame,
    metric: str = "rmse",
    maximize: Optional[bool] = None,
    n_iterations: int = 100,
    pred_columns: Optional[Sequence[str]] = None,
    pred_prefix: str = "pred_",
    label_col: str = "Label",
    early_stopping_rounds: Optional[int] = None,
    tol: float = 0.0,
    use_best_iteration: bool = True,
    verbose: bool = False,
) -> Tuple[CaruanaGreedyEnsembler, np.ndarray]:
    """
    Convenience wrapper.
    Returns (fitted_ensembler, ensemble_predictions).
    """
    ensembler = CaruanaGreedyEnsembler(
        metric=metric,
        maximize=maximize,
        n_iterations=n_iterations,
        pred_columns=pred_columns,
        pred_prefix=pred_prefix,
        label_col=label_col,
        early_stopping_rounds=early_stopping_rounds,
        tol=tol,
        use_best_iteration=use_best_iteration,
        verbose=verbose,
    )
    ensembler.fit(df)
    pred = ensembler.predict(df)
    return ensembler, pred
