from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    fold: int
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
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        stratified: Optional[bool] = None,
        pred_columns: Optional[Sequence[str]] = None,
        pred_prefix: str = "pred_",
        label_col: str = "Label",
        early_stopping_rounds: Optional[int] = None,
        tol: float = 0.0,
        verbose: bool = False,
    ) -> None:
        metric = metric.lower()
        if metric not in _METRICS:
            valid = ", ".join(sorted(_METRICS.keys()))
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {valid}")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be > 0")
        if n_splits <= 0:
            raise ValueError("n_splits must be > 0")
        if early_stopping_rounds is not None and early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be > 0 when provided")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        metric_fn, metric_default_maximize = _METRICS[metric]
        self.metric_name = metric
        self.metric_fn = metric_fn
        self.maximize = metric_default_maximize if maximize is None else maximize
        self.n_iterations = n_iterations
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified
        self.pred_columns = list(pred_columns) if pred_columns is not None else None
        self.pred_prefix = pred_prefix
        self.label_col = label_col
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.verbose = verbose

        self.pred_columns_: List[str] = []
        self.selected_models_: List[str] = []
        self.model_counts_: Dict[str, int] = {}
        self.model_weights_: Dict[str, float] = {}
        self.history_: List[SelectionStep] = []
        self.best_score_: Optional[float] = None
        self.best_iteration_: Optional[int] = None
        self.fold_model_weights_: List[Dict[str, float]] = []
        self.fold_results_: List[Dict[str, Any]] = []

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

    def _validate_target_and_predictions(
        self, y: np.ndarray, pred_matrix: np.ndarray
    ) -> None:
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

    def _fit_single_fold(
        self,
        y: np.ndarray,
        pred_matrix: np.ndarray,
        pred_cols: Sequence[str],
        fold: int,
    ) -> Dict[str, Any]:
        n_samples, n_models = pred_matrix.shape
        if n_samples == 0:
            raise ValueError(f"Fold {fold}: empty training data.")

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
            selected_name = str(pred_cols[candidate_best_idx])
            history.append(
                SelectionStep(
                    fold=fold,
                    iteration=it,
                    selected_model=selected_name,
                    score=float(candidate_best_score),
                )
            )

            if self.verbose:
                print(
                    f"[fold={fold:02d}][{it:03d}] "
                    f"add={selected_name}, score={candidate_best_score:.8f}"
                )

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
                        f"[fold={fold:02d}] Early stopping at iteration {it} "
                        f"(no improvement for {self.early_stopping_rounds} rounds)."
                    )
                break

        if not history:
            raise RuntimeError("Ensemble history is empty after fitting.")

        if best_iter == 0:
            best_iter = 1
            best_score = history[0].score

        final_steps = history[:best_iter]
        final_models = [s.selected_model for s in final_steps]
        counts = dict(Counter(final_models))
        total = sum(counts.values())
        weights = {k: v / total for k, v in counts.items()}

        return {
            "fold": fold,
            "selected_models": final_models,
            "model_counts": counts,
            "model_weights": weights,
            "history": history,
            "best_score": float(best_score),
            "best_iteration": int(best_iter),
        }

    def _kfold_indices(self, n_samples: int) -> List[np.ndarray]:
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        return [fold.astype(int) for fold in np.array_split(indices, self.n_splits)]

    def _stratified_kfold_indices(self, y: np.ndarray) -> List[np.ndarray]:
        unique_labels = set(np.unique(y))
        if unique_labels != {0.0, 1.0}:
            raise ValueError("Stratified split requires binary labels in {0,1}.")

        folds: List[List[int]] = [[] for _ in range(self.n_splits)]
        rng = np.random.default_rng(self.random_state)
        for label in (0.0, 1.0):
            label_indices = np.where(y == label)[0]
            if self.shuffle:
                rng.shuffle(label_indices)
            for i, idx in enumerate(label_indices):
                folds[i % self.n_splits].append(int(idx))

        fold_arrays: List[np.ndarray] = []
        for i, fold in enumerate(folds, start=1):
            if not fold:
                raise ValueError(
                    f"Fold {i} is empty after stratified split. "
                    "Reduce n_splits or set stratified=False."
                )
            fold_arrays.append(np.array(sorted(fold), dtype=int))
        return fold_arrays

    def _build_validation_folds(self, y: np.ndarray) -> List[np.ndarray]:
        if self.n_splits == 1:
            return [np.arange(len(y), dtype=int)]

        use_stratified = (
            self.stratified
            if self.stratified is not None
            else self.metric_name in {"logloss", "auc"}
        )
        if use_stratified:
            return self._stratified_kfold_indices(y)
        return self._kfold_indices(len(y))

    def _predict_with_weights(
        self, pred_matrix: np.ndarray, pred_cols: Sequence[str], weights: Dict[str, float]
    ) -> np.ndarray:
        weight_vector = np.array([weights.get(c, 0.0) for c in pred_cols], dtype=float)
        return pred_matrix @ weight_vector

    def fit(self, df: pd.DataFrame) -> "CaruanaGreedyEnsembler":
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found.")

        pred_cols = self._resolve_pred_columns(df)
        y = df[self.label_col].to_numpy(dtype=float)
        pred_matrix = df[pred_cols].to_numpy(dtype=float)

        self._validate_target_and_predictions(y, pred_matrix)
        n_samples = len(y)
        if self.n_splits > n_samples:
            raise ValueError(
                f"n_splits ({self.n_splits}) cannot exceed sample size ({n_samples})."
            )

        if self.n_splits == 1:
            fold_fit = self._fit_single_fold(
                y=y,
                pred_matrix=pred_matrix,
                pred_cols=pred_cols,
                fold=1,
            )
            full_pred = self._predict_with_weights(
                pred_matrix, pred_cols, fold_fit["model_weights"]
            )
            try:
                validation_score = float(self.metric_fn(y, full_pred))
            except ValueError:
                validation_score = None

            self.pred_columns_ = pred_cols
            self.selected_models_ = list(fold_fit["selected_models"])
            self.model_counts_ = dict(fold_fit["model_counts"])
            self.model_weights_ = dict(fold_fit["model_weights"])
            self.history_ = list(fold_fit["history"])
            self.best_score_ = (
                validation_score
                if validation_score is not None
                else float(fold_fit["best_score"])
            )
            self.best_iteration_ = int(fold_fit["best_iteration"])
            self.fold_model_weights_ = [dict(fold_fit["model_weights"])]
            self.fold_results_ = [
                {
                    "fold": 1,
                    "train_size": int(n_samples),
                    "validation_size": int(n_samples),
                    "best_score": fold_fit["best_score"],
                    "best_iteration": fold_fit["best_iteration"],
                    "validation_score": validation_score,
                    "selected_models": fold_fit["selected_models"],
                    "model_counts": fold_fit["model_counts"],
                    "model_weights": fold_fit["model_weights"],
                    "history": [s.__dict__ for s in fold_fit["history"]],
                }
            ]
            return self

        val_folds = self._build_validation_folds(y)
        fold_results: List[Dict[str, Any]] = []
        all_selected_models: List[str] = []
        all_history: List[SelectionStep] = []
        fold_weights: List[Dict[str, float]] = []

        for fold_id, val_idx in enumerate(val_folds, start=1):
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]

            if len(train_idx) == 0:
                raise ValueError(
                    f"Fold {fold_id}: training size is 0. Reduce n_splits."
                )
            if len(val_idx) == 0:
                raise ValueError(
                    f"Fold {fold_id}: validation size is 0. Reduce n_splits."
                )
            if self.metric_name == "auc":
                train_labels = set(np.unique(y[train_idx]))
                if train_labels != {0.0, 1.0}:
                    raise ValueError(
                        f"Fold {fold_id}: AUC optimization requires both classes "
                        "to exist in training split."
                    )

            fold_fit = self._fit_single_fold(
                y=y[train_idx],
                pred_matrix=pred_matrix[train_idx],
                pred_cols=pred_cols,
                fold=fold_id,
            )
            fold_pred_val = self._predict_with_weights(
                pred_matrix[val_idx], pred_cols, fold_fit["model_weights"]
            )
            try:
                validation_score = float(self.metric_fn(y[val_idx], fold_pred_val))
            except ValueError:
                validation_score = None

            if self.verbose:
                print(
                    f"[fold={fold_id:02d}] train_best={fold_fit['best_score']:.8f}, "
                    f"val_score={validation_score}"
                )

            fold_results.append(
                {
                    "fold": fold_id,
                    "train_size": int(len(train_idx)),
                    "validation_size": int(len(val_idx)),
                    "best_score": fold_fit["best_score"],
                    "best_iteration": fold_fit["best_iteration"],
                    "validation_score": validation_score,
                    "selected_models": fold_fit["selected_models"],
                    "model_counts": fold_fit["model_counts"],
                    "model_weights": fold_fit["model_weights"],
                    "history": [s.__dict__ for s in fold_fit["history"]],
                }
            )

            all_selected_models.extend(fold_fit["selected_models"])
            all_history.extend(fold_fit["history"])
            fold_weights.append(fold_fit["model_weights"])

        counts = dict(Counter(all_selected_models))
        total = sum(counts.values())
        aggregate_weights = {k: v / total for k, v in counts.items()}

        validation_scores = [
            float(f["validation_score"])
            for f in fold_results
            if f["validation_score"] is not None
        ]
        if validation_scores:
            best_score = float(np.mean(validation_scores))
        else:
            best_score = float(np.mean([float(f["best_score"]) for f in fold_results]))
        best_iteration = int(
            round(np.mean([int(f["best_iteration"]) for f in fold_results]))
        )

        self.pred_columns_ = pred_cols
        self.selected_models_ = all_selected_models
        self.model_counts_ = counts
        self.model_weights_ = aggregate_weights
        self.history_ = all_history
        self.best_score_ = best_score
        self.best_iteration_ = best_iteration
        self.fold_model_weights_ = fold_weights
        self.fold_results_ = fold_results
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fold_model_weights_ and not self.model_weights_:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        missing = [c for c in self.pred_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required prediction columns in DataFrame: {missing}")

        pred_matrix = df[self.pred_columns_].to_numpy(dtype=float)
        if np.isnan(pred_matrix).any():
            raise ValueError("Prediction columns contain NaN.")

        if self.fold_model_weights_:
            fold_preds = [
                self._predict_with_weights(pred_matrix, self.pred_columns_, w)
                for w in self.fold_model_weights_
            ]
            return np.mean(np.vstack(fold_preds), axis=0)
        return self._predict_with_weights(
            pred_matrix, self.pred_columns_, self.model_weights_
        )

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.__dict__ for s in self.history_])


def greedy_ensemble(
    df: pd.DataFrame,
    metric: str = "rmse",
    maximize: Optional[bool] = None,
    n_iterations: int = 100,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
    stratified: Optional[bool] = None,
    pred_columns: Optional[Sequence[str]] = None,
    pred_prefix: str = "pred_",
    label_col: str = "Label",
    early_stopping_rounds: Optional[int] = None,
    tol: float = 0.0,
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
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        stratified=stratified,
        pred_columns=pred_columns,
        pred_prefix=pred_prefix,
        label_col=label_col,
        early_stopping_rounds=early_stopping_rounds,
        tol=tol,
        verbose=verbose,
    )
    ensembler.fit(df)
    pred = ensembler.predict(df)
    return ensembler, pred
