from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from caruana_greedy_ensemble import CaruanaGreedyEnsembler


def _parse_pred_columns(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return cols or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caruana-style greedy ensemble selection (with replacement).")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="ensemble_output.csv", help="Output CSV path with ensemble predictions")
    parser.add_argument("--report-path", default="ensemble_report.json", help="Output JSON path with selected models and history")
    parser.add_argument("--id-col", default="ID", help="ID column name")
    parser.add_argument("--label-col", default="Label", help="Label column name")
    parser.add_argument("--pred-prefix", default="pred_", help="Prefix for prediction columns")
    parser.add_argument("--pred-columns", default=None, help="Comma-separated prediction columns. If set, pred-prefix is ignored.")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae", "r2", "spearman", "pcc", "pearson", "logloss", "auc"], help="Optimization metric")
    parser.add_argument("--maximize", action="store_true", help="Force maximize metric (default behavior depends on metric)")
    parser.add_argument("--n-iterations", type=int, default=100, help="Maximum number of greedy additions")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds for K-fold greedy ensemble fitting")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle before fold split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for fold split")
    parser.add_argument("--stratified", action="store_true", help="Force stratified fold split (binary label only)")
    parser.add_argument("--early-stopping-rounds", type=int, default=None, help="Stop if no improvement for this many rounds")
    parser.add_argument("--tol", type=float, default=0.0, help="Minimum improvement threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_path)

    df = pd.read_csv(input_path)
    pred_columns = _parse_pred_columns(args.pred_columns)

    ensembler = CaruanaGreedyEnsembler(
        metric=args.metric,
        maximize=True if args.maximize else None,
        n_iterations=args.n_iterations,
        n_splits=args.n_splits,
        shuffle=not args.no_shuffle,
        random_state=args.random_state,
        stratified=True if args.stratified else None,
        pred_columns=pred_columns,
        pred_prefix=args.pred_prefix,
        label_col=args.label_col,
        early_stopping_rounds=args.early_stopping_rounds,
        tol=args.tol,
        verbose=args.verbose,
    )
    ensembler.fit(df)
    pred = ensembler.predict(df)

    out_df = pd.DataFrame({"ensemble_pred": pred})
    if args.id_col in df.columns:
        out_df.insert(0, args.id_col, df[args.id_col].values)
    if args.label_col in df.columns:
        out_df[args.label_col] = df[args.label_col].values
    out_df.to_csv(output_path, index=False)

    report = {
        "metric": ensembler.metric_name,
        "maximize": ensembler.maximize,
        "n_splits": ensembler.n_splits,
        "best_score": ensembler.best_score_,
        "best_iteration": ensembler.best_iteration_,
        "n_selected_models": len(ensembler.selected_models_),
        "selected_models": ensembler.selected_models_,
        "model_counts": ensembler.model_counts_,
        "model_weights": ensembler.model_weights_,
        "fold_model_weights": ensembler.fold_model_weights_,
        "fold_results": ensembler.fold_results_,
        "history": ensembler.history_frame().to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] output: {output_path}")
    print(f"[DONE] report: {report_path}")
    print(f"[CV] n_splits={ensembler.n_splits}")
    print(f"[BEST] score={ensembler.best_score_}, iter={ensembler.best_iteration_}")
    print(f"[AGG_WEIGHTS] {ensembler.model_weights_}")


if __name__ == "__main__":
    main()
