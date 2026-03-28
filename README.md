# Greedy_Ens

Caruana-style Greedy Ensemble（重複選択あり）を実行するための最小実装です。

## ファイル構成

- `caruana_greedy_ensemble.py`
  - Greedy Ensemble本体
  - `fit()`でモデル選択（`pred_*`を1本ずつ追加）
  - `predict()`で最終アンサンブル予測を出力
- `run_greedy_ensemble.py`
  - CSV入力で実行できるCLI
  - 予測CSVと選択履歴JSONを出力

## 環境構築（Conda -> uv）

### 1. Conda環境を作成・有効化

```powershell
conda env create -f environment.yml
conda activate greedy-ens
```

### 2. uvで依存を同期

```powershell
uv sync
```

補足:

- 依存定義は`pyproject.toml`にあります。
- `uv sync`で`.venv`が作成され、実行時は`uv run ...`が使えます。

### 3. 実行

```powershell
uv run python run_greedy_ensemble.py --input train_pred.csv --metric r2
```

## 前提データ

DataFrame（またはCSV）に以下の列を持たせます。

- `ID`（任意）
- `Label`（正解ラベル）
- `pred_0`, `pred_1`, ...（各モデル予測）

## Pythonから使う例

```python
import pandas as pd
from caruana_greedy_ensemble import CaruanaGreedyEnsembler

df = pd.read_csv("train_pred.csv")

ens = CaruanaGreedyEnsembler(
    metric="r2",            # rmse / mae / r2 / spearman / pcc / pearson / logloss / auc
    n_iterations=100,
    label_col="Label",
    pred_prefix="pred_",
    early_stopping_rounds=20
)

ens.fit(df)
pred = ens.predict(df)

print("best_score:", ens.best_score_)
print("best_iteration:", ens.best_iteration_)
print("model_weights:", ens.model_weights_)
print("history head:")
print(ens.history_frame().head())
```

## CaruanaGreedyEnsemblerの使い方

基本フローは以下です。

1. `CaruanaGreedyEnsembler(...)`で設定
2. `fit(train_or_trainval_df)`でGreedyにモデル選択（重複あり）
3. `predict(any_df_with_pred_cols)`でアンサンブル予測

### コンストラクタ引数

- `metric` (`str`, default=`"rmse"`)
  - 最適化指標。`"rmse"`, `"mae"`, `"r2"`, `"spearman"`, `"pcc"`, `"pearson"`, `"logloss"`, `"auc"` から選択。
- `maximize` (`Optional[bool]`, default=`None`)
  - `None`なら指標ごとに自動判定。`r2/spearman/pcc/pearson/auc`は最大化、`rmse/mae/logloss`は最小化。
- `n_iterations` (`int`, default=`100`)
  - モデルを追加する最大反復回数。
- `pred_columns` (`Optional[Sequence[str]]`, default=`None`)
  - 使う予測列を明示指定。指定時は`pred_prefix`を使わない。
- `pred_prefix` (`str`, default=`"pred_"`)
  - `pred_columns`未指定時に、このprefixで予測列を自動検出。
- `label_col` (`str`, default=`"Label"`)
  - 正解ラベル列名。
- `early_stopping_rounds` (`Optional[int]`, default=`None`)
  - 改善が止まった場合の早期終了ラウンド数。
- `tol` (`float`, default=`0.0`)
  - 改善とみなす最小差分。微小変化を無視したい時に使う。
- `verbose` (`bool`, default=`False`)
  - 各反復の選択結果を標準出力に表示。

補足:

- 最終重みは常に「best iterationまでの追加履歴」で確定します（固定仕様）。

### 主なメソッド

- `fit(df: pd.DataFrame) -> CaruanaGreedyEnsembler`
  - `label_col`と予測列を使って重複許容のGreedy選択を実行。
- `predict(df: pd.DataFrame) -> np.ndarray`
  - 学習済み重みでアンサンブル予測。`Label`列は不要（予測列のみ必要）。
- `history_frame() -> pd.DataFrame`
  - 各反復の`iteration`, `selected_model`, `score`を返す。

### fit後に参照できる属性

- `best_score_`: ベストスコア
- `best_iteration_`: ベスト反復番号
- `selected_models_`: 最終的に採用された列名リスト（重複あり）
- `model_counts_`: 列ごとの採用回数
- `model_weights_`: `model_counts_`を正規化した重み
- `history_`: 全反復の履歴

### Train+Valで学習してTestへ適用する例

```python
import pandas as pd
from caruana_greedy_ensemble import CaruanaGreedyEnsembler

train_val_df = pd.read_csv("train_val_pred.csv")  # ID, Label, pred_*
test_df = pd.read_csv("test_pred.csv")            # ID, pred_*（Label不要）

ens = CaruanaGreedyEnsembler(metric="r2", n_iterations=200, early_stopping_rounds=30)
ens.fit(train_val_df)
test_pred = ens.predict(test_df)

pd.DataFrame({"ID": test_df["ID"], "ensemble_pred": test_pred}).to_csv(
    "test_ensemble_pred.csv", index=False
)
```

## CLI実行例

```powershell
uv run python run_greedy_ensemble.py `
  --input train_pred.csv `
  --output ensemble_output.csv `
  --report-path ensemble_report.json `
  --metric spearman `
  --n-iterations 100 `
  --early-stopping-rounds 20
```

## 選べる評価指標

- `rmse`（小さいほど良い）
- `mae`（小さいほど良い）
- `r2`（大きいほど良い）
- `spearman`（大きいほど良い）
- `pcc` / `pearson`（大きいほど良い）
- `logloss`（小さいほど良い, 2値分類向け）
- `auc`（大きいほど良い, 2値分類向け）

`ensemble_output.csv`:

- `ID`（存在する場合）
- `ensemble_pred`
- `Label`（存在する場合）

`ensemble_report.json`:

- ベストスコア
- ベスト反復回数
- 選択されたモデル列（重複含む）
- モデル重み（選択頻度ベース）
- 全反復の履歴
