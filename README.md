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
    early_stopping_rounds=20,
    use_best_iteration=True
)

ens.fit(df)
pred = ens.predict(df)

print("best_score:", ens.best_score_)
print("best_iteration:", ens.best_iteration_)
print("model_weights:", ens.model_weights_)
print("history head:")
print(ens.history_frame().head())
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
