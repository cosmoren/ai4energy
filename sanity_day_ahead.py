import os
import sys
from pathlib import Path
import contextlib
import io
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from datasets.folsom_day_ahead import FolsomDayAheadDataset


def _rmse(a, b):
    return np.sqrt(np.nanmean((a - b) ** 2))


def _mae(a, b):
    return np.nanmean(np.abs(a - b))


def _mbe(a, b):
    return np.nanmean(a - b)


def run_forecast(target, horizon, root_dir="/mnt/nfs/yuan/Folsom", write_hdf: bool = False):
    with contextlib.redirect_stdout(io.StringIO()):
        train_ds = FolsomDayAheadDataset(
            root_dir=root_dir,
            split="train",
            target=target,
            horizon=horizon,
        )
        test_ds = FolsomDayAheadDataset(
            root_dir=root_dir,
            split="test",
            target=target,
            horizon=horizon,
        )

    # Use __getitem__ output (batched) for features/targets; convert to numpy on the sklearn side.
    train_batch = next(iter(DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, num_workers=0)))
    test_batch = next(iter(DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0)))

    train_y_kt = train_batch["target"].numpy().reshape(-1)
    test_y_kt = test_batch["target"].numpy().reshape(-1)
    train_clear = train_batch["clear_sky"].numpy().reshape(-1)
    test_clear = test_batch["clear_sky"].numpy().reshape(-1)
    train_elev = train_batch["elevation"].numpy().reshape(-1)
    test_elev = test_batch["elevation"].numpy().reshape(-1)

    # Ground-truth irradiance + NAM baseline from __getitem__ output (dataset self-contained)
    ycol = f"{target}_{horizon}"
    train_y = train_batch["actual"].numpy().reshape(-1)
    test_y = test_batch["actual"].numpy().reshape(-1)
    train_nam = train_batch["nam_irr"].numpy().reshape(-1)
    test_nam = test_batch["nam_irr"].numpy().reshape(-1)

    # Structured feature blocks from __getitem__ (batched, flat dict)
    train_endo = train_batch["endo"].numpy()
    test_endo = test_batch["endo"].numpy()

    train_cc = train_batch["nam_cc"].numpy()
    test_cc = test_batch["nam_cc"].numpy()

    train_nam_feat = train_batch["nam"].numpy().reshape(len(train_ds), -1)
    test_nam_feat = test_batch["nam"].numpy().reshape(len(test_ds), -1)

    X_sets = {
        "endo": (train_endo, test_endo),
        "exo": (np.concatenate([train_endo, train_cc], axis=1), np.concatenate([test_endo, test_cc], axis=1)),
        "endo+NAM": (
            np.concatenate([train_endo, train_nam_feat], axis=1),
            np.concatenate([test_endo, test_nam_feat], axis=1),
        ),
        "exo+NAM": (
            np.concatenate([train_endo, train_cc, train_nam_feat], axis=1),
            np.concatenate([test_endo, test_cc, test_nam_feat], axis=1),
        ),
    }

    models = {
        "OLS": linear_model.LinearRegression(),
        "Ridge": linear_model.RidgeCV(cv=10),
        "Lasso": linear_model.LassoCV(cv=10, n_jobs=-1, max_iter=10000),
    }

    # Build forecast table like official script (keeps Train/Test rows)
    train_idx = train_batch["timestamp"]
    test_idx = test_batch["timestamp"]
    train = pd.DataFrame(index=train_idx)
    test = pd.DataFrame(index=test_idx)
    ccol = f"{target}_clear_{horizon}"
    ktcol = f"{target}_kt_{horizon}"
    train[ycol], train[ktcol], train[ccol] = train_y, train_y_kt, train_clear
    test[ycol], test[ktcol], test[ccol] = test_y, test_y_kt, test_clear

    # NAM baseline column (irradiance, no clear-sky multiplication for day-ahead)
    train[f"{target}_nam"] = train_nam
    test[f"{target}_nam"] = test_nam
    train.loc[train_elev < 5, f"{target}_nam"] = np.nan
    test.loc[test_elev < 5, f"{target}_nam"] = np.nan

    preds_test = {("NAM", "NAM"): test[f"{target}_nam"].values.astype(float)}

    for feat_name, (Xtr, Xte) in X_sets.items():
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        for mname, model in models.items():
            model.fit(Xtr, train_y_kt)
            pred_kt = model.predict(Xte)
            pred = pred_kt * test_clear
            pred[test_elev < 5] = np.nan
            colname = f"{target}_{mname.lower()}_{feat_name.replace('+', '_').lower()}"
            test[colname] = pred
            preds_test[(mname, feat_name)] = pred

    cols = train.columns[train.columns.str.startswith(f"{target}")].tolist()
    train, test = train[cols], test[cols]
    train["dataset"] = "Train"
    test["dataset"] = "Test"
    df = pd.concat([train, test], axis=0)
    df["target"] = target
    df["horizon"] = horizon

    if write_hdf:
        os.makedirs("forecasts", exist_ok=True)
        df.to_hdf(
            os.path.join("forecasts", f"forecasts_day-ahead_horizon={horizon}_{target}.h5"),
            key="df",
            mode="w",
        )

    # Test-only metrics after clear-sky (already in irradiance), baseline is NAM
    y_true = test[ycol].values.astype(float)
    y_true[test_elev < 5] = np.nan
    base = preds_test[("NAM", "NAM")]
    rmse_base = _rmse(y_true, base)

    rows = []
    for (mname, feat_name), y_pred in preds_test.items():
        rmse = _rmse(y_true, y_pred)
        mae = _mae(y_true, y_pred)
        mbe = _mbe(y_true, y_pred)
        skill = 0.0 if mname == "NAM" else (1.0 - rmse / rmse_base if rmse_base and not np.isnan(rmse_base) else np.nan)
        rows.append(
            {
                "task": "day-ahead",
                "target": target.upper(),
                "model": mname,
                "features": feat_name,
                "RMSE": rmse,
                "MAE": mae,
                "MBE": mbe,
                "Skill": skill,
            }
        )
    return rows


if __name__ == "__main__":
    root_dir = "/mnt/nfs/yuan/Folsom"
    write_hdf = False
    targets = ["ghi", "dni"]
    horizons = ["26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h"]

    per_run_rows = []
    for t in targets:
        for h in horizons:
            per_run_rows.extend(run_forecast(t, h, root_dir, write_hdf=write_hdf))

    df = pd.DataFrame(per_run_rows)
    console = Console()

    def show_table(title: str, out: pd.DataFrame):
        table = Table(title=title, header_style="bold", show_lines=False)
        table.add_column("task", style="dim")
        table.add_column("model", style="cyan")
        table.add_column("features", style="green")
        table.add_column("MAE", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("Skill", justify="right")
        table.add_column("MBE", justify="right")
        for _, r in out.iterrows():
            table.add_row(
                str(r["task"]),
                str(r["model"]),
                str(r["features"]),
                f"{r['MAE']:.3f}" if pd.notna(r["MAE"]) else "nan",
                f"{r['RMSE']:.3f}" if pd.notna(r["RMSE"]) else "nan",
                f"{r['Skill']:.3f}" if pd.notna(r["Skill"]) else "nan",
                f"{r['MBE']:.3f}" if pd.notna(r["MBE"]) else "nan",
            )
        console.print(table)

    for tgt in ["GHI", "DNI"]:
        out = (
            df[df["target"] == tgt]
            .groupby(["task", "model", "features"], as_index=False)[["MAE", "RMSE", "Skill", "MBE"]]
            .mean()
            .sort_values(["model", "features"])
        )
        show_table(f"{tgt} RESULTS (mean over horizons)", out)
