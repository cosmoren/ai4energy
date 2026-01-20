from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))

from datasets.folsom_day_ahead import FolsomDayAheadDataModule
from models.day_ahead_model import DayAhead


def _ensure_bth(x: torch.Tensor, num_targets: int, num_horizons: int) -> torch.Tensor:
    """
    Ensure tensor is shaped [B, T, H] (handles dataset squeeze conventions).
    """
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        # [B, H] when T==1 or [B, T] when H==1
        if num_targets == 1 and x.shape[1] == num_horizons:
            return x.unsqueeze(1)
        if num_horizons == 1 and x.shape[1] == num_targets:
            return x.unsqueeze(2)
    if x.dim() == 1:
        return x.view(-1, 1, 1)
    raise ValueError(f"Unexpected shape: {tuple(x.shape)}")


def _nan_rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y - yhat) ** 2)))


def _nan_mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y - yhat)))


def _nan_mbe(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(y - yhat))


def _fmt_num(x: float | None, ndigits: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "nan"
    return f"{float(x):.{ndigits}f}"


def _normalize_baseline_dict(baseline_metrics: dict | None) -> dict[str, dict[str, float]]:
    """
    Accept either:
      A) {"ghi": {"RMSE": 74.0, "MAE": 50.2, "MBE": 1.5, "Skill_vs_NAM": 33.8}, "dni": {...}}
      B) {"RMSE_ghi": 74.0, "MAE_ghi": 50.2, ...}  (also supports lowercase metric names)
    Returns normalized: {target: {metric: value}}
    """
    if not baseline_metrics:
        return {}

    # Nested dict form
    if any(isinstance(v, dict) for v in baseline_metrics.values()):
        out: dict[str, dict[str, float]] = {}
        for tgt, md in baseline_metrics.items():
            if not isinstance(md, dict):
                continue
            out[str(tgt)] = {}
            for k, v in md.items():
                out[str(tgt)][str(k)] = float(v)
        return out

    # Flat form: metric_target
    out: dict[str, dict[str, float]] = {}
    for k, v in baseline_metrics.items():
        if not isinstance(k, str):
            continue
        parts = k.split("_")
        if len(parts) < 2:
            continue
        tgt = parts[-1]
        metric = "_".join(parts[:-1])
        metric_norm = {
            "rmse": "RMSE",
            "mae": "MAE",
            "mbe": "MBE",
            "skill": "Skill_vs_NAM",
            "skill_vs_nam": "Skill_vs_NAM",
        }.get(metric.lower(), metric)
        out.setdefault(str(tgt), {})[metric_norm] = float(v)
    return out


def _score_for_ranking(metric: str, val: float) -> float:
    """
    Convert metric value into a score where higher is better, for ranking methods.
    - RMSE/MAE: lower is better -> score = -val
    - MBE: closer to 0 is better -> score = -abs(val)
    - Skill_vs_NAM: higher is better -> score = val
    """
    if metric in ("RMSE", "MAE"):
        return -val
    if metric == "MBE":
        return -abs(val)
    if metric == "Skill_vs_NAM":
        return val
    return -val


def _print_multi_method_tables(
    method_means: dict[str, dict[str, dict[str, float]]],
    targets: list[str],
):
    """
    Print one table per metric.
    Rows: method name
    Columns: targets (GHI, DNI, ...)
    Cells: mean over horizons.
    Coloring: green = best, red = worst for each metric+target across methods.
    """
    metrics = ["RMSE", "MAE", "MBE", "Skill_vs_NAM"]
    methods = list(method_means.keys())

    for metric in metrics:
        table = Table(title=f"{metric} (mean over horizons)", header_style="bold")
        table.add_column("Method", justify="left")
        for tgt in targets:
            table.add_column(str(tgt).upper(), justify="right")

        # Determine best/worst method per target for this metric
        best: dict[str, str] = {}
        worst: dict[str, str] = {}
        for tgt in targets:
            scores = []
            for m in methods:
                v = method_means.get(m, {}).get(tgt, {}).get(metric)
                if v is None:
                    continue
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    continue
                scores.append((m, _score_for_ranking(metric, float(v))))
            if not scores:
                continue
            scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
            best[tgt] = scores_sorted[0][0]
            worst[tgt] = scores_sorted[-1][0]

        for m in methods:
            row = [m]
            for tgt in targets:
                v = method_means.get(m, {}).get(tgt, {}).get(metric)
                s = _fmt_num(v)
                if tgt in best and m == best[tgt]:
                    s = f"[green]{s}[/green]"
                elif tgt in worst and m == worst[tgt]:
                    s = f"[red]{s}[/red]"
                row.append(s)
            table.add_row(*row)

        Console().print(table)


def run_evaluation(
    checkpoint_path: str,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "cuda",
    root_dir: str | None = None,
    target: str | None = None,
    horizon: str | None = None,
    save_preds: str | None = None,
):
    ckpt_path = Path(checkpoint_path)
    config_path_candidates = [
        ckpt_path.parent / "config.yaml",
        ckpt_path.parent.parent / "config.yaml",
    ]
    config_path = next((p for p in config_path_candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            "Could not find config.yaml next to checkpoint. Tried: "
            + ", ".join(str(p) for p in config_path_candidates)
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))

    if root_dir is not None:
        data_cfg["root_dir"] = root_dir
    if target is not None:
        data_cfg["target"] = target
    if horizon is not None:
        data_cfg["horizon"] = horizon

    data_cfg["batch_size"] = batch_size
    data_cfg["num_workers"] = num_workers
    data_cfg["drop_last"] = False

    dm = FolsomDayAheadDataModule(**data_cfg)
    dm.setup("test")

    if dm.test_dataset is None:
        raise RuntimeError("test_dataset is None after setup('test').")
    if len(dm.test_dataset) == 0:
        raise RuntimeError("test_dataset length is 0 (likely due to strict dropna filtering).")

    num_targets = len(dm.test_dataset.targets)
    num_horizons = len(dm.test_dataset.horizons)
    endo_dim = len(dm.test_dataset.feature_cols_endo)
    target_names = list(dm.test_dataset.targets)
    horizon_names = list(dm.test_dataset.horizons)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model; override dims to match dataset (must match checkpoint weights)
    model = DayAhead.load_from_checkpoint(
        str(ckpt_path),
        endo_dim=endo_dim,
        num_targets=num_targets,
        num_horizons=num_horizons,
    )
    model = model.to(device)
    model.eval()

    preds_kt = []
    y_true = []
    y_base = []
    clear = []
    elev = []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            yhat_kt = model(batch)  # [B,T,H]
            preds_kt.append(yhat_kt.detach().cpu())

            y_true.append(_ensure_bth(batch["actual"], num_targets, num_horizons).detach().cpu())
            y_base.append(_ensure_bth(batch["nam_irr"], num_targets, num_horizons).detach().cpu())
            clear.append(_ensure_bth(batch["clear_sky"], num_targets, num_horizons).detach().cpu())
            elev.append(batch["elevation"].detach().cpu())  # [B,H] or [B] if H==1

    preds_kt = torch.cat(preds_kt, dim=0)  # [N,T,H]
    y_true = torch.cat(y_true, dim=0)
    y_base = torch.cat(y_base, dim=0)
    clear = torch.cat(clear, dim=0)
    elev = torch.cat(elev, dim=0)

    if save_preds is not None:
        outp = Path(save_preds)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.save(outp, preds_kt.numpy())
        print("saved preds_kt:", str(outp))

    # kt -> irradiance
    y_pred = preds_kt * clear

    # Apply nighttime mask (elevation < 5)
    if elev.dim() == 1:
        elev_bh = elev.view(-1, 1)
    else:
        elev_bh = elev
    mask = (elev_bh < 5).unsqueeze(1).expand(-1, num_targets, -1)  # [N,T,H]

    y_true_np = y_true.numpy().astype(float)
    y_pred_np = y_pred.numpy().astype(float)
    y_base_np = y_base.numpy().astype(float)

    y_true_np[mask.numpy()] = np.nan
    y_pred_np[mask.numpy()] = np.nan
    y_base_np[mask.numpy()] = np.nan

    # Metrics per target/horizon
    rows = []
    for ti in range(num_targets):
        for hi in range(num_horizons):
            yt = y_true_np[:, ti, hi]
            yp = y_pred_np[:, ti, hi]
            yb = y_base_np[:, ti, hi]

            rmse_b = _nan_rmse(yt, yb)
            rmse = _nan_rmse(yt, yp)
            mae = _nan_mae(yt, yp)
            mbe = _nan_mbe(yt, yp)
            skill = 0.0 if (rmse_b == 0.0 or np.isnan(rmse_b)) else float(1.0 - rmse / rmse_b)

            rows.append(
                {
                    "target": str(target_names[ti]),
                    "horizon": str(horizon_names[hi]),
                    "RMSE": rmse,
                    "MAE": mae,
                    "MBE": mbe,
                    "Skill_vs_NAM": skill,
                }
            )

    # Print separate summaries for each target (e.g., GHI vs DNI)
    print("test samples:", preds_kt.shape[0], "targets:", target_names, "horizons:", horizon_names)
    model_means: dict[str, dict[str, float]] = {}
    for tgt in target_names:
        tgt_rows = [r for r in rows if r["target"] == tgt]
        rmse_mean = float(np.nanmean([r["RMSE"] for r in tgt_rows]))
        mae_mean = float(np.nanmean([r["MAE"] for r in tgt_rows]))
        mbe_mean = float(np.nanmean([r["MBE"] for r in tgt_rows]))
        skill_mean = float(np.nanmean([r["Skill_vs_NAM"] for r in tgt_rows]))
        model_means[str(tgt)] = {
            "RMSE": rmse_mean,
            "MAE": mae_mean,
            "MBE": mbe_mean,
            "Skill_vs_NAM": skill_mean,
        }
        print(f"[{tgt.upper()}] mean MAE: {mae_mean} mean MBE: {mbe_mean} mean RMSE: {rmse_mean} mean Skill_vs_NAM: {skill_mean}")

    model_name = str(config.get("experiment", {}).get("run_name") or config.get("wandb", {}).get("name") or ckpt_path.stem or "model")
    return model_name, model_means



if __name__ == "__main__":
    import random
    import string
    from dataclasses import dataclass

    @dataclass
    class ModelEvaluation:
        checkpoint_path: str
        name_postfix: str | None = None
        def __post_init__(self):
            if self.name_postfix is None:
                self.name_postfix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    @dataclass
    class BaselineMetrics:
        name: str
        ghi: dict[str, float]
        dni: dict[str, float]

    experiments = [
        ModelEvaluation(checkpoint_path=""),
        BaselineMetrics(name="regression_baseline", ghi={"RMSE": 74.0, "MAE": 50.2, "MBE": 1.5, "Skill_vs_NAM": 0.338}, dni={"RMSE": 185.0, "MAE": 136.4, "MBE": 11.6, "Skill_vs_NAM": 0.228}),
    ]
    # Optional overrides (None means use what was saved in config.yaml)
    root_dir = None  # e.g. "/mnt/nfs/yuan/Folsom"
    target = None    # e.g. "ghi" or "dni"
    horizon = None   # e.g. "26h"

    batch_size = 256
    num_workers = 8
    device = "cuda"
    save_preds = None  # e.g. "/tmp/preds_kt.npy"

    # Evaluate multiple models + multiple baselines and print them in the same tables.
    method_means: dict[str, dict[str, dict[str, float]]] = {}
    targets_all: set[str] = set()

    for exp in experiments:
        if isinstance(exp, ModelEvaluation):
            name, means = run_evaluation(
                checkpoint_path=exp.checkpoint_path,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                root_dir=root_dir,
                target=target,
                horizon=horizon,
                save_preds=save_preds,
            )
            # Ensure every model instance shows up even if base `name` is the same.
            # Use the provided/random postfix to make the method key unique.
            method = f"{name}_{exp.name_postfix}" if exp.name_postfix else name
            for tgt, md in means.items():
                targets_all.add(str(tgt))
                method_means.setdefault(method, {})[str(tgt)] = dict(md)
        elif isinstance(exp, BaselineMetrics):
            method = exp.name
            method_means.setdefault(method, {})
            method_means[method]["ghi"] = dict(exp.ghi)
            method_means[method]["dni"] = dict(exp.dni)
            targets_all.update(["ghi", "dni"])
        else:
            raise TypeError(f"Unknown experiment type: {type(exp)}")

    _print_multi_method_tables(method_means=method_means, targets=sorted(targets_all))

