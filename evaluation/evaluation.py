import os
import numpy as np
import pandas as pd

class Evaluation:
    """
    Unified evaluation class for:
        - day-ahead
        - intra-day
        - intra-hour

    Input:
        model outputs in kt, shape [N, T]

    Output:
        pandas DataFrame with RMSE / MAE / MBE / Skill
    """

    def __init__(self, base_dir="/mnt/glusterfs/Planning/data/folsom/unzipped"):
        self.base_dir = base_dir

    # =====================================================
    # Metrics
    # =====================================================
    @staticmethod
    def rmse(y, yhat):
        return np.sqrt(np.nanmean((y - yhat) ** 2))

    @staticmethod
    def mae(y, yhat):
        return np.nanmean(np.abs(y - yhat))

    @staticmethod
    def mbe(y, yhat):
        return np.nanmean(y - yhat)

    # =====================================================
    # Task configuration
    # =====================================================
    def _task_config(self, eval_type, target):
        """
        Returns task-specific configuration.
        """
        if eval_type == "intra-hour":
            return {
                "horizons": ["5min","10min","15min","20min","25min","30min"],
                "horizon_unit": "min",
                "baseline_name": "sp",
                "baseline_col": lambda h: f"B({target}_kt|5min)",
                "endo_file": "Irradiance_features_intra-hour.csv",
                "exo_file": "Sky_image_features_intra-hour.csv",
                "target_file": "Target_intra-hour.csv",
            }

        if eval_type == "intra-day":
            return {
                "horizons": ["30min","60min","90min","120min","150min","180min"],
                "horizon_unit": "min",
                "baseline_name": "sp",
                "baseline_col": lambda h: f"B({target}_kt|30min)",
                "endo_file": "Irradiance_features_intra-day.csv",
                "exo_file": "Sat_image_features_intra-day.csv",
                "target_file": "Target_intra-day.csv",
            }

        if eval_type == "day-ahead":
            return {
                "horizons": ["26h","27h","28h","29h","30h","31h","32h","33h",
                             "34h","35h","36h","37h","38h","39h"],
                "horizon_unit": "h",
                "baseline_name": "nam",
                "baseline_col": lambda h: f"nam_{target}_{h}",
                "endo_file": "Irradiance_features_day-ahead.csv",
                "exo_file": "NAM_nearest_node_day-ahead.csv",
                "target_file": "Target_day-ahead.csv",
            }

        raise ValueError(f"Unknown eval_type: {eval_type}")

    # =====================================================
    # Main evaluation entry
    # =====================================================
    def eval(
        self,
        eval_type: str,
        target: str,
        model_name: str,
        result: np.ndarray,
    ):
        """
        Evaluate a model output.

        Args:
            eval_type: 'day-ahead' | 'intra-day' | 'intra-hour'
            target: 'ghi' | 'dni'
            model_name: name of your model (e.g. 'cnn', 'transformer')
            result: np.ndarray, shape [N_test, T_horizons], in kt

        Returns:
            pd.DataFrame
        """
        cfg = self._task_config(eval_type, target)
        horizons = cfg["horizons"]

        if result.shape[1] != len(horizons):
            raise ValueError(
                f"Result shape mismatch: got T={result.shape[1]}, "
                f"expected {len(horizons)}"
            )

        # load test data
        inpEndo = pd.read_csv(
            os.path.join(self.base_dir, cfg["endo_file"]),
            parse_dates=True,
            index_col=0,
        )
        inpExo = pd.read_csv(
            os.path.join(self.base_dir, cfg["exo_file"]),
            parse_dates=True,
            index_col=0,
        )
        tar = pd.read_csv(
            os.path.join(self.base_dir, cfg["target_file"]),
            parse_dates=True,
            index_col=0,
        )

        results = {target: {}}

        for t_idx, horizon in enumerate(horizons):
            out = {}

            cols = [
                f"{target}_{horizon}",
                f"{target}_kt_{horizon}",
                f"{target}_clear_{horizon}",
                f"elevation_{horizon}",
            ]

            test = inpEndo[inpEndo.index.year == 2016]
            test = test.join(inpExo[inpEndo.index.year == 2016], how="inner")
            test = test.join(tar[tar.index.year == 2016], how="inner")
            feature_cols_endo = inpEndo.filter(regex=target).columns.tolist()
            feature_cols_exo = feature_cols_endo + inpExo.columns.tolist()
            test  = test[cols + feature_cols_exo].dropna()

            y_true = test[f"{target}_{horizon}"].values
            clear = test[f"{target}_clear_{horizon}"].values
            elev = test[f"elevation_{horizon}"].values

            # model prediction
            y_pred = np.clip(result[:, t_idx], 0, 1.1)
            y_pred = y_pred * clear
            y_pred[elev < 5] = np.nan

            out[model_name] = {
                "y_true": y_true,
                "y_pred": y_pred,
            }

            # baseline
            base = test[cfg["baseline_col"](horizon)].values * clear
            base[elev < 5] = np.nan

            out[cfg["baseline_name"]] = {
                "y_true": y_true,
                "y_pred": base,
            }

            results[target][horizon] = out

        return self.calc_metrics(
            results,
            target,
            baseline_name=cfg["baseline_name"],
            horizon_unit=cfg["horizon_unit"],
        )

    # =====================================================
    # Metrics table
    # =====================================================
    def calc_metrics(self, results, target, baseline_name, horizon_unit):
        rows = []

        for horizon, models in results[target].items():
            row = {"horizon": horizon}

            y_true = models[baseline_name]["y_true"]
            y_base = models[baseline_name]["y_pred"]
            rmse_base = self.rmse(y_true, y_base)

            for model_name, data in models.items():
                y_pred = data["y_pred"]

                row[f"{model_name}_rmse"] = self.rmse(y_true, y_pred)
                row[f"{model_name}_mae"]  = self.mae(y_true, y_pred)
                row[f"{model_name}_mbe"]  = self.mbe(y_true, y_pred)

                if model_name != baseline_name:
                    row[f"{model_name}_skill"] = (
                        1.0 - row[f"{model_name}_rmse"] / rmse_base
                    )

            rows.append(row)

        df = pd.DataFrame(rows)

        df["h"] = df["horizon"].str.replace(horizon_unit, "").astype(int)
        df = df.sort_values("h").drop(columns="h")

        # Mean row (paper-style)
        mean_row = {"horizon": "Mean"}
        for col in df.columns:
            if col != "horizon":
                mean_row[col] = df[col].mean()

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        return df

if __name__ == "__main__":
    # a demo
    test_result = np.zeros((48357, 6)) # 48357 non-nan samples in intra-day dataset, 6 horizons.
    # test_result in kt, no units
    # eval outputs RMSE / MAE / MBE / Skill

    evaluator = Evaluation()
    df = evaluator.eval(
        eval_type="intra-hour",
        target="ghi",
        model_name="my_model",
        result=test_result,   # shape [N, 6]
    )
    print(df)