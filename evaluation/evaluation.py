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

    def __init__(self, base_dir="/mnt/nfs/yuan/Folsom"):
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

        # Sanity check: print dataset size before filtering
        test_before_filter = inpEndo[inpEndo.index.year == 2016]
        test_before_filter = test_before_filter.join(inpExo[inpEndo.index.year == 2016], how="inner")
        test_before_filter = test_before_filter.join(tar[tar.index.year == 2016], how="inner")
        print(f"Dataset size before filtering: {len(test_before_filter)} samples")

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
            # Don't dropna here - we'll filter after aligning with predictions
            
            # Get all required columns first
            y_true_all = test[f"{target}_{horizon}"].values
            clear_all = test[f"{target}_clear_{horizon}"].values
            elev_all = test[f"elevation_{horizon}"].values
            
            # Prepare model predictions for all samples (before filtering)
            # result shape is [N_test, T_horizons] where N_test should match test dataset size
            if result.shape[0] != len(test):
                raise ValueError(
                    f"Prediction shape mismatch: got {result.shape[0]} predictions, "
                    f"expected {len(test)} samples from test dataset"
                )
            
            y_pred_all = np.clip(result[:, t_idx], 0, 1.1)
            y_pred_all = y_pred_all * clear_all
            y_pred_all[elev_all < 5] = np.nan
            
            # Now filter NaN values from ground truth and apply same mask to predictions
            # This ensures both GT and predictions are filtered identically
            valid_mask = ~np.isnan(y_true_all)
            # Also filter out low elevation angles
            valid_mask = valid_mask & (elev_all >= 5)
            
            y_true = y_true_all[valid_mask]
            y_pred = y_pred_all[valid_mask]
            clear = clear_all[valid_mask]
            elev = elev_all[valid_mask]

            out[model_name] = {
                "y_true": y_true,
                "y_pred": y_pred,
            }

            # baseline - apply same filtering mask
            if eval_type == "day-ahead":
                base_all = test[cfg["baseline_col"](horizon)].values
            else:
                base_all = test[cfg["baseline_col"](horizon)].values * clear_all
            base_all[elev_all < 5] = np.nan
            
            # Apply same valid_mask to baseline
            base = base_all[valid_mask]

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
    test_result = np.zeros((48401, 6)) # 48357 non-nan samples in intra-day dataset, 6 horizons.
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