import numpy as np
import torch
import xgboost as xgb

def dataset_to_xy(dataset):
    """
    Convert FolsomIntraDayDataset to numpy X, y

    Returns:
        X: [N, 2, 6, 5]
        y: [N, 2, 6]
    """
    X_list = []
    y_list = []
    y_clear_list = []

    for i in range(len(dataset)):
        sample = dataset[i]

        #irr = sample["irradiance"].reshape(-1).numpy()  # [36]
        #img = sample["images"].reshape(-1).numpy()      # [frames * ch * 10 * 10]
        xgb_input = sample["xgb_input"] # [2, 6, 5]
        X = xgb_input
        #X = np.concatenate([irr, img])

        X_list.append(X)
        y_list.append(sample["target"].numpy())
        #y_list.append(sample["target"].numpy() - sample["clear_kt"].numpy())   # [2,6] predict the residual

        y_clear_list.append(sample["clear_kt"].numpy())  # [2,6]

    X = np.stack(X_list)
    y = np.stack(y_list)
    y_clear = np.stack(y_clear_list)

    return X, y, y_clear

class IntraDayXGB:
    def __init__(self):
        self.models = {}  # (target, horizon) -> model

    def fit(self, X, y):
        """
        X: [N, 2, 6, 5]
        y: [N, 2, 6]
        """
        for t_idx, t_name in enumerate(["ghi", "dni"]):
            for h in range(6):
                x_th = X[:, t_idx, h, :] # [N, 5]
                y_th = y[:, t_idx, h] # [N]

                model = xgb.XGBRegressor(
                    n_estimators=500, # 500
                    max_depth=6, # 6
                    learning_rate=0.05, # 0.05
                    subsample=0.8, # 0.8
                    colsample_bytree=0.8, # 0.8
                    objective="reg:squarederror", # "reg:squarederror"
                    tree_method="hist", # "hist"
                    random_state=42, # 42
                )

                #model.fit(X, y_th)
                model.fit(x_th, y_th)
                self.models[(t_idx, h)] = model

    def predict(self, X, y_clear):
        """
        inputs:
            X: [N, 2, 6 ,5]
        Returns:
            pred: [N, 2, 6]
        """
        N = X.shape[0]
        preds = np.zeros((N, 2, 6), dtype=np.float32)

        for (t_idx, h), model in self.models.items():
            preds[:, t_idx, h] = model.predict(X[:, t_idx, h, :])

        #return preds + y_clear
        return preds
