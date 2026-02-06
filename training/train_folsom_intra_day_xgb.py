import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path to import datasets
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom_intra_day import FolsomIntraDayDataset
from models.xgb_utils import dataset_to_xy, IntraDayXGB
from evaluation.evaluation import Evaluation

# load data
train_ds = FolsomIntraDayDataset(split="train")
test_ds  = FolsomIntraDayDataset(split="test")

X_train, y_train, y_clear_train = dataset_to_xy(train_ds)
X_test, y_test, y_clear_test = dataset_to_xy(test_ds)

# train
model = IntraDayXGB()
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test, y_clear_test)  # [N,2,6]

# eval
evaluator = Evaluation()

df_ghi = evaluator.eval(
    eval_type="intra-day",
    target="ghi",
    model_name="xgb_intra_day",
    result=pred[:, 0, :],
)

df_dni = evaluator.eval(
    eval_type="intra-day",
    target="dni",
    model_name="xgb_intra_day",
    result=pred[:, 1, :],
)

print("**** GHI ****")
print(df_ghi)
print("**** DNI ****")
print(df_dni)
