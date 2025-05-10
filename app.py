import os
import pandas as pd
import torch
from flask import Flask, render_template, request
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
class FocalLoss(nn.Module):
   def init(self, alpha=0.9, gamma=8.0, reduction='mean'):
    super(FocalLoss, self).init()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction


def forward(self, inputs, targets):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
    pt = torch.exp(-BCE_loss)
    focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

    if self.reduction == 'mean':
        return focal_loss.mean()
    elif self.reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

class FocalLossMetric(Metric):
   def __init__(self, alpha=0.9, gamma=8.0):
    super().__init__()
    self.focal = FocalLoss(alpha, gamma)
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

   def update(self, preds: torch.Tensor, target: torch.Tensor):
    loss = self.focal(preds, target)
    self.total += loss.detach() * target.size(0)
    self.count += target.size(0)
   
   def compute(self):
    return self.total / self.count

# Load training dataset and model checkpoint (adjust paths as needed)
training = torch.load("training.pkl")
model = TemporalFusionTransformer.load_from_checkpoint("tft_model_1.ckpt")

# ----------- Routes ------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Load CSV as DataFrame
            df = pd.read_csv(filepath)
            df["TIME"] = pd.to_datetime(df["TIME"])
            df = df.sort_values("TIME")
            df["time_idx"] = (df["TIME"] - df["TIME"].min()).dt.days
            df["location_id"] = df["LATITUDE"].astype(str) + "_" + df["LONGITUDE"].astype(str)
            df["FLOOD"] = 0  # Adding a placeholder for flood column

            # Create TimeSeriesDataSet for prediction
            predict_dataset = TimeSeriesDataSet.from_dataset(
                training, df, predict=True, stop_randomization=True
            )
            loader = predict_dataset.to_dataloader(train=False, batch_size=1)
            raw_predictions = model.predict(loader, mode="raw")  # returns namedtuple
            logits = raw_predictions.prediction  # this is a Tensor
            flood_proba = torch.sigmoid(logits).view(-1).detach().cpu().numpy()

            # Include both flood probabilities and rainfall values
            days = list(range(1, len(flood_proba) + 1))
            rainfall_values = df["RAINFALL"].values  # Extract rainfall values from the DataFrame
            result = list(zip(days, rainfall_values, flood_proba))
            prediction = [(int(day), float(rainfall), float(prob)) for day, rainfall, prob in result]

            return render_template("result.html", prediction=prediction)
        else:
            return "Invalid file. Please upload a CSV."
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
