import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # you can comment it out if you are on Windows
import matplotlib.pyplot as plt

min_x = -10
max_x = 10
learning_rate = 0.01
epochs = 500

# 1. Define the math function
def math_function(x):
    #return x ** 3 + 4 * x
    return np.sin(x) + 0.3 * x * np.cos(x) + 0.1 * x  
    #return np.sin(x) * (-2) * (x ** 2)  + 3 * (x ** 2) * np.cos(x) - 88 * (x ** 2) * np.tan(x) + 4 * x + 34

# 2. Create a synthetic dataset
class MathFunctionDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = np.linspace(min_x, max_x, num_samples, dtype=np.float32)
        self.y = math_function(self.x).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_val = self.x[idx]
        features = np.array([x_val, np.sin(x_val), np.cos(x_val), np.tan(x_val)],  dtype=np.float32)
        return torch.tensor(features), torch.tensor([self.y[idx]])

# 3. Define the model using PyTorch Lightning
class FunctionApproximator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

# 4. Train the model
def train_model():
    dataset = MathFunctionDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FunctionApproximator()
    trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False, logger=False)
    trainer.fit(model, dataloader)

    return model

# 5. Evaluate the model
def evaluate_model(model):
    x_test = torch.linspace(min_x, max_x, 200).unsqueeze(1)
    features = torch.cat([x_test, torch.sin(x_test), torch.cos(x_test), torch.tan(x_test)], dim=1)
    y_pred = model(features).detach().numpy()
    y_true = math_function(x_test.numpy())

    plt.plot(x_test.numpy(), y_true, label="True Function")
    plt.plot(x_test.numpy(), y_pred, label="NN Prediction")
    plt.legend()
    plt.title("Function Approximation")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)
