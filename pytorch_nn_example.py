import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Dataset with explicit float32 tensors
class MyDataset(Dataset):
    def __init__(self):
        self.x = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
        self.y = [0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor([self.x[idx]], dtype=torch.float32),
            torch.tensor([self.y[idx]], dtype=torch.float32),
        )

# Lightning module with corrected initialization, loss, and optimizer
class MyPyTorch(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Initialize with small random values
        # Layer 0
        self.w0_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b0_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w0_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b0_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w0_2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b0_2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w0_3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b0_3 = nn.Parameter(torch.randn(1, requires_grad=True))

        # Layer 1
        self.w1_0_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_0_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_1_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_1_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_2_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_2_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_3_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w1_3_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b1_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b1_1 = nn.Parameter(torch.randn(1, requires_grad=True))

        # Layer 2 (final)
        self.w2_0 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b2 = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, input):
        relu0_0 = F.relu(input * self.w0_0 + self.b0_0)
        relu0_1 = F.relu(input * self.w0_1 + self.b0_1)
        relu0_2 = F.relu(input * self.w0_2 + self.b0_2)
        relu0_3 = F.relu(input * self.w0_3 + self.b0_3)

        edges_relu1_0 = relu0_0 * self.w1_0_0 + relu0_1 * self.w1_1_0 + relu0_2 * self.w1_2_0 + relu0_3 * self.w1_3_0
        edges_relu2_0 = relu0_0 * self.w1_0_1 + relu0_1 * self.w1_1_1 + relu0_2 * self.w1_2_1 + relu0_3 * self.w1_3_1

        relu1_0 = F.relu(edges_relu1_0 + self.b1_0)
        relu1_1 = F.relu(edges_relu2_0 + self.b1_1)
        
        output = relu1_0 * self.w2_0 + relu1_1 * self.w2_1 + self.b2
        return output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

class GraphRender():
    # Visualize learned function
    def show(self, model, dataset, min_x, max_x, step):
        x_test = torch.linspace(min_x, max_x, step).unsqueeze(1)
        y_pred = model(x_test).detach().numpy()

        plt.plot(x_test.numpy(), y_pred, label="Prediction")
        plt.scatter(dataset.x, dataset.y, color="red", label="Data Points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Learned Function after Training")
        plt.show()

# DataLoader with batch_size=1 (fine for tiny dataset)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Train
model = MyPyTorch()
trainer = L.Trainer(max_epochs=1000, enable_checkpointing=False, logger=False)
trainer.fit(model, train_dataloaders=dataloader)

# Render
render = GraphRender()
render.show(model, dataset, 0, 1, 10)