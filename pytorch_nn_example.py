import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim


# Dataset with explicit float32 tensors
class MyDataset(Dataset):
    def __init__(self):
        self.x = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
        self.y = [0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx] #* 2 - 1   # Normalize x to [-1, 1]
        y = self.y[idx] #* 2 - 1   # Normalize y to [-1, 1]
        return (
            torch.tensor([x], dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
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

        self.learning_rate = 0.005

    def forward(self, input):
        act0_0 = F.tanh(input * self.w0_0 + self.b0_0)
        act0_1 = F.tanh(input * self.w0_1 + self.b0_1)
        act0_2 = F.tanh(input * self.w0_2 + self.b0_2)
        act0_3 = F.tanh(input * self.w0_3 + self.b0_3)

        edges_relu1_0 = act0_0 * self.w1_0_0 + act0_1 * self.w1_1_0 + act0_2 * self.w1_2_0 + act0_3 * self.w1_3_0
        edges_relu2_0 = act0_0 * self.w1_0_1 + act0_1 * self.w1_1_1 + act0_2 * self.w1_2_1 + act0_3 * self.w1_3_1

        act1_0 = F.tanh(edges_relu1_0 + self.b1_0)
        act1_1 = F.tanh(edges_relu2_0 + self.b1_1)
        
        output = act1_0 * self.w2_0 + act1_1 * self.w2_1 + self.b2
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = SGD(self.parameters(), lr=0.01, momentum=0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        return {"optimizer": optimizer, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        if loss < 0.002:
            print(f'Stopping early since the loss is small. Epoch: {self.current_epoch}')
            print("\n=== Parameter values in net ===\n")
            for name, param in self.named_parameters():
                print(f"{name}: {param.shape}\n{param.data}\n")
            self.trainer.should_stop = True

        return loss
    
class GraphRender():
    # Visualize learned function
    def show(self, model, dataset, min_x, max_x, step):
        x_test = torch.linspace(min_x, max_x, step).unsqueeze(1)
        y_pred = model(x_test).detach().squeeze()

        # Denormalize for visualization
        def denorm(t):
            return t
            #return (t + 1) / 2
        
        x_test_denorm = denorm(x_test.squeeze()).numpy()
        y_pred_denorm = denorm(y_pred).numpy()

        plt.plot(x_test_denorm, y_pred_denorm, label="Prediction")
        plt.scatter(dataset.x, dataset.y, color="red", label="Data Points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Learned Function after Training")
        plt.show()

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Train
model = MyPyTorch()
trainer = L.Trainer(max_epochs=5000)
trainer.fit(model, train_dataloaders=dataloader)

# Render
render = GraphRender()
render.show(model, dataset, 0, 1, 10)