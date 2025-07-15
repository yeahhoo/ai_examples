import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

learning_rate = 0.01
epochs = 15

class DataGenerator:
    def __init__(self, max_x, max_y, max_z, step):
        self.data = []
        self.labels = []
        for x in range(0, max_x + 1, step):
            for y in range(0, max_y + 1, step):
                for z in range(0, max_z + 1, step):
                    self.data.append((float(x), float(y), float(z)))
                    self.labels.append(self.calc_right_answer(x, y, z))

    def calc_func_result(self, x, y, z):
        return 4 * x + 2 * y - 3 * z

    # function that defines right answer for a combination of (x, y, z)
    # can only return values 0, 1, 2. 
    def calc_right_answer(self, x, y, z):
        result = self.calc_func_result(x, y, z)
        if result <= 10:
            return 0
        elif result <= 40:
            return 1
        else:
            return 2

    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels

class ClassificatorNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = self.loss_fn(logits, y)
    #     acc = (logits.argmax(dim=1) == y).float().mean()
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def test_predict(self, x, y, z):
        raw_func_result = data_gen.calc_func_result(x, y, z)
        y_true = data_gen.calc_right_answer(x, y, z)
        probabilities = self.predict_probabilities(torch.tensor([x, y, z]))
        print(f'true_answer: {y_true}, func_result: {raw_func_result}, probabilities: {probabilities.detach().numpy()}')
    
    def predict_probabilities(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=0)
        return torch.round(probabilities * 100) / 100
    
    def my_train_model(self, generator):
        data_tensor = torch.tensor(generator.get_data(), dtype=torch.float32)
        labels_tensor = torch.tensor(generator.get_labels(), dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Train the model
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self, dataloader)

data_gen = DataGenerator(10, 10, 10, 2)
model = ClassificatorNetwork()
model.my_train_model(data_gen)

model.test_predict(0.0, 1.0, 4.0)
model.test_predict(2.0, 8.0, 1.0)
model.test_predict(9.0, 5.0, 0.0)
model.test_predict(8.3, 2.1, 4.2)
model.test_predict(2.4, 5.1, 3.9)