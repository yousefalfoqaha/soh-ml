from pathlib import Path

import torch
from torch.utils.data import DataLoader

import mf4_to_hdf
from dataset import McusDataset
from lstm_model import LstmModel

MCUS_TRAIN = ["mcu1", "mcu2"]
MCUS_VALID = ["mcu3"]
MCUS_TEST = ["mcu4"]
DATA_PATH = Path("../data")
RASTER_FREQ = 0.1
TARGET_CHANNELS = ["U", "I", "Temp[1]", "Qneg"]
N_EPOCHS = 30


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mf4_to_hdf.convert(
        data_path=DATA_PATH,
        mcus=MCUS_TRAIN,
        raster=RASTER_FREQ,
        target_channels=TARGET_CHANNELS,
    )

    hdf_data_path = DATA_PATH.joinpath("hdf")
    dataset_train = McusDataset(
        mcus=MCUS_TRAIN, data_path=hdf_data_path, window_length=1000
    )

    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    model = LstmModel(input_size=2, hidden_size=64, output_size=2).to(device)


def train(model, optimizer, criterion, train_loader, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}")


if __name__ == "__main__":
    main()
