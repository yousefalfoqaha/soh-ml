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
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
WINDOW_LENGTH = 1000


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_mcus = MCUS_TRAIN + MCUS_VALID + MCUS_TEST
    mf4_to_hdf.convert(
        data_path=DATA_PATH,
        mcus=all_mcus,
        raster=RASTER_FREQ,
        target_channels=TARGET_CHANNELS,
    )

    hdf_data_path = DATA_PATH.joinpath("hdf")

    dataset_train = McusDataset(
        mcus=MCUS_TRAIN, data_path=hdf_data_path, window_length=WINDOW_LENGTH
    )
    dataset_valid = McusDataset(
        mcus=MCUS_VALID, data_path=hdf_data_path, window_length=WINDOW_LENGTH
    )

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

    model = LstmModel(input_size=2, hidden_size=HIDDEN_SIZE, output_size=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = torch.nn.HuberLoss()

    train_and_validate(
        model, optimizer, criterion, loader_train, loader_valid, N_EPOCHS, device
    )


def train_and_validate(
    model, optimizer, criterion, train_loader, valid_loader, n_epochs, device
):
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        total_valid_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in valid_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                y_val_pred = model(X_val)
                val_loss = criterion(y_val_pred, y_val)

                total_valid_loss += val_loss.item()

        mean_train_loss = total_train_loss / len(train_loader)
        mean_valid_loss = total_valid_loss / len(valid_loader)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train Loss: {mean_train_loss:.5f} | "
            f"Valid Loss: {mean_valid_loss:.5f}"
        )


if __name__ == "__main__":
    main()
