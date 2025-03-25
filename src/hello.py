import polars as pl
import torch as th
import torchvision as tv
from tqdm import tqdm


def preprocess_data(df_path):
    df = pl.read_csv(df_path)

    human_readable = df.select(
        [
            "sequence",
            "label",
            "length",
        ]
    )

    X = df.select([col for col in df.columns if "x" in col])
    y = df.select([col for col in df.columns if "y" in col])

    X = th.tensor(X.to_numpy()).unsqueeze(-1).float()
    y = th.tensor(y.to_numpy())
    y += 1

    y = th.nn.functional.one_hot(y.long()).float()

    return human_readable, X, y


def accuracy(y_pred, y_true):
    y_pred = th.argmax(y_pred, dim=-1)
    y_true = th.argmax(y_true, dim=-1)
    return (y_pred == y_true).float().mean()


def main():
    # =======================================================
    # Data Preprocessing
    # =======================================================

    # Get data
    data_path = "data/Protein_Secondary_Structure/"
    tain_file = "protein-secondary-structure.train.csv"
    test_file = "protein-secondary-structure.test.csv"

    train_human_readable, train_X, train_y = preprocess_data(data_path + tain_file)
    test_human_readable, test_X, test_y = preprocess_data(data_path + test_file)

    print(train_human_readable)
    print(f"X {train_X.shape}, y {train_y.shape}")

    print(test_human_readable)
    print(f"X {test_X.shape}, y {test_y.shape}")

    # Create datasets
    train_dataset = th.utils.data.TensorDataset(train_X, train_y)
    test_dataset = th.utils.data.TensorDataset(test_X, test_y)

    # Create dataloaders
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # =======================================================
    # Model Training
    # =======================================================

    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "mps"
        if th.backends.mps.is_available()
        else "cpu"
    )

    # Create LSTM model
    lstm = th.nn.LSTM(input_size=1, hidden_size=4, num_layers=2, batch_first=True).to(
        device
    )

    # Create optimizer
    optimizer = th.optim.AdamW(lstm.parameters(), lr=0.001)
    criterion = tv.ops.sigmoid_focal_loss
    metric = accuracy

    # Train model
    for epoch in tqdm(range(10)):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = lstm(X)
            loss = criterion(out, y, reduction="mean")
            metric_value = metric(out, y)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}, Metric: {metric_value.item()}")


if __name__ == "__main__":
    main()
