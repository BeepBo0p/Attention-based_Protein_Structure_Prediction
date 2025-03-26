import os
import time

import matplotlib.pyplot as plt
import polars as pl
import torch as th
import torchvision as tv


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
    # Select the class with the highest probability
    y_pred = th.argmax(y_pred, dim=-1)
    y_true = th.argmax(y_true, dim=-1)
    
    y_real = th.zeros_like(y_pred) == y_true
    y_real = y_real.float() == 0
    y_real = y_real.float()
    
    # Calculate accuracy 
    accuracy = (y_pred == y_true).float() * y_real
    accuracy = accuracy.sum(dim=-1) / y_real.sum(dim=-1)
    return accuracy.float().mean()
    return (y_pred == y_true).float().mean()


def f1_score(y_pred, y_true):
    pass


def l2_regularization(model, device, lambda_=0.01):
    l2_reg = th.tensor(0.0).to(device)
    for param in model.parameters():
        l2_reg += th.norm(param)
    return lambda_ * l2_reg


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
    lstm = th.nn.LSTM(
        input_size=1, hidden_size=4, num_layers=2, batch_first=True, dropout=0.5
    ).to(device)

    bidirectional_lstm = th.nn.Sequential(
        th.nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        ),
        th.nn.Linear(8, 4),
    ).to(device)

    model = bidirectional_lstm

    # Create optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=0.001)
    criterion = tv.ops.sigmoid_focal_loss
    regularizer = l2_regularization
    metric = accuracy
    epochs = 50
    best_test_loss = float("inf")
    patience = int(epochs * 0.01)
    counter = 0

    train_info = {
        "Epoch": [],
        "Train Loss": [],
        "Train Metric": [],
        "Test Loss": [],
        "Test Metric": [],
        "Best Test Loss": [],
        "Counter": [],
    }

    id = str(int(time.time()))
    out_dir = f"out/{model.__class__.__name__}/{id}"
    os.makedirs(out_dir, exist_ok=True)

    # Train model
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_metric = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # If the model is bidirectional we need to handle the hidden state
            if isinstance(model, th.nn.Sequential):
                for layer in model:
                    if isinstance(layer, th.nn.LSTM):
                        out, _ = layer(X)
                    else:
                        out = layer(out)

            # Otherwise, ram data through the model as usual
            else:
                out, _ = model(X)
            loss = criterion(out, y, reduction="mean")
            if regularizer:
                loss += regularizer(model, device)
            metric_value = metric(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_metric += metric_value.item()
        train_loss /= len(train_loader)
        train_metric /= len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.3f}, Train Metric: {train_metric:.3f}",
            end=", ",
        )

        # Test
        model.eval()
        test_loss = 0
        test_metric = 0
        with th.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                # If the model is bidirectional we need to handle the hidden state
                if isinstance(model, th.nn.Sequential):
                    for layer in model:
                        if isinstance(layer, th.nn.LSTM):
                            out, _ = layer(X)
                        else:
                            out = layer(out)
                # Otherwise, ram data through the model as usual
                else:
                    out, _ = model(X)
                loss = criterion(out, y, reduction="mean")
                metric_value = metric(out, y)
                test_loss += loss.item()
                test_metric += metric_value.item()
        test_loss /= len(test_loader)
        test_metric /= len(test_loader)
        print(f"Test Loss: {test_loss:.3f}, Test Metric: {test_metric:.3f}", end=", ")

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            counter = 0

            # Save best model
            th.save(model.state_dict(), out_dir + "/model.pth")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

        print(f"Best Test Loss: {best_test_loss:.3f} ({counter}/{patience})")

        # Save training info
        train_info["Epoch"].append(epoch)
        train_info["Train Loss"].append(train_loss)
        train_info["Train Metric"].append(train_metric)
        train_info["Test Loss"].append(test_loss)
        train_info["Test Metric"].append(test_metric)
        train_info["Best Test Loss"].append(best_test_loss)
        train_info["Counter"].append(counter)

    # Save training info
    train_info = pl.DataFrame(train_info)
    train_info.write_csv(out_dir + "/_train_info.csv")

    # Side-by-side Plots for Loss and Metric
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Loss
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(
        train_info["Epoch"],
        train_info["Train Loss"],
        label="Train Loss",
        color="blue",
        linestyle="-",
    )
    ax1.plot(
        train_info["Epoch"],
        train_info["Test Loss"],
        label="Test Loss",
        color="blue",
        linestyle="--",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Plot Metric
    ax2.set_title("Metric")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.plot(
        train_info["Epoch"],
        train_info["Train Metric"],
        label="Train Metric",
        color="red",
        linestyle="-",
    )
    ax2.plot(
        train_info["Epoch"],
        train_info["Test Metric"],
        label="Test Metric",
        color="red",
        linestyle="--",
    )
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # Title and save
    plt.suptitle("Training and Testing Loss and Metric")
    fig.tight_layout()
    plt.savefig(out_dir + f"/{id}_side_by_side_info.png")
    plt.close()
    # =======================================================
    # Model Evaluation
    # =======================================================

    # Load best model
    model.load_state_dict(th.load(out_dir + "/model.pth"))


if __name__ == "__main__":
    main()
