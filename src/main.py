import json
import os
import time

import matplotlib.pyplot as plt

# import monai
# import monai.losses
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

    # Create a mask to ignore padding
    y_real = th.zeros_like(y_pred) == y_true
    y_real = y_real.float() == 0
    y_real = y_real.float()

    # Calculate accuracy
    accuracy = (y_pred == y_true).float() * y_real
    accuracy = accuracy.sum(dim=-1) / y_real.sum(dim=-1)
    return accuracy.float().mean()
    return (y_pred == y_true).float().mean()


def precision(y_pred, y_true):
    # Select the class with the highest probability
    y_pred = th.argmax(y_pred, dim=-1)
    y_true = th.argmax(y_true, dim=-1)

    # Create a mask to ignore padding
    y_real = th.zeros_like(y_pred) == y_true
    y_real = y_real.float() == 0
    y_real = y_real.float()

    # Calculate Precision
    tp = (y_pred == y_true).float() * y_real
    tp = tp.sum(dim=-1)
    fp = (y_pred != y_true).float() * y_real
    fp = fp.sum(dim=-1)
    precision = tp / (tp + fp + 1e-8)
    return precision.mean()


def recall(y_pred, y_true):
    # Select the class with the highest probability
    y_pred = th.argmax(y_pred, dim=-1)
    y_true = th.argmax(y_true, dim=-1)

    # Create a mask to ignore padding
    y_real = th.zeros_like(y_pred) == y_true
    y_real = y_real.float() == 0
    y_real = y_real.float()

    # Calculate Precision
    tp = (y_pred == y_true).float() * y_real
    tp = tp.sum(dim=-1)
    fn = (y_pred != y_true).float() * y_real
    fn = fn.sum(dim=-1)
    recall = tp / (tp + fn + 1e-8)
    return recall.mean()


def f1_score(y_pred, y_true):
    # Select the class with the highest probability
    y_pred = th.argmax(y_pred, dim=-1)
    y_true = th.argmax(y_true, dim=-1)

    # Create a mask to ignore padding
    y_real = th.zeros_like(y_pred) == y_true
    y_real = y_real.float() == 0
    y_real = y_real.float()

    # Calculate Precision
    tp = (y_pred == y_true).float() * y_real
    tp = tp.sum(dim=-1)
    fp = (y_pred != y_true).float() * y_real
    fp = fp.sum(dim=-1)
    precision = tp / (tp + fp + 1e-8)

    # Calculate Recall
    fn = (y_pred != y_true).float() * y_real
    fn = fn.sum(dim=-1)
    recall = tp / (tp + fn + 1e-8)

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.mean()


def l2_regularization(model, device, lambda_=0.01):
    l2_reg = th.tensor(0.0).to(device)
    nr_params = 0
    for param in model.parameters():
        l2_reg += th.norm(param)
        nr_params += 1
    return lambda_ * l2_reg / nr_params


def l1_regularization(model, device, lambda_=0.01):
    l1_reg = th.tensor(0.0).to(device)
    nr_params = 0
    for param in model.parameters():
        l1_reg += th.norm(param, p=1)
        nr_params += 1
    return lambda_ * l1_reg / nr_params


def lmax_regularization(model, device, lambda_=0.01):
    lmax_reg = th.tensor(0.0).to(device)
    for param in model.parameters():
        lmax_reg = (
            th.norm(param, p=float("inf"))
            if th.norm(param, p=float("inf")) > lmax_reg
            else lmax_reg
        )
    return lambda_ * lmax_reg


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
        input_size=1, hidden_size=4, num_layers=8, batch_first=True, dropout=0.25
    ).to(device)

    """
    bidirectional_lstm = th.nn.Sequential(
        th.nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=8,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        ),
        th.nn.Linear(8, 4),
    ).to(device)
    """

    model = lstm

    # Create optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=0.001)
    criterion = (
        tv.ops.focal_loss.sigmoid_focal_loss
    )  # th.nn.CrossEntropyLoss() #monai.losses.dice.DiceCELoss()
    regularizer = lmax_regularization  # l2_regularization
    metric = f1_score
    epochs = int(1e6)
    best_test_loss = float("inf")
    best_test_metric = -1
    patience = epochs  # int(epochs * 0.01)
    counter = 0

    train_settings = {
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__
        if isinstance(criterion, th.nn.Module)
        else criterion.__name__,
        "regularizer": regularizer.__name__ if regularizer else None,
        "metric": metric.__name__,
        "epochs": epochs,
        "patience": patience,
        "counter": counter,
    }

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

    # Save training settings as yaml
    import yaml

    with open(out_dir + "/train_settings.yaml", "w") as file:
        yaml.dump(train_settings, file)

    try:
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
                if isinstance(model, th.nn.LSTM):
                    out, _ = model(X)

                # Otherwise, ram data through the model as usual
                else:
                    out = model(X)
                loss = criterion(
                    out, y, reduction="mean"
                ) + 0.001 * th.nn.CrossEntropyLoss()(
                    out, y
                )  # if hasattr(criterion, "reduction") else criterion(out, y)
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
                    if isinstance(model, th.nn.LSTM):
                        out, _ = model(X)
                    else:
                        out = model(X)
                    loss = criterion(
                        out, y, reduction="mean"
                    ) + 0.001 * th.nn.CrossEntropyLoss()(
                        out, y
                    )  # if hasattr(criterion, "reduction") else criterion(out, y)
                    metric_value = metric(out, y)
                    test_loss += loss.item()
                    test_metric += metric_value.item()
            test_loss /= len(test_loader)
            test_metric /= len(test_loader)
            print(
                f"Test Loss: {test_loss:.3f}, Test Metric: {test_metric:.3f}", end=", "
            )

            # Early stopping
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                counter = 0

                # Save best model
                th.save(model.state_dict(), out_dir + "/model.pth")

            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

            print(f"Best Test Metric: {best_test_metric:.3f} ({counter}/{patience})")

            # Save training info
            train_info["Epoch"].append(epoch)
            train_info["Train Loss"].append(train_loss)
            train_info["Train Metric"].append(train_metric)
            train_info["Test Loss"].append(test_loss)
            train_info["Test Metric"].append(test_metric)
            train_info["Best Test Loss"].append(best_test_loss)
            train_info["Counter"].append(counter)

    except KeyboardInterrupt:
        print("Training interrupted")

    except Exception as e:
        raise e

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
    ax2.set_ylim(0, 1)
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

    # Perform inference on the test set
    predictions = []
    truths = []
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    avg_accuracy = []

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
        predictions.append(out)
        truths.append(y)

        avg_precision.append(precision(out, y))
        avg_recall.append(recall(out, y))
        avg_f1.append(f1_score(out, y))
        avg_accuracy.append(accuracy(out, y))

    avg_precision = th.stack(avg_precision).mean()
    avg_recall = th.stack(avg_recall).mean()
    avg_f1 = th.stack(avg_f1).mean()
    avg_accuracy = th.stack(avg_accuracy).mean()

    print(f"Precision: {avg_precision:.3f}")
    print(f"Recall: {avg_recall:.3f}")
    print(f"F1 Score: {avg_f1:.3f}")
    print(f"Accuracy: {avg_accuracy:.3f}")

    predictions = th.cat(predictions)
    truths = th.cat(truths)

    predictions = th.argmax(predictions, dim=-1)
    truths = th.argmax(truths, dim=-1)

    # Convert to numpy
    predictions = predictions.cpu().numpy()
    truths = truths.cpu().numpy()

    # Subtract 1 to get the original labels
    predictions -= 1
    truths -= 1

    sequences = []
    true_sequences = []

    for i in range(predictions.shape[0]):
        # Get the index of the first -1
        end = list(truths[i]).index(-1)

        # Get the sequence and append it to the list
        sequences.append(predictions[i][:end])
        true_sequences.append(truths[i][:end])
        gt = truths[i][:end]

        print(f"Sequence: {i}")
        print(f"Prediction: {sequences[-1]}")
        print(f"Truth: {gt}")
        print()

    # Convert the sequences back to strings using the label_to_idx mapping in data/
    with open(data_path + "label_to_idx.json", "r") as file:
        label_to_idx = json.load(file)

    # Invert the mapping
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}

    # Add the padding character
    idx_to_label["-1"] = "-"

    # Convert the sequences to strings
    sequences = [[idx_to_label[str(idx)] for idx in sequence] for sequence in sequences]
    true_sequences = [
        [idx_to_label[str(idx)] for idx in sequence] for sequence in true_sequences
    ]

    # Print the sequences
    for i, sequence in enumerate(sequences):
        print(f"Sequence: {i}")
        print(f"Prediction: \t{''.join(sequence)}")
        print(f"Truth: \t\t{''.join(true_sequences[i])}")
        print()


if __name__ == "__main__":
    main()
