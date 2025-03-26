import json

import numpy as np
import polars as pl
import torch as th
import torchvision as tv


# =======================================================
# Utility Functions
# =======================================================
def accuracy(y_pred, y_true):
    y_pred = th.argmax(y_pred, dim=1)
    y_true = th.argmax(y_true, dim=1)
    return th.mean((y_pred == y_true).float())


def convert_sequence_to_string(sequence, mapping):
    return "".join([mapping[seq] for seq in sequence])


# =======================================================
# Data Preprocessing Definitions
# =======================================================


def preprocess_data(df_path):
    df = pl.read_csv(df_path)

    human_readable = df.select(
        [
            "sequence",
            "label",
            "length",
        ]
    )

    X = df.select([col for col in df.columns if "x" in col]).to_numpy()
    Y = df.select([col for col in df.columns if "y" in col]).to_numpy()
    length = df["length"].to_numpy()

    # Convert to lists
    X = [list(row) for row in X]
    Y = [list(row) for row in Y]
    length = [int(row) for row in length]

    if len(X) != len(Y) or len(X) != len(length):
        raise ValueError("Input and output sequences must have the same length")

    for i, (x, y) in enumerate(zip(X, Y)):
        # Find the first negative value in the sequence
        idx = next((i for i, x in enumerate(x) if x < 0), None)
        if idx is not None:
            x = x[:idx]
            y = y[:idx]

            X[i] = x
            Y[i] = y

    if any([len(x) != len(y) for x, y in zip(X, Y)]):
        raise ValueError("Input and output sequences must have the same length")

    # Concatenate all sequences
    X = np.array([item for sublist in X for item in sublist])
    Y = np.array([item for sublist in Y for item in sublist])

    if len(X) != len(Y):
        raise ValueError("Input and output sequences must have the same length")

    return human_readable, X, Y, length


class GenomeTokenDataset(th.utils.data.Dataset):
    def __init__(self, X, Y, indice_information, sequence_length=20, one_hot=True):
        super(GenomeTokenDataset, self).__init__()
        self.indice_information = indice_information
        self.sequence_length = sequence_length

        self.genome_sequence = []
        self.genome_label = []

        for i, _ in enumerate(X):
            indices_in_sequence = np.arange(i, i + sequence_length)

            if any([idx >= len(X) for idx in indices_in_sequence]):
                continue

            if any([idx in indice_information for idx in indices_in_sequence[1:]]):
                continue

            lower = np.min(indices_in_sequence)
            upper = np.max(indices_in_sequence) + 1

            if upper - lower != sequence_length:
                continue

            self.genome_sequence.append(X[lower:upper])
            self.genome_label.append(Y[lower:upper])

        for i, row in enumerate(self.genome_sequence):
            if len(row) != sequence_length:
                print("Genome sequence checks failed")
                print(row)
                print(len(row))
                print(f"Index: {i} / {len(self.genome_sequence)}")
                exit()

            # If any sequence value is negative, print the row
            if any([sequence < 0 for sequence in row]):
                print("Genome sequence checks failed")
                print(row)
                print(f"Index: {i} / {len(self.genome_sequence)}")
                exit()

        for i, row in enumerate(self.genome_label):
            if len(row) != sequence_length:
                print("Genome label checks failed")
                print(row)
                print(len(row))
                print(f"Index: {i} / {len(self.genome_label)}")
                exit()

            # If any label value is negative, print the row
            if any([label < 0 for label in row]):
                print("Genome label checks failed")
                print(row)
                print(f"Index: {i} / {len(self.genome_label)}")
                exit()

        self.genome_sequence = np.array(self.genome_sequence)
        self.genome_label = np.array(self.genome_label)

        self.genome_sequence = th.tensor(self.genome_sequence, dtype=th.float32)
        self.genome_label = th.tensor(self.genome_label, dtype=th.float32)

        # Convert labels to one-hot encoding
        if one_hot:
            self.genome_label = th.nn.functional.one_hot(
                self.genome_label.clone().detach().long(), num_classes=3
            ).float()

        # Unsqueeze the sequence
        self.genome_sequence = self.genome_sequence.long()

    def __len__(self):
        return len(self.genome_sequence)

    def __getitem__(self, idx):
        return self.genome_sequence[idx], self.genome_label[idx]


# =======================================================
# Model Definition
# =======================================================


class GenomeTokenLSTM(th.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GenomeTokenLSTM, self).__init__()
        self.embedding = th.nn.Embedding(20, input_size)
        self.lstm = th.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = th.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        b, s = x.shape

        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = th.nn.functional.relu(x)
        x = self.fc(x)
        x = th.nn.functional.softmax(x, dim=-1)
        return x.view(b, s, -1)


class GenomeTokenAttention(th.nn.Module):
    def __init__(self, embed_dim, hidden_size, output_size, num_layers=2):
        super(GenomeTokenAttention, self).__init__()
        self.embedding = th.nn.Embedding(20, embed_dim)
        self.mha = th.nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.group_norm = th.nn.GroupNorm(8, embed_dim)
        self.mlps = th.nn.ModuleList(
            [th.nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)]
        )
        self.fc = th.nn.Linear(embed_dim, output_size)

    def _sinusoidal_positional_embedding(self, n_position, embed_dim):
        pos_enc = np.array(
            [
                [
                    pos / np.power(10000, 2 * (i // 2) / embed_dim)
                    for i in range(embed_dim)
                ]
                if pos != 0
                else np.zeros(embed_dim)
                for pos in range(n_position)
            ]
        )

        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])

        return th.from_numpy(pos_enc).float()

    def forward(self, x):
        b, s = x.shape

        pos = np.arange(s)
        pos = self._sinusoidal_positional_embedding(s, 256).to(x.device)

        pos = pos.unsqueeze(0).expand(b, -1, -1)

        x = self.embedding(x) + pos
        x_res = x
        x, _ = self.mha(x, x, x)
        x = x + x_res
        x = x.permute(0, 2, 1)
        x = self.group_norm(x)
        x = x.permute(0, 2, 1)
        x = th.nn.functional.relu(x)
        for mlp in self.mlps:
            x = mlp(x)
            x = th.nn.functional.relu(x)
        x = self.fc(x)
        x = th.nn.functional.softmax(x, dim=-1)
        return x.view(b, s, -1)


def main():
    # =======================================================
    # Data Preprocessing
    # =======================================================

    # Get data
    data_path = "data/Protein_Secondary_Structure/"
    tain_file = "protein-secondary-structure.train.csv"
    test_file = "protein-secondary-structure.test.csv"

    train_human_readable, train_X, train_y, train_length = preprocess_data(
        data_path + tain_file
    )
    test_human_readable, test_X, test_y, test_length = preprocess_data(
        data_path + test_file
    )

    # Set train-val split
    train_val_split = 0.8
    # Create Dataset
    train_dataset = GenomeTokenDataset(train_X, train_y, train_length)
    train_dataset, val_dataset = th.utils.data.random_split(
        train_dataset,
        [
            int(train_val_split * len(train_dataset)),
            len(train_dataset) - int(train_val_split * len(train_dataset)),
        ],
    )

    train_dataloader = th.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_dataloader = th.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # =======================================================
    # Model Training
    # =======================================================

    # Device
    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "mps"
        if th.backends.mps.is_available()
        else "cpu"
    )

    # Model Parameters
    input_size = 1
    hidden_size = 128
    output_size = 3
    num_layers = 2

    # Model
    model = GenomeTokenLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    model = GenomeTokenAttention(256, hidden_size, output_size, num_layers).to(device)

    # Loss Function
    loss_fn = tv.ops.focal_loss.sigmoid_focal_loss
    loss_fn = th.nn.CrossEntropyLoss()
    metric_fn = accuracy

    # Optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=0.001)

    # Early Stopping
    patience = 5
    best_val_loss = np.inf
    counter = 0

    try:
        # Training Loop
        for epoch in range(100):
            model.train()

            train_losses = []
            train_metrics = []

            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                if isinstance(loss_fn, th.nn.CrossEntropyLoss):
                    loss = loss_fn(y_pred, y)
                elif callable(loss_fn):
                    loss = loss_fn(y_pred, y, alpha=0.25, gamma=2.0, reduction="mean")
                else:
                    loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                metric = metric_fn(y_pred, y)

                train_losses.append(loss.item())
                train_metrics.append(metric.item())

            train_loss = np.mean(train_losses)
            train_metric = np.mean(train_metrics)

            print(
                f"Epoch: {epoch} | Train , Loss: {train_loss:.3f}, Acc: {train_metric:.3f}",
                end=" | ",
            )

            model.eval()
            val_losses = []
            val_metrics = []

            with th.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    if isinstance(loss_fn, th.nn.CrossEntropyLoss):
                        loss = loss_fn(y_pred, y)
                    elif callable(loss_fn):
                        loss = loss_fn(
                            y_pred, y, alpha=0.25, gamma=2.0, reduction="mean"
                        )
                    else:
                        loss = loss_fn(y_pred, y)
                    metric = metric_fn(y_pred, y)

                    val_losses.append(loss.item())
                    val_metrics.append(metric.item())

            val_loss = np.mean(val_losses)
            val_metric = np.mean(val_metrics)

            print(
                f"Val , Loss: {val_loss:.3f}, Acc: {val_metric:.3f} | Patience: {patience - counter}"
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0

                # Save the model
                pass

            else:
                counter += 1
                if counter >= patience:
                    print("Early Stopping")
                    break

    except KeyboardInterrupt:
        print("Training interrupted")

    except Exception as e:
        raise e

    finally:
        pass

    # =======================================================
    # Inference evaluation
    # =======================================================

    # Split the test dataset into its constituent sequences
    test_sequences = []
    test_sequences_labels = []

    for i, length in enumerate(test_length):
        test_sequences.append(test_X[i : i + length])
        test_sequences_labels.append(test_y[i : i + length])

    print(f"Number of test sequences: {len(test_sequences)}")

    # Perform prediction on the test split
    predicted_labels = []
    prediction_accuracies = []
    for seq, label in zip(test_sequences, test_sequences_labels):
        # Convert the sequence to a tensor
        x = th.tensor(seq).long().unsqueeze(0).to(device)

        # Get the predicted label
        y_pred = model(x)
        predicted_label = th.argmax(y_pred, dim=-1).squeeze().cpu().numpy()
        predicted_labels.append(predicted_label)

        # Get the prediction accuracy
        prediction_accuracy = (predicted_label == label).mean()
        prediction_accuracies.append(prediction_accuracy)

    if any(
        [len(seq) != len(label) for seq, label in zip(test_sequences, predicted_labels)]
    ):
        raise ValueError(
            "The predicted labels and test sequences must have the same length"
        )

    # Load in character mapping and label mapping
    with open("data/Protein_Secondary_Structure/char_to_idx.json", "r") as f:
        char_mapping = json.load(f)

    with open("data/Protein_Secondary_Structure/label_to_idx.json", "r") as f:
        label_mapping = json.load(f)

    # Reverse the mapping
    idx_to_char = {v: k for k, v in char_mapping.items()}
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # Convert the sequences to strings
    test_sequences = [
        convert_sequence_to_string(seq, idx_to_char) for seq in test_sequences
    ]
    test_sequences_labels = [
        convert_sequence_to_string(seq, idx_to_label) for seq in test_sequences_labels
    ]
    predicted_labels = [
        convert_sequence_to_string(seq, idx_to_label) for seq in predicted_labels
    ]

    print(f"Average Prediction Accuracy: {np.mean(prediction_accuracies):.3f}")

    for i, (seq, label, prediction, acc) in enumerate(
        zip(
            test_sequences,
            test_sequences_labels,
            predicted_labels,
            prediction_accuracies,
        )
    ):
        print(f"Sequence {i + 1}, Accuracy: {acc:.3f}")
        print(f"Ground Truth: \t{label}")
        print(f"Prediction: \t{prediction}")
        print()

    # Save the predictions to a text file
    with open("predictions.txt", "w") as f:
        f.write(f"Average Prediction Accuracy: {np.mean(prediction_accuracies):.3f}\n")
        f.write("\n")
        for i, (seq, label, prediction, acc) in enumerate(
            zip(
                test_sequences,
                test_sequences_labels,
                predicted_labels,
                prediction_accuracies,
            )
        ):
            f.write(f"Sequence {i + 1}, Accuracy: {acc:.3f}\n")
            f.write(f"Ground Truth: \t{label}\n")
            f.write(f"Prediction: \t{prediction}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
