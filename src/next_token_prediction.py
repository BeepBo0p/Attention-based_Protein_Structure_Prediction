import polars as pl
import torch as th
import numpy as np


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
                th.tensor(self.genome_label, dtype=th.int64), num_classes=3
            )

    def __len__(self):
        return len(self.genome_sequence)

    def __getitem__(self, idx):
        return self.genome_sequence[idx], self.genome_label[idx]


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

    print(train_human_readable)

    print(test_human_readable)

    # Create Dataset
    train_dataset = GenomeTokenDataset(train_X, train_y, train_length)
    test_dataset = GenomeTokenDataset(test_X, test_y, test_length)

    train_first = train_dataset[0]
    test_first = test_dataset[0]

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")
    print(
        f"First Element Shapes (train): {train_first[0].shape}, {train_first[1].shape}"
    )
    print(f"First Element Shapes (test): {test_first[0].shape}, {test_first[1].shape}")

    train_dataloader = th.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_dataloader = th.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    for x, y in train_dataloader:
        print(x.shape)
        print(y.shape)
        break

    for x, y in test_dataloader:
        print(x.shape)
        print(y.shape)
        break

    # =======================================================
    # Model Training
    # =======================================================


if __name__ == "__main__":
    main()
