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

    for idx, x, y in zip(length, X, Y):
        x = x[:idx]
        y = y[:idx]

    # Concatenate all sequences
    X = np.array([item for sublist in X for item in sublist])
    Y = np.array([item for sublist in Y for item in sublist])

    # Convert to tensors
    X = th.tensor(X)
    Y = th.tensor(Y)

    print(X.shape, Y.shape)

    return human_readable, X, y, length


class GenomeTokenDataset(th.utils.data.Dataset):
    def __init__(self, X, y, indice_information, sequence_length=100):
        self.X = X
        self.y = y
        self.indice_information = indice_information
        self.sequence_length = sequence_length

        self.genome_sequence = []
        self.genome_label = []

        for i, (x, y) in enumerate(zip(X, y)):
            indices_in_sequence = np.arange(i, i + sequence_length)

            if any([idx >= len(x) for idx in indices_in_sequence]):
                continue

            if any([idx in indice_information for idx in indices_in_sequence[1:]]):
                continue

            self.genome_sequence.append(x[indices_in_sequence])
            self.genome_label.append(y[indices_in_sequence])

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

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")


if __name__ == "__main__":
    main()
