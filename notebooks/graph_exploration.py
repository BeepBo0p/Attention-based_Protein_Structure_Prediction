import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import networkx as nx
    return mo, nx, pl


@app.cell
def _(mo):
    mo.md(
        r"""
        # Exploratory Data Analysis
    
        ## Raw data information
        """
    )
    return


@app.cell
def function_declarations():
    def read_sequences(file_path):
        """
        Read sequences from a file, ignoring lines that begin with '#' or ' '.
        Sequences start with '<>' and end with a line containing 'end'.

        Args:
            file_path (str): Path to the file to read

        Returns:
            List of sequence strings
        """
        raw_data = []
        current_sequence = None

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()

                # Skip comments and lines starting with spaces
                if line.startswith("#") or line.startswith(" "):
                    continue

                # Start of a sequence
                if line.startswith("<>"):
                    current_sequence = [line]
                # End of a sequence
                elif line == "end" and current_sequence is not None:
                    raw_data.append("\n".join(current_sequence[1:]))
                    current_sequence = None
                # Part of a sequence
                elif current_sequence is not None:
                    current_sequence.append(line)

        sequences, labels = [], []

        # For each sequence, split it into its sequence and label parts
        for sequence in raw_data:
            parts = sequence.split("\n")

            # Join the sequence parts together
            parts = "".join(parts)

            # Remove any whitespace
            parts = parts.replace(" ", "")

            sequence = "".join(
                [char for char, i in zip(parts, range(0, len(parts))) if i % 2 == 0]
            )
            label = "".join(
                [char for char, i in zip(parts, range(0, len(parts))) if i % 2 == 1]
            )

            if len(sequence) != len(label):
                raise ValueError("Sequence and label lengths do not match")

            sequences.append(sequence)
            labels.append(label)

        return sequences, labels


    def pad_sequences(sequences, max_length):
        """
        Pad sequences to the specified length by repeating the sequence.

        Args:
            sequences (List[str]): List of sequences to pad
            max_length (int): Maximum length of the padded sequences

        Returns:
            List of padded sequences
        """
        padded_sequences = []

        for sequence in sequences:

            # Pad the sequence with spaces
            padded_sequence = sequence.ljust(max_length, " ")
            padded_sequences.append(padded_sequence)

        return padded_sequences
    return pad_sequences, read_sequences


@app.cell
def sequence_processing(pad_sequences, read_sequences):
    # =============================================================================
    # Read Sequences
    # =============================================================================

    data_path = "data/Protein_Secondary_Structure/protein-secondary-structure.train"

    sequences, labels = read_sequences(data_path)

    for seq, label in zip(sequences[:5], labels[:5]):
        print(f"Sequence: \t{seq}")
        print(f"Label: \t\t{label}")
        print(f"Length: \t{len(seq)}")
        print()

    # =============================================================================
    # Pad Sequences
    # =============================================================================

    max_length = 512  # max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, max_length)
    padded_labels = pad_sequences(labels, max_length)

    print("\n\n============================================================\n\n")

    for seq, padded_seq in zip(sequences[:5], padded_sequences[:5]):
        print(f"Sequence: \t\t{seq}")
        print(f"Padded Seq: \t{padded_seq}")
        print(f"Length: \t\t{len(padded_seq)}")
        print()

    for label, padded_label in zip(labels[:5], padded_labels[:5]):
        print(f"Label: \t\t\t{label}")
        print(f"Padded Label: \t{padded_label}")
        print(f"Length: \t\t{len(padded_label)}")
        print()
    return (
        data_path,
        label,
        labels,
        max_length,
        padded_label,
        padded_labels,
        padded_seq,
        padded_sequences,
        seq,
        sequences,
    )


@app.cell
def _(mo, sequences):
    mo.md(f"Number of sequences: {len(sequences)}")
    return


@app.cell
def _(mo, sequences):
    mo.md(
        f"Shortest sequence is {min(len(seq) for seq in sequences)} and longest is {max(len(seq) for seq in sequences)} chars long"
    )
    return


@app.cell
def data_analysis(labels, sequences):
    # Extract the unique characters from the sequences
    unique_chars = set("".join(sequences))
    unique_labels = set("".join(labels))

    print(f"Unique Characters: {unique_chars}")
    print(f"Unique Labels: {unique_labels}")
    return unique_chars, unique_labels


@app.cell
def _(mo, unique_chars):
    mo.md(f"There are {len(unique_chars)} unique characters in the sequences.")
    return


@app.cell
def _(mo, unique_labels):
    mo.md(f"And {len(unique_labels)} unique labels in the sequences.")
    return


@app.cell
def _(labels, sequences, unique_chars, unique_labels):
    # Create mapping dictionaries for characters and labels
    char_to_idx = {char: idx for idx, char in enumerate(sorted(unique_chars))}
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Convert sequences and labels to numeric representations
    numeric_sequences = []
    numeric_labels = []

    for _seq, _label in zip(sequences, labels):
        numeric_seq = [char_to_idx[char] for char in _seq]
        numeric_label = [label_to_idx[char] for char in _label]

        numeric_sequences.append(numeric_seq)
        numeric_labels.append(numeric_label)

    # Display examples of the numeric conversion
    for i in range(3):
        print(f"Original sequence: {sequences[i][:20]}...")
        print(f"Numeric sequence: {numeric_sequences[i][:20]}...")
        print(f"Original label: {labels[i][:20]}...")
        print(f"Numeric label: {numeric_labels[i][:20]}...")
        print()

    # Create reverse mapping for reference
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Save the reverse mapping as json files
    import json

    json.dump(char_to_idx, open("data/Protein_Secondary_Structure/char_to_idx.json", "w"))
    json.dump(label_to_idx, open("data/Protein_Secondary_Structure/label_to_idx.json", "w"))


    print("Character mapping:")
    print(char_to_idx)
    print("\nLabel mapping:")
    print(label_to_idx)
    return (
        char_to_idx,
        i,
        idx_to_char,
        idx_to_label,
        json,
        label_to_idx,
        numeric_label,
        numeric_labels,
        numeric_seq,
        numeric_sequences,
    )


@app.cell
def _(labels, numeric_labels, numeric_sequences, pl, sequences):
    import numpy as np

    # Convert the numeric sequences and labels to numpy arrays
    # We'll pad them to the same length for consistency
    X = np.ones((len(numeric_sequences), 512), dtype=np.int32) * -1
    y = np.ones((len(numeric_labels), 512), dtype=np.int32) * -1

    # Fill the arrays with the numeric sequences and labels
    for _i, (_seq, _label) in enumerate(zip(numeric_sequences, numeric_labels)):
        seq_len = len(_seq)
        X[_i, :seq_len] = _seq
        y[_i, :seq_len] = _label

    # Create a polars DataFrame
    # First, let's create a dictionary with our data
    data_dict = {
        "sequence": sequences,
        "label": labels,
        "length": [len(seq) for seq in sequences],
    }

    print(f"x shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_dict = {f"x{i}": X[:, i] for i in range(X.shape[1])}
    y_dict = {f"y{i}": y[:, i] for i in range(y.shape[1])}

    data_dict.update(X_dict)
    data_dict.update(y_dict)

    # Create the DataFrame
    df = pl.DataFrame(data_dict)

    # Display the DataFrame
    print("DataFrame shape:", df.shape)
    print("\nDataFrame schema:")
    print(df.schema)

    # Save the DataFrame to a csv file
    df.write_csv("data/Protein_Secondary_Structure/protein-secondary-structure.train.csv")

    df
    return X, X_dict, data_dict, df, np, seq_len, y, y_dict


if __name__ == "__main__":
    app.run()
