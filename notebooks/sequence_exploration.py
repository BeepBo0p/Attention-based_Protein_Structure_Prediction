import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def read_sequences():
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
                [
                    char
                    for char, i in zip(parts, range(0, len(parts)))
                    if i % 2 == 0
                ]
            )
            label = "".join(
                [
                    char
                    for char, i in zip(parts, range(0, len(parts)))
                    if i % 2 == 1
                ]
            )

            if len(sequence) != len(label):
                raise ValueError("Sequence and label lengths do not match")

            sequences.append(sequence)
            labels.append(label)

        return sequences, labels
    return (read_sequences,)


@app.cell
def _(read_sequences):
    data_path = "/home/beepboop/Desktop/Subjects/bioinfo_project_frfr/data/Protein_Secondary_Structure/protein-secondary-structure.train"

    sequences, labels = read_sequences(data_path)

    for seq, label in zip(sequences[:5], labels[:5]):
        print(f"Sequence: \t{seq}")
        print(f"Label: \t\t{label}")
        print(f"Length: \t{len(seq)}")
        print()
    return data_path, label, labels, seq, sequences


if __name__ == "__main__":
    app.run()
