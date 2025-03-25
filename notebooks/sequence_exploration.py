import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    return alt, mo, pl


@app.cell
def _(mo):
    mo.md(
        """
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
            while len(sequence) < max_length:
                sequence += sequence

            padded_sequences.append(sequence[:max_length])

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

    max_length = max(len(seq) for seq in sequences)
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
def _(mo):
    mo.md(
        """## Investigating the effect of data augmentation on the distribution of characters in the sequences"""
    )
    return


@app.cell
def _(mo):
    mo.md("""### Distribution of characters in the raw sequences""")
    return


@app.cell
def _(alt, mo, pl, sequences, unique_chars):
    # Create a frequency table for the characters in the sequences
    char_freq = {char: 0 for char in unique_chars}

    for sequence in sequences:
        for char in sequence:
            char_freq[char] += 1

    char_freq = {
        k: v
        for k, v in sorted(char_freq.items(), key=lambda item: item[1], reverse=True)
    }

    char_freq = pl.DataFrame(
        {
            "Character": list(char_freq.keys()),
            "Frequency": list(char_freq.values()),
        }
    )

    # Get the sum of the frequencies
    total_freq = char_freq["Frequency"].sum()

    # Divide the frequency of each character by the total frequency to get the percentage
    char_freq = char_freq.with_columns(Percent=pl.col("Frequency") / total_freq * 100)

    char_freq.select("Character", "Percent")

    mo.ui.altair_chart(
        alt.Chart(char_freq.to_pandas())
        .mark_bar()
        .encode(x="Character", y="Percent", color="Character")
    )
    return char, char_freq, sequence, total_freq


@app.cell
def _(mo):
    mo.md("""### Distribution of characters in the padded sequences""")
    return


@app.cell
def _(alt, mo, padded_sequences, pl, unique_chars):
    # Create a frequency table for the characters in the sequences
    padded_char_freq = {pchar: 0 for pchar in unique_chars}

    for p_sequence in padded_sequences:
        for pchar in p_sequence:
            padded_char_freq[pchar] += 1

    pad_char_freq = {
        k: v
        for k, v in sorted(
            padded_char_freq.items(), key=lambda item: item[1], reverse=True
        )
    }

    pad_char_freq = pl.DataFrame(
        {
            "Character": list(padded_char_freq.keys()),
            "Frequency": list(padded_char_freq.values()),
        }
    )

    # Get the sum of the frequencies
    pad_total_freq = pad_char_freq["Frequency"].sum()

    # Divide the frequency of each character by the total frequency to get the percentage
    pad_char_freq = pad_char_freq.with_columns(
        Percent=pl.col("Frequency") / pad_total_freq * 100
    )

    pad_char_freq.select("Character", "Percent")

    mo.ui.altair_chart(
        alt.Chart(pad_char_freq.to_pandas())
        .mark_bar()
        .encode(x="Character", y="Percent", color="Character")
    )
    return p_sequence, pad_char_freq, pad_total_freq, padded_char_freq, pchar


@app.cell
def _(mo):
    mo.md(
        """### Difference between the original and padded sequences in the distribution of characters"""
    )
    return


@app.cell
def _(alt, char_freq, mo, pad_char_freq, pl):
    diff = char_freq.join(pad_char_freq, on="Character", how="inner")

    diff = diff.with_columns(
        Diff=(pl.col("Percent") - pl.col("Percent_right")) / pl.col("Percent") * 100
    )

    average_diff = diff["Diff"].mean()
    std_diff = diff["Diff"].std()
    maximal_diff = max(abs(diff["Diff"].max()), abs(diff["Diff"].min()))

    diff.select("Character", "Diff")

    mo.ui.altair_chart(
        alt.Chart(diff.to_pandas())
        .mark_bar()
        .encode(x="Character", y="Diff", color="Character")
    )
    return average_diff, diff, maximal_diff, std_diff


@app.cell
def _(average_diff, maximal_diff, mo, std_diff):
    mo.md(
        f"The average difference in the distribution of characters between the original and padded sequences is **{average_diff:.2f}%**, with a standard deviation of **{std_diff:.4f}** and a maximal difference of **{maximal_diff:.2f}%**"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""Overall, the distribution of characters in the padded sequences is similar to that of the original sequences. This indicates that the padding process does not significantly alter the distribution of characters in the sequences, except for certain outliers. We progress using the padded sequences for- training the machine learning model."""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""## Investigating the effect of data augmentation on the distribution of labels in the sequences"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Distribution of Labels in the raw sequences""")
    return


@app.cell
def _(alt, labels, mo, pl, unique_labels):
    label_freq = {label: 0 for label in unique_labels}

    for l_label in labels:
        for label_char in l_label:
            label_freq[label_char] += 1

    label_freq = {
        k: v
        for k, v in sorted(label_freq.items(), key=lambda item: item[1], reverse=True)
    }

    label_freq = pl.DataFrame(
        {
            "Label": list(label_freq.keys()),
            "Frequency": list(label_freq.values()),
        }
    )

    # Get the sum of the frequencies
    total_label_freq = label_freq["Frequency"].sum()

    label_freq = label_freq.with_columns(
        Percent=pl.col("Frequency") / total_label_freq * 100
    )

    label_freq.select("Label", "Percent")

    mo.ui.altair_chart(
        alt.Chart(label_freq.to_pandas())
        .mark_bar()
        .encode(x="Label", y="Percent", color="Label")
    )
    return l_label, label_char, label_freq, total_label_freq


@app.cell
def _(mo):
    mo.md(r"""### Distribution of Labels in the padded sequences""")
    return


@app.cell
def _(alt, mo, padded_labels, pl, unique_labels):
    p_label_freq = {label: 0 for label in unique_labels}

    for pl_label in padded_labels:
        for p_l in pl_label:
            p_label_freq[p_l] += 1

    p_label_freq = {
        k: v
        for k, v in sorted(p_label_freq.items(), key=lambda item: item[1], reverse=True)
    }

    p_label_freq = pl.DataFrame(
        {
            "Label": list(p_label_freq.keys()),
            "Frequency": list(p_label_freq.values()),
        }
    )

    # Get the sum of the frequencies
    p_total_label_freq = p_label_freq["Frequency"].sum()

    # Divide the frequency of each character by the total frequency to get the percentage
    p_label_freq = p_label_freq.with_columns(
        Percent=pl.col("Frequency") / p_total_label_freq * 100
    )

    p_label_freq.select("Label", "Percent")

    mo.ui.altair_chart(
        alt.Chart(p_label_freq.to_pandas())
        .mark_bar()
        .encode(x="Label", y="Percent", color="Label")
    )
    return p_l, p_label_freq, p_total_label_freq, pl_label


@app.cell
def _(mo):
    mo.md(
        r"""### Difference between the original and padded sequences in the distribution of labels"""
    )
    return


@app.cell
def _(alt, label_freq, mo, p_label_freq, pl):
    label_diff = label_freq.join(p_label_freq, on="Label", how="inner")

    label_diff = label_diff.with_columns(
        Diff=((pl.col("Percent") - pl.col("Percent_right")) / pl.col("Percent") * 100)
    )

    print(label_diff.head())

    average_label_diff = label_diff["Diff"].mean()
    std_label_diff = label_diff["Diff"].std()
    maximal_label_diff = max(
        abs(label_diff["Diff"].max()), abs(label_diff["Diff"].min())
    )

    label_diff.select("Label", "Diff")

    mo.ui.altair_chart(
        alt.Chart(label_diff.to_pandas())
        .mark_bar()
        .encode(x="Label", y="Diff", color="Label")
    )
    return average_label_diff, label_diff, maximal_label_diff, std_label_diff


@app.cell
def _(average_label_diff, maximal_label_diff, mo, std_label_diff):
    mo.md(
        f"""The average difference in the distribution of labels between the original and padded sequences is **{average_label_diff:.2f}%**, with a standard deviation of **{std_label_diff:.4f}** and a maximal difference of **{maximal_label_diff:.2f}%**"""
    )
    return


if __name__ == "__main__":
    app.run()
