import sys


def read_sequences(file_path):
    """
    Read sequences from a file, ignoring lines that begin with '#' or ' '.
    Sequences start with '<>' and end with a line containing 'end'.

    Args:
        file_path (str): Path to the file to read

    Returns:
        List of sequence strings
    """
    sequences = []
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
                sequences.append("\n".join(current_sequence))
                current_sequence = None
            # Part of a sequence
            elif current_sequence is not None:
                current_sequence.append(line)

    return sequences


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        sequences = read_sequences(file_path)
        print(f"Found {len(sequences)} sequences.")
    else:
        print("Please provide a file path as an argument.")
