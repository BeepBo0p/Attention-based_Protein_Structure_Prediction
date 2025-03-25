# Predicting secondary structure
Make a tool for predicting secondary structure of a protein. It is known that it is possible to use data
from e.g. PSI-Blast (multiple alignment) to improve such predictions. It is not necessary to do that in
this project, but this aspect should at least be discussed in the report.
http://www.compbio.dundee.ac.uk/jpred/about.shtml


# Dataset Information

url to dataset: https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Protein+Secondary+Structure)

## Additional Information

This is a data set used by Ning Qian and Terry Sejnowski in their study using a neural net to predict the secondary structure of certain globular proteins [1].  The idea is to take a linear sequence of amino acids and to predict, for each of these amino acids, what secondary structure it is a part of within the protein.  There are three choices: alpha-helix, beta-sheet, and random-coil.  The data set contains both a large set of training data and a distinct set of data that can be used for testing the resulting network.  Qian and Sejnowski use a Nettalk-like approach and report an accuracy of 64.3% on the test set, and they speculate that this is about the best that can be done using only local context.

There is also a domain theory in the folder, donated and created by Jude Shavlik & Rich Maclin

## Has Missing Values?

No
