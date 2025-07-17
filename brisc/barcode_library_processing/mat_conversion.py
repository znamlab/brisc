import scipy.io
import pandas as pd
import sys

inputdir = sys.argv[1]
outdir = sys.argv[2]
samplename = sys.argv[3]

for i in range(1, 4):
    mat = scipy.io.loadmat(f"{inputdir}_editd{i}_collapsed.mat")
    counts = mat["collapsed"][0][0][0]
    sequences = mat["collapsed"][0][0][1]
    sequence_strings = ["".join(chr(i) for i in row) for row in sequences]

    count_df = pd.DataFrame(counts, columns=["counts"])
    sequence_df = pd.DataFrame(sequence_strings, columns=["sequences"])

    # Concatenate the dataframes horizontally
    df = pd.concat([count_df, sequence_df], axis=1)
    df.to_csv(
        f"{outdir}/{samplename}_bowtie_ed{i}.txt", header=False, index=False, sep="\t"
    )
