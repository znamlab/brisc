import editdistance
import sys

nemo_input = sys.argv[1]
nemo_output = sys.argv[2]
barcodetype = sys.argv[3]
print(sys.argv[3], flush=True)

with open(nemo_output, "w") as outfile:
    with open(nemo_input) as file:
        # Proceed one line at a time to prevent reading whole large file into memory
        if barcodetype == "20bp_continuous":
            for line in file:
                # If exact match, immediately keep line
                if (
                    line[12:43] == "GAAATGCCCTGAGTCCACCCCGGCCGACTGC"
                    and line[63:79] == "GGCGCCTAGCCGGTCA"
                ):
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:12] + line[43:63] + "\n")
                # Else, if there is one mismatch in SAM handle and none in Barcode handle, keep line
                elif (
                    editdistance.eval(line[12:43], "GAAATGCCCTGAGTCCACCCCGGCCGACTGC")
                    < 2
                    and line[63:79] == "GGCGCCTAGCCGGTCA"
                ):
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:12] + line[43:63] + "\n")

        elif barcodetype == "20bp_bipartite":
            for line in file:
                # If exact match, immediately keep line
                if (
                    line[12:35] == "GAAATGCCCTGAGTCCACCCCGG"
                    and line[45:61] == "CCGGTCGGCCGACTGC"
                ):
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:12] + line[35:45] + line[61:71] + "\n")
                # Else, if there is one mismatch in SAM handle and none in Barcode handle, keep line
                elif (
                    editdistance.eval(line[12:35], "GAAATGCCCTGAGTCCACCCCGG") < 2
                    and line[45:61] == "CCGGTCGGCCGACTGC"
                ):
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:12] + line[35:45] + line[61:71] + "\n")

        elif barcodetype == "BC2_bipartite":
            for line in file:
                # If exact match, immediately keep line
                if (
                    line[12:35] == "GAAATGCCCTGAGTCCACCCCGG"
                    and line[45:61] == "CCGGTCGGCCGACTGC"
                ):
                    # Indexes of UMI, BC2, assuming tab separation
                    outfile.write(line[0:12] + line[61:71] + "\n")
                # Else, if there is one mismatch in SAM handle and none in Barcode handle, keep line
                elif (
                    editdistance.eval(line[12:35], "GAAATGCCCTGAGTCCACCCCGG") < 2
                    and line[45:61] == "CCGGTCGGCCGACTGC"
                ):
                    # Indexes of UMI, BC2, assuming tab separation
                    outfile.write(line[0:12] + line[61:71] + "\n")

        elif barcodetype == "20bp_wickersham_N2c":
            for line in file:
                # If exact match, immediately keep line
                if line[20:37] == "ATTGACAGGGTGCCAGA" and line[57:60] == "AAT":
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:20] + line[37:57] + "\n")
                # Else, if there is one mismatch in SAM handle and none in Barcode handle, keep line
                elif (
                    editdistance.eval(line[20:37], "ATTGACAGGGTGCCAGA") < 2
                    and line[57:60] == "AAT"
                ):
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:20] + line[37:57] + "\n")

        elif barcodetype == "20bp_wickersham_B19G":
            for line in file:
                # If exact match, immediately keep line
                if line[20:37] == "ACAAAATGCCGGAGCT" and line[56:60] == "AGCA":
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:20] + line[36:56] + "\n")
                # Else, if there is one mismatch in SAM handle and none in Barcode handle, keep line
                elif (
                    editdistance.eval(line[20:37], "ACAAAATGCCGGAGCT") < 2
                    and line[56:60] == "AGCA"
                ):
                    # Indexes of UMI, BC1, BC2, assuming tab separation
                    outfile.write(line[0:20] + line[36:56] + "\n")
        else:
            raise ValueError("Select proper barcodetype")
