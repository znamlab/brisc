#!/bin/bash
# Job name
#SBATCH --job-name=CollapseBarcodes
# Number of tasks in job script
#SBATCH --ntasks=1
# Wall time limit
#SBATCH --time=1-00:00:00
# Partition
#SBATCH --partition=ncpu
# CPUs assigned to tasks
#SBATCH --cpus-per-task=32
# MEMORY assigned to tasks
#SBATCH --mem=256G
# Notifications
# %x expands to the job-name (set per sample via sbatch -J)
#SBATCH --output=/nemo/lab/znamenskiyp/home/users/becalia/logs/slurm_logs/%x.out

# Load modules
if command -v ml &>/dev/null; then
  ml purge
  ml Anaconda3/2024.10-1
  ml Trimmomatic
    ml FastQC
    ml Anaconda3
    ml Bowtie
    ml matlab/R2019a
else
  echo "Module system is not available on node: $CURRENT_NODE"
  resubmit_job
fi

eval "$(/camp/apps/eb/software/Anaconda3/2024.10-1/bin/conda shell.bash hook)"
conda activate brisc

#. ~/.bashrc
#Required packages = editdistance


#Select barcode type from "20bp_continuous", "20bp_bipartite", "BC2_bipartite", 20bp_wickersham_N2c or 20bp_wickersham_B19G
barcodetype="20bp_continuous"

mkdir -p /nemo/lab/znamenskiyp/home/shared/projects/barcode_diversity_analysis/
basedir=/nemo/lab/znamenskiyp/home/shared/projects/barcode_diversity_analysis/
mkdir -p /nemo/lab/znamenskiyp/home/shared/projects/barcode_diversity_analysis/collapsed_barcodes/
outdir=/nemo/lab/znamenskiyp/home/shared/projects/barcode_diversity_analysis/collapsed_barcodes/
codedir=/nemo/lab/znamenskiyp/home/users/becalia/code/brisc/brisc/barcode_library_processing


# Search fastqs.txt manifest for sample name and corresponding fastq file
# fastqs.txt must contain two comma separated columns, a list of the sample names and corresponding paths for the FASTQ files to be processed
trim_whitespace() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

fastq_manifest="${codedir}/fastqs.txt"
if [[ ! -f "$fastq_manifest" ]]; then
    echo "FASTQ manifest not found: $fastq_manifest" >&2
    exit 1
fi

sample_list=()
fastq_list=()
while IFS=',' read -r manifest_sample manifest_fastq || [[ -n "$manifest_sample" || -n "$manifest_fastq" ]]; do
    manifest_sample="$(trim_whitespace "$manifest_sample")"
    manifest_fastq="$(trim_whitespace "$manifest_fastq")"
    if [[ -z "$manifest_sample" && -z "$manifest_fastq" ]]; then
        continue
    fi
    if [[ -z "$manifest_sample" || -z "$manifest_fastq" ]]; then
        echo "Malformed entry in $fastq_manifest: '$manifest_sample,$manifest_fastq'" >&2
        exit 1
    fi
    sample_list+=("$manifest_sample")
    fastq_list+=("$manifest_fastq")
done < "$fastq_manifest"

if [[ ${#sample_list[@]} -ne ${#fastq_list[@]} ]]; then
    echo "Mismatch between sample and FASTQ counts in $fastq_manifest" >&2
    exit 1
fi

selected_fastq=""
for i in "${!sample_list[@]}"; do
    if [[ "${sample_list[$i]}" == "$samplename" ]]; then
        selected_fastq="${fastq_list[$i]}"
        break
    fi
done

if [[ -z "$selected_fastq" ]]; then
    echo "Sample $samplename not found in $fastq_manifest" >&2
    exit 1
fi


if [[ "$subsample" != "False" && -n "$subsample" ]]
then
    # Set up scratchdir for writing intermediate files and outdir for final results
    # Subsample and samplename received from batch script
    mkdir -p ${basedir}/subsample_${subsample}
    mkdir -p ${basedir}/subsample_${subsample}/${samplename}
    scratchdir=${basedir}/subsample_${subsample}/${samplename}
    mkdir -p ${outdir}/subsample_${subsample}/${samplename}
    outdir=${outdir}/subsample_${subsample}/${samplename}
else
    mkdir -p ${basedir}/${samplename}
    scratchdir=${basedir}/${samplename}
    outdir=${outdir}/${samplename}
fi

# List of P5 and P7 adapter sequences
adapt="${codedir}/nextera_rhAMP.fa"

echo "$selected_fastq"
outlog="${outdir}/${samplename}trimlog.txt"
outfile="${scratchdir}/${samplename}_trim.fastq"
echo $outfile
# Trim raw data to remove short reads and reads with low quality base calls
java -jar $EBROOTTRIMMOMATIC/trimmomatic-0.36.jar SE -threads 23 -trimlog $outlog "$selected_fastq" $outfile ILLUMINACLIP:"${adapt}":2:30:10 SLIDINGWINDOW:6:20 LEADING:3 MINLEN:101
echo "Completed Trimmomatic trimming"


for f in ${scratchdir}/${samplename}_trim.fastq;
do
	# Generate read quality visualisations
	mkdir -p ${outdir}/fastqc_trim
	fastqc --threads=23 --outdir="${outdir}/fastqc_trim" $f


done

# Take trimmed fastqs and split into just sequence data
sed -n '2~4p' ${scratchdir}/${samplename}_trim.fastq > ${scratchdir}/${samplename}_seq.fastq

echo "Completed FastQC and splitting"


#Perform pattern matching on flanking sequences of the barcode
#Discards any sequences that have incorrect flankers
python pattern_match.py ${scratchdir}/${samplename}_seq.fastq ${scratchdir}/${samplename}_seq_cleaned.fastq ${barcodetype}

echo "Completed flanker pattern matching"


mkdir -p "${scratchdir}/Bowtie/"
bowtiepath="${scratchdir}/Bowtie/${samplename}"

if [[ "$subsample" != "False" && -n "$subsample" ]]
then
    # Generate a random sample of ${subsample} lines
    shuf -n ${subsample} ${basedir}/${samplename}/${samplename}_seq_cleaned.fastq > ${bowtiepath}_sub_cleaned.fastq
else
    cp ${scratchdir}/${samplename}_seq_cleaned.fastq ${bowtiepath}_sub_cleaned.fastq
fi

#Remove duplicate UMIs with uniq
sort --parallel=31 < ${bowtiepath}_sub_cleaned.fastq > ${bowtiepath}_sorted_cleaned.fastq
echo "First sort complete"
uniq -c < ${bowtiepath}_sorted_cleaned.fastq > ${bowtiepath}_numsorted_cleaned.fastq
echo "Numbered sort complete"
sort -nr < ${bowtiepath}_numsorted_cleaned.fastq > ${bowtiepath}_finalsorted_cleaned.fastq
echo "Final sort complete"

awk '{print $1}' ${bowtiepath}_finalsorted_cleaned.fastq > ${bowtiepath}_counts.txt
awk '{print $2}' ${bowtiepath}_finalsorted_cleaned.fastq > ${bowtiepath}_UMIVBC_seq.txt
echo "Splitting complete"

#Cut out just the viral barcode and save with a number index
cut -b 13-32 ${bowtiepath}_UMIVBC_seq.txt > ${bowtiepath}_VBC_seq.txt
nl ${bowtiepath}_VBC_seq.txt > ${bowtiepath}_VBC_seq_indexed.txt
cut -b 1-12 ${bowtiepath}_UMIVBC_seq.txt > ${bowtiepath}_UMI_seq.txt
nl ${bowtiepath}_UMI_seq.txt > ${bowtiepath}_UMI_seq_indexed.txt

#Then count duplicate viral sequences and save these counts and a list of unique sequences
sort --parallel=31 -k 2 < ${bowtiepath}_VBC_seq_indexed.txt > ${bowtiepath}_sorted_VBC_seq.txt
echo "First VBC sort complete"
uniq -c -f 1 < ${bowtiepath}_sorted_VBC_seq.txt > ${bowtiepath}_numsorted_VBC_seq.txt
echo "Numbered VBC sort complete"
#sort reverse numerically by the first field and then by the third field, seq, if first is identical
sort -k1,1rn -k3,3 ${bowtiepath}_numsorted_VBC_seq.txt > ${bowtiepath}_finalsorted_VBC_seq.txt
echo "Final VBC sort complete"

awk '{print $1}' ${bowtiepath}_finalsorted_VBC_seq.txt > ${bowtiepath}_revnum_VBC_counts.txt
awk '{print $2}' ${bowtiepath}_finalsorted_VBC_seq.txt > ${bowtiepath}_revnum_VBC_index.txt
awk '{print $3}' ${bowtiepath}_finalsorted_VBC_seq.txt > ${bowtiepath}_revnum_VBC_seq.txt
echo "Splitting VBCs complete"

nl ${bowtiepath}_revnum_VBC_seq.txt > ${bowtiepath}_numbered_VBC_seq.txt
awk '{print ">" $1 "\n" $2}' ${bowtiepath}_numbered_VBC_seq.txt > ${bowtiepath}_fasta_VBC_seq.txt


mkdir -p ${bowtiepath}_indexes/fastaBC

# First make index files for performing alignment
# -q sets quiet

bowtie-build -q ${bowtiepath}_fasta_VBC_seq.txt ${bowtiepath}_indexes/fastaBC

#Perform bowtie alignment with edit distance 1 , 2 and 3
# Report alignments with at most -v mismatches.
# -f The query input files are FASTA files with assumed 40 Phred score
# --best Order alignments from best to worst
# -a Report all valid alignments up to -v mismatches

bowtie -v 1 -p 31 -f --best -a ${bowtiepath}_indexes/fastaBC ${bowtiepath}_fasta_VBC_seq.txt ${bowtiepath}_bowtiealignment_editd1_VBC.txt

# Bowtie output fields are:
# 1 - Name of read that aligned
# 3 - Reference strand aligned to

awk '{print $1}' ${bowtiepath}_bowtiealignment_editd1_VBC.txt > ${bowtiepath}_bowtiealignment_editd1_BC1.txt
awk '{print $3}' ${bowtiepath}_bowtiealignment_editd1_VBC.txt > ${bowtiepath}_bowtiealignment_editd1_BC3.txt

bowtie -v 2 -p 31 -f --best -a ${bowtiepath}_indexes/fastaBC ${bowtiepath}_fasta_VBC_seq.txt ${bowtiepath}_bowtiealignment_editd2_VBC.txt

awk '{print $1}' ${bowtiepath}_bowtiealignment_editd2_VBC.txt > ${bowtiepath}_bowtiealignment_editd2_BC1.txt
awk '{print $3}' ${bowtiepath}_bowtiealignment_editd2_VBC.txt > ${bowtiepath}_bowtiealignment_editd2_BC3.txt

bowtie -v 3 -p 31 -f --best -a ${bowtiepath}_indexes/fastaBC ${bowtiepath}_fasta_VBC_seq.txt ${bowtiepath}_bowtiealignment_editd3_VBC.txt

awk '{print $1}' ${bowtiepath}_bowtiealignment_editd3_VBC.txt > ${bowtiepath}_bowtiealignment_editd3_BC1.txt
awk '{print $3}' ${bowtiepath}_bowtiealignment_editd3_VBC.txt > ${bowtiepath}_bowtiealignment_editd3_BC3.txt
echo "Bowtie complete"

cd ${codedir}
matlab -nodesktop -nodisplay -batch "batchviruslibrary_matlab ${bowtiepath}"
echo "Matlab collapse complete, all done"

#Convert matplot .mat file to .txt output
python mat_conversion.py ${bowtiepath} ${outdir} ${samplename}
echo "${samplename} Sequencing collapse complete"
