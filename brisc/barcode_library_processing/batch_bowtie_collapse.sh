#!/bin/bash
# Job name
#SBATCH --job-name=CollapseAllBarcodes
# Number of tasks in job script
#SBATCH --ntasks=1
# Wall time limit
#SBATCH --time=0:10:00
# Partition
#SBATCH --partition=ncpu
# CPUs assigned to tasks
#SBATCH --cpus-per-task=1
# MEMORY assigned to tasks
#SBATCH --mem=4G

cd /brisc/barcode_library_processing/

sample_names=("PBC20")

for SAMPLENAME in "${sample_names[@]}"; do
	echo "Processing sample: $SAMPLENAME"
	sbatch -J "${SAMPLENAME}_bowtiecollapse" --export=subsample=$SUBSAMPLE,samplename=$SAMPLENAME bowtie_collapse.sh
done
