#!/bin/bash
# Submit one brainmapper SLURM job per mouse.
# Voxel sizes are z y x (order expected by brainmapper -v flag).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A VOXEL_SIZES
VOXEL_SIZES["BRAC11791.3d"]="5 2 2"
VOXEL_SIZES["BRAC11791.3e"]="5 2 2"
VOXEL_SIZES["BRAC11791.3f"]="8 2 2"
VOXEL_SIZES["BRAC11913.3a"]="5 2 2"
VOXEL_SIZES["BRAC11913.3b"]="5 2 2"
VOXEL_SIZES["BRAC11913.3c"]="8 2 2"
for mouse in "${!VOXEL_SIZES[@]}"; do
    voxel="${VOXEL_SIZES[$mouse]}"
    echo "Submitting job for ${mouse} (voxel: ${voxel})"
    sbatch \
        --job-name="brainmapper_${mouse}" \
        --export="MOUSE=${mouse},VOXEL_SIZE=${voxel}" \
        "${SCRIPT_DIR}/brainmapper_bash.sh"
done
