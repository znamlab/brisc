#!/bin/bash

#SBATCH -p ga100 # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 2-0:0 # time (D-HH:MM)
#SBATCH -o brainmapper_%j.out

root_path="/nemo/lab/znamenskiyp/data/instruments/raw_data/projects/"
project="rabies_barcoding"
mouse="${MOUSE:-BRAC11913.3a}"
atlas="allen_mouse_10um"
cell_file="$root_path/$project/$mouse/stitchedImages_100/3"
background_file="$root_path/$project/$mouse/stitchedImages_100/2"
model="/nemo/lab/znamenskiyp/home/shared/resources/cellfinder_resources/cellfinder_training/tail_cortex_brisc_revision/trained_model/model.keras"
output_dir="/nemo/lab/znamenskiyp/home/shared/projects/$project/$mouse/cellfinder_results_010"

echo "Loading conda environment"
source ~/.bashrc
conda activate brainglobe
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/brainglobe/lib/

echo "Running brainmapper with parameters:"
echo "  mouse:           $mouse"
echo "  voxel size (-v): ${VOXEL_SIZE:-5 2 2}"
echo "  cell_file:       $cell_file"
echo "  background_file: $background_file"
echo "  output_dir:      $output_dir"
echo "  atlas:           $atlas"
echo "  model:           $model"
echo ""
brainmapper -s $cell_file -b $background_file -o $output_dir -v ${VOXEL_SIZE:-5 2 2}  --orientation psl  --trained-model $model --atlas $atlas --soma-diameter 11 --log-sigma-size 0.3 --threshold 6.0 --tiled-threshold 8.0 --ball-xy-size 6 --ball-z-size 10 --ball-overlap-fraction 0.6 --soma-spread-factor 1.1
