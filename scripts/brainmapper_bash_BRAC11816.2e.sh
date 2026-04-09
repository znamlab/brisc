#!/bin/bash

#SBATCH -p ga100 # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 2-0:0 # time (D-HH:MM)
#SBATCH -o brainmapper_BRAC11816.2e.out

root_path="/nemo/lab/znamenskiyp/data/instruments/raw_data/projects/"
project="becalia_rabies_barseq"
mouse="BRAC11816.2e"
atlas="allen_mouse_10um"
# Imaged at the swc, signal is ch2
cell_file="$root_path/$project/$mouse/stitchedImages_100/2"
background_file="$root_path/$project/$mouse/stitchedImages_100/3"
model="/nemo/lab/znamenskiyp/home/shared/resources/cellfinder_resources/cellfinder_training/tail_cortex_brisc_revision/trained_model/model.keras"
output_dir="/nemo/lab/znamenskiyp/home/shared/projects/$project/$mouse/cellfinder_results"

echo "Loading conda environment"
source ~/.bashrc
conda activate brainglobe
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/brainglobe/lib/

echo "Running brainmapper with parameters:"
echo "  mouse:           $mouse"
echo "  voxel size (-v): 5 1 1 "
echo "  cell_file:       $cell_file"
echo "  background_file: $background_file"
echo "  output_dir:      $output_dir"
echo "  atlas:           $atlas"
echo "  model:           $model"
echo ""
brainmapper -s $cell_file -b $background_file -o $output_dir -v 5 1 1 --orientation psl  --trained-model $model --atlas $atlas --soma-diameter 20 --log-sigma-size 0.4 --threshold 8.0 --tiled-threshold 8.0 --ball-xy-size 6 --ball-z-size 10 --ball-overlap-fraction 0.6
