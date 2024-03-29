#!/bin/bash
 
#these are all the default values anyway 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1 
 
# the job name and output file
#SBATCH --job-name=dens_ratio #(default is the script name)
#SBATCH --output=myout.txt #(default is ~/slurm-<job ID>.out)
 
# time limit and memory allocation 
#SBATCH --time=0-24:00:00 #(2 days and 0 hours, default is 24 hours) 
#SBATCH --mem=100GB #(8 GB, default is 1G)
 
#again, these are the defaults anyway
#SBATCH --partition=short
#SBATCH --account=shared

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
 
python -u plot_compare_glider_path_geometry.py
