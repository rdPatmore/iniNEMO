#!/bin/bash
 
#these are all the default values anyway 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1  
#SBATCH --threads-per-core=1 
 
# the job name and output file
#SBATCH --job-name=srho #(default is the script name)
#SBATCH --output=out_prof.txt #(default is ~/slurm-<job ID>.out)
 
# time limit and memory allocation 
#SBATCH --time=0-8:00:00 #(2 days and 0 hours, default is 24 hours) 
#SBATCH --mem=100G #(8 GB, default is 1G)
 
#again, these are the defaults anyway
#export OMP_NUM_THREADS=1
 
python -u ../Physics/calc_glider_relevant_diags.py
