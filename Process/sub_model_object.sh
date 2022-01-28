#!/bin/bash
 
#these are all the default values anyway 
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1 
 
# the job name and output file
#SBATCH --job-name=nemo_mean #(default is the script name)
#SBATCH --output=myout.txt #(default is ~/slurm-<job ID>.out)
 
# time limit and memory allocation 
#SBATCH --time=0-24:00:00 #(2 days and 0 hours, default is 24 hours) 
#SBATCH --mem=100GB #(8 GB, default is 1G)
 
#again, these are the defaults anyway
#SBATCH --partition=cluster
#SBATCH --account=shared
#export OMP_NUM_THREADS=8
 
python -u model_object.py
