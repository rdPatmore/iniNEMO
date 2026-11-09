#!/bin/bash

# Slurm job options (job-name, job time)
#SBATCH --job-name=data_analysis
#SBATCH --time=12:30:00
#SBATCH --ntasks=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=n02-bas             
#SBATCH --partition=serial
#SBATCH --qos=serial

# Define memory required for this jobs. By default, you would 
# get just under 2 GB, but you can ask for up to 125 GB.
#SBATCH --mem=125GB

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically 
#   using threading.
export OMP_NUM_THREADS=1

module load cray-python

python -u process.py

