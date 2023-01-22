#!/bin/bash 

#SBATCH --partition=high-mem
#SBATCH -o %j.out 
#SBATCH -e %j.err
#SBATCH --time=06:00:00
#SBATCH --mem=100000

# executable 
#python -u calc_mld.py
python -u calc_reynolds.py
