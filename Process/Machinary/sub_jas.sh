#!/bin/bash 

#SBATCH --partition=long-serial
#SBATCH -o %j.out 
#SBATCH -e %j.err
#SBATCH --time=06:00:00
#SBATCH --mem=100000

# executable 
#python -u calc_mld.py
python -u ../Budgets/calc_KE.py
