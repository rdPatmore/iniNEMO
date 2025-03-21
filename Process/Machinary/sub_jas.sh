#!/bin/bash 

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1 

#SBATCH --partition=high-mem
#SBATCH -o %j.out 
#SBATCH -e %j.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G

# executable 
#python -u calc_mld.py
conda activate coast

path='/home/users/ryapat30/iniNEMO/'
export PYTHONPATH=$path:$PYTHONPATH

python -u ../Budgets/calc_KE.py
