#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH --mem=4GB
#SBATCH --account=Education-EEMCS-Courses-WI4450

# run CG for a n^3 Poisson problem
n=100
export OMP_NUM_THREADS=1
srun ./main_gmres_poisson.x $n 
