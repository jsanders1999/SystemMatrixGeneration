#!/bin/sh

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450
#SBATCH --time=00:05:00

# run GMRES for a n^3 Poisson problem
n=180

srun ./main_gmres_poisson_global.x $n
srun ./main_gmres_poisson_win.x $n
srun ./main_gmres_poisson_cart.x $n

