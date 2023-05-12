#!/bin/sh

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450
#SBATCH --time=00:05:00

# run GMRES for a n^3 Poisson problem
n=50

make clean > /dev/null
make STENCIL=GLOBAL_COMM > /dev/null
srun ./main_gmres_poisson.x $n

make clean > /dev/null
make STENCIL=ONE_SIDED > /dev/null
srun ./main_gmres_poisson.x $n

make clean > /dev/null
make STENCIL=MPI_CART > /dev/null
srun ./main_gmres_poisson.x $n

