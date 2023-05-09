#!/bin/sh

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 1
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450
#SBATCH --time=00:05:00

# run GMRES for a n^3 Poisson problem
n=100

#TODO it is not necessary to compile on the compute node, but here we do it anyway!
make clean > /dev/null
make STENCIL=GLOBAL_COMM > /dev/null
srun ./main_gmres_poisson.x $n

make clean > /dev/null
make STENCIL=ONE_SIDED > /dev/null
srun ./main_gmres_poisson.x $n

make clean > /dev/null
make STENCIL=MPI_CART > /dev/null
srun ./main_gmres_poisson.x $n

