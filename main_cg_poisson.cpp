#include "operations.hpp"
#include "cg_solver.hpp"
#include "timer.hpp"

#include <iostream>
#include <cmath>
#include <limits>

#include <cmath>
#include <mpi.h>

// Main program that solves the 3D Poisson equation
// on a unit cube. The grid size (nx,ny,nz) can be 
// passed to the executable like this:
//
// ./main_cg_poisson <nx> <ny> <nz>
//
// or simply ./main_cg_poisson <nx> for ny=nz=nx.
// If no arguments are given, the default nx=ny=nz=128 is used.
//
// Boundary conditions and forcing term f(x,y,z) are
// hard-coded in this file. See README.md for details
// on the PDE and boundary conditions.

// Forcing term
double f(double x, double y, double z){
	return z*sin(2*M_PI*x)*std::sin(M_PI*y) + 8*z*z*z;
}

// boundary condition at z=0
double g_0(double x, double y){
	return x*(1.0-x)*y*(1-y);
}

stencil3d laplace3d_stencil(int nx, int ny, int nz, double dx, double dy, double dz){
	//if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
	stencil3d L;
	L.nx=nx; L.ny=ny; L.nz=nz;
	L.value_c =  2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
	L.value_n = -1.0/(dy*dy);
	L.value_e = -1.0/(dx*dx);
	L.value_s = -1.0/(dy*dy);
	L.value_w = -1.0/(dx*dx);
	L.value_t = -1.0/(dz*dz);
	L.value_b = -1.0/(dz*dz);
	return L;
}

int main(int argc, char* argv[]){
	// initialize MPI. This always has to be called first.
	// to set up the internal data structures of the library.
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int nx, ny, nz;

	if      (argc==1) {nx=64;           ny=64;           nz=64;}
	else if (argc==2) {nx=atoi(argv[1]); ny=nx;            nz=nx;}
	else if (argc==4) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]);}
	else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)"<<std::endl; exit(-1);}
	if (ny<0) ny=nx;
	if (nz<0) nz=nx;

	// create the domain decomposition
	block_params BP = create_blocks(nx, ny, nz);


	if (rank==0) {
		std::cout << "Domain decomposition:"<<std::endl;
		std::cout << "Grid is           ["<<nx << " x "<< ny << " x " << nz << "]"<<std::endl;
		std::cout << "Processor grid is ["<<BP.bx_sz << " x "<<BP.by_sz<<" x "<<BP.bz_sz << "]"<<std::endl;
	}
	
	//ordered printing for nicer output
	for (int p=0; p<size; p++){
		if (rank==p)std::cout << "Processor " << p << " grid is ["<<BP.bx_sz << " x "<<BP.by_sz<<" x "<<BP.bz_sz << "]"<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

	// Laplace operator
	stencil3d L = laplace3d_stencil(BP.bx_sz, BP.by_sz, BP.bz_sz, dx, dy, dz);

	// The stencil needs to take the block_params object along
	// so that the offsets and neighbors can be determined
	// inside the apply function
	//L.BP=BP; //TODO should we do this instead?

	long loc_n = BP.bx_sz*BP.by_sz*BP.bz_sz;
	// solution vector: start with a 0 vector
	double *x = new double[loc_n];
	init(loc_n, x, 0.0);

	// right-hand side
	double *b = new double[loc_n];
	init(loc_n, b, 0.0);

	// initialize the rhs with f(x,y,z) in the interior of the domain
	for (int iz=0; iz<BP.bz_sz; iz++){
		double z = (BP.bz_start+iz)*dz;
		for (int iy=0; iy<BP.by_sz; iy++){
			double y = (BP.by_start+iy)*dy;
			for (int ix=0; ix<BP.bx_sz; ix++){
				double x = (BP.bx_start+ix)*dx;
				b[L.index_c(ix,iy,iz)] = f(x,y,z);
			}
		}
	}
	// Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
	if (BP.bz_start==0) {
		for (int iy=0; iy<BP.by_sz; iy++) {
			for (int ix=0; ix<BP.bx_sz; ix++){
				b[L.index_c(ix,iy,0)] -= L.value_b*g_0((BP.bx_start+ix)*dx, (BP.by_start+iy)*dy);
			}
		}
	}

	// solve the linear system of equations using CG
	int numIter, maxIter=1000;
	double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

	try {   //Timer t("cg solver");
		cg_solver(&L, &BP, loc_n, x, b, tol, maxIter, &resNorm, &numIter, 1);
	} catch(std::exception e) {
		std::cerr << "Caught an exception in cg_solve: " << e.what() << std::endl;
		exit(-1);
	}
	delete [] x;
	delete [] b;

	if (rank==0) Timer::summarize();

	// no MPI calls must be issued after this one...
	MPI_Finalize();
	return 0;
}
