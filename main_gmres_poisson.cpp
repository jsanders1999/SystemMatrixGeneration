#include "operations.hpp"
#include "gmres_solver.hpp"
#include "timer.hpp"

#include <iostream>
#include <cmath>
#include <limits>

#include <cmath>

// Main program that solves the 3D Poisson equation
// on a unit cube. The grid size (nx,ny,nz) can be 
// passed to the executable like this:
//
// ./main_gmres_poisson <nx> <ny> <nz>
//
// or simply ./main_gmres_poisson <nx> for ny=nz=nx.
// If no arguments are given, the default nx=ny=nz=128 is used.
//
// Boundary conditions and forcing term f(x,y,z) are
// hard-coded in this file. See README.md for details
// on the PDE and boundary conditions.

// Forcing term
double f(double x, double y, double z)
{
  return z*sin(2*M_PI*x)*std::sin(M_PI*y) + 8*z*z*z;
}

// boundary condition at z=0
double g_0(double x, double y)
{
  return x*(1.0-x)*y*(1-y);
}

// Stencil for [∇^2 + d/dx] (Convection-diffusion with convection in x direction)
stencil3d laplace3d_stencil(int nx, int ny, int nz, double dx, double dy, double dz)
{
  //if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
  stencil3d L;
  L.nx=nx; L.ny=ny; L.nz=nz;
  
  L.value_c =  2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
  L.value_n = -1.0/(dy*dy);
  L.value_e = -1.0/(dx*dx) + 0.5/dx;
  L.value_s = -1.0/(dy*dy);
  L.value_w = -1.0/(dx*dx) - 0.5/dx;
  L.value_t = -1.0/(dz*dz);
  L.value_b = -1.0/(dz*dz);

  #ifdef USE_DIAG
  {
    std::cout << "USE_POLY case for order: " << order <<std::endl;
    L.value_n = L.value_n / L.value_c;
    L.value_e = L.value_e / L.value_c;
    L.value_s = L.value_s / L.value_c;
    L.value_w = L.value_w / L.value_c;
    L.value_t = L.value_t / L.value_c;
    L.value_b = L.value_b / L.value_c;
    L.value_c = 1.0;
  }
  #endif

  return L;
}

int main(int argc, char* argv[])
{
  // initialize MPI. This always has to be called first
  // to set up the internal data structures of the library.
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int nx, ny, nz;
  int order=0; //The order of the polynomial preconditioner

  if      (argc==1) {nx=64;            ny=64;            nz=64;            order = 2;}
  else if (argc==2) {nx=atoi(argv[1]); ny=nx;            nz=nx;            order = 2;}
  else if (argc==3) {nx=atoi(argv[1]); ny=nx;            nz=nx;            order = atoi(argv[2]);}
  else if (argc==4) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]); order = 2;         }
  else if (argc==5) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]); order = atoi(argv[4]);}
  else {std::cerr << "Invalid number of arguments (should be 0, 1, 2 or 3, or 4)"<<std::endl; exit(-1);}
  if (ny<0) ny=nx;
  if (nz<0) nz=nx;

  // total number of unknowns
 // int n=nx*ny*nz;

  // create the domain decomposition
  block_params BP;
  BP = create_blocks(nx,ny,nz);

  if (rank==0)
  { 
     std::cout << "Grid is ["<<nx << " x "<< ny << " x " << nz << "] and we have " << size << " processes." << std::endl;
  }

  //ordered printing for nicer output
  /*for (int p=0; p<size; p++){
     if (rank==p)std::cout << "Processor " << p << " grid is ["<<BP.bx_sz << " x "<<BP.by_sz<<" x "<<BP.bz_sz << "]"<<std::endl;
     MPI_Barrier(MPI_COMM_WORLD);
  }*/

  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

  // Laplace operator
  stencil3d L = laplace3d_stencil(BP.bx_sz, BP.by_sz, BP.bz_sz, dx, dy, dz);
  
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

  #ifdef USE_DIAG
  {
     for (int i=0; i<loc_n; i++) b[i] = b[i] /( 2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz) );   
  }
  #endif

  // solve the linear system of equations using GMRES
  int numIter, maxIter=500;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  try {
  //Timer t("gmres solver");
  #ifdef USE_POLY
     std::cout << "USE_POLY case for order: " << order <<std::endl;
     polygmres_solver(&L, &BP, loc_n, x, b, tol, maxIter, &resNorm, &numIter, order, 1);
  #else
     gmres_solver(&L, &BP, loc_n, x, b, tol, maxIter, &resNorm, &numIter, 1);
  #endif
  } catch(std::exception e)
  {
    std::cerr << "Caught an exception in gmres_solve: " << e.what() << std::endl;
    exit(-1);
  }
  delete [] x;
  delete [] b;
 
 // MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) Timer::summarize(); 

  MPI_Finalize();
  return 0;
}

