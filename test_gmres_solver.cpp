#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "gmres_solver.hpp"
#include "cg_solver.hpp"

#include <iostream>
#include <cmath>
#include <limits>




TEST(gmres_solver, gmres_solver_symm_stencil)
{
/*This is a test for gmres_solver which solves the linear system Ax=b
  Here we use a simple symmetric 3d stencil and an all-one vector as b, by which 
  we know the correct solution easily.*/
  const int nx=17, ny=7, nz=11;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  block_params BP = create_blocks(nx, ny, nz);  
  
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  stencil3d S;
  S.nx=BP.bx_sz; S.ny=BP.by_sz; S.nz=BP.bz_sz;
  S.value_c =  2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
  S.value_n = -1.0/(dy*dy);
  S.value_e = -1.0/(dx*dx);
  S.value_s = -1.0/(dy*dy);
  S.value_w = -1.0/(dx*dx);
  S.value_t = -1.0/(dz*dz);
  S.value_b = -1.0/(dz*dz);

  const int n = BP.bx_sz*BP.by_sz*BP.bz_sz;

  double *x = new double[n]; // solution vector x
  double *b = new double[n]; // right hand side vector b
  double *r = new double[n]; // residual r=Ax-b

  init(n, x, 0.0); // solution starts with [0,0,...]
  init(n, b, 1.0); // right hand side b=[1,1,...] 

  // solve the linear system of equations using GMRES
  int numIter, maxIter=100;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  gmres_solver(&S, &BP, n, x, b, tol, maxIter, &resNorm, &numIter, 0);

  apply_stencil3d(&S, &BP, x, r); // r = op * x
  axpby(n, 1.0, b, -1.0, r); // r = b - r

  double err=std::sqrt(dot(n, r, r))/std::sqrt(dot(n,b,b));
  EXPECT_NEAR(err, 0.0, std::sqrt(std::numeric_limits<double>::epsilon()));
  
  delete [] x;
  delete [] b;
  delete [] r;

}


// Forcing term, needed for the GMRES CG comparison test
double f(double x, double y, double z){
	return z*sin(2*M_PI*x)*std::sin(M_PI*y) + 8*z*z*z;
}

// boundary condition at z=0, needed for the GMRES CG comparison test
double g_0(double x, double y){
	return x*(1.0-x)*y*(1-y);
}

TEST(gmres_solver, gmres_and_cg_comparison)
{
/*This is a test for gmres_solver which solves the linear system Ax=b
  We compare the result of this solver with one obtained with a CG solver
  Here we use a simple symmetric 3d stencil and an all-one vector as b, by which 
  we know the correct solution easily.*/
  const int nx=17, ny=7, nz=11;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  block_params BP = create_blocks(nx, ny, nz);  
  
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  stencil3d S;
  S.nx=BP.bx_sz; S.ny=BP.by_sz; S.nz=BP.bz_sz;
  S.value_c =  2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
  S.value_n = -1.0/(dy*dy);
  S.value_e = -1.0/(dx*dx);
  S.value_s = -1.0/(dy*dy);
  S.value_w = -1.0/(dx*dx);
  S.value_t = -1.0/(dz*dz);
  S.value_b = -1.0/(dz*dz);

  const int n = BP.bx_sz*BP.by_sz*BP.bz_sz;

  double *x = new double[n]; // GMRES solution vector x
  double *y = new double[n]; // CG solution vector y
  double *z = new double[n]; // Difference in solutions vector z = x - y
  double *b = new double[n]; // right hand side vector b
  double *r = new double[n]; // residual r=Ax-b

  init(n, x, 0.0); // solution starts with [0,0,...]
  init(n, y, 0.0); // solution starts with [0,0,...]
  init(n, z, 0.0); // solution starts with [0,0,...]

  // initialize b with f(x,y,z) in the interior of the domain
	init(n, b, 0.0);

	for (int iz=0; iz<BP.bz_sz; iz++){
		double z = (BP.bz_start+iz)*dz;
		for (int iy=0; iy<BP.by_sz; iy++){
			double y = (BP.by_start+iy)*dy;
			for (int ix=0; ix<BP.bx_sz; ix++){
				double x = (BP.bx_start+ix)*dx;
				b[S.index_c(ix,iy,iz)] = f(x,y,z);
			}
		}
	}

	// Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
	if (BP.bz_start==0) {
		for (int iy=0; iy<BP.by_sz; iy++) {
			for (int ix=0; ix<BP.bx_sz; ix++){
				b[S.index_c(ix,iy,0)] -= S.value_b*g_0((BP.bx_start+ix)*dx, (BP.by_start+iy)*dy);
			}
		}
	}

  // solve the linear system of equations using GMRES
  int numIter, maxIter=500;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  gmres_solver(&S, &BP, n, x, b, tol, maxIter, &resNorm, &numIter, 1);
  cg_solver(&S, &BP, n, y, b, tol, maxIter, &resNorm, &numIter, 1);

  //z = x-y
  axpby(1.0, x, 0.0, z);
  axpby(-1.0, y, 1.0, z);


  double rel_err=std::sqrt(dot(n, z, z))/std::sqrt(dot(n, b, b));
  EXPECT_NEAR(rel_err, 0.0, std::sqrt(std::numeric_limits<double>::epsilon()));
  
  delete [] x;
  delete [] y;
  delete [] z;
  delete [] b;
  delete [] r;

}

TEST(gmres_solver, gmres_solver_asymm_stencil)
{
/*This is a test for gmres_solver which solves the linear system Ax=b
  Here we use a simple asymmetric 3d stencil and an all-one vector as b, by which 
  we know the correct solution easily.*/
  const int nx=12, ny=12, nz=11;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  block_params BP = create_blocks(nx, ny, nz);  
  
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  stencil3d S;
  S.nx=BP.bx_sz; S.ny=BP.by_sz; S.nz=BP.bz_sz;
  S.value_c =  2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
  S.value_n = -1.0/(dy*dy);
  S.value_e = -1.0/(dx*dx) + 0.5/dx;
  S.value_s = -1.0/(dy*dy);
  S.value_w = -1.0/(dx*dx) - 0.5/dx;
  S.value_t = -1.0/(dz*dz);
  S.value_b = -1.0/(dz*dz);

  const int n = BP.bx_sz*BP.by_sz*BP.bz_sz;

  double *x = new double[n]; // solution vector x
  double *b = new double[n]; // right hand side vector b
  double *r = new double[n]; // residual r=Ax-b

  init(n, x, 0.0); // solution starts with [0,0,...]
  init(n, b, 1.0); // right hand side b=[1,1,...] 

  // solve the linear system of equations using GMRES
  int numIter, maxIter=100;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  gmres_solver(&S, &BP, n, x, b, tol, maxIter, &resNorm, &numIter, 0);

  apply_stencil3d(&S, &BP, x, r); // r = op * x
  axpby(n, 1.0, b, -1.0, r); // r = b - r

  double err=std::sqrt(dot(n, r, r))/std::sqrt(dot(n,b,b));
  EXPECT_NEAR(err, 0.0, std::sqrt(std::numeric_limits<double>::epsilon()));
  
  delete [] x;
  delete [] b;
  delete [] r;

}
