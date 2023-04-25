#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "gmres_solver.hpp"

#include <iostream>
#include <cmath>
#include <limits>

TEST(gmres_solver, gmres_solver_symm_stencil)
{
/*This is a test for gmres_solver which solves the linear system Ax=b
  Here we use a simple 3d stencil and an all-one vector as b, by which 
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

TEST(gmres_solver, gmres_solver_asymm_stencil)
{
/*This is a test for gmres_solver which solves the linear system Ax=b
  Here we use a simple 3d stencil and an all-one vector as b, by which 
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
  S.value_e = -1.0/(dx*dx);
  S.value_s = -1.1/(dy*dy);
  S.value_w = -1.1/(dx*dx);
  S.value_t = -0.9/(dz*dz);
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
