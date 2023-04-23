#include "gtest_mpi.hpp"

#include "operations.hpp"

#include <iostream>


// note: you may add any number of tests to verify
// your code behaves correctly, but do not change
// the existing tests.

TEST(stencil, bounds_check)
{
  stencil3d S;
  S.nx=5;
  S.ny=3;
  S.nz=2;
  EXPECT_THROW(S.index_c(-1,0,0), std::runtime_error);
  EXPECT_THROW(S.index_c(S.nx,0,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,-1,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,S.ny,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,0,-1), std::runtime_error);
  EXPECT_THROW(S.index_c(0,0,S.nz), std::runtime_error);
}

TEST(stencil, index_order_kji)
{
  stencil3d S;
  S.nx=50;
  S.ny=33;
  S.nz=21;

  int i=10, j=15, k=9;

  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i-1,j,k)+1);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j-1,k)+S.nx);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j,k-1)+S.nx*S.ny);
}

TEST(operations, init)
{
  const int n=15;
  double x[n];
  for (int i=0; i<n; i++) x[i]=double(i+1);

  double val=42.0;
  init(n, x, val);

  double err=0.0;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(x[i]-val));

  // note: EXPECT_NEAR uses a tolerance relative to the size of the target,
  // near 0 this is very small, so we use an absolute test instead by 
  // comparing to 1 instead of 0.
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}


TEST(operations, dot) {
  const int n=150;
  double x[n], y[n];

  for (int i=0; i<n; i++)
  {
    x[i] = double(i+1);
    y[i] = 1.0/double(i+1);
  }

  double res = dot(n, x, y); // The results of dot(x,y) should be equal to the length n
  EXPECT_NEAR(res, (double)n, n*std::numeric_limits<double>::epsilon());
}


TEST(operations, axpby)
{
  const int n=10;
  double x[n], y[n];
  
  double a=2.0; 
  double b=2.0;

  for (int i=0; i<n; i++)
  {
     x[i] = double(i+1)/2.0;
     y[i] = double(n-i-1)/2.0;
  }

  axpby(n, a, x, b, y); // The results of axpby should be the array in which every element is equal to the length n
  
  double err=0.0;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(y[i]-double(n)));
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}


//TODO Add tests that use multiple MPI processes.
TEST(operations, stencil3d_symmetric){
	const int nx=3, ny=3, nz=3;
	const int n=nx*ny*nz;
	double* e=new double[n];
	for (int i=0; i<n; i++) e[i]=0.0;
	double* A=new double[n*n];
	for (int ix=0; ix<n*n; ix++) {
		A[ix]=1.0;
	}

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	block_params BP = create_blocks(nx, ny, nz);
	stencil3d S;
	S.nx=BP.bx_sz; S.ny=BP.by_sz; S.nz=BP.bz_sz;
	S.value_c = 8;
	S.value_n = 2;
	S.value_e = 4;
	S.value_s = 2;
	S.value_w = 4;
	S.value_b = 1;
	S.value_t = 1;

	for (int ix=0; ix<n; ix++){
		e[ix]=1.0;
		if (ix>0) {
			e[ix-1]=0.0;
		}
		apply_stencil3d(&S, &BP, e, A+ix*n);
	}

	if (rank == 1) {
		int wrong_entries=0;
		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++)
			{
				if (A[i*n+j]!=A[j*n+i]) wrong_entries++;
			}
		}
		EXPECT_EQ(0, wrong_entries);

		if (wrong_entries)
		{
			std::cout << "Your matrix (computed on a 3x3x3 grid by apply_stencil(I)) is ..."<<std::endl;
			for (int j=0; j<n; j++){
				for (int i=0; i<n; i++){
					std::cout << A[i*n+j] << " ";
				}
				std::cout << std::endl;
			}
		}
	}
	delete [] e;
	delete [] A;
}


TEST(operations, given_rotation)
{
  int k = 3;
  double h[k+2] = {-0.2910, -0.0411, -0.1517, 0.1386, 0.2944};
  double cs[k+2] = {0.9753,  0.2696, -0.5138, 0.0, 0.0};
  double sn[k+2] = {0.2208,  0.9630,  0.8579, 0.0, 0.0};
  double resh[k+2] = {-0.292887, -0.139571, 0.151877, 0.294843, 0.0}; 

  given_rotation(k, h, cs, sn);
 
  double err=0.0;
  for (int i=0; i<k+2; i++) err = std::max(err, std::abs(h[i]-resh[i]));
  EXPECT_NEAR(1.0+err, 1.0, 1e-5);
}


TEST(operations, arnoldi)
{
  int k = 0;
  int nx=2, ny=2, nz=2;
  int n = nx*ny*nz;
 
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  block_params BP = create_blocks(nx, ny, nz);
  stencil3d S;
  S.nx=BP.bx_sz; S.ny=BP.by_sz; S.nz=BP.bz_sz;
  S.value_c = 6;
  S.value_n = 1;
  S.value_e = 1;
  S.value_s = 1;
  S.value_w = 1;
  S.value_b = 1;
  S.value_t = 1;

  // Q and H are stored by column major
  double* Q = new double[2*n];
  double* H = new double[2];
  // initialize Q and H
  init(n, Q, -0.3536);
  init(n,Q+n,0.0);
  init(2, H, 0.0);

  arnoldi(k, Q, H, &S, &BP); 

  double res_q[n];
  double res_h[2];
  init(n, res_q, 0.3536);
  res_h[0]=9.0024; res_h[1]=0.0024; 
   
  double err = 0.0;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(Q[(k+1)*n+i]-res_q[i]));
  for (int i=0; i<2; i++) err = std::max(err, std::abs(H[i]-res_h[i]));
  
  EXPECT_NEAR(1.0+err, 1.0, 1e-4);

  free(Q);
  free(H);
}

























