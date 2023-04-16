#include "gtest_mpi.hpp"
#include "operations.hpp"
#include <iostream>
#include <mpi.h>

TEST(DotTest, DotProductIsCorrect) {
	int n = 300;
	int rank;
	double x[n], y[n];
	for (int i = 0; i < n; i++) {
		x[i] = (double)i+1.0;
		y[i] = 1.0 / (double)(i+1);
	}
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	double dot_product = dot(n, x, y, comm);

	if (rank == 0) {
		EXPECT_DOUBLE_EQ(dot_product, (double)n);
	}
}

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

TEST(stencil, index_order_kji){
  stencil3d S;
  S.nx=50;
  S.ny=33;
  S.nz=21;

  int i=10, j=15, k=9;

  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i-1,j,k)+1);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j-1,k)+S.nx);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j,k-1)+S.nx*S.ny);
}

TEST(operations, init){
  const int n=15;
  double x[n];
  for (int i=0; i<n; i++) x[i]=double(i+1);

  double val=42.0;
  init(n, x, val);

  double err=0.0;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(x[i]-val));

  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}

TEST(operations,stencil3d_symmetric){
	const int nx=2, ny=2, nz=2;
	const int n=nx*ny*nz;
	double* e=new double[n];
	for (int i=0; i<n; i++) e[i]=0.0;
	double* A=new double[n*n];

	stencil3d S;
	S.nx=nx; S.ny=ny; S.nz=nz;
	S.value_c = 8;
	S.value_n = 2;
	S.value_e = 4;
	S.value_s = 2;
	S.value_w = 4;
	S.value_b = 1;
	S.value_t = 1;

	for (int i=0; i<n; i++){
		e[i]=1.0;
		if (i>0) e[i-1]=0.0;
		apply_stencil3d(&S, e, A+i*n, MPI_COMM_WORLD);
	}

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
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
