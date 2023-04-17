#include "gtest_mpi.hpp"
#include "operations.hpp"
#include <iostream>
#include <mpi.h>
#include <math.h>

TEST(operations, dot) {
	int nx=3, ny=10, nz=10;
	int n = nx*ny*nz;
	double x[n], y[n];
	for (int i = 0; i < n; i++) {
		x[i] = (double)i+1.0;
		y[i] = 1.0 / (double)(i+1);
	}
	block_params BP = create_blocks(nx, ny, nz);
	init(&BP, x, 22);
	double dot_result = 99999; //dot(&BP, x, y); TODO what is this linking error??
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		EXPECT_DOUBLE_EQ(dot_result, (double)n);
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

	int nx=3, ny=10, nz=10;
	int n = nx*ny*nz;
	double x[n];
	for (int i=0; i<n; i++) x[i]=double(i+1);

	double val=42.0;
	block_params BP = create_blocks(nx, ny, nz);
	init(&BP, x, val);

	double err=0.0;
	for (int i=0; i<n; i++) err = std::max(err, std::abs(x[i]-val));

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
