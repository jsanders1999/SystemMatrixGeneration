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
	double dot_result = dot(&BP, x, y); 
	
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

	block_params BP;
	// Choose number of blocks in x y and z directions
	BP.bkx = ceil(pow(size, 1.0/3.0));
	BP.bky = ceil(sqrt(size/BP.bkx));
	BP.bkz = ceil(size / (BP.bkx * BP.bky));
	// Calculate the sizes (ignoring divisibility)
	BP.bx_sz = (nx + BP.bkx - 1) / BP.bkx;
	BP.by_sz = (ny + BP.bky - 1) / BP.bky;
	BP.bz_sz = (nz + BP.bkz - 1) / BP.bkz;
	// Calculate index in each direction
	BP.bx_idx = rank % BP.bkx;
	BP.by_idx = (rank / BP.bkx) % BP.bky;
	BP.bz_idx = rank / (BP.bkx * BP.bky);
	// Grid points are often not perfectly divisible into blocks, we handle that here.
	// x-direction
	int xb_start = BP.bx_idx * BP.bx_sz;
	int xb_end = xb_start + BP.bx_sz - 1;
	if (xb_end >= nx) {
		BP.bx_sz = nx - xb_start;
	}
	// y-direction
	int yb_start = BP.by_idx * BP.by_sz;
	int yb_end = yb_start + BP.by_sz - 1;
	if (yb_end >= ny) {
		BP.by_sz = ny - yb_start;
	}
	// z-direction
	int zb_start = BP.bz_idx * BP.bz_sz;
	int zb_end = zb_start + BP.bz_sz - 1;
	if (zb_end >= nz) {
		BP.bz_sz = nz - zb_start;
	}
	// Save rank of neighbours
	BP.rank_w = BP.bx_idx > 0          ? rank - 1               : MPI_PROC_NULL; //west
	BP.rank_e = BP.bx_idx < BP.bkx - 1 ? rank + 1               : MPI_PROC_NULL; //east
	BP.rank_s = BP.by_idx > 0          ? rank - BP.bkx          : MPI_PROC_NULL; //south
	BP.rank_n = BP.by_idx < BP.bky - 1 ? rank + BP.bkx          : MPI_PROC_NULL; //north
	BP.rank_b = BP.bz_idx > 0          ? rank - BP.bkx * BP.bky : MPI_PROC_NULL; //bot
	BP.rank_t = BP.bz_idx < BP.bkz - 1 ? rank + BP.bkx * BP.bky : MPI_PROC_NULL; //top
	
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
