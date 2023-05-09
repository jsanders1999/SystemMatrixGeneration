#include "gtest_mpi.hpp"
#include "operations.hpp"
#include <iostream>
#include <mpi.h>
#include <math.h>

TEST(operations, dot) {
	int nx=3, ny=10, nz=10;
	int n = nx*ny*nz;
	block_params BP = create_blocks(nx, ny, nz);
	int loc_n = BP.bx_sz*BP.by_sz*BP.bz_sz;
	double x[loc_n];
	double y[loc_n];
	for (int i = 0; i < loc_n; i++) {
		x[i] = (double)i+1.0;
		y[i] = 1.0 / (double)(i+1);
	}
	
	double dot_result = dot(loc_n, x, y); //TODO what is this linking error??
	
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
	block_params BP = create_blocks(nx, ny, nz);
	int n_loc = BP.bx_sz*BP.by_sz*BP.bz_sz;
	double x[n_loc];
	for (int i=0; i<n_loc; i++) x[i]=double(i+1);


	double val=42.0;
	init(n_loc, x, val);

	double err=0.0;
	for (int i=0; i<n_loc; i++) err = std::max(err, std::abs(x[i]-val));

	EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}

TEST(operations, stencil3d_symmetric){
	const int nx=6, ny=7, nz=4;
	const int n=nx*ny*nz;

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

	int n_loc = S.nx*S.ny*S.nz;
	double* e=new double[n];
	for (int i=0; i<n_loc; i++) e[i]=0.0;
	double* A=new double[n_loc*(n_loc+1)];
	for (int ix=0; ix<n_loc*(n_loc+1); ix++) {
		A[ix]=0.0;
	}
	int p_sz;
	for (int p=0; p<size; p++) {	
		if (rank==p) p_sz = n_loc;
		MPI_Bcast(&p_sz, 1, MPI_INT, p, MPI_COMM_WORLD);
		for (int ix=0; ix<p_sz; ix++){
			if (rank==p) e[ix]=1.0;
			if (rank==p) {
				apply_stencil3d(&S, &BP, e, A+ix*n_loc);
			}else {
				apply_stencil3d(&S, &BP, e, A+n_loc*n_loc);
			}
			e[ix]=0.0;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		int wrong_entries=0;
		for (int i=0; i<n_loc; i++) {
			for (int j=0; j<n_loc; j++)
			{
				if (A[i*n_loc+j]!=A[j*n_loc+i]) wrong_entries++;
			}
		}
		EXPECT_EQ(0, wrong_entries);

		if (wrong_entries) {
			std::cout << "Your matrix (computed on a 3x3x3 grid by apply_stencil(I)) is ..."<<std::endl;
			for (int j=0; j<n_loc; j++){
				for (int i=0; i<n_loc; i++){
					std::cout << A[i+j*n_loc] << " ";
				}
				std::cout << std::endl;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
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


/*TEST(operations, arnoldi)
{
  int k = 0;
  int n, nx=2, ny=2, nz=2;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  block_params BP = create_blocks(nx, ny, nz);
  stencil3d S;
  S.nx=BP.bx_sz; S.ny=BP.by_sz; S.nz=BP.bz_sz;
  n = S.nx * S.ny * S.nz;
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

  delete [] Q;
  delete [] H;
}*/




