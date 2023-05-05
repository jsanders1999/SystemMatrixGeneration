#include <mpi.h>
#include <math.h>
#include "operations.hpp"
#include <iostream>
#include "timer.hpp"

void print_array(const double* arr, int const size) {
	printf("[ ");
	for (int i = 0; i < size; i++) {
		printf("%.2f ", arr[i]);
	}
	printf("]\n");
}

void init(int const local_n, double* x, double const value) {
	//Timer timerinit("1. Init operation");
	for (int id=0; id<local_n; id++) {
		x[id] = value;
	}
	return;
}

double dot(int const local_n, double const* x, double const* y) {
	Timer timerdot("1. dot operation");
	double local_dot = 0.0;
	
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	for (int id = 0; id < local_n; id++) {
		local_dot += x[id] * y[id];
	}

	double global_dot;
	MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return global_dot;
}

void axpby(int const local_n, double const a, double const* x, double const b, double* y){
	Timer timeraxpby("2. axpby operation");
	for (int id=0; id<local_n; id++) {
		y[id] = a*x[id]+b*y[id];
	}
	return;
}

void axpby(int const local_n, double const a, double const* x, 
	      double const b, double const* y, double const c, double* z){
        Timer timeraxpby("2. axpby operation");
	for (int id=0; id<local_n; id++) {
		z[id] = a*x[id]+b*y[id]+c*z[id];
	}
	return;
}

// y = a*x+y
void axpy(int const local_n, double const a, double const* x, double* y){
	Timer timeraxpby("2. axpby operation");
	for (int id=0; id<local_n; id++) {
		y[id] += a*x[id];
	}
	return;
}

void apply_stencil3d(stencil3d const* S, block_params const* BP, double const* u, double* v) {
	Timer timerstencil("3. Stencil operation");
	
	//TODO will these fit on the stack for our grid sizes? 
	double send_west_buffer[BP->by_sz*BP->bz_sz];
	double recv_west_buffer[BP->by_sz*BP->bz_sz]; 
	double send_east_buffer[BP->by_sz*BP->bz_sz];  
	double recv_east_buffer[BP->by_sz*BP->bz_sz];  
	
	double send_south_buffer[BP->bx_sz*BP->bz_sz];    
	double recv_south_buffer[BP->bx_sz*BP->bz_sz];    
	double send_north_buffer[BP->bx_sz*BP->bz_sz];    
	double recv_north_buffer[BP->bx_sz*BP->bz_sz];    
	
	double send_bot_buffer[BP->bx_sz*BP->by_sz];   
	double recv_bot_buffer[BP->bx_sz*BP->by_sz];  
	double send_top_buffer[BP->bx_sz*BP->by_sz]; 
	double recv_top_buffer[BP->bx_sz*BP->by_sz]; 
	
	//If we have neighbour in a direction, we communicate the bdry points both ways, otherwise we set the recv_buffer to zero.
	MPI_Request requests[12];
	MPI_Status statuses[12];
	MPI_Comm comm = MPI_COMM_WORLD;//BP->comm; gives a segmentation fault for some reason? @elias help
	//west
	if (BP->rank_w != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int iy = 0; iy < BP->by_sz; iy++, id++) {
				send_west_buffer[id] = u[S->index_c(0, iy, iz)];
			}
		}
		MPI_Isend(send_west_buffer, BP->by_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_w, 0, comm, &requests[0]);
		MPI_Irecv(recv_west_buffer, BP->by_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_w, 1, comm, &requests[1]);
	} else {
		 for (int id=0; id<BP->by_sz*BP->bz_sz; id++) {
			 recv_west_buffer[id]=0.0;
		 }
	}
	//east
	if (BP->rank_e != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int iy = 0; iy < BP->by_sz; iy++, id++) {
				send_east_buffer[id] = u[S->index_c(BP->bx_sz-1, iy, iz)];
			}
		}
		MPI_Isend(send_east_buffer, BP->by_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_e, 1, comm, &requests[2]);
		MPI_Irecv(recv_east_buffer, BP->by_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_e, 0, comm, &requests[3]);
	} else {
		 for (int id=0; id<BP->by_sz*BP->bz_sz; id++) {
			 recv_east_buffer[id]=0.0;
		 }
	}
	//south
	if (BP->rank_s != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				send_south_buffer[id] = u[S->index_c(ix, 0, iz)];
			}
		}
		MPI_Isend(send_south_buffer, BP->bx_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_s, 0, comm, &requests[4]);
		MPI_Irecv(recv_south_buffer, BP->bx_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_s, 1, comm, &requests[5]);
	} else {
		for (int id=0; id<BP->bx_sz*BP->bz_sz; id++) {
			 recv_south_buffer[id]=0.0;
		}
	}
	//north
	if (BP->rank_n != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				send_north_buffer[id] = u[S->index_c(ix, BP->by_sz-1, iz)];
			}
		}
		MPI_Isend(send_north_buffer, BP->bx_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_n, 1, comm, &requests[6]);
		MPI_Irecv(recv_north_buffer, BP->bx_sz * BP->bz_sz, MPI_DOUBLE, BP->rank_n, 0, comm, &requests[7]);
	} else {
		 for (int id=0; id<BP->bx_sz*BP->bz_sz; id++) {
			 recv_north_buffer[id]=0.0;
		 }
	}
	//bot
	if (BP->rank_b != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < BP->by_sz; iy++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				send_bot_buffer[id] = u[S->index_c(ix, iy, 0)];
			}
		}
		MPI_Isend(send_bot_buffer, BP->bx_sz * BP->by_sz, MPI_DOUBLE, BP->rank_b, 0, comm, &requests[8]);
		MPI_Irecv(recv_bot_buffer, BP->bx_sz * BP->by_sz, MPI_DOUBLE, BP->rank_b, 1, comm, &requests[9]);
	} else {
		 for (int id=0; id<BP->by_sz*BP->bx_sz; id++) {
			 recv_bot_buffer[id]=0.0;
		 }
	}
	//top
	if (BP->rank_t != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < BP->by_sz; iy++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				send_top_buffer[id] = u[S->index_c(ix, iy, BP->bz_sz-1)];
			}
		}
		MPI_Isend(send_top_buffer, BP->bx_sz * BP->by_sz, MPI_DOUBLE, BP->rank_t, 1, comm, &requests[10]);
		MPI_Irecv(recv_top_buffer, BP->bx_sz * BP->by_sz, MPI_DOUBLE, BP->rank_t, 0, comm, &requests[11]);
	} else {
		 for (int id=0; id<BP->by_sz*BP->bx_sz; id++) {
			 recv_top_buffer[id]=0.0;
		 }
	}
	
	//Wait for recv buffers to be filled
	int neighbours[] = {BP->rank_w, BP->rank_e, BP->rank_s, BP->rank_n, BP->rank_b, BP->rank_t};
	for (int id=0; id<6; id++)
		if (neighbours[id] != MPI_PROC_NULL)
			MPI_Wait(&requests[2*id+1], MPI_STATUS_IGNORE);
	
	//Use buffers for edge points, apply the stencil.
	for (int iz=0; iz<BP->bz_sz; iz++) {
		for (int iy=0; iy<BP->by_sz; iy++) {
			for (int ix=0; ix<BP->bx_sz; ix++) {
				int ew_id = BP->by_sz*iz+iy; 
				int ns_id = BP->bx_sz*iz+ix; 
				int tb_id = BP->bx_sz*iy+ix; 
				double accum = S->value_c * u[S->index_c(ix, iy, iz)];
				accum += S->value_w * ((ix==0            ) ? recv_west_buffer[ew_id]  : u[S->index_w(ix, iy, iz)]);
				accum += S->value_e * ((ix==0+BP->bx_sz-1) ? recv_east_buffer[ew_id]  : u[S->index_e(ix, iy, iz)]);
				accum += S->value_s * ((iy==0            ) ? recv_south_buffer[ns_id] : u[S->index_s(ix, iy, iz)]);
				accum += S->value_n * ((iy==0+BP->by_sz-1) ? recv_north_buffer[ns_id] : u[S->index_n(ix, iy, iz)]);
				accum += S->value_b * ((iz==0            ) ? recv_bot_buffer[tb_id]   : u[S->index_b(ix, iy, iz)]);
				accum += S->value_t * ((iz==0+BP->bz_sz-1) ? recv_top_buffer[tb_id]   : u[S->index_t(ix, iy, iz)]);
				v[S->index_c(ix, iy, iz)] = accum;
			}
		}
	}
	return;
}

block_params create_blocks(int const nx, int const ny, int const nz) {
	//Timer timerblock("4. Block creation operation");
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	block_params BP;
	// Choose number of blocks in x y and z directions
	BP.bkx = ceil(pow(size, 1.0/3.0));
	BP.bky = ceil(sqrt(size/BP.bkx));
	BP.bkz = ceil(size / (BP.bkx * BP.bky));

	if (rank==0 && size!=BP.bkx*BP.bky*BP.bkz) {
		printf("ERROR: Invalid number of MPI processes (%d), can't create blocks. Try using %d instead.\n", size, BP.bkx*BP.bky*BP.bkz);
		exit(1);
	}
	// Calculate the sizes (ignoring divisibility)
	BP.bx_sz = (nx + BP.bkx - 1) / BP.bkx;
	BP.by_sz = (ny + BP.bky - 1) / BP.bky;
	BP.bz_sz = (nz + BP.bkz - 1) / BP.bkz;
	// Calculate index in each direction
	BP.bx_idx = rank % BP.bkx;
	BP.by_idx = (rank / BP.bkx) % BP.bky;
	BP.bz_idx = rank / (BP.bkx * BP.bky);
	// Save start index (gridpoint) in x y and z directions
	BP.bx_start = BP.bx_idx*BP.bx_sz;
	BP.by_start = BP.by_idx*BP.by_sz;
	BP.bz_start = BP.bz_idx*BP.bz_sz;
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
	return BP;
}

block_params create_blocks_cart(int const nx, int const ny, int const nz) {
	//Timer timerblock("4. Block creation operation");
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	block_params BP;
	// Choose number of blocks in x y and z directions
	BP.bkx = ceil(pow(size, 1.0/3.0));
	BP.bky = ceil(sqrt(size/BP.bkx));
	BP.bkz = ceil(size / (BP.bkx * BP.bky));

	if (rank==0 && size!=BP.bkx*BP.bky*BP.bkz) {
		printf("ERROR: Invalid number of MPI processes (%d), can't create blocks. Try using %d instead.\n", size, BP.bkx*BP.bky*BP.bkz);
		exit(1);
	}

	int dim[3] = {BP.bkx, BP.bky, BP.bkz }; //Dimensions of the cartesian grid
	int periodical[3] = {0, 0, 0}; //Whether eacht dimention is periodic or not (not in our case)
	int reorder = 1; // Whether MPI is allowed to reorder processes to speed up computation
	MPI_Comm cart_comm; //new communicator to store the cartesian communicator

	MPI_Cart_create(MPI_COMM_WORLD, 3, dim, periodical, reorder, &cart_comm); //Create the cartesian topolgy and store it in a new MPI communicator
	MPI_Comm_rank(cart_comm, &rank); //get the rank into the new communicator
	int coord[3]; //The catesian index of the current process
	MPI_Cart_coords(cart_comm, rank, 3, coord);
	for (int p=0; p<size; p++){
      if (rank==p)std::cout << "Processor " << p << " coordinates are ("<< coord[0] << ", "<< coord[1] <<", "<< coord[2] << ")"<<std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
	}

	// Calculate the sizes (ignoring divisibility)
	BP.bx_sz = (nx + BP.bkx - 1) / BP.bkx;
	BP.by_sz = (ny + BP.bky - 1) / BP.bky;
	BP.bz_sz = (nz + BP.bkz - 1) / BP.bkz;
	// Calculate index in each direction
	BP.bx_idx = rank % BP.bkx;
	BP.by_idx = (rank / BP.bkx) % BP.bky;
	BP.bz_idx = rank / (BP.bkx * BP.bky);
	// Save start index (gridpoint) in x y and z directions
	BP.bx_start = BP.bx_idx*BP.bx_sz;
	BP.by_start = BP.by_idx*BP.by_sz;
	BP.bz_start = BP.bz_idx*BP.bz_sz;
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
	MPI_Cart_shift(cart_comm, 0, -1, &rank, &BP.rank_w);
	MPI_Cart_shift(cart_comm, 0, 1, &rank, &BP.rank_e);
	MPI_Cart_shift(cart_comm, 1, -1, &rank, &BP.rank_s);
	MPI_Cart_shift(cart_comm, 1, 1, &rank, &BP.rank_n);
	MPI_Cart_shift(cart_comm, 2, -1, &rank, &BP.rank_b);
	MPI_Cart_shift(cart_comm, 2, 1, &rank, &BP.rank_t);
	BP.comm = cart_comm;
	return BP;
}





// apply given rotation
void given_rotation(int const k, double* h, double* cs, double* sn){
  Timer timergivens("4. Givens rotation");

  double temp, t, cs_k, sn_k;
  for (int i=0; i<k; i++)
  {
     temp = cs[i] * h[i] + sn[i] * h[i+1];
     h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1];
     h[i] = temp;
  }
  
  // update the next sin cos values for rotation
  t = std::sqrt( h[k]*h[k] + h[k+1]*h[k+1] );
  cs[k] = h[k]/t;
  sn[k] = h[k+1]/t;

  // eliminate H(i+1,i)
  h[k] = cs[k]*h[k] + sn[k]*h[k+1];
  h[k+1] = 0.0;

  return;
}

// Arnoldi function (without preconditioning)
void arnoldi(int const k, double* Q, double* h, stencil3d const* op, block_params const* BP) {
  Timer timerArnoldi("5. Arnoldi function");
  int n = op->nx * op->ny * op->nz;
  apply_stencil3d(op, BP, Q+k*n, Q+(k+1)*n);
	
  //#pragma omp parallel for
  for (int i=0; i<=k; i++)
  {
    h[i] = dot(n, Q+(k+1)*n, Q+i*n);
    axpby(n, -h[i], Q+i*n, 1.0, Q+(k+1)*n);
  }

  h[k+1] = std::sqrt(dot(n, Q+(k+1)*n, Q+(k+1)*n));
  for (int i=0; i<n; i++)
    Q[(k+1)*n+i] = Q[(k+1)*n+i] / h[k+1];
 
 return; 
}

// Arnoldi function (with polynomial preconditioning)
void arnoldi(int const k, double* Q, double* h, stencil3d const* op, block_params const* BP, int verbose) {
  Timer timerArnoldi("5. Arnoldi function");
  int n = op->nx * op->ny * op->nz;
  double *x1 = new double[n];
  double *x2 = new double[n];

  apply_stencil3d(op, BP, Q+k*n, Q+(k+1)*n);
  apply_stencil3d(op, BP, Q+(k+1)*n, x1);
  apply_stencil3d(op, BP,x1, x2);
  axpby(n, -3.0, x1, 1.0, x2, 3.0, Q+(k+1)*n);

  //#pragma omp parallel for
  for (int i=0; i<=k; i++)
  {
    h[i] = dot(n, Q+(k+1)*n, Q+i*n);
    axpby(n, -h[i], Q+i*n, 1.0, Q+(k+1)*n);
  }

  h[k+1] = std::sqrt(dot(n, Q+(k+1)*n, Q+(k+1)*n));
  for (int i=0; i<n; i++)
    Q[(k+1)*n+i] = Q[(k+1)*n+i] / h[k+1];
 
 delete [] x1;
 delete [] x2;

 return; 
}
