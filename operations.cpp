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
#if defined(STENCIL_ONE_SIDED)
	Timer t("apply_stencil3d one sided");
#elif defined(STENCIL_GLOBAL_COMM)
	Timer t("apply_stencil3d global comm");
#elif defined(STENCIL_MPI_CART)
	Timer t("apply_stencil3d mpi cart");
#endif
	int sz_ew = BP->by_sz*BP->bz_sz;
	int sz_ns = BP->bx_sz*BP->bz_sz;
	int sz_tb = BP->bx_sz*BP->by_sz;

	//west
	if (BP->rank_w != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int iy = 0; iy < BP->by_sz; iy++, id++) {
				BP->send_west_buffer[id] = u[S->index_c(0, iy, iz)];
			}
		}
	}
	//east
	if (BP->rank_e != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int iy = 0; iy < BP->by_sz; iy++, id++) {
				BP->send_east_buffer[id] = u[S->index_c(BP->bx_sz-1, iy, iz)];
			}
		}
	}
	//south
	if (BP->rank_s != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				BP->send_south_buffer[id] = u[S->index_c(ix, 0, iz)];
			}
		}
	}
	//north
	if (BP->rank_n != MPI_PROC_NULL) {
		int id = 0;
		for (int iz = 0; iz < BP->bz_sz; iz++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				BP->send_north_buffer[id] = u[S->index_c(ix, BP->by_sz-1, iz)];
			}
		}
	}
	//bot
	if (BP->rank_b != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < BP->by_sz; iy++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				BP->send_bot_buffer[id] = u[S->index_c(ix, iy, 0)];
			}
		}
	}
	//top
	if (BP->rank_t != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < BP->by_sz; iy++) {
			for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
				BP->send_top_buffer[id] = u[S->index_c(ix, iy, BP->bz_sz-1)];
			}
		}
	}
	///////////////// COMMUNICATION START /////////////////////
	#if defined(STENCIL_ONE_SIDED)
		// Wait until neigbours' buffers are filled.
		MPI_Win_fence(0, BP->win_e);
		MPI_Win_fence(0, BP->win_w);
		MPI_Win_fence(0, BP->win_n);
		MPI_Win_fence(0, BP->win_s);
		MPI_Win_fence(0, BP->win_t);
		MPI_Win_fence(0, BP->win_b);
		// Read from neighbours' buffers.
		if (BP->rank_e != MPI_PROC_NULL) {
			MPI_Get(BP->recv_east_buffer, sz_ew, MPI_DOUBLE, BP->rank_e, 0, sz_ew, MPI_DOUBLE, BP->win_e);
		}
		if (BP->rank_w != MPI_PROC_NULL) {
			MPI_Get(BP->recv_west_buffer, sz_ew, MPI_DOUBLE, BP->rank_w, 0, sz_ew, MPI_DOUBLE, BP->win_w);
		}
		if (BP->rank_n != MPI_PROC_NULL) {
			MPI_Get(BP->recv_north_buffer, sz_ns, MPI_DOUBLE, BP->rank_n, 0, sz_ns, MPI_DOUBLE, BP->win_n);
		}
		if (BP->rank_s != MPI_PROC_NULL) {
			MPI_Get(BP->recv_south_buffer, sz_ns, MPI_DOUBLE, BP->rank_s, 0, sz_ns, MPI_DOUBLE, BP->win_s);
		}
		if (BP->rank_t != MPI_PROC_NULL) {
			MPI_Get(BP->recv_top_buffer, sz_tb, MPI_DOUBLE, BP->rank_t, 0, sz_tb, MPI_DOUBLE, BP->win_t);
		}
		if (BP->rank_b != MPI_PROC_NULL) {
			MPI_Get(BP->recv_bot_buffer, sz_tb, MPI_DOUBLE, BP->rank_b, 0, sz_tb, MPI_DOUBLE, BP->win_b);
		}
		// Wait until we are done reading neighbours' buffers.
		MPI_Win_fence(0, BP->win_e);
		MPI_Win_fence(0, BP->win_w);
		MPI_Win_fence(0, BP->win_n);
		MPI_Win_fence(0, BP->win_s);
		MPI_Win_fence(0, BP->win_t);
		MPI_Win_fence(0, BP->win_b);
	#elif defined(STENCIL_GLOBAL_COMM) || defined(STENCIL_MPI_CART)
		MPI_Request requests[12];
		MPI_Status statuses[12];
		//west
		if (BP->rank_w != MPI_PROC_NULL) {
			MPI_Isend(BP->send_west_buffer, sz_ew, MPI_DOUBLE, BP->rank_w, 0, BP->comm, &requests[0]);
			MPI_Irecv(BP->recv_west_buffer, sz_ew, MPI_DOUBLE, BP->rank_w, 1, BP->comm, &requests[1]);
		}
		//east
		if (BP->rank_e != MPI_PROC_NULL) {
			MPI_Isend(BP->send_east_buffer, sz_ew, MPI_DOUBLE, BP->rank_e, 1, BP->comm, &requests[2]);
			MPI_Irecv(BP->recv_east_buffer, sz_ew, MPI_DOUBLE, BP->rank_e, 0, BP->comm, &requests[3]);
		}
		//south
		if (BP->rank_s != MPI_PROC_NULL) {
			MPI_Isend(BP->send_south_buffer, sz_ns, MPI_DOUBLE, BP->rank_s, 0, BP->comm, &requests[4]);
			MPI_Irecv(BP->recv_south_buffer, sz_ns, MPI_DOUBLE, BP->rank_s, 1, BP->comm, &requests[5]);
		}
		//north
		if (BP->rank_n != MPI_PROC_NULL) {
			MPI_Isend(BP->send_north_buffer, sz_ns, MPI_DOUBLE, BP->rank_n, 1, BP->comm, &requests[6]);
			MPI_Irecv(BP->recv_north_buffer, sz_ns, MPI_DOUBLE, BP->rank_n, 0, BP->comm, &requests[7]);
		}
		//bot
		if (BP->rank_b != MPI_PROC_NULL) {
			MPI_Isend(BP->send_bot_buffer, sz_tb, MPI_DOUBLE, BP->rank_b, 0, BP->comm, &requests[8]);
			MPI_Irecv(BP->recv_bot_buffer, sz_tb, MPI_DOUBLE, BP->rank_b, 1, BP->comm, &requests[9]);
		}
		//top
		if (BP->rank_t != MPI_PROC_NULL) {
			MPI_Isend(BP->send_top_buffer, sz_tb, MPI_DOUBLE, BP->rank_t, 1, BP->comm, &requests[10]);
			MPI_Irecv(BP->recv_top_buffer, sz_tb, MPI_DOUBLE, BP->rank_t, 0, BP->comm, &requests[11]);
		}
		
		//Wait for recv buffers to be filled
		int neighbours[] = {BP->rank_w, BP->rank_e, BP->rank_s, BP->rank_n, BP->rank_b, BP->rank_t};
		for (int id=0; id<6; id++)
			if (neighbours[id] != MPI_PROC_NULL)
				MPI_Wait(&requests[2*id+1], MPI_STATUS_IGNORE);
	#else
		#error "No stencil version specified"
	#endif
	//Use buffers for edge points, apply the stencil.
	for (int iz=0; iz<BP->bz_sz; iz++) {
		for (int iy=0; iy<BP->by_sz; iy++) {
			for (int ix=0; ix<BP->bx_sz; ix++) {
				int ew_id = BP->by_sz*iz+iy; 
				int ns_id = BP->bx_sz*iz+ix; 
				int tb_id = BP->bx_sz*iy+ix; 
				double accum = S->value_c * u[S->index_c(ix, iy, iz)];
				accum += S->value_w * ((ix==0            ) ? BP->recv_west_buffer[ew_id]  : u[S->index_w(ix, iy, iz)]);
				accum += S->value_e * ((ix==0+BP->bx_sz-1) ? BP->recv_east_buffer[ew_id]  : u[S->index_e(ix, iy, iz)]);
				accum += S->value_s * ((iy==0            ) ? BP->recv_south_buffer[ns_id] : u[S->index_s(ix, iy, iz)]);
				accum += S->value_n * ((iy==0+BP->by_sz-1) ? BP->recv_north_buffer[ns_id] : u[S->index_n(ix, iy, iz)]);
				accum += S->value_b * ((iz==0            ) ? BP->recv_bot_buffer[tb_id]   : u[S->index_b(ix, iy, iz)]);
				accum += S->value_t * ((iz==0+BP->bz_sz-1) ? BP->recv_top_buffer[tb_id]   : u[S->index_t(ix, iy, iz)]);
				v[S->index_c(ix, iy, iz)] = accum;
			}
		}
	}
	return;
}

block_params create_blocks(int const nx, int const ny, int const nz) {
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
	#ifdef STENCIL_MPI_CART
		int dim[3] = {BP.bkz, BP.bky, BP.bkx}; //Dimensions of the cartesian grid
		int periodical[3] = {0, 0, 0}; //Whether eacht dimention is periodic or not (not in our case)
		MPI_Comm cart_comm; //new communicator to store the cartesian communicator
		MPI_Cart_create(MPI_COMM_WORLD, 3, dim, periodical, 0, &cart_comm); //Create the cartesian topolgy and store it in a new MPI communicator
		BP.comm = cart_comm;
	#else
		BP.comm = MPI_COMM_WORLD;
	#endif

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
	#ifdef STENCIL_MPI_CART
		MPI_Cart_shift(cart_comm, 2, -1, &rank, &BP.rank_w);
		MPI_Cart_shift(cart_comm, 2, 1, &rank, &BP.rank_e);
		MPI_Cart_shift(cart_comm, 1, -1, &rank, &BP.rank_s);
		MPI_Cart_shift(cart_comm, 1, 1, &rank, &BP.rank_n);
		MPI_Cart_shift(cart_comm, 0, -1, &rank, &BP.rank_b);
		MPI_Cart_shift(cart_comm, 0, 1, &rank, &BP.rank_t);
	#else
		BP.rank_w = BP.bx_idx > 0          ? rank - 1               : MPI_PROC_NULL; //west
		BP.rank_e = BP.bx_idx < BP.bkx - 1 ? rank + 1               : MPI_PROC_NULL; //east
		BP.rank_s = BP.by_idx > 0          ? rank - BP.bkx          : MPI_PROC_NULL; //south
		BP.rank_n = BP.by_idx < BP.bky - 1 ? rank + BP.bkx          : MPI_PROC_NULL; //north
		BP.rank_b = BP.bz_idx > 0          ? rank - BP.bkx * BP.bky : MPI_PROC_NULL; //bot
		BP.rank_t = BP.bz_idx < BP.bkz - 1 ? rank + BP.bkx * BP.bky : MPI_PROC_NULL; //top
	#endif

	BP.send_west_buffer = (double*) calloc(BP.by_sz*BP.bz_sz, sizeof(double));
	BP.recv_west_buffer = (double*) calloc(BP.by_sz*BP.bz_sz, sizeof(double));
	BP.send_east_buffer = (double*) calloc(BP.by_sz*BP.bz_sz, sizeof(double));
	BP.recv_east_buffer = (double*) calloc(BP.by_sz*BP.bz_sz, sizeof(double));
	
	BP.send_south_buffer = (double*) calloc(BP.bx_sz*BP.bz_sz, sizeof(double));
	BP.recv_south_buffer = (double*) calloc(BP.bx_sz*BP.bz_sz, sizeof(double));
	BP.send_north_buffer = (double*) calloc(BP.bx_sz*BP.bz_sz, sizeof(double));
	BP.recv_north_buffer = (double*) calloc(BP.bx_sz*BP.bz_sz, sizeof(double));
	
	BP.send_bot_buffer = (double*) calloc(BP.bx_sz*BP.by_sz, sizeof(double));
	BP.recv_bot_buffer = (double*) calloc(BP.bx_sz*BP.by_sz, sizeof(double));
	BP.send_top_buffer = (double*) calloc(BP.bx_sz*BP.by_sz, sizeof(double));
	BP.recv_top_buffer = (double*) calloc(BP.bx_sz*BP.by_sz, sizeof(double));
	#ifdef STENCIL_ONE_SIDED
		int sz_ew = BP.by_sz*BP.bz_sz;
		int sz_ns = BP.bx_sz*BP.bz_sz;
		int sz_tb = BP.bx_sz*BP.by_sz;
		MPI_Win win_e, win_w, win_s, win_n, win_t, win_b;
		MPI_Win_create(BP.send_east_buffer, sz_ew * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_w);
		MPI_Win_create(BP.send_west_buffer, sz_ew * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_e);
		MPI_Win_create(BP.send_north_buffer, sz_ns * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_s);
		MPI_Win_create(BP.send_south_buffer, sz_ns * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_n);
		MPI_Win_create(BP.send_top_buffer, sz_tb * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);
		MPI_Win_create(BP.send_bot_buffer, sz_tb * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_t);
		BP.win_e = win_e;
		BP.win_w = win_w;
		BP.win_n = win_n;
		BP.win_s = win_s;
		BP.win_t = win_t;
		BP.win_b = win_b;
	#endif

	return BP;
}


// polynomial preconditioning
void polynomial(stencil3d* op, block_params const* BP, double* x, double* t1, double* t2, int n, int order)
{
  // initialization
  apply_stencil3d(op, BP, x, t1);
  axpby(n, 1.0, x, -1.0, t1);  
  axpby(n, 1.0, t1, 1.0, x);

  for (int i=1; i<order; i++)
  {
     apply_stencil3d(op, BP, t1, t2);
     axpby(n, -1.0, t2, 1.0, t1); 
     axpby(n, 1.0, t1, 1.0, x);
  }
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
void arnoldi(int const k, double* Q, double* h, stencil3d* op, block_params const* BP, int order) {
  Timer timerArnoldi("5. Arnoldi function");
  int n = op->nx * op->ny * op->nz;
  double *x1 = new double[n];
  double *x2 = new double[n];

  apply_stencil3d(op, BP, Q+k*n, Q+(k+1)*n);
  polynomial(op, BP, Q+(k+1)*n, x1, x2, n, order);
  
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



