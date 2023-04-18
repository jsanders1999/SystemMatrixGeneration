#include <mpi.h>
#include <math.h>
#include "operations.hpp"

void print_array(const double* arr, int size) {
	printf("[ ");
	for (int i = 0; i < size; i++) {
		printf("%.2f ", arr[i]);
	}
	printf("]\n");
}

void init(int const local_n, double* x, double const value) {
	for (int id=0; id<local_n; id++) {
		x[id] = value;
	}
	return;
}

double dot(int const local_n, double const* x, double const* y) {
	double local_dot = 0.0;

	for (int id = 0; id < local_n; id++) {
		local_dot += x[id] * y[id];
	}

	double global_dot;
	MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return global_dot;
}

void axpby(int const local_n, double a, double const* x, double b, double* y){
	for (int id=0; id<local_n; id++) {
		y[id] = a*x[id]+b*y[id];
	}
	return;
}

void apply_stencil3d(stencil3d const* S, block_params const* BP, double const* u, double* v) {
	
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
	MPI_Comm comm = MPI_COMM_WORLD;
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
				int ew_id = BP->bz_sz*iz+iy; 
				int ns_id = BP->bz_sz*iz+ix; 
				int tb_id = BP->by_sz*iy+ix; 
				//printf("%d %d %d, %d %d %d\n", ix, iy, iz, BP->bz_sz, BP->by_sz, BP->bz_sz);
				double accum = S->value_c * u[S->index_c(ix, iy, iz)]; //TODO make branchless
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

block_params create_blocks(int nx, int ny, int nz) {
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
	return BP;
}
