#include <mpi.h>
#include <math.h>
#include "operations.hpp"

//Used for dividing work in apply_stencil3d
void get_block_parameters(int nx, int ny, int nz, MPI_Comm comm, int *nbx, int *nby, int *nbz, int *ixb, int *iyb, int *izb, int* neighbours) {
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	printf("rank: %d, size: %d\n", rank, size);
	
	//Number of blocks
	// EXAMPLE: size=48 -> blocks_x=ceil(3.42)=3; blocks_y=sqrt(16)=4; blocks_z=4 
	int blocks_x = ceil(pow(size, 1.0/3.0)); 
	int blocks_y = ceil(sqrt(size/blocks_x));
	int blocks_z = ceil(size / (blocks_x * blocks_y));

	// block size
	int bx_sz = (nx + blocks_x - 1) / blocks_x;
	int by_sz = (ny + blocks_y - 1) / blocks_y;
	int bz_sz = (nz + blocks_z - 1) / blocks_z;

	// block indices
	int bx_idx = rank % blocks_x;
	int by_idx = (rank / blocks_x) % blocks_y;
	int bz_idx = rank / (blocks_x * blocks_y);

	// Grid points are often not perfectly divisible into blocks, fix this here.
	// x-direction
	int xb_start = bx_idx * bx_sz;
	int xb_end = xb_start + bx_sz - 1;
	if (xb_end >= nx) {
		bx_sz = nx - xb_start;
	}
	// y-direction
	int yb_start = by_idx * by_sz;
	int yb_end = yb_start + by_sz - 1;
	if (yb_end >= ny) {
		by_sz = ny - yb_start;
	}
	// z-direction
	int zb_start = bz_idx * bz_sz;
	int zb_end = zb_start + bz_sz - 1;
	if (zb_end >= nz) {
		bz_sz = nz - zb_start;
	}
	
	//Set outputs for this block
	*nbx = bx_sz;
	*nby = by_sz;
	*nbz = bz_sz;
	*ixb = xb_start;
	*iyb = yb_start;
	*izb = zb_start;

	//Save idx's of our neighbours
	int west  = bx_idx > 0            ? rank - 1                   : MPI_PROC_NULL;
	int east  = bx_idx < blocks_x - 1 ? rank + 1                   : MPI_PROC_NULL;
	int south = by_idx > 0            ? rank - blocks_x            : MPI_PROC_NULL;
	int north = by_idx < blocks_y - 1 ? rank + blocks_x            : MPI_PROC_NULL;
	int bot   = bz_idx > 0            ? rank - blocks_x * blocks_y : MPI_PROC_NULL;
	int top   = bz_idx < blocks_z - 1 ? rank + blocks_x * blocks_y : MPI_PROC_NULL;
	
	neighbours[0] = west;
	neighbours[1] = east;
	neighbours[2] = south;
	neighbours[3] = north;
	neighbours[4] = bot;
	neighbours[5] = top;
}

double dot(long n, double const* x, double const* y, MPI_Comm comm) {
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	double local_dot = 0.0;
	long local_n = n / size;
	long remainder = n % size;
	long start_ix = rank * local_n;
	if (rank < remainder) {
		local_n++;
		start_ix += rank;
	} else {
		start_ix += remainder;
	}

	for (int ix = 0; ix < local_n; ix++) {
		local_dot += x[start_ix + ix] * y[start_ix + ix];
	}

	double global_dot;
	MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);

	return global_dot;
}

//TODO Not yet parallel
void axpby(long n, double a, double const* x, double b, double* y){
	for (int ix=0; ix<n; ix++) {
		y[ix] = a*x[ix]+b*y[ix];
	}
	return;
}

//TODO Not yet parallel
void init(long n, double* x, double const value) {
	for (int ix=0; ix<n; ix++) {
		x[ix] = value;
	}
	return;
}

void apply_stencil3d(stencil3d const* S, double const* u, double* v, MPI_Comm comm) {
	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;

	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int bx_sz, by_sz, bz_sz, bx_start, by_start, bz_start, neighbours[6];
	get_block_parameters(nx, ny, nz, comm, &bx_sz, &by_sz, &bz_sz, &bx_start, &by_start, &bz_start, neighbours);

	printf("block parameters: %d, %d, %d, start ids: %d, %d, %d\n", bx_sz, by_sz, bz_sz, bx_start, by_start, bz_start);
	/*
	double* send_west_buffer  = (double*) malloc(by_sz * bz_sz * sizeof(double));
	double* recv_west_buffer  = (double*) malloc(by_sz * bz_sz * sizeof(double));
	double* send_east_buffer = (double*) malloc(by_sz * bz_sz * sizeof(double));
	double* recv_east_buffer = (double*) malloc(by_sz * bz_sz * sizeof(double));
	
	double* send_south_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	double* recv_south_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	double* send_north_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	double* recv_north_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	
	double* send_bot_buffer  = (double*) malloc(bx_sz * by_sz * sizeof(double));
	double* recv_bot_buffer  = (double*) malloc(bx_sz * by_sz * sizeof(double));
	double* send_top_buffer = (double*) malloc(bx_sz * by_sz * sizeof(double));
	double* recv_top_buffer = (double*) malloc(bx_sz * by_sz * sizeof(double));
	*/
	double send_west_buffer[by_sz*bz_sz] = {0.0};//TODO will these fit on the stack for our grid sizes? 
	double recv_west_buffer[by_sz*bz_sz] = {0.0}; 
	double send_east_buffer[by_sz*bz_sz] = {0.0};  
	double recv_east_buffer[by_sz*bz_sz] = {0.0};  
	
	double send_south_buffer[bx_sz*bz_sz] = {0.0};    
	double recv_south_buffer[bx_sz*bz_sz] = {0.0};    
	double send_north_buffer[bx_sz*bz_sz] = {0.0};    
	double recv_north_buffer[bx_sz*bz_sz] = {0.0};    
	
	double send_bot_buffer[bx_sz*by_sz] = {0.0};   
	double recv_bot_buffer[bx_sz*by_sz] = {0.0};  
	double send_top_buffer[bx_sz*by_sz] = {0.0}; 
	double recv_top_buffer[bx_sz*by_sz] = {0.0}; 
	
	//If we have neighbour in a direction, we communicate the bdry points northh ways, otherwise we set the recv_buffer to zero.
	//TODO loop ordering should be changed.
	MPI_Request requests[12];
	MPI_Status statuses[12];

	//west
	if (neighbours[0] != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < by_sz; iy++) {
			for (int iz = 0; iz < bz_sz; iz++, id++) {
				send_west_buffer[id] = v[S->index_c(bx_start+0, by_start+iy, bz_start+iz)];
			}
		}
		MPI_Isend(send_west_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[0], 0, comm, &requests[0]);
		MPI_Irecv(recv_west_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[0], 1, comm, &requests[1]);
	} else {
		 for (int id=0; id<by_sz*bz_sz; id++) {
			 recv_west_buffer[id]=0.0;
		 }
	}
	//east
	if (neighbours[1] != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < by_sz; iy++) {
			for (int iz = 0; iz < bz_sz; iz++, id++) {
				send_east_buffer[id] = v[S->index_c(bx_start+bz_sz-1, iy+by_start, bz_start+iz)];
			}
		}
		MPI_Isend(send_east_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[1], 1, comm, &requests[2]);
		MPI_Irecv(recv_east_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[1], 0, comm, &requests[3]);
	} else {
		 for (int id=0; id<by_sz*bz_sz; id++) {
			 recv_east_buffer[id]=0.0;
		 }
	}
	//south
	if (neighbours[2] != MPI_PROC_NULL) {
		int id = 0;
		for (int ix = 0; ix < bx_sz; ix++) {
			for (int iz = 0; iz < bz_sz; iz++, id++) {
				send_south_buffer[id] = v[S->index_c(bx_start+ix, by_start+0, bz_start+iz)];
			}
		}
		MPI_Isend(send_south_buffer, bx_sz * bz_sz, MPI_DOUBLE, neighbours[2], 0, comm, &requests[4]);
		MPI_Irecv(recv_south_buffer, bx_sz * bz_sz, MPI_DOUBLE, neighbours[2], 1, comm, &requests[5]);
	} else {
		for (int id=0; id<bx_sz*bz_sz; id++) {
			 recv_south_buffer[id]=0.0;
		}
	}
	//north
	if (neighbours[3] != MPI_PROC_NULL) {
		int id = 0;
		for (int ix = 0; ix < bx_sz; ix++) {
			for (int iz = 0; iz < bz_sz; iz++, id++) {
				send_north_buffer[id] = v[S->index_c(bx_start+ix, by_start+by_sz-1, bz_start+iz)];
			}
		}
		MPI_Isend(send_north_buffer, bx_sz * bz_sz, MPI_DOUBLE, neighbours[3], 1, comm, &requests[6]);
		MPI_Irecv(recv_north_buffer, bx_sz * bz_sz, MPI_DOUBLE, neighbours[3], 0, comm, &requests[7]);
	} else {
		 for (int id=0; id<bx_sz*bz_sz; id++) {
			 recv_north_buffer[id]=0.0;
		 }
	}
	//bot
	if (neighbours[4] != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < by_sz; iy++) {
			for (int ix = 0; ix < bx_sz; ix++, id++) {
				send_bot_buffer[id] = v[S->index_c(bx_start+ix, iy+by_start, bz_start+0)];
			}
		}
		MPI_Isend(send_bot_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[4], 0, comm, &requests[8]);
		MPI_Irecv(recv_bot_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[4], 1, comm, &requests[9]);
	} else {
		 for (int id=0; id<by_sz*bx_sz; id++) {
			 recv_bot_buffer[id]=0.0;
		 }
	}
	//top
	if (neighbours[5] != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < by_sz; iy++) {
			for (int ix = 0; ix < bx_sz; ix++, id++) {
				send_top_buffer[id] = v[S->index_c(bx_start+ix, iy+by_start, bz_start+bz_sz-1)];
			}
		}
		MPI_Isend(send_top_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[5], 1, comm, &requests[10]);
		MPI_Irecv(recv_top_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[5], 0, comm, &requests[11]);
	} else {
		 for (int id=0; id<by_sz*bx_sz; id++) {
			 recv_top_buffer[id]=0.0;
		 }
	}
	
	//For the interior of the block, perform the operations as usual.
	for (int iz=bz_start; iz<bz_start+bz_sz; iz++) {
		for (int iy=by_start; iy<by_start+by_sz; iy++) {
			for (int ix=bx_start; ix<bx_start+bx_sz; ix++) {
				int ew_id = by_sz*iz+iy; 
				int ns_id = bx_sz*ix+iz; 
				int tb_id = by_sz*iy+ix; 
				double accum = S->value_c * u[S->index_c(ix, iy, iz)]; //TODO make branchless
				accum += S->value_b * ((iz==bz_start        ) ? recv_bot_buffer[tb_id]   : u[S->index_b(ix, iy, iz)]);
				accum += S->value_t * ((iz==bz_start+bz_sz-1) ? recv_top_buffer[tb_id]   : u[S->index_t(ix, iy, iz)]);
				accum += S->value_s * ((iy==by_start        ) ? recv_south_buffer[ns_id] : u[S->index_s(ix, iy, iz)]);
				accum += S->value_n * ((iy==by_start+by_sz-1) ? recv_north_buffer[ns_id] : u[S->index_n(ix, iy, iz)]);
				accum += S->value_w * ((ix==bx_start        ) ? recv_west_buffer[ew_id]  : u[S->index_w(ix, iy, iz)]);
				accum += S->value_e * ((ix==bx_start+bx_sz-1) ? recv_east_buffer[ew_id]  : u[S->index_e(ix, iy, iz)]);
				//printf("ix=%d, iy=%d, iz=%d\n", ix, iy, iz);
				//if (iz==bz_start)
				//	printf("%.1f ", recv_bot_buffer[tb_id]);
				//	printf("%.1f ", recv_west_buffer[tb_id]);
				v[S->index_c(ix, iy, iz)] = accum;
			}
		}
	}
	return;
}

///Test the 
stencil3d laplace3d_stencil(int nx, int ny, int nz){
	if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
	stencil3d L;
	L.nx=nx; L.ny=ny; L.nz=nz;
	double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
	L.value_c = 2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
	L.value_n =  -1.0/(dy*dy);
	L.value_e = -1.0/(dx*dx);
	L.value_s =  -1.0/(dy*dy);
	L.value_w = -1.0/(dx*dx);
	L.value_t = -1.0/(dz*dz);
	L.value_b = -1.0/(dz*dz);
	return L;
}

void print_array(double arr[], int size) {
	printf("[ ");
	for (int i = 0; i < size; i++) {
		printf("%.2f ", arr[i]);
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	
	int nx = 10;
	int ny = 10;
	int nz = 10;
	double u[nx*ny*nz]={5.0, 5.0, 5.0};
	double v[nx*ny*nz]={1.0, 0.0, 2.0};
	stencil3d L = laplace3d_stencil(nx, ny, nz);
	apply_stencil3d(&L, u, v, comm);
	print_array(v, 5);
	double dot_res = dot(nx*ny*nz, u, v, comm);
	if (rank==0)
		printf("Dot result %f\n", dot_res);
	
	MPI_Finalize();
	return 0;
}
