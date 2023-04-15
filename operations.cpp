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
	int left   = bx_idx > 0            ? rank - 1                   : MPI_PROC_NULL;
	int right  = bx_idx < blocks_x - 1 ? rank + 1                   : MPI_PROC_NULL;
	int bottom = by_idx > 0            ? rank - blocks_x            : MPI_PROC_NULL;
	int top    = by_idx < blocks_y - 1 ? rank + blocks_x            : MPI_PROC_NULL;
	int back   = bz_idx > 0            ? rank - blocks_x * blocks_y : MPI_PROC_NULL;
	int front  = bz_idx < blocks_z - 1 ? rank + blocks_x * blocks_y : MPI_PROC_NULL;
	
	neighbours[0] = left;
	neighbours[1] = right;
	neighbours[2] = bottom;
	neighbours[3] = top;
	neighbours[4] = back;
	neighbours[5] = front;
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
	double* send_left_buffer  = (double*) malloc(by_sz * bz_sz * sizeof(double));
	double* recv_left_buffer  = (double*) malloc(by_sz * bz_sz * sizeof(double));
	double* send_right_buffer = (double*) malloc(by_sz * bz_sz * sizeof(double));
	double* recv_right_buffer = (double*) malloc(by_sz * bz_sz * sizeof(double));
	
	double* send_top_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	double* recv_top_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	double* send_bot_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	double* recv_bot_buffer   = (double*) malloc(bx_sz * bz_sz * sizeof(double));
	
	double* send_back_buffer  = (double*) malloc(bx_sz * by_sz * sizeof(double));
	double* recv_back_buffer  = (double*) malloc(bx_sz * by_sz * sizeof(double));
	double* send_front_buffer = (double*) malloc(bx_sz * by_sz * sizeof(double));
	double* recv_front_buffer = (double*) malloc(bx_sz * by_sz * sizeof(double));
	*/
	double send_left_buffer[by_sz*bz_sz];//TODO will these fit on the stack for our grid sizes? 
	double recv_left_buffer[by_sz*bz_sz]; 
	double send_right_buffer[by_sz*bz_sz];  
	double recv_right_buffer[by_sz*bz_sz];  
	
	double send_top_buffer[bx_sz*bz_sz];    
	double recv_top_buffer[bx_sz*bz_sz];    
	double send_bot_buffer[bx_sz*bz_sz];    
	double recv_bot_buffer[bx_sz*bz_sz];    
	
	double send_back_buffer[bx_sz*by_sz];   
	double recv_back_buffer[bx_sz*by_sz];  
	double send_front_buffer[bx_sz*by_sz]; 
	double recv_front_buffer[bx_sz*by_sz]; 
	
	//If we have neighbour in a direction, we communicate the bdry points both ways, otherwise we set the recv_buffer to zero.
	//TODO loop ordering should be changed.
	MPI_Request requests[12];
	MPI_Status statuses[12];

	for (int id=0; id<6; id++)
		printf("%d ", neighbours[id]);
	printf("\n");

	//Left
	
	if (neighbours[0] != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < by_sz; iy++) {
			for (int iz = 0; iz < bz_sz; iz++, id++) {
				//printf("id left: %d %d %d\n", bx_start, by_start+iy, bz_start+iz);
				send_left_buffer[id] = v[S->index_c(bx_start+0, by_start+iy, bz_start+iz)];
			}
		}
		//MPI_Isend(send_left_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[0], 0, comm, &requests[0]);
		//MPI_Irecv(recv_left_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[0], 0, comm, &requests[1]);
		MPI_Sendrecv(send_left_buffer, by_sz*by_sz, MPI_DOUBLE, neighbours[0], 7, recv_left_buffer, by_sz*bz_sz, MPI_DOUBLE, neighbours[1], 7, comm, &statuses[0]);
	} else {
		 for (int id=0; id<by_sz*bz_sz; id++) {
			 recv_left_buffer[id]=0.0;
		 }
	}
	//Right
	if (neighbours[1] != MPI_PROC_NULL) {
		int id = 0;
		for (int iy = 0; iy < by_sz; iy++) {
			for (int iz = 0; iz < bz_sz; iz++, id++) {
				//printf("id right: %d %d %d\n", bx_start+bx_sz-1, by_start+iy, bz_start+iz);
				send_right_buffer[id] = v[S->index_c(bx_start+bz_sz-1, iy+by_start, bz_start+iz)];
			}
		}
		//MPI_Isend(send_right_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[1], 0, comm, &requests[2]);
		//MPI_Irecv(recv_right_buffer, by_sz * bz_sz, MPI_DOUBLE, neighbours[1], 0, comm, &requests[3]);
		MPI_Sendrecv(send_right_buffer, by_sz*by_sz, MPI_DOUBLE, neighbours[1], 7, recv_right_buffer, by_sz*bz_sz, MPI_DOUBLE, neighbours[0], 7, comm, &statuses[1]);
	} else {
		 for (int id=0; id<by_sz*bz_sz; id++) {
			 recv_right_buffer[id]=0.0;
		 }
	}
	//MPI_Wait(&requests[1], &statuses[1]);
	/*
	//Bottom
	if (neighbours[2] != MPI_PROC_NULL) {
		for (int ix = 0; ix < bx_sz; ix++) {
			for (int iz = 0; iz < bz_sz; iz++) {
				send_bot_buffer[S->index_c(bx_start+ix, by_start+0, bz_start+iz)] = v[S->index_c(bx_start+ix, by_start+0, bz_start+iz)];
			}
		}
		MPI_Isend(send_bot_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[2], 0, comm, &requests[4]);
		MPI_Irecv(recv_bot_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[2], 0, comm, &requests[5]);
	} else {
		for (int id=0; id<bx_sz*bz_sz; id++) {
			 recv_bot_buffer[id]=0.0;
		}
	}

	//Top
	if (neighbours[3] != MPI_PROC_NULL) {
		for (int ix = 0; ix < bx_sz; ix++) {
			for (int iz = 0; iz < bz_sz; iz++) {
				send_top_buffer[S->index_c(bx_start+ix, by_start+by_sz-1, bz_start+iz)] = v[S->index_c(bx_start+ix, by_start+by_sz-1, bz_start+iz)];
			}
		}
		MPI_Isend(send_top_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[3], 0, comm, &requests[6]);
		MPI_Irecv(recv_top_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[3], 0, comm, &requests[7]);
	} else {
		 for (int id=0; id<bx_sz*bz_sz; id++) {
			 recv_top_buffer[id]=0.0;
		 }
	}

	//Back
	if (neighbours[4] != MPI_PROC_NULL) {
		for (int iy = 0; iy < by_sz; iy++) {
			for (int ix = 0; ix < bx_sz; ix++) {
				send_back_buffer[S->index_c(bx_start+ix, by_start+iy, bz_start+0)] = v[S->index_c(bx_start+ix, iy+by_start, bz_start+0)];
			}
		}
		MPI_Isend(send_back_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[4], 0, comm, &requests[8]);
		MPI_Irecv(recv_back_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[4], 0, comm, &requests[9]);
	} else {
		 for (int id=0; id<by_sz*bx_sz; id++) {
			 recv_back_buffer[id]=0.0;
		 }
	}

	//Front
	if (neighbours[5] != MPI_PROC_NULL) {
		for (int iy = 0; iy < by_sz; iy++) {
			for (int ix = 0; ix < bx_sz; ix++) {
				send_front_buffer[S->index_c(bx_start+ix, by_start+iy, bz_start+bz_sz-1)] = v[S->index_c(bx_start+ix, iy+by_start, bz_start+bz_sz-1)];
			}
		}
		MPI_Isend(send_front_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[5], 0, comm, &requests[10]);
		MPI_Irecv(recv_front_buffer, bx_sz * by_sz, MPI_DOUBLE, neighbours[5], 0, comm, &requests[11]);
	} else {
		 for (int id=0; id<by_sz*bx_sz; id++) {
			 recv_front_buffer[id]=0.0;
		 }
	}
	*/
	
	
	/*
	//TODO The case where no neighbour exists needs to be handled in some way here.
	//When we have our response, treat the 6 border cases separately. 
	MPI_Wait(&requests[1], &statuses[1]);
	//TODO left case
	MPI_Wait(&requests[3], &statuses[3]);
	//TODO right case
	MPI_Wait(&requests[5], &statuses[5]);
	//TODO top case
	MPI_Wait(&requests[7], &statuses[7]);
	//TODO bottom case
	MPI_Wait(&requests[9], &statuses[9]);
	//TODO back case
	MPI_Wait(&requests[11], &statuses[11]);
	//TODO front case
	*/
	//For the interior of the block, perform the operations as usual.
	for (int iz=bz_start+1; iz<bz_start+bz_sz-1; iz++) {
		for (int iy=by_start+1; iy<by_start+by_sz-1; iy++) {
			for (int ix=bx_start+1; ix<bx_start+bx_sz-1; ix++) {
				v[S->index_c(ix, iy, iz)] = S->value_c * u[S->index_c(ix, iy, iz)]
				+ S->value_b * u[S->index_b(ix, iy, iz)]
				+ S->value_t * u[S->index_t(ix, iy, iz)]
				+ S->value_s * u[S->index_s(ix, iy, iz)]
				+ S->value_n * u[S->index_n(ix, iy, iz)]
				+ S->value_w * u[S->index_w(ix, iy, iz)]
				+ S->value_e * u[S->index_e(ix, iy, iz)];
				printf("ix=%d, iy=%d, iz=%d\n", ix, iy, iz);
			}
		}
	}
	/*
	free(send_left_buffer);
	free(recv_left_buffer);
	free(send_right_buffer);
	free(recv_right_buffer);
	free(send_top_buffer);
	free(recv_top_buffer); 
	free(send_bot_buffer);
	free(recv_bot_buffer);
	free(send_back_buffer);
	free(recv_back_buffer);
	free(send_front_buffer);
	free(recv_front_buffer);
	*/
	return;
}

///Test the 
stencil3d laplace3d_stencil(int nx, int ny, int nz)
{
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
	
	int nx = 5;
	int ny = 5;
	int nz = 5;
	double u[nx*ny*nz]={5.0, 5.0, 5.0};
	double v[nx*ny*nz]={1.0, 0.0, 2.0};
	stencil3d L = laplace3d_stencil(nx, ny, nz);
	apply_stencil3d(&L, u, v, comm);
	double dot_res = dot(nx*ny*nz, u, v, comm);
	if (rank==0)
		printf("Dot result %f\n", dot_res);
	
	MPI_Finalize();
	return 0;
}
