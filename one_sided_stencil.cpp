#include <mpi.h>
#include <math.h>
#include "operations.hpp"
#include <iostream>
#include "timer.hpp"

void apply_stencil3d(stencil3d const* S, block_params const* BP, double const* u, double* v) {
	Timer t("apply_stencil MPI_Win");
	
	//buffer sizes
	int sz_ew = BP->by_sz*BP->bz_sz;
	int sz_ns = BP->bx_sz*BP->bz_sz;
	int sz_tb = BP->bx_sz*BP->by_sz;
	//Initialize buffers with zeros; TODO, we call calloc only if we need it. Should be faster!
	double* west_buffer   = (double*) calloc(sz_ew, sizeof(double)); 
	double* east_buffer   = (double*) calloc(sz_ew, sizeof(double)); 
	double* south_buffer  = (double*) calloc(sz_ns, sizeof(double)); 
	double* north_buffer  = (double*) calloc(sz_ns, sizeof(double)); 
	double* bot_buffer    = (double*) calloc(sz_tb, sizeof(double)); 
	double* top_buffer    = (double*) calloc(sz_tb, sizeof(double)); 
	double* west_recv_buf  = (double*) calloc(sz_ew, sizeof(double)); 
	double* east_recv_buf  = (double*) calloc(sz_ew, sizeof(double)); 
	double* south_recv_buf = (double*) calloc(sz_ns, sizeof(double)); 
	double* north_recv_buf = (double*) calloc(sz_ns, sizeof(double)); 
	double* bot_recv_buf   = (double*) calloc(sz_tb, sizeof(double)); 
	double* top_recv_buf   = (double*) calloc(sz_tb, sizeof(double));
	{
		Timer t("apply_stencil: 1)fill bufs");
		
		//If we have a neighbour, we fill the buffer.
		//west
		if (BP->rank_w != MPI_PROC_NULL) {
			int id = 0;
			for (int iz = 0; iz < BP->bz_sz; iz++) {
				for (int iy = 0; iy < BP->by_sz; iy++, id++) {
					west_buffer[id] = u[S->index_c(0, iy, iz)];
				}
			}
		}
		//east
		if (BP->rank_e != MPI_PROC_NULL) {
			int id = 0;
			for (int iz = 0; iz < BP->bz_sz; iz++) {
				for (int iy = 0; iy < BP->by_sz; iy++, id++) {
					east_buffer[id] = u[S->index_c(BP->bx_sz-1, iy, iz)];
				}
			}
		}
		//south
		if (BP->rank_s != MPI_PROC_NULL) {
			int id = 0;
			for (int iz = 0; iz < BP->bz_sz; iz++) {
				for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
					south_buffer[id] = u[S->index_c(ix, BP->by_sz-1, iz)];
				}
			}
		}
		//north
		if (BP->rank_n != MPI_PROC_NULL) {
			int id = 0;
			for (int iz = 0; iz < BP->bz_sz; iz++) {
				for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
					north_buffer[id] = u[S->index_c(ix, BP->by_sz-1, iz)];
				}
			}
		}
		//bot
		if (BP->rank_b != MPI_PROC_NULL) {
			int id = 0;
			for (int iy = 0; iy < BP->by_sz; iy++) {
				for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
					bot_buffer[id] = u[S->index_c(ix, iy, 0)];
				}
			}
		}
		//top
		if (BP->rank_t != MPI_PROC_NULL) {
			int id = 0;
			for (int iy = 0; iy < BP->by_sz; iy++) {
				for (int ix = 0; ix < BP->bx_sz; ix++, id++) {
					top_buffer[id] = u[S->index_c(ix, iy, BP->bz_sz-1)];
				}
			}
		}
	}
	MPI_Win win_e, win_w, win_s, win_n, win_t, win_b;
	{
		Timer t("apply_stencil: 2)win_create");
		//For each direction, create a MPI_Window so that other processes can read from it. //TODO, this takes some time, it would be faster if we didn't do it every time the function is called.
		MPI_Win_create(east_buffer, sz_ew * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_e);
		MPI_Win_create(west_buffer, sz_ew * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_w);
		MPI_Win_create(north_buffer, sz_ns * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_n);
		MPI_Win_create(south_buffer, sz_ns * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_s);
		MPI_Win_create(top_buffer, sz_tb * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_t);
		MPI_Win_create(bot_buffer, sz_tb * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);
	}
	{
		Timer t("apply_stencil: 3)first fence");
		//We don't want to start reading before all processes is at this stage, so we sync.
		MPI_Win_fence(0, win_e);
		MPI_Win_fence(0, win_w);
		MPI_Win_fence(0, win_n);
		MPI_Win_fence(0, win_s);
		MPI_Win_fence(0, win_t);
		MPI_Win_fence(0, win_b);
	}
	{
		Timer t("apply_stencil: 4)get");
		//Read from neighbours if we have any.
		if (BP->rank_e != MPI_PROC_NULL) {
			MPI_Get(east_recv_buf, sz_ew, MPI_DOUBLE, BP->rank_e, 0, sz_ew, MPI_DOUBLE, win_e);
		}
		if (BP->rank_w != MPI_PROC_NULL) {
			MPI_Get(west_recv_buf, sz_ew, MPI_DOUBLE, BP->rank_w, 0, sz_ew, MPI_DOUBLE, win_w);
		}
		if (BP->rank_n != MPI_PROC_NULL) {
			MPI_Get(north_recv_buf, sz_ns, MPI_DOUBLE, BP->rank_n, 0, sz_ns, MPI_DOUBLE, win_n);
		}
		if (BP->rank_s != MPI_PROC_NULL) {
			MPI_Get(south_recv_buf, sz_ns, MPI_DOUBLE, BP->rank_s, 0, sz_ns, MPI_DOUBLE, win_s);
		}
		if (BP->rank_t != MPI_PROC_NULL) {
			MPI_Get(top_recv_buf, sz_tb, MPI_DOUBLE, BP->rank_t, 0, sz_tb, MPI_DOUBLE, win_t);
		}
		if (BP->rank_b != MPI_PROC_NULL) {
			MPI_Get(bot_recv_buf, sz_tb, MPI_DOUBLE, BP->rank_b, 0, sz_tb, MPI_DOUBLE, win_b);
		}
	}
	{
		Timer t("apply_stencil: 5)second fence");
		//Again, we don't want to read from the buffers if they are not filled, so we sync.
		MPI_Win_fence(0, win_e);
		MPI_Win_fence(0, win_w);
		MPI_Win_fence(0, win_n);
		MPI_Win_fence(0, win_s);
		MPI_Win_fence(0, win_t);
		MPI_Win_fence(0, win_b);
	}
	{
		Timer t("apply_stencil: 6)final step");
		//Use buffers for edge points, apply the stencil.
		for (int iz=0; iz<BP->bz_sz; iz++) {
			for (int iy=0; iy<BP->by_sz; iy++) {
				for (int ix=0; ix<BP->bx_sz; ix++) {
					int ew_id = BP->by_sz*iz+iy; 
					int ns_id = BP->bx_sz*iz+ix; 
					int tb_id = BP->bx_sz*iy+ix; 
					double accum = S->value_c * u[S->index_c(ix, iy, iz)];
					accum += S->value_w * ((ix==0            ) ? west_recv_buf[ew_id]  : u[S->index_w(ix, iy, iz)]);
					accum += S->value_e * ((ix==0+BP->bx_sz-1) ? east_recv_buf[ew_id]  : u[S->index_e(ix, iy, iz)]);
					accum += S->value_s * ((iy==0            ) ? south_recv_buf[ns_id] : u[S->index_s(ix, iy, iz)]);
					accum += S->value_n * ((iy==0+BP->by_sz-1) ? north_recv_buf[ns_id] : u[S->index_n(ix, iy, iz)]);
					accum += S->value_b * ((iz==0            ) ? bot_recv_buf[tb_id]   : u[S->index_b(ix, iy, iz)]);
					accum += S->value_t * ((iz==0+BP->bz_sz-1) ? top_recv_buf[tb_id]   : u[S->index_t(ix, iy, iz)]);
					v[S->index_c(ix, iy, iz)] = accum;
				}
			}
		}
	}
	return;
}
