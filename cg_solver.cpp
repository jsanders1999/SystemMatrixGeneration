#include "cg_solver.hpp"
#include "operations.hpp" //TODO linking??
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
//#include "timer.hpp"

void cg_solver(stencil3d const* op, block_params const* BP, int n, double* x, double const* b, double tol, int maxIter, double* resNorm, int* numIter, int verbose) {
	if (op->nx * op->ny * op->nz != n)
	{
		throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
	}
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double *p = new double[n];
	double *q = new double[n];
	double *r = new double[n];

	double alpha, beta, rho=1.0, rho_old=0.0;

	// r = b - op * x
	apply_stencil3d(op, BP, x, r);
	axpby(n, 1.0, b, -1.0, r);
	// p = q = 0
	init(n, p, 0.0);
	init(n, q, 0.0);

	// start CG iteration
	int iter = -1;
	while (true)
	{
		iter++;
		// rho = <r, r>
		rho = dot(n, r, r);
		if (verbose && rank==0 && iter%10==0){
			std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
		}
		if ((std::sqrt(rho) < tol) || (iter > maxIter)){
			break;
		}
		if (rho_old==0.0) {
			alpha = 0.0;
		} else {
			alpha = rho / rho_old;
		}
		// p = r + alpha * p
		axpby(n, 1.0, r, alpha, p);
		// q = op * p
		apply_stencil3d(op, BP, p, q);
		// beta = <p,q>
		beta = dot(n, p, q);
		alpha = rho / beta;
		// x = x + alpha * p
		axpby(n, alpha, p, 1.0, x);
		// r = r - alpha * q
		axpby(n, -alpha, q, 1.0, r);
		std::swap(rho_old, rho);
	}

	// clean up
	delete [] p;
	delete [] q;
	delete [] r;
	
	// return number of iterations and achieved residual
	*resNorm = rho;
	*numIter = iter;
	return;
}
