#pragma once

#include "operations.hpp"

// run Conjugate Gradient iterations to solve the linear system
// op*x=b, where op is the 7-point stencil representation of a linear
// operator. The function returns if the 2-norm of the residual reaches
// tol, or the number of iterations reaches maxIter. The residual norm
// is returned in *resNorm, the number of iterations in *numIter.
void cg_solver(stencil3d const* op, block_params const* BP, int n, double* x, double const* b,
        double  tol,     int  maxIter,
        double* resNorm, int* numIter,
        int verbose=1);
