#include "gtest_mpi.hpp"
#include "operations.hpp"
#include <iostream>
#include <mpi.h>

TEST(DotTest, DotProductIsCorrect) {
	int n = 300;
	int rank;
	double x[n], y[n];
	for (int i = 0; i < n; i++) {
		x[i] = (double)i+1.0;
		y[i] = 1.0 / (double)(i+1);
	}
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	printf("Rank: %d\n", rank);
	double dot_product = dot(n, x, y, comm);

	if (rank == 0) {
		EXPECT_DOUBLE_EQ(dot_product, (double)n);
	}
}

