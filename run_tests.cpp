#ifdef USE_MPI
#include <mpi.h>
#endif

#include "gtest_mpi.hpp"

int main(int argc, char **argv) {
    int test_result = 0;

    testing::InitGoogleTest(&argc, argv);

    int rank;
    int num_procs;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    // on MPI ranks != 0 remove the default output listeners if there are any
    if (rank != 0) {
        ::testing::TestEventListener* defaultListener = ::testing::UnitTest::GetInstance()->listeners().default_result_printer();
        ::testing::UnitTest::GetInstance()->listeners().Release(defaultListener);
        delete defaultListener;

        ::testing::TestEventListener* defaultXMLListener = ::testing::UnitTest::GetInstance()->listeners().default_xml_generator();
        ::testing::UnitTest::GetInstance()->listeners().Release(defaultXMLListener);
        delete defaultXMLListener;
    }
	test_result = RUN_ALL_TESTS();
#ifdef USE_MPI
    MPI_Finalize();
#endif

    return test_result;
}
