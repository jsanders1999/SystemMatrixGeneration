#ifdef USE_MPI
#include <mpi.h>
#endif

#include "gtest_mpi.hpp"

GTEST_API_ int main(int argc, char **argv) {
    int test_result = 0;

    testing::InitGoogleTest(&argc, argv);

    int num_processes[] = {1, 2, 8}; // number of processes to test with
    int num_processes_size = sizeof(num_processes)/sizeof(num_processes[0]);

#ifdef USE_MPI
	MPI_Init(&argc, &argv);
    for (int i = 0; i < num_processes_size; i++) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

        if (rank != 0) {
            // on MPI ranks != 0 remove the default output listeners if there are any
            ::testing::TestEventListener* defaultListener = ::testing::UnitTest::GetInstance()->listeners().default_result_printer();
            ::testing::UnitTest::GetInstance()->listeners().Release(defaultListener);
            delete defaultListener;

            ::testing::TestEventListener* defaultXMLListener = ::testing::UnitTest::GetInstance()->listeners().default_xml_generator();
            ::testing::UnitTest::GetInstance()->listeners().Release(defaultXMLListener);
            delete defaultXMLListener;
        }

        // set the number of MPI processes for the test
        int mpi_size = num_processes[i];
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        //::testing::TestEnvironment::GetInstance()->SetMPIWorldComm(MPI_COMM_WORLD, mpi_size);

        // run the tests
        test_result = RUN_ALL_TESTS();
    }
	// finalize MPI
	MPI_Finalize();
#endif

    return test_result;
}
