#ifdef USE_MPI
#include <mpi.h>
#endif

#include "gtest_mpi.hpp"

int main(int argc, char **argv) {
    int test_result = 0;

    testing::InitGoogleTest(&argc, argv);

    int rank = 0;
    int num_procs = 1;

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

    // run tests with 1, 2, and 8 processes
    for (int num_procs_iter : {1, 2, 8}) {
        if (num_procs_iter <= num_procs) {
            MPI_Comm comm;
            MPI_Comm_split(MPI_COMM_WORLD, rank < num_procs_iter, rank, &comm);

            if (MPI_COMM_NULL != comm) {
                MPI_Comm_rank(comm, &rank);

                if (rank == 0) {
                    std::cout << "Running tests with " << num_procs_iter << " processes..." << std::endl;
                }

                if (num_procs_iter > 1) {
                    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
                    delete listeners.Release(listeners.default_result_printer());
                    listeners.Append(new ::testing::TestEventListenerMPI(comm));
                }

                test_result = RUN_ALL_TESTS();

                if (num_procs_iter > 1) {
                    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
                    delete listeners.Release(listeners.default_result_printer());
                    listeners.Append(new ::testing::TestEventListenerMPI(MPI_COMM_WORLD));
                }

                MPI_Comm_free(&comm);
            }
        }
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return test_result;
}
