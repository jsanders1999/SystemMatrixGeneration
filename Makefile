# first:
#module load 2022r2
#module load openmpi

CXX = mpic++
CXXFLAGS = -O2 -g -lm -lstdc++

SRCS = operations.cpp
EXEC = operations.x

operations.o: operations.hpp
gtest_mpi.o: gtest_mpi.hpp

TEST_SRCS = test_operations.cpp
TEST_EXEC = test_operations.x

all: $(EXEC)

$(EXEC): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(EXEC)

run_tests.x: run_tests.cpp ${TEST_SRCS} operations.o gtest_mpi.o
	$(CXX) $(CXXFLAGS) $^ -DUSE_MPI -o $@

test: run_tests.x
	mpirun -np 1 ./run_tests.x
	mpirun -np 2 ./run_tests.x
	mpirun -np 8 ./run_tests.x

$(TEST_EXEC): $(SRCS) $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(TEST_SRCS) -o $(TEST_EXEC)

solver: main_cg_poisson.cpp cg_solver.cpp cg_solver.hpp operations.hpp
	$(CXX) $(CXXFLAGS) main_cg_poisson.cpp cg_solver.cpp -o solver.x
	mpirun -np 1 ./solver.x 64
	mpirun -np 18 ./solver.x 64

clean:
	rm -f $(EXEC) $(TEST_EXEC)
 					
