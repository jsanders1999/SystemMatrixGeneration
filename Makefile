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

run_tests.x: run_tests.cpp ${TEST_SRCS} gtest_mpi.o operations.o
	$(CXX) $(CXXFLAGS) $^ -DUSE_MPI -o $@

test: run_tests.x
	./run_tests.x

$(TEST_EXEC): $(SRCS) $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(TEST_SRCS) -o $(TEST_EXEC)

clean:
	rm -f $(EXEC) $(TEST_EXEC)
 					
