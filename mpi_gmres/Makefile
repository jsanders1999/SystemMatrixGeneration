# How to use
# > module load 2022r2 nvhpc
# > make TARGET=cpu
# or
# > make TARGET=gpu
# or
# to run the test: make test


CXX=mpic++
CXX_FLAGS=-O2 -g -fopenmp -std=c++17
#CXX_FLAGS=-O3 -march=native -g -fopenmp -std=c++17
DEFS=-DNDEBUG

#default target (built when typing just "make")
default: run_tests.x main_gmres_poisson.x main_benchmarks.x

# general rule to comple a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} ${DEFS} $<

#define some dependencies on headers
operations.o: operations.hpp timer.hpp
gmres_solver.o: gmres_solver.hpp operations.hpp timer.hpp
gtest_mpi.o: gtest_mpi.hpp

TEST_SOURCES=test_operations.cpp test_gmres_solver.cpp timer.o
MAIN_OBJ=main_gmres_poisson.o gmres_solver.o operations.o timer.o 
BCMK_OBJ=main_benchmarks.o operations.o timer.o
 
run_tests.x: run_tests.cpp ${TEST_SOURCES} gtest_mpi.o operations.o gmres_solver.o 
	${CXX} ${CXX_FLAGS} ${DEFS}  -o run_tests.x $^

main_gmres_poisson.x: ${MAIN_OBJ}
	${CXX} ${CXX_FLAGS} ${DEFS} -o main_gmres_poisson.x $^

main_benchmarks.x: ${BCMK_OBJ} 
	${CXX} ${CXX_FLAGS} ${DEFS} -o main_benchmarks.x $^

test: run_tests.x
	./run_tests.x

clean:
	-rm *.o *.x

# phony targets are run regardless of dependencies being up-to-date
PHONY: clean, test

