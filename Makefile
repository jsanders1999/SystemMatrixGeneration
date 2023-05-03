# first:
#module load 2022r2
#module load openmpi

CXX=mpic++
CXX_FLAGS=-O2 -g -fopenmp -lm -std=c++17
#CXX_FLAGS=-O3 -march=native -g -fopenmp -std=c++17
DEFS=-DUSE_POLY

#main_gmres_poisson.o: FLAGS+=-DUSE_POLY

#default target (built when typing just "make")
default: run_tests.x main_cg_poisson.x main_gmres_poisson.x main_polyg_poisson.x main_diagg_poisson.x

# general rule to compile a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS}  $<

#define some dependencies on headers
operations.o: operations.hpp timer.hpp
gmres_solver.o: gmres_solver.hpp operations.hpp timer.hpp
polygmres_solver.o: gmres_solver.hpp operations.hpp timer.hpp
cg_solver.o: cg_solver.hpp operations.hpp timer.hpp
gtest_mpi.o: gtest_mpi.hpp

TEST_SOURCES=test_operations.cpp test_gmres_solver.cpp timer.o
MAIN_CG_OBJ=main_cg_poisson.o cg_solver.o operations.o timer.o 
MAIN_GMRES_OBJ=main_gmres_poisson.cpp gmres_solver.o operations.o timer.o 
MAIN_DIAGG_OBJ=main_gmres_poisson.cpp gmres_solver.o operations.o timer.o
MAIN_POLYG_OBJ=main_gmres_poisson.cpp polygmres_solver.o operations.o timer.o
 
run_tests.x: run_tests.cpp ${TEST_SOURCES} gtest_mpi.o operations.o gmres_solver.o cg_solver.o
	${CXX} ${CXX_FLAGS} -DUSE_MPI -o run_tests.x $^

main_cg_poisson.x: ${MAIN_CG_OBJ}
	${CXX} ${CXX_FLAGS} -o main_cg_poisson.x $^

main_gmres_poisson.x: ${MAIN_GMRES_OBJ}
	${CXX} ${CXX_FLAGS} -o main_gmres_poisson.x $^

main_polyg_poisson.x: ${MAIN_POLYG_OBJ}
	${CXX} ${CXX_FLAGS} -DUSE_POLY -o main_polyg_poisson.x $^

main_diagg_poisson.x: ${MAIN_DIAGG_OBJ}
	${CXX} ${CXX_FLAGS} -DUSE_DIAG -o main_diagg_poisson.x $^

test: run_tests.x
	mpirun -np 1 ./run_tests.x
	mpirun -np 2 ./run_tests.x
	mpirun -np 8 ./run_tests.x
	mpirun -np 18 ./run_tests.x --mca orte_base_help_aggregate 0

cg_solver: main_cg_poisson.x
	mpirun -np 1 ./main_cg_poisson.x 64
	mpirun -np 18 ./main_cg_poisson.x 64

gmres_solver: main_gmres_poisson.x
	mpirun -np 1 ./main_gmres_poisson.x 64
	mpirun -np 18 ./main_gmres_poisson.x 64

clean:
	-rm *.o *.x

# phony targets are run regardless of dependencies being up-to-date
PHONY: clean, test

