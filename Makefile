# first:
#module load 2022r2
#module load openmpi



CXX=g++#mpic++
CXX_FLAGS=-O2 -g -fopenmp -std=c++17 

#default target (built when typing just "make")
default: main_integraltest.x

# general rule to compile a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS}  $<


#define some dependencies on headers


MAIN_INT_OBJ=main_integraltest.o


main_integraltest.x: ${MAIN_CG_OBJ}
	${CXX} ${CXX_FLAGS} -o main_integraltest.x $^




#test: run_tests.x
#	./run_tests.x


clean:
	-rm *.o *.x

# phony targets are run regardless of dependencies being up-to-date
PHONY: clean, test

