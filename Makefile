CXX = mpic++
CXXFLAGS = -O2 -g -lm -lstdc++


SRCS = operations.cpp
EXEC = operations.x

all: $(EXEC)

$(EXEC): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(EXEC)

clean:
	rm -f $(EXEC)

