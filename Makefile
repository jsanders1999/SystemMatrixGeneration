CXX = mpic++
CXXFLAGS = -O2 -g -lm -lgtest -lstdc++

SRCS = operations.cpp
EXEC = operations.x

TEST_SRCS = operations_test.cpp
TEST_EXEC = operations_test.x

all: $(EXEC)

$(EXEC): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(EXEC)

test: $(TEST_EXEC)
	./$(TEST_EXEC)

$(TEST_EXEC): $(SRCS) $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(TEST_SRCS) -o $(TEST_EXEC) -lgtest -lgtest_main -lpthread

clean:
	rm -f $(EXEC) $(TEST_EXEC)
