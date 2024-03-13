CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
HDRS = SolverCG.h LidDrivenCavity.h
OBJS = SolverCG.o LidDrivenCavity.o
OBJS_MAIN = LidDrivenCavitySolver.o
OBJS_TESTS = UnitTests.o
LDLIBS = -llapack -lblas -lboost_program_options -lboost_unit_test_framework
TARGET = solver
TARGET_TESTS = unittests
TARGET_DOXY = doxygen_file

default: $(TARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	
$(TARGET): $(OBJS_MAIN) $(OBJS) 
	$(CXX) -o $@ $^ $(LDLIBS)

$(TARGET_TESTS): $(OBJS_TESTS) $(OBJS) 
	$(CXX) -o $@ $^ $(LDLIBS)
	
.PHONY: clean	
clean:	
	-rm -f *.o $(TARGET) $(TARGET_TESTS)
	
.PHONY: docs
docs:
	@doxygen -g $(TARGET_DOXY)