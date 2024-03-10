CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
HDRS = SolverCG.h LidDrivenCavity.h
OBJS = SolverCG.o LidDrivenCavity.o
OBJS_MAIN = LidDrivenCavitySolver.o
OBJS_TESTS = UnitTests.o
LDLIBS = -llapack -lblas -lboost_program_options
TARGET = solver
TARGET_TESTS = unittests

default: $(TARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TARGET): $(OBJS) $(OBJS_MAIN)
	$(CXX) -o $@ $^ $(LDLIBS)

$(TARGET_TESTS): $(OBJS) $(OBJS_TESTS)
	$(CXX) -o $@ $^ $(LDLIBS)
	
.PHONY: clean	

clean:	
	-rm -f *.o $(TARGET) $(TARGET_TESTS)