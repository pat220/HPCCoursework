CXX = mpicxx -fopenmp
CXXFLAGS = -std=c++11 -Wall -O3
HDRS = SolverCG.h LidDrivenCavity.h MPIGridCommunicator.h
OBJS = SolverCG.o LidDrivenCavity.o MPIGridCommunicator.o
OBJS_MAIN = LidDrivenCavitySolver.o
OBJS_TESTS_SCG = UnitTestSCG.o
OBJS_TESTS_LDC = UnitTestLDC.o
LDLIBS = -lblas -lboost_program_options -lboost_unit_test_framework #-lboost_chrono #-lboost_timer
TARGET = solver
TARGET_TESTS_SCG = unittest_scg # tried to get unitests as a target with both unit tests but when doing that 1 test fails whilst separately they do not - to do with MPI
TARGET_TESTS_LDC = unittest_ldc # tried to get unitests as a target with both unit tests but when doing that 1 test fails whilst separately they do not - to do with MPI
TARGET_TESTS = unittests
TARGET_PROFILER = solver_profiler # it has special flags inside
TARGET_DOXY = doxygen_file


default: $(TARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -g -o $@ -c $<
	
$(TARGET): $(OBJS_MAIN) $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

# Target for displaying the help message to run unit tests separately
$(TARGET_TESTS): $(TARGET_TESTS_SCG) $(TARGET_TESTS_LDC)
	@echo "WARNING FROM DESIGNER: Making the unittests on the same file does not work. Try executing them separately with -./unittest_scg- and -./unittest_ldc-"

$(TARGET_TESTS_SCG): $(OBJS_TESTS_SCG) $(OBJS) 
	$(CXX) -o $@ $^ $(LDLIBS)

$(TARGET_TESTS_LDC): $(OBJS_TESTS_LDC) $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

$(TARGET_PROFILER): $(OBJS_MAIN) $(OBJS)
	export OMPI_CXX=g++-10
	$(CXX) -o $@ $^ $(LDLIBS)

doc:
	doxygen -g $(TARGET_DOXY)
	doxygen $(TARGET_DOXY)


.PHONY: clean	
clean:	
	-rm -f *.o $(TARGET) $(TARGET_PROFILER) $(TARGET_TESTS_SCG) $(TARGET_TESTS_LDC) $(TARGET_DOXY) *.out
	
.PHONY: docs
docs:
	@doxygen -g $(TARGET_DOXY)