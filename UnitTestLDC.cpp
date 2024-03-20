#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

#define BOOST_TEST_MODULE UnitTestLDC
#define IDX(I,J) ((J)*Nx_local + (I))
#define IDX_GLOBAL(I, J) ((J) * Nx + (I))

#include <boost/program_options.hpp>
#include <boost/test/included/unit_test.hpp>

#include "LidDrivenCavity.h"
#include "SolverCG.h"
#include "MPIGridCommunicator.h"
#include "mpi.h"
#include <cmath> // Include the cmath library for sqrt function


/**
* @file LidDrivenCavitySolver.cpp
* @brief Solving the vorticity-stream function formulation of the
* incompressible Navier-Stokes equations in 2D using the finite difference method.
*
* @param argc Number of command-line arguments.
* @param argv Array of command-line arguments.
*/

bool compareFiles(const string& file1, const string& file2) {
    ifstream stream1(file1);
    ifstream stream2(file2);

    if (!stream1.is_open() || !stream2.is_open()) {
        cerr << "Error opening files for comparison." << endl;
        return false;
    }

    vector<double> vec1, vec2;
    string line1, line2;
    
    // Read numbers from the first file and store them in vec1
    while (getline(stream1, line1)) {
        istringstream iss(line1);
        double num;
        
        while (iss >> num) {
            vec1.push_back(num);
        }
    }
    // Close the files
    stream1.close();

    // Read numbers from the second file and store them in vec2
    while (getline(stream2, line2)) {
        istringstream iss(line2);
        double num;
        
        while (iss >> num) {
            vec2.push_back(num);
        }
    }
    // Close the files
    stream2.close();

    // Check if the vectors have the same size
    if (vec1.size() != vec2.size()) {
        cerr << "Vectors have different sizes!" << endl;
        return false;
    }

    // Check if each element in the vectors is equal
    for (int i = 0; i < (int)vec1.size(); ++i) {
        BOOST_CHECK_SMALL(abs(vec1[i]- vec2[i]), 1e-3);
    }

    return true;
}

struct MPIFixture {
    public:
        explicit MPIFixture() {
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout << "Initialising MPI" << endl;
            MPI_Init(&argc, &argv);
        }

        ~MPIFixture() {
            cout << "Finalising MPI" << endl;
            MPI_Finalize();
        }

        int argc;
        char **argv;
};
BOOST_GLOBAL_FIXTURE(MPIFixture);


// Boost Test Case for file comparison LidDrivenCavitySolver
BOOST_AUTO_TEST_CASE(LidDrivenCavitySolver_file_comparison) {

    LidDrivenCavity* solver = new LidDrivenCavity();

    // Hardcoded values
    double Lx = 1.0;
    double Ly = 1.0;
    int Nx = 9;
    int Ny = 9;
    double dt = 0.01;
    double T = 1.0;
    double Re = 10.0;


    // Initialise MPI within the struct

    // Get the total number of processes and the rank of the current process
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create a Cartesian topology
    int p = sqrt(size);
    int dims[2] = {p, p};  // pxp grid
    int periods[2] = {0, 0};  // No periodicity
    int coords[2];
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    solver->SetDomainSize(Lx, Ly);
    solver->SetGridSize(Nx, Ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);
    solver->SetLocalVariables(Nx, Ny, p, coords);

    solver->Initialise(cart_comm, coords, p);

    solver->WriteSolution("TestInputLidDrivenCavity.txt");

    solver->Integrate();

    solver->WriteSolution("TestOutputLidDrivenCavity.txt");

    delete solver; // Release the allocated memory

    if (rank == 0){
        cout << "Testing input files for LidDrivenCavity: " << endl;
        BOOST_CHECK(compareFiles("TestInputLidDrivenCavity.txt", "BaselineInputLidDrivenCavity.txt"));
    
        cout << "Testing output files for LidDrivenCavity: " << endl;
        BOOST_CHECK(compareFiles("TestOutputLidDrivenCavity.txt", "BaselineOutputLidDrivenCavity.txt"));
    }
    
}
