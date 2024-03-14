#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define BOOST_TEST_MODULE UnitTests
#define IDX(I,J) ((J)*Nx + (I))

#include <boost/program_options.hpp>
#include <boost/test/included/unit_test.hpp>

#include "LidDrivenCavity.h"
#include "SolverCG.h"
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

    string line1, line2;
    while (!stream1.eof() && !stream2.eof()) {
        getline(stream1, line1);
        getline(stream2, line2);
        if (line1 != line2) {
            // Check if the characters are equal within a tolerance (there are some changes in the last digits of the order of e-310)
            double val1, val2;
            if (sscanf(line1.c_str(), "%lf", &val1) == 1 && sscanf(line2.c_str(), "%lf", &val2) == 1) {
                double diff = fabs(val1 - val2);
                if (diff > 1e-3) {
                    cout << "Files are not equal." << endl;
                    return false;
                }
            } else {
                cout << "Files are not equal." << endl;
                return false;
            }
        }
    }

    if (stream1.eof() && stream2.eof()) {
        cout << "Files are equal." << endl;
        return true;
    }

    cout << "Files are not equal (different number of lines)." << stream1.eof() << stream2.eof() << endl;
    return false;
}

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

    int argc = 2;
    char** argv = new char*[argc];

    // Initialize MPI
    MPI_Init(NULL, NULL);

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
    solver->SetLocalVariables(Nx, Ny, p, rank);

    solver->PrintConfiguration();

    solver->Initialise();

    solver->GetInfoMPI(cart_comm, rank, size, coords, p);

    solver->WriteSolution("TestInputLidDrivenCavity.txt");

    solver->Integrate();

    solver->WriteSolution("TestOutputLidDrivenCavity.txt");

    delete solver; // Release the allocated memory

    // Finalise MPI
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    cout << "Testing input files for LidDrivenCavity: " << endl;
    BOOST_CHECK(compareFiles("TestInputLidDrivenCavity.txt", "BaselineInputLidDrivenCavity.txt"));
 
    cout << "Testing output files for LidDrivenCavity: " << endl;
    BOOST_CHECK(compareFiles("TestOutputLidDrivenCavity.txt", "BaselineOutputLidDrivenCavity.txt"));
    
}

// Boost Test Case for file comparison SolverCG
BOOST_AUTO_TEST_CASE(SolverCG_file_comparison) {

    // // Hardcoded values
    double Lx = 1.0;
    double Ly = 1.0;
    int Nx = 9;
    int Ny = 9;

    // Sinusoidal test case with analytical solution, which can be used to test
    // the Poisson solver

    double dx = Lx / (Nx-1);
    double dy = Ly / (Ny-1);
    int Npts = Nx * Ny;

    double* v   = new double[Npts]();
    double* s   = new double[Npts]();

    // Declare and initialize the variable "cg"
    SolverCG* cg = new SolverCG(Nx, Ny, dx, dy);

    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    

    // Solve Poisson problem
    cg->Solve(v, s);

    // Write the solution to file
    ofstream file;
    file.open("TestOutputSolverCG.txt");
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            file << s[IDX(i,j)] << " ";
        }
        file << endl;
    }
    file.close();

    //Check it against the baseline
    cout << "Testing output files for SolverCG: " << endl;
    BOOST_CHECK(compareFiles("TestOutputSolverCG.txt", "BaselineOutputSolverCG.txt"));
    
 
}