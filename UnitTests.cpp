#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define BOOST_TEST_MODULE UnitTests
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

    string line1, line2;
    stringstream ss1, ss2;
    bool stream1_eof = stream1.eof();
    bool stream2_eof = stream2.eof();
    
    while (!stream1_eof && !stream2_eof) {
        getline(stream1, line1);
        getline(stream2, line2);
        // Read each character in a line from a .txt file and go to the next line when finished
        ss1.str(line1);
        ss2.str(line2);

        for (int i = 0; i < 9; i++) { // hard coded 9 as it is Nx = 9
            double val1, val2;
            ss1 >> val1;
            ss2 >> val2;
            // cout << val1 << " " << val2 << endl;
            if (abs(val1 - val2) > 1e-3) {
                cout << "Files are not equal." << endl;
                return false;
            }
        }

        stream1_eof = stream1.eof();
        stream2_eof = stream2.eof();
    }

    if (stream1.eof() && stream2.eof()) {
        cout << "Files are equal." << endl;
        return true;
    }

    cout << "Files are not equal (different number of lines): " << stream1.eof() << " and " << stream2.eof() << endl;
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
    solver->SetLocalVariables(Nx, Ny, p, coords);

    solver->Initialise(cart_comm, coords, p);

    solver->WriteSolution("TestInputLidDrivenCavity.txt");

    solver->Integrate();

    solver->WriteSolution("TestOutputLidDrivenCavity.txt");

    delete solver; // Release the allocated memory

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

    // Initialize MPI
    // MPI_Init(NULL, NULL);

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

    // Set up starting and end points not to include boundaries (0, Nx/Ny - 1)
    double extra_x = Nx % p;
    double extra_y = Ny % p;
    int min_points_x = (Nx - extra_x) / p;
    int min_points_y = (Ny - extra_y) / p;

    int Nx_local, Ny_local;
    int x_first, x_last, y_first, y_last;

    // Calculate the starting and ending points of each process
    if (coords[1] < extra_x)
    {
        min_points_x++;
        x_first = coords[1] * min_points_x; // global coordinate
        x_last = (coords[1] + 1) * min_points_x; // global coordinate
        Nx_local = x_last - x_first;
    }
    else
    {
        x_first = (min_points_x + 1) * extra_x + min_points_x * (coords[1] - extra_x); // global coordinate
        x_last = (min_points_x + 1) * extra_x + min_points_x * (coords[1] - extra_x + 1); // global coordinate
        Nx_local = x_last - x_first;
    }

    if (coords[0] < extra_y)
    {
        min_points_y++;
        y_last = abs( Ny -(coords[0]) * min_points_y); // global coordinate
        y_first = abs( Ny -(coords[0] + 1) * min_points_y); // global coordinate
        Ny_local = y_last - y_first;
    }
    else
    {
        y_last = abs( Ny - ((min_points_y + 1) * extra_y + min_points_y * (coords[0] - extra_y))); // global coordinate
        y_first = abs( Ny -((min_points_y + 1) * extra_y + min_points_y * (coords[0] - extra_y + 1))); // global coordinate
        Ny_local = y_last - y_first;
    }

    int start_x = coords[1] == 0 ? 1 : 0;
    int end_x = coords[1] == p - 1 ? Nx_local - 1 : Nx_local;
    int start_y = coords[0] == p - 1 ? 1 : 0;
    int end_y = coords[0] == 0 ? Ny_local - 1 : Ny_local;


    MPIGridCommunicator* mpiGridCommunicator = new MPIGridCommunicator(cart_comm, Nx_local, Ny_local, start_x, end_x, start_y, end_y, coords, p);

    // Declare and initialize the variable "cg"
    SolverCG* cg = new SolverCG(Nx, Ny, dx, dy, mpiGridCommunicator);

    double* v   = new double[Nx_local*Ny_local]();
    double* s   = new double[Nx_local*Ny_local]();

    const int k = 3;
    const int l = 3;
    const int var1 = (k * k + l * l);

    for (int i = x_first; i < x_first + Nx_local; ++i) {
        double var_i = sin(M_PI * k * i * dx);
        for (int j = y_first; j < y_first + Ny_local; ++j) {
            double var_j = sin(M_PI * l * j * dy);
            v[IDX(i - x_first,j - y_first)] = -M_PI * M_PI * var1 * var_i * var_j;
        }
    }

    // ofstream file;
    // file.open("TestOutputSolverCG.txt", ios::app);
    // for (int r = 0; r < p*p; ++r) {
    //     if (r == rank) {
    //         file << "Rank: " << rank << endl;
    //         file << "x first: " << x_first << " x last: " << x_last << " y first: " << y_first << " y last: " << y_last << endl;
    //         for (int j = 0; j < Ny_local; ++j) {
    //             for (int i = 0; i < Nx_local; ++i) {
    //                 file << v[IDX(i,j)] << " ";
    //             }
    //             file << endl;
    //         } 
    //         MPI_Barrier(cart_comm);
    //     }
    // }
    // file.close();

    // Solve Poisson problem
    cout << "Solving Poisson problem using SolverCG" << endl;
    double* outputArray_v = new double[Npts]();
    double* global_v = new double[Npts]();

    MPI_Barrier(mpiGridCommunicator->cart_comm);

    for (int j = y_first; j < y_last; ++j)
    {
        for (int i = x_first; i < x_last; ++i)
        {
            outputArray_v[IDX_GLOBAL(i, j)] = v[IDX(i - x_first, j - y_first)];
        }
    }

    MPI_Allreduce(outputArray_v, global_v, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Barrier(mpiGridCommunicator->cart_comm);

    cg->Solve(global_v, s);

    // Write the solution to file
    // Gather the arrays into actual size

    ofstream file;
    file.open("TestOutputSolverCG.txt");
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            file << s[IDX_GLOBAL(i,j)] << " ";
        }
        file << endl;
    }
    file.close();

    //Check it against the baseline
    cout << "Testing output files for SolverCG: " << endl;
    BOOST_CHECK(compareFiles("TestOutputSolverCG.txt", "BaselineOutputSolverCG.txt"));
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    // Finalise MPI
    
 
}