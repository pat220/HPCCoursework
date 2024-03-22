#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

#define BOOST_TEST_MODULE UnitTestSCG
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

    MPIGridCommunicator* mpiGridCommunicatorCG = new MPIGridCommunicator(cart_comm, Nx_local, Ny_local, coords, p);
    SolverCG* cgCG = new SolverCG(Nx_local, Ny_local, dx, dy, mpiGridCommunicatorCG);

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

    cout << "Solving Poisson problem using SolverCG" << endl;
    cgCG->Solve(v, s);
    
    double* outputArray_s = new double[Npts]();
    double* global_s = new double[Npts]();

    MPI_Barrier(mpiGridCommunicatorCG->cart_comm);

    for (int j = y_first; j < y_last; ++j)
    {
        for (int i = x_first; i < x_last; ++i)
        {
            outputArray_s[IDX_GLOBAL(i, j)] = s[IDX(i - x_first, j - y_first)];
        }
    }

    MPI_Allreduce(outputArray_s, global_s, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicatorCG->cart_comm);
    MPI_Barrier(mpiGridCommunicatorCG->cart_comm);

    // Write the solution to file
    // Gather the arrays into actual size
    if (rank == 0){
        ofstream file;
        file.open("TestOutputSolverCG.txt");
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                file << global_s[IDX_GLOBAL(i,j)] << " ";
            }
            file << endl;
        }
        file.close();
        //Check it against the baseline
        cout << "Testing output files for SolverCG: " << endl;
        BOOST_CHECK(compareFiles("TestOutputSolverCG.txt", "BaselineOutputSolverCG.txt"));
    }

    
    // Finalise MPI within the struct
    delete cgCG;
    delete mpiGridCommunicatorCG;

    delete [] v;
    delete [] s;
    delete [] outputArray_s;
    delete [] global_s;
    
}