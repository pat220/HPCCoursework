#include <iostream>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"
#include "mpi.h"
#include <omp.h>
#include <cmath> // Include the cmath library for sqrt function

/// @file LidDrivenCavitySolver.cpp
/// @brief Solving the vorticity-stream function formulation of the
/// incompressible Navier-Stokes equations in 2D using the finite difference method.
/// @param argc Number of command-line arguments.
/// @param argv Array of command-line arguments.
/// @return 0 if the program terminates successfully, 1 otherwise.

int main(int argc, char **argv)
{

    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1.0),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        // Print help message
        cout << opts << endl;
        return 0;
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processes and the rank of the current process
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Print error message if the number of processes is not a perfect square or exceeds 16
    if (sqrt(size) != floor(sqrt(size)) || size > 16) {
        if (rank == 0) {
            cout << "The number of processes must be a perfect square and less or equal to 16" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Create a Cartesian topology
    int p = sqrt(size); // Number of processes in each dimension
    int dims[2] = {p, p};  // pxp grid
    int periods[2] = {0, 0};  // No periodicity
    int coords[2];
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Create the solver object
    LidDrivenCavity* solver = new LidDrivenCavity();

    // Set the parameters
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());
    solver->SetLocalVariables(vm["Nx"].as<int>(), vm["Ny"].as<int>(), p, coords);

    // Initialise the solver
    solver->Initialise(cart_comm, coords, p);

    // Print the parameters
    solver->WriteSolution("ic.txt");

    // Integrate in time    
    solver->Integrate();

    // Write the final solution
    solver->WriteSolution("final.txt");

    // Release the allocated memory
    delete solver; 

    // Finalise MPI
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;

}
