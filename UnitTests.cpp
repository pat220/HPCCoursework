#include <iostream>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"


/**
* @file LidDrivenCavitySolver.cpp
* @brief Solving the vorticity-stream function formulation of the
* incompressible Navier-Stokes equations in 2D using the finite difference method.
*
* @param argc Number of command-line arguments.
* @param argv Array of command-line arguments.
*/

int main(int argc, char **argv)
{
    LidDrivenCavity* solver = new LidDrivenCavity();

    // Hardcoded values
    double Lx = 1.0;
    double Ly = 1.0;
    int Nx = 201;
    int Ny = 201;
    double dt = 0.005;
    double T = 50.0;
    double Re = 1000.0;

    solver->SetDomainSize(Lx, Ly);
    solver->SetGridSize(Nx, Ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);

    solver->PrintConfiguration();

    solver->Initialise();

    solver->WriteSolution("ic.txt");

    solver->Integrate();

    solver->WriteSolution("Baseline.txt");

    delete solver; // Release the allocated memory

    return 0;
}
