#include <iostream>
#include <fstream>
#include <string>

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

bool compareFiles(const string& file1, const string& file2) {
    ifstream stream1(file1);
    ifstream stream2(file2);

    if (!stream1.is_open() || !stream2.is_open()) {
        cerr << "Error opening files for comparison." << endl;
        return false;
    }

    string line1, line2;
    while (getline(stream1, line1) && getline(stream2, line2)) {
        if (line1 != line2) {
            cout << "Files are not equal." << endl;
            return false;
        }
    }

    if (stream1.eof() && stream2.eof()) {
        cout << "Files are equal." << endl;
        return true;
    }

    cout << "Files are not equal (different number of lines)." << stream1.eof() << stream2.eof() << endl;
    return false;
}

int main(int argc, char **argv)
{
    LidDrivenCavity* solver = new LidDrivenCavity();

    // Hardcoded values
    double Lx = 1.0;
    double Ly = 1.0;
    int Nx = 9;
    int Ny = 9;
    double dt = 0.01;
    double T = 1.0;
    double Re = 10.0;

    solver->SetDomainSize(Lx, Ly);
    solver->SetGridSize(Nx, Ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);

    solver->PrintConfiguration();

    solver->Initialise();

    solver->WriteSolution("ic.txt");

    solver->Integrate();

    solver->WriteSolution("Test1.txt");

    delete solver; // Release the allocated memory
    
    // Compare Test1.txt with Baseline.txt
    if (compareFiles("Test1.txt", "Baseline.txt")) {
        cout << "The solutions match." << endl;
    } else {
        cout << "The solutions do not match." << endl;
    }

    return 0;
}
