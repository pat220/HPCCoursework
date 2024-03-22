#pragma once

#include <string>
using namespace std;

#include "mpi.h"

class SolverCG;
class MPIGridCommunicator;

/// @class LidDrivenCavity
/// @headerfile LidDrivenCavity.h
/// @details The class LidDrivenCavity solves the vorticity-stream function formulation of the incompressible Navier-Stokes equations in 2D using the finite difference method. The domain is a square cavity with a moving lid.
/// The class uses the conjugate gradient method to solve the pressure Poisson equation. The class uses MPI for parallel computing and OMP for threads.
/// The class has methods to set the domain size, grid size, time step, final time, Reynolds number, and local variables.
/// The class has methods to initialise the solver, integrate the solution, write the solution to a file, and print the configuration.
/// The class has private methods to clean up the solver, clean up the buffers, initialise the buffers, update the dx and dy values, advance the solution, calculate the node vorticity, calculate the interior vorticity, and time advance the vorticity.
class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);
    void SetLocalVariables(int Nx, int Ny, int p, int *coords);

    void Initialise(MPI_Comm comm, int *coords, int p);
    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();

private:
    double* v   = nullptr;
    double* vnew = nullptr;
    double* s   = nullptr;
    double* tmp = nullptr;

    double dt   = 0.01;
    double T    = 1.0;
    int NSteps = ceil(T / dt);
    double dx;
    double dy;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;

    double rank = 0; // MPI rank default value
    double size = 0; // MPI size default value
    
    int* coords = nullptr;
    int p = 1;

    int Nx_local = Nx;
    int Ny_local = Ny;
    int Npts_local = Npts;

    int x_first;
    int x_last;
    int y_first;
    int y_last;

    int start_x;
    int end_x;
    int start_y;
    int end_y;

    double dxi;
    double dyi;
    double dx2i;
    double dy2i;
    
    SolverCG* cg = nullptr;
    SolverCG* cg_whole = nullptr;
    MPIGridCommunicator* mpiGridCommunicator = nullptr;

    double* receiveBufferTopV = nullptr;
    double* receiveBufferBottomV = nullptr;
    double* receiveBufferLeftV = nullptr;
    double* receiveBufferRightV = nullptr;

    double* receiveBufferTopS = nullptr;
    double* receiveBufferBottomS = nullptr;
    double* receiveBufferLeftS = nullptr;
    double* receiveBufferRightS = nullptr;

    void CleanUp();
    void CleanUpBuffers();
    void InitialiseBuffers();
    void UpdateDxDy();
    void Advance();
    void NodeVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads);
    void InteriorVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads);
    void TimeAdvanceVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads);
};

