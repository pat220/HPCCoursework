#pragma once

#include <string>
using namespace std;

#include "mpi.h"

class SolverCG;
class MPIGridCommunicator;

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
    void NodeVorticity();
    void InteriorVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads);
    void TimeAdvanceVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads);
};

