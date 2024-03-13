#pragma once

#include <string>
using namespace std;

#include "mpi.h"

class SolverCG;

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

    void Initialise();
    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();
    void GetInfoMPI(MPI_Comm comm, int rank, int size, int* coords, int p);

private:
    double* v   = nullptr;
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
    MPI_Comm comm;
    int* coords;
    int p;

    int Nx_local;
    int Ny_local;
    
    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();
    void Advance();
};

