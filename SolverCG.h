#pragma once

/// @headerfile SolverCG.h
/// @brief The SolverCG class solves a linear system of equations using the conjugate gradient method.
/// @details SolverCG class has methods to solve the linear system of equations, apply the operator, precondition the system,
/// impose boundary conditions, initialise buffers, and clean up buffers. It uses MPI for parallel computing and teh class MpiGridCommunicator for communication between ranks.

class MPIGridCommunicator;
class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy, MPIGridCommunicator* mpiGridCommunicator);
    ~SolverCG();

    void Solve(double* b, double* x);

private:
    double dx;
    double dy;
    double dx2i;
    double dy2i;
    double factor;
    int Nx;
    int Ny;
    int Nx_local;
    int Ny_local;
    double* r;
    double* k;
    double* z;
    double* t;

    const int p;
    const int* coords;
    
    MPIGridCommunicator* mpiGridCommunicator = nullptr;

    double* receiveBufferTop = nullptr;
    double* receiveBufferBottom = nullptr;
    double* receiveBufferLeft = nullptr;
    double* receiveBufferRight = nullptr;

    void ApplyOperator(double* k, double* t, int threadid, int nthreads);
    void Precondition(double* k, double* t);
    void ImposeBC(double* k);
    void InitialiseBuffers();
    void CleanUpBuffers();

};

