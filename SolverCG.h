#pragma once

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

    double* receiveBufferTopV = nullptr;
    double* receiveBufferBottomV = nullptr;
    double* receiveBufferLeftV = nullptr;
    double* receiveBufferRightV = nullptr;

    double* receiveBufferTopS = nullptr;
    double* receiveBufferBottomS = nullptr;
    double* receiveBufferLeftS = nullptr;
    double* receiveBufferRightS = nullptr;

    void ApplyOperator(double* k, double* t);
    void Precondition(double* k, double* t);
    void ImposeBC(double* k);
    void InitialiseBuffers();
    void CleanUpBuffers();

};

