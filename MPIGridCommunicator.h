#ifndef MPIGRIDCOMMUNICATOR_H
#define MPIGRIDCOMMUNICATOR_H

#include "mpi.h"
// Include necessary MPI headers here

class MPIGridCommunicator {
public:
    // Constructor
    MPIGridCommunicator(MPI_Comm comm, int Nx_local, int Ny_local, int* coords, int p);
    // Destructor
    ~MPIGridCommunicator();
    
    // Send and receive edges
    void SendReceiveEdges(const double* varArray,
    double* receiveBufferTop, double* receiveBufferBottom, double* receiveBufferLeft, double* receiveBufferRight);

    const MPI_Comm cart_comm;

    const int Nx_local;
    const int Ny_local;
    int start_x;
    int end_x;
    int start_y;
    int end_y;
    const int* coords;
    const int p;

private:
    double* sendBufferTop;
    double* sendBufferBottom;
    double* sendBufferLeft;
    double* sendBufferRight;
};

#endif // MPIGRIDCOMMUNICATOR_H
