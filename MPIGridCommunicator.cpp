#include "MPIGridCommunicator.h"
#include "mpi.h"
#include <iostream>
using namespace std;


#define IDX(I, J) ((J) * Nx_local + (I))

MPIGridCommunicator::MPIGridCommunicator(MPI_Comm comm, int Nx_local, int Ny_local, int start_x, int end_x, int start_y, int end_y, int* coords, int p)
    : cart_comm(comm), Nx_local(Nx_local), Ny_local(Ny_local), start_x(start_x), end_x(end_x), start_y(start_y), end_y(end_y), coords(coords), p(p)
{
    // Initialize the grid communicator
    this->sendBufferTop = new double[Nx_local];
    this->sendBufferBottom = new double[Nx_local];
    this->sendBufferLeft = new double[Ny_local];
    this->sendBufferRight = new double[Ny_local];

    // cout << "Rank " << coords[0] << " " << coords[1] << " Nx_local " << Nx_local << " Ny_local " << Ny_local << " start_x " << start_x << " end_x " << end_x << " start_y " << start_y << " end_y " << end_y << endl;
}

MPIGridCommunicator::~MPIGridCommunicator()
{
    delete[] sendBufferTop;
    delete[] sendBufferBottom;
    delete[] sendBufferLeft;
    delete[] sendBufferRight;
}

void MPIGridCommunicator::SendReceiveEdges(const double* varArray, double* receiveBufferTop, double* receiveBufferBottom, double* receiveBufferLeft, double* receiveBufferRight)
{
    int bottomRank, topRank, rightRank, leftRank;
    MPI_Cart_shift(cart_comm, 0, 1, &topRank, &bottomRank);
    MPI_Cart_shift(cart_comm, 1, 1, &leftRank, &rightRank);
    MPI_Request request;

    if (bottomRank != MPI_PROC_NULL) {
        // Send data to bottom members
        for (int i = 0; i < Nx_local; ++i) {
            sendBufferBottom[i] = varArray[IDX(i, end_y)];
        }
        MPI_Isend(sendBufferBottom, Nx_local, MPI_DOUBLE, bottomRank, 0, cart_comm, &request);
    } 
    if (topRank != MPI_PROC_NULL) {
        // Receive data from top members 
        MPI_Irecv(receiveBufferTop, Nx_local, MPI_DOUBLE, topRank, 0, cart_comm, &request);
    }

    if (topRank != MPI_PROC_NULL) {
        // Send data to top members
        for (int i = 0; i < Nx_local; ++i) {
            sendBufferTop[i] = varArray[IDX(i, start_y)];
        }
        MPI_Isend(sendBufferTop, Nx_local, MPI_DOUBLE, topRank, 0, cart_comm, &request);
    }
    if (bottomRank != MPI_PROC_NULL) {
        // Receive data from bottom members 
        MPI_Irecv(receiveBufferBottom, Nx_local, MPI_DOUBLE, bottomRank, 0, cart_comm, &request);
    }
    
    if (rightRank != MPI_PROC_NULL) {
        // Send data to right members
        for (int j = 0; j < Ny_local; ++j) {
            sendBufferRight[j] = varArray[IDX(end_x, j)];
        }
        MPI_Isend(sendBufferRight, Ny_local, MPI_DOUBLE, rightRank, 0, cart_comm, &request);
    }
    if (leftRank != MPI_PROC_NULL) {
        // Receive data from left members 
        MPI_Irecv(receiveBufferLeft, Ny_local, MPI_DOUBLE, leftRank,0, cart_comm, &request);
    }

    if (leftRank != MPI_PROC_NULL) {
        // Send data to left members
        for (int j = 0; j < Ny_local; ++j) {
            sendBufferLeft[j] = varArray[IDX(start_x, j)];
        }
        MPI_Isend(sendBufferLeft, Ny_local, MPI_DOUBLE, leftRank, 0, cart_comm, &request);
    }
    if (rightRank != MPI_PROC_NULL) {
        // Receive data from right members 
        MPI_Irecv(receiveBufferRight, Ny_local, MPI_DOUBLE, rightRank, 0, cart_comm, &request);
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);
}


