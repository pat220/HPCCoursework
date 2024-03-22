#include "MPIGridCommunicator.h"
#include "mpi.h"
#include <iostream>
using namespace std;

#define IDX(I, J) ((J) * Nx_local + (I))

/// @class MPIGridCommunicator
/// @brief Class to communicate between ranks in the grid. It holds 8 arrays to store the sending buffers and store the receiving buffers. The 4 receiving buffers are used by classes outside

/// @brief Constructor for the MPIGridCommunicator class. Hold the parameters and creates the buffers to send.
/// @param comm MPI communicator
/// @param Nx_local Number of subgrid points in x direction for the rank
/// @param Ny_local Number of subgrid points in y direction for the rank
/// @param start_x Starting index in x direction for the rank (1 or 0 depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param end_x Ending index in x direction for the rank (Nx_local - 1 or Nx_local depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param start_y Starting index in y direction for the rank (1 or 0 depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param end_y Ending index in y direction for the rank (Nx_local - 1 or Nx_local depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param coords Coordinates of the rank in the cartesian communicator
/// @param p Grid size (pxp grid)
MPIGridCommunicator::MPIGridCommunicator(MPI_Comm comm, int Nx_local, int Ny_local, int* coords, int p)
    : cart_comm(comm), Nx_local(Nx_local), Ny_local(Ny_local), coords(coords), p(p)
{
    // Set the start and end indices for the subgrid not to include 0 and Nx/Ny - 1 in global grid
    this->start_x = coords[1] == 0 ? 1 : 0;
    this->end_x = coords[1] == p - 1 ? Nx_local - 1 : Nx_local;
    this->start_y = coords[0] == p - 1 ? 1 : 0;
    this->end_y = coords[0] == 0 ? Ny_local - 1 : Ny_local;

    // Initialize the grid communicator
    this->sendBufferTop = new double[Nx_local];
    this->sendBufferBottom = new double[Nx_local];
    this->sendBufferLeft = new double[Ny_local];
    this->sendBufferRight = new double[Ny_local];
    
}

/// @brief Destructor for the MPIGridCommunicator class. Cleans the buffers
MPIGridCommunicator::~MPIGridCommunicator()
{
    delete[] sendBufferTop;
    delete[] sendBufferBottom;
    delete[] sendBufferLeft;
    delete[] sendBufferRight;
}

/// @brief Send and receive the edges of the subgrid to the neighbouring ranks
/// @param varArray Array of the variable to send
/// @param receiveBufferTop Buffer to store the received data from the top rank
/// @param receiveBufferBottom Buffer to store the received data from the bottom rank
/// @param receiveBufferLeft Buffer to store the received data from the left rank
/// @param receiveBufferRight Buffer to store the received data from the right rank
void MPIGridCommunicator::SendReceiveEdges(const double* varArray, double* receiveBufferTop, double* receiveBufferBottom, double* receiveBufferLeft, double* receiveBufferRight)
{
    int bottomRank, topRank, rightRank, leftRank;
    MPI_Cart_shift(cart_comm, 0, 1, &topRank, &bottomRank);
    MPI_Cart_shift(cart_comm, 1, 1, &leftRank, &rightRank);
    MPI_Request request;

    if (topRank == MPI_PROC_NULL && bottomRank == MPI_PROC_NULL && leftRank == MPI_PROC_NULL && rightRank == MPI_PROC_NULL) {
        return; // case for only 1 process
    }

    if (bottomRank != MPI_PROC_NULL) {
        // Send data to bottom members
        for (int i = 0; i < Nx_local; ++i) {
            sendBufferBottom[i] = varArray[IDX(i, 0)];
        }
        MPI_Isend(sendBufferBottom, Nx_local, MPI_DOUBLE, bottomRank, 0, cart_comm, &request);
    } 
    if (topRank != MPI_PROC_NULL) {
        // Receive data from top members 
        MPI_Irecv(receiveBufferTop, Nx_local, MPI_DOUBLE, topRank, 0, cart_comm, &request);
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    if (topRank != MPI_PROC_NULL) {
        // Send data to top members
        for (int i = 0; i < Nx_local; ++i) {
            sendBufferTop[i] = varArray[IDX(i, Ny_local-1)];
        }
        MPI_Isend(sendBufferTop, Nx_local, MPI_DOUBLE, topRank, 0, cart_comm, &request);
    }
    if (bottomRank != MPI_PROC_NULL) {
        // Receive data from bottom members 
        MPI_Irecv(receiveBufferBottom, Nx_local, MPI_DOUBLE, bottomRank, 0, cart_comm, &request);
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    if (rightRank != MPI_PROC_NULL) {
        // Send data to right members
        for (int j = 0; j < Ny_local; ++j) {
            sendBufferRight[j] = varArray[IDX(Nx_local-1, j)];
        }
        MPI_Isend(sendBufferRight, Ny_local, MPI_DOUBLE, rightRank, 0, cart_comm, &request);
    }
    if (leftRank != MPI_PROC_NULL) {
        // Receive data from left members 
        MPI_Irecv(receiveBufferLeft, Ny_local, MPI_DOUBLE, leftRank,0, cart_comm, &request);
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    if (leftRank != MPI_PROC_NULL) {
        // Send data to left members
        for (int j = 0; j < Ny_local; ++j) {
            sendBufferLeft[j] = varArray[IDX(0, j)];
        }
        MPI_Isend(sendBufferLeft, Ny_local, MPI_DOUBLE, leftRank, 0, cart_comm, &request);
    }
    if (rightRank != MPI_PROC_NULL) {
        // Receive data from right members 
        MPI_Irecv(receiveBufferRight, Ny_local, MPI_DOUBLE, rightRank, 0, cart_comm, &request);
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    return;
}


