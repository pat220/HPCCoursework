#include <iostream>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>

class MPIGridCommunicator;
using namespace std;

#include <cblas.h>
#include "mpi.h"
#include "SolverCG.h"
#include "MPIGridCommunicator.h"

#define IDX(I, J) ((J) * Nx_local + (I))
#define IDX_GLOBAL(I, J) ((J) * Nx + (I))

SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy, MPIGridCommunicator* mpiGridCommunicator)
    : p(mpiGridCommunicator->p), coords(mpiGridCommunicator->coords), mpiGridCommunicator(mpiGridCommunicator)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    Nx_local = mpiGridCommunicator->Nx_local;
    Ny_local = mpiGridCommunicator->Ny_local;
    int n = Nx*Ny;

    r = new double[n];
    k = new double[n];
    z = new double[n];
    t = new double[n]; //temp

    InitialiseBuffers();

}

SolverCG::~SolverCG()
{
    delete[] r;
    delete[] k;
    delete[] z;
    delete[] t;
}

void SolverCG::Solve(double* b, double* x) {

    // Open the file for writing
    std::ofstream outputFile("testingB.txt", std::ios::app); // Append mode

    unsigned int n = Nx_local*Ny_local;
    int g;
    double alpha, alpha_global;
    double beta, beta_global;
    double eps;
    double tol = 0.001;

    // Reduce SUM to compute global dot of b and x and sqrt them
    double dot_local = cblas_ddot(n, b, 1, b, 1);
    double dot_global;
    MPI_Allreduce(&dot_local, &dot_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    eps = sqrt(dot_global);

    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return;
    }

    ApplyOperator(x, t); // parallelised inside

    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    ImposeBC(r); // parallelised inside

    cblas_daxpy(n, -1.0, t, 1, r, 1);
    Precondition(r, z); // parallelised inside
    cblas_dcopy(n, z, 1, k, 1);        // k_0 = r_0
    


    g = 0;
    do {
        g++;
        // Perform action of Nabla^2 * p
        ApplyOperator(k, t); // parallelised inside
        
        alpha = cblas_ddot(n, t, 1, k, 1);  // alpha = p_k^T A p_k
        MPI_Allreduce(&alpha, &alpha_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);

        alpha = cblas_ddot(n, r, 1, z, 1) / alpha_global; // compute alpha_k
        MPI_Allreduce(&alpha, &alpha_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);

        beta  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
        MPI_Allreduce(&beta, &beta_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);

        cblas_daxpy(n,  alpha_global, k, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha_global, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        // Compute local to global 2norm with dot products
        dot_local = cblas_ddot(n, r, 1, r, 1);
        MPI_Allreduce(&dot_local, &dot_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
        eps = sqrt(dot_global);
        if (eps < tol*tol) {
            break;
        }
        
        Precondition(r, z);

        beta = cblas_ddot(n, r, 1, z, 1) / beta_global;
        MPI_Allreduce(&beta, &beta_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
        // cout << rank << " beta2 is: in g = " << beta << endl;

        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta_global, k, 1, t, 1);
        cblas_dcopy(n, t, 1, k, 1);

    } while (g < 5000); // Set a maximum number of iterations

    if (g == 5000) {
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }

    cout << "Converged in " << g << " iterations. eps = " << eps << endl;

}

void SolverCG::ApplyOperator(double* in, double* out) {
    
    // Obtain start and end points to yield results from 1 to < Nx/Ny -1
    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    int rank;
    MPI_Comm_rank(mpiGridCommunicator->cart_comm, &rank);
    // cout << "Rank " << rank << " has start_x = " << start_x << " and end_x = " << end_x << " and start_y = " << start_y << " and end_y = " << end_y << endl;
    
    mpiGridCommunicator->SendReceiveEdges(in, receiveBufferTop, receiveBufferBottom, receiveBufferLeft, receiveBufferRight);

    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    for (int j = start_y; j < end_y; ++j) {
        for (int i = start_x; i < end_x; ++i) {

            double leftNeighborValueV = (coords[1] > 0 && i == 0) ? receiveBufferLeft[j] : in[IDX(i - 1, j)];
            double rightNeighborValueV = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRight[j] : in[IDX(i + 1, j)];
            double botomNeighborValueV = (coords[0] < p - 1 && j == 0) ? receiveBufferBottom[i] : in[IDX(i, j - 1)];
            double topNeighborValueV = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTop[i] : in[IDX(i, j + 1)];

            out[IDX(i,j)] = ( -     leftNeighborValueV
                              + 2.0*in[IDX(i,   j)]
                              -     rightNeighborValueV)*dx2i
                          + ( -     topNeighborValueV
                              + 2.0*in[IDX(i,   j)]
                              -     botomNeighborValueV)*dy2i;

            // cout << rank << " has out[" << i << "][" << j << "] = " << out[IDX(i,j)] << endl;
          
        }
    }
}

void SolverCG::Precondition(double* in, double* out) {
    int i, j;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 2.0*(dx2i + dy2i);

    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    for (i = start_x; i < end_x; ++i) {
        for (j = start_y; j < end_y; ++j) {
            out[IDX(i,j)] = in[IDX(i,j)]/factor;
        }
    }

    // Boundaries
    if (coords[0] == 0) //top
    {   
        for (int i = 0; i < Nx_local; ++i) {
            out[IDX(i, Ny_local-1)] = in[IDX(i, Ny_local-1)];
        }
    }
    else if (coords[0] == p - 1) //bottom
    {
        for (int i = 0; i < Nx_local; ++i) {
            out[IDX(i, 0)] = in[IDX(i,0)];
        }
    }
    
    if (coords[1] == 0) //left
    {
        for (int j = 0; j < Ny_local; ++j) {
            out[IDX(0, j)] = in[IDX(0, j)];
        }
    }
    else if (coords[1] == p - 1) //right
    {
        for (int j = 0; j < Ny_local; ++j) {
            out[IDX(Nx_local - 1, j)] = in[IDX(Nx_local - 1, j)];
        }
    }
}

void SolverCG::InitialiseBuffers()
{
    CleanUpBuffers();

    receiveBufferTop = new double[Nx_local];
    receiveBufferBottom = new double[Nx_local];
    receiveBufferLeft = new double[Ny_local];
    receiveBufferRight = new double[Ny_local];
}

void SolverCG::CleanUpBuffers()
{
    if (receiveBufferTop)
    {
        delete[] receiveBufferTop;
        delete[] receiveBufferBottom;
        delete[] receiveBufferLeft;
        delete[] receiveBufferRight;

    }
}

void SolverCG::ImposeBC(double* inout) {
    // Boundaries

    if (coords[0] == 0) // top
    {
        for (int i = 0; i < Nx_local; ++i) {
            inout[IDX(i, Ny_local-1)] = 0.0;
        }
    }
    else if (coords[0] == p - 1) // bottom
    {
        for (int i = 0; i < Nx_local; ++i) {
            inout[IDX(i, 0)] = 0.0;
        }
    }
    
    if (coords[1] == 0) // left
    {
        for (int j = 0; j < Ny_local; ++j) {
            inout[IDX(0, j)] = 0.0;
        }
    }
    else if (coords[1] == p - 1) // right
    {
        for (int j = 0; j < Ny_local; ++j) {
            inout[IDX(Nx_local-1, j)] = 0.0;
        }
    }
}