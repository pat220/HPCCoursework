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
#include <omp.h>
#include "SolverCG.h"
#include "MPIGridCommunicator.h"

#define IDX(I, J) ((J) * Nx_local + (I))
#define IDX_GLOBAL(I, J) ((J) * Nx + (I))

/// @class SolverCG
/// @brief Class to solve the mathematical expression using the Conjugate Gradient method

/// @brief Construct a new SolverCG::SolverCG object
/// @param pNx Number of grid points in x direction
/// @param pNy Number of grid points in y direction
/// @param pdx Grid spacing in x direction
/// @param pdy Grid spacing in y direction
/// @param mpiGridCommunicator Communicator object, another class used to communicate between ranks
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

/// @brief Destroy the SolverCG::SolverCG object
SolverCG::~SolverCG()
{
    delete[] r;
    delete[] k;
    delete[] z;
    delete[] t;
}

/// @brief Solve the mathematical expression using the Conjugate Gradient method
/// @param b Right hand side of the equation (vorticity)
/// @param x Solution to the equation (stream function)
void SolverCG::Solve(double* b, double* x) {

    // Initialise variables
    unsigned int n = Nx_local*Ny_local;
    int g;
    double alpha, alpha_global;
    double beta, beta_global;
    double eps;
    double tol = 0.001;

    // Obtain norm of b by doing dot product locally, then summing globally and taking square root
    double dot_local = cblas_ddot(n, b, 1, b, 1);
    double dot_global;
    MPI_Allreduce(&dot_local, &dot_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    eps = sqrt(dot_global);

    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return;
    }

    ApplyOperator(x, t, 0, 1); // parallelised inside

    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    ImposeBC(r); // parallelised inside

    cblas_daxpy(n, -1.0, t, 1, r, 1);
    Precondition(r, z); // parallelised inside
    cblas_dcopy(n, z, 1, k, 1);        // k_0 = r_0

    // Start the CG loop parallelised with OpenMP
    int nthreads, threadid;
    bool shouldBreak = false;
    #pragma omp parallel default(shared) private(threadid, g)
    {
        threadid = omp_get_thread_num();
        if(threadid==0) {
            nthreads = omp_get_num_threads();
            }
        g = 0;
        do {
            g++;

            // Perform action of Nabla^2 * p
            #pragma omp barrier
            ApplyOperator(k, t, threadid, nthreads); // parallelised inside
            #pragma omp barrier

            if(threadid == 0) {
                alpha = cblas_ddot(n, t, 1, k, 1);  // alpha = p_k^T A p_k
                MPI_Allreduce(&alpha, &alpha_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm); // sum globally

                alpha = cblas_ddot(n, r, 1, z, 1) / alpha_global; // compute alpha_k
                MPI_Allreduce(&alpha, &alpha_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm); // sum globally

                beta = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
                MPI_Allreduce(&beta, &beta_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm); // sum globally

                cblas_daxpy(n, alpha_global, k, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
                cblas_daxpy(n, -alpha_global, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

                // Compute local to global norm with dot products as earlier
                dot_local = cblas_ddot(n, r, 1, r, 1);
                MPI_Allreduce(&dot_local, &dot_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
                eps = sqrt(dot_global);
                if (eps < tol * tol) {
                    shouldBreak = true;
                }

                Precondition(r, z);

                beta = cblas_ddot(n, r, 1, z, 1) / beta_global;
                MPI_Allreduce(&beta, &beta_global, 1, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm); // sum globally
                // cout << rank << " beta2 is: in g = " << beta << endl;

                cblas_dcopy(n, z, 1, t, 1);
                cblas_daxpy(n, beta_global, k, 1, t, 1);
                cblas_dcopy(n, t, 1, k, 1);
            }
            #pragma omp barrier
            if(shouldBreak){ break; }
            cout << g << endl;
        } while (g < 5000); // Set a maximum number of iterations

        if (threadid == 0 && g == 5000) {
            cout << "FAILED TO CONVERGE" << endl;
            exit(-1);
        }
    }
    // cout << "Converged in " << g << " iterations. eps = " << eps << endl; // commented out to test time only
}

/// @brief Apply the operator to the input vector
/// @param in Input vector
/// @param out Output vector
/// @param threadid Thread ID
/// @param nthreads Number of threads
void SolverCG::ApplyOperator(double* in, double* out, int threadid, int nthreads) {

    // Obtain start and end points to yield results from 1 to < Nx/Ny -1
    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    if(threadid == 0){
        int rank;
        MPI_Comm_rank(mpiGridCommunicator->cart_comm, &rank);
        // cout << "Rank " << rank << " has start_x = " << start_x << " and end_x = " << end_x << " and start_y = " << start_y << " and end_y = " << end_y << endl;
        mpiGridCommunicator->SendReceiveEdges(in, receiveBufferTop, receiveBufferBottom, receiveBufferLeft, receiveBufferRight);
    }
    
    #pragma omp barrier

    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    for (int j = start_y+threadid; j < end_y; j += nthreads) {
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
        }
    }
}

/// @brief Apply the preconditioner to the input vector
/// @param in Input vector
/// @param out Output vector
/// @brief Apply a factor to the inner points
void SolverCG::Precondition(double* in, double* out) {

    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 1.0/2.0/(dx2i + dy2i);

    // Obtain start and end points to yield results from 1 to < Nx/Ny -1
    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    // Inner points
    for (int j = start_y; j < end_y; ++j) {
        for (int i = start_x; i < end_x; ++i) {
            out[IDX(i,j)] = in[IDX(i,j)]*factor;
        }
    }

    // Boundaries
    if (coords[0] == 0) //top
    {   
        for (int i = 0; i < Nx_local; ++i) {
            out[IDX(i, Ny_local-1)] = in[IDX(i, Ny_local-1)];
        }
    }
    
    if (coords[0] == p - 1) //bottom
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
    
    if (coords[1] == p - 1) //right
    {
        for (int j = 0; j < Ny_local; ++j) {
            out[IDX(Nx_local - 1, j)] = in[IDX(Nx_local - 1, j)];
        }
    }
}

/// @brief Initialise the buffers
void SolverCG::InitialiseBuffers()
{
    CleanUpBuffers();

    receiveBufferTop = new double[Nx_local];
    receiveBufferBottom = new double[Nx_local];
    receiveBufferLeft = new double[Ny_local];
    receiveBufferRight = new double[Ny_local];
}

/// @brief Clean up the buffers
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

/// @brief Impose the boundary conditions
/// @param inout Input and output vector. The boundary conditions are imposed on this vector
void SolverCG::ImposeBC(double* inout) {
    
    // Boundaries
    if (coords[0] == 0) // top
    {
        for (int i = 0; i < Nx_local; ++i) {
            inout[IDX(i, Ny_local-1)] = 0.0;
        }
    }
    
    if (coords[0] == p - 1) // bottom
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
    
    if (coords[1] == p - 1) // right
    {
        for (int j = 0; j < Ny_local; ++j) {
            inout[IDX(Nx_local-1, j)] = 0.0;
        }
    }
}