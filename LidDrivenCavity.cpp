#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>

using namespace std;

#include <cblas.h>

#define IDX(I, J) ((J) * Nx_local + (I))
#define IDX_GLOBAL(I, J) ((J) * Nx + (I))

#include "LidDrivenCavity.h"
#include "SolverCG.h"
#include "MPIGridCommunicator.h"
#include "mpi.h"
#include <omp.h>

/// @class LidDrivenCavity
/// @brief  Class for the lid-driven cavity problem
/// @note   The class solves the lid-driven cavity problem using the vorticity-stream function formulation and the conjugate gradient solver in class SolverCG


/// @brief  Constructor
/// @note   Initialises the class
LidDrivenCavity::LidDrivenCavity()
{
}

/// @brief  Destructor
/// @note   Cleans up the class by calling the CleanUp function
LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

/// @brief  Set the domain size
/// @param  xlen    Length of the domain in the x-direction
/// @param  ylen    Length of the domain in the y-direction
/// @note   This function sets the domain size and updates the dx and dy values calling the UpdateDxDy function
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

/// @brief  Set the grid size
/// @param  nx  Number of grid points in the x-direction
/// @param  ny  Number of grid points in the y-direction
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
}

/// @brief  Set the time step size
/// @param  deltat  Time step size
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

/// @brief  Set the final time
/// @param  finalt  Final time
void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

/// @brief  Set the Reynolds number and setting the inverse of it to multiply it in some amthematical expresssions instead of dividing it for faster computation
/// @param  re  Reynolds number
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0 / re;
}

/// @brief  Set the local variables for each process. Sets the subgrid size of each rank based on their coordinates in the cartesian grid and how big the grid pxp is
/// @param  Nx      Number of grid points in the x-direction
/// @param  Ny      Number of grid points in the y-direction
/// @param  p       Square root of number of processes (pxp grid)
/// @param  coords  Coordinates of the current process in the cartesian grid
void LidDrivenCavity::SetLocalVariables(int Nx, int Ny, int p, int* coords)
{
    // Calculate starting and ending points of each process
    // Divide number of points over the number of processos and get minimum number of points each needs
    double extra_x = Nx % p;
    double extra_y = Ny % p;
    int min_points_x = (Nx - extra_x) / p;
    int min_points_y = (Ny - extra_y) / p;

    // Calculate the starting and ending points of each process in global coordinates
    if (coords[1] < extra_x)
    {
        min_points_x++;
        this->x_first = coords[1] * min_points_x; // global coordinate
        this->x_last = (coords[1] + 1) * min_points_x; // global coordinate
        this->Nx_local = x_last - x_first;
    }
    else
    {
        this->x_first = (min_points_x + 1) * extra_x + min_points_x * (coords[1] - extra_x); // global coordinate
        this->x_last = (min_points_x + 1) * extra_x + min_points_x * (coords[1] - extra_x + 1); // global coordinate
        this->Nx_local = x_last - x_first;
    }

    if (coords[0] < extra_y)
    {
        min_points_y++;
        this->y_last = Ny - (coords[0]) * min_points_y; // global coordinate
        this->y_first = Ny - (coords[0] + 1) * min_points_y; // global coordinate
        this->Ny_local = y_last - y_first;
    }
    else
    {
        this->y_last = Ny - ((min_points_y + 1) * extra_y + min_points_y * (coords[0] - extra_y)); // global coordinate
        this->y_first = Ny - ((min_points_y + 1) * extra_y + min_points_y * (coords[0] - extra_y + 1)); // global coordinate
        this->Ny_local = y_last - y_first;
    }

    UpdateDxDy();
}

/// @brief  Initialise the MPI grid communicator and the conjugate gradient solver. Also initialise the local arrays for each process (v, vnew, s), the buffers for the communication and the start and end points of each subgrid so that in the global grid the mathematical expressions go from 1 to < Nx/Ny - 1
/// @param  comm    MPI communicator
/// @param  coords  Coordinates of the current process in the cartesian grid
/// @param  p       Square root of number of processes (pxp grid)
void LidDrivenCavity::Initialise(MPI_Comm comm, int *coords, int p)
{
    CleanUp();
    InitialiseBuffers();
    this->NSteps = ceil(T / dt);

    
    // Set up the MPI grid communicator and the conjugate gradient solver
    mpiGridCommunicator = new MPIGridCommunicator(comm, Nx_local, Ny_local, coords, p);

    // Set the start and ending points for the interior vorticity so that it goes from 1 to < Nx/Ny - 1 from mpiGridCommunicator
    this->start_x = mpiGridCommunicator->start_x;
    this->end_x = mpiGridCommunicator->end_x;
    this->start_y = mpiGridCommunicator->start_y;
    this->end_y = mpiGridCommunicator->end_y;


    cg = new SolverCG(Nx_local, Ny_local, dx, dy, mpiGridCommunicator);
    this->coords = coords;
    this->p = p;

    // Set up the local arrays
    v = new double[Npts_local](); // local
    vnew = new double[Npts_local](); // local
    s = new double[Npts_local](); // local
    
}

/// @brief  Integrate the vorticity field by advancing it in time as many times as NSteps
/// @note   SolverCG is called to solve for the stream function inside
void LidDrivenCavity::Integrate()
{   
    // Integrate the vorticity field in time
    for (int t = 0; t < NSteps; ++t)
    {
        // std::cout << "Step: " << setw(8) << t
        //           << "  Time: " << setw(8) << t * dt
        //           << std::endl;
        Advance();
    }
}

/// @brief  Write the solution to a file
/// @param  file    File name
/// @note   The solution is written to a file with the format: x y v s u0 u1, where x and y are the coordinates, v is the vorticity, s is the stream function, u0 is the x-component of the velocity and u1 is the y-component of the velocity
///         u0 and u1 are computed here. The file is written by the rank 0 process
void LidDrivenCavity::WriteSolution(std::string file)
{   
    int rank;
    MPI_Comm_rank(mpiGridCommunicator->cart_comm, &rank);

    // Start and ending points for the interior vorticity so that it goes from 1 to < Nx/Ny - 1 from mpiGridCommunicator
    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    // Create the x-direction and y-direciton velocity arrays
    double *u0 = new double[Npts_local]();
    double *u1 = new double[Npts_local]();

    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS); // Initialised this with S not to store more data

    // Calculate the x-direction and y-direction velocity arrays
    for (int j = start_y; j < end_y; ++j)
    {
        for (int i = start_x; i < end_x; ++i)
        {   
            double rightNeighborValue = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j] : s[IDX(i + 1, j)];
            double topNeighborValue = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i] : s[IDX(i, j + 1)];

            u0[IDX(i, j)] = (topNeighborValue - s[IDX(i, j)]) / dy;
            u1[IDX(i, j)] = -(rightNeighborValue - s[IDX(i, j)]) / dx;
        }
    }

    // Set the boundary conditions for the x-direction velocity
    if (coords[0] == 0){
        for (int i = 0; i < Nx_local; ++i)
        {
            u0[IDX(i, Ny_local - 1)] = U;
        }
    } 

    // Create the output arrays, where the outputArray is the global array used by each rank
    // (populated with 0 everywhere but where the local array is) and the global array is the array with the overall values of the grid
    double* outputArray_u0 = new double[Npts]();
    double* outputArray_u1 = new double[Npts]();
    double* outputArray_v = new double[Npts]();
    double* outputArray_s = new double[Npts]();

    double* global_u0 = new double[Npts]();
    double* global_u1 = new double[Npts]();
    double* global_v = new double[Npts]();
    double* global_s = new double[Npts]();

    MPI_Request request;
    MPI_Barrier(mpiGridCommunicator->cart_comm);

    // Send the local arrays to the output arrays
    for (int j = y_first; j < y_last; ++j)
    {
        for (int i = x_first; i < x_last; ++i)
        {
            outputArray_u0[IDX_GLOBAL(i, j)] = u0[IDX(i - x_first, j - y_first)];
            outputArray_u1[IDX_GLOBAL(i, j)] = u1[IDX(i - x_first, j - y_first)];
            outputArray_v[IDX_GLOBAL(i, j)] = v[IDX(i - x_first, j - y_first)];
            outputArray_s[IDX_GLOBAL(i, j)] = s[IDX(i - x_first, j - y_first)];
        }
    }

    MPI_Barrier(mpiGridCommunicator->cart_comm);
    
    // Reduce the output arrays to the global arrays
    MPI_Allreduce(outputArray_u0, global_u0, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Allreduce(outputArray_u1, global_u1, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Allreduce(outputArray_v, global_v, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Allreduce(outputArray_s, global_s, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Barrier(mpiGridCommunicator->cart_comm);


    // Write the solution to a file
    if (rank == 0){
        std::ofstream f(file.c_str());
        std::cout << "Writing file " << file << std::endl;
        int k = 0;
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                k = IDX_GLOBAL(i, j);

                // og
                f << i * dx << " " << j * dy << " " << global_v[k] << " " << global_s[k]
                        << " " << global_u0[k] << " " << global_u1[k] << std::endl;

            }
            f << std::endl;
        }
        f.close();

        delete[] u0;
        delete[] u1;
        delete[] outputArray_u0;
        delete[] outputArray_u1;
        delete[] outputArray_v;
        delete[] outputArray_s;
        delete[] global_u0;
        delete[] global_u1;
        delete[] global_v;
        delete[] global_s;
    }

}

/// @brief  Print the configuration of the simulation
/// @note   Checks if the time-step restriction is satisfied and prints the configuration of the simulation
void LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << Nx << " x " << Ny << endl;
    cout << "Spacing:   " << dx << " x " << dy << endl;
    cout << "Length:    " << Lx << " x " << Ly << endl;
    cout << "Grid pts:  " << Npts << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T / dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25)
    {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy * Re << endl; // 0.25 * dx * dy / nu
        exit(-1);
    }
}

/// @brief  Clean up the class by deleting the arrays and the objects
void LidDrivenCavity::CleanUp()
{
    if (v)
    {
        delete[] v;
        delete[] vnew;
        delete[] s;
        delete cg;
        delete mpiGridCommunicator;
    }
}

/// @brief  Update the dx and dy values. Also update the total number of points in the local grid and global grid. It also calculates 1/dx, 1/dy, 1/dx^2 and 1/dy^2 for faster computation in the mathematical expressions
void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx - 1);
    dy = Ly / (Ny - 1);
    Npts_local = Nx_local * Ny_local;
    Npts = Nx * Ny;

    this->dxi = 1.0 / dx;
    this->dyi = 1.0 / dy;
    this->dx2i = 1.0 / dx / dx;
    this->dy2i = 1.0 / dy / dy;
}

/// @brief  Calculates the vorticity at the boundary nodes, at the interior nodes and advance the vorticity in time. It also solves for the stream function using the conjugate gradient solver in SolveCG
/// @note   The vorticity at the boundary nodes is calculated first, then the vorticity at the interior nodes is calculated and advanced in time
void LidDrivenCavity::Advance()
{
    // Parallelising using OpenMP
    int nthreads, threadid;
    #pragma omp parallel default(shared) private(threadid)
    {
        threadid = omp_get_thread_num();
        if(threadid==0) {
            nthreads = omp_get_num_threads();
        }

        NodeVorticity(start_x, end_x, start_y, end_y, threadid, nthreads);
        #pragma omp barrier

        InteriorVorticity(start_x, end_x, start_y, end_y, threadid, nthreads); // change vnew
        #pragma omp barrier

        TimeAdvanceVorticity(start_x, end_x, start_y, end_y, threadid, nthreads); // change v with vnew
        #pragma omp barrier

    }

    cg->Solve(v, s);
}

/// @brief  Initialise the buffers for the communication
void LidDrivenCavity::InitialiseBuffers()
{
    CleanUpBuffers();

    receiveBufferTopV = new double[Nx_local];
    receiveBufferBottomV = new double[Nx_local];
    receiveBufferLeftV = new double[Ny_local];
    receiveBufferRightV = new double[Ny_local];

    receiveBufferTopS = new double[Nx_local];
    receiveBufferBottomS = new double[Nx_local];
    receiveBufferLeftS = new double[Ny_local];
    receiveBufferRightS = new double[Ny_local];

}

/// @brief  Clean up the buffers for the communication
void LidDrivenCavity::CleanUpBuffers()
{
    if (receiveBufferTopS)
    {
        delete[] receiveBufferTopV;
        delete[] receiveBufferBottomV;
        delete[] receiveBufferLeftV;
        delete[] receiveBufferRightV;

        delete[] receiveBufferTopS;
        delete[] receiveBufferBottomS;
        delete[] receiveBufferLeftS;
        delete[] receiveBufferRightS;
    }
}

/// @brief  Calculate the vorticity at the boundary nodes with its corresponding mathematical expression
/// @param  threadid    Thread id
/// @param  nthreads    Number of threads
void LidDrivenCavity::NodeVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads)
{   

    // Corners first
    if (coords[0] == 0 && coords[1] == 0) // top left
    {
        for (int i = startX+threadid; i < endX; i += nthreads)
        {
            vnew[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = startY+threadid; j < endY; j += nthreads)
        {
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    #pragma omp barrier
    
    if (coords[0] == 0 && coords[1] == p - 1) // top right
    {
        for (int i = startX+threadid; i < endX; i += nthreads)
        {
            vnew[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = startY+threadid; j < endY; j += nthreads)
        {
            vnew[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }
    #pragma omp barrier
    
    if (coords[0] == p - 1 && coords[1] == 0) // bottom left
    {
        for (int i = startX+threadid; i < endX; i += nthreads)
        {
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = startY+threadid; j < endY; j += nthreads)
        {
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    #pragma omp barrier
    
    if (coords[0] == p - 1 && coords[1] == p - 1) // bottom right
    {
        for (int i = startX; i < endX; i += nthreads)
        {
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = startY+threadid; j < endY; j += nthreads)
        {
            vnew[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }
    #pragma omp barrier


    // Edges
    if (coords[0] == 0 && !(coords[1] == 0 || coords[1] == p - 1)) // top
    {
        for (int i = startX+threadid; i < endX; i += nthreads)
        {
            vnew[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
    }
    #pragma omp barrier
    
    if (coords[0] == p - 1 && !(coords[1] == 0 || coords[1] == p - 1)) //bottom
    {
        for (int i = startX+threadid; i < endX; i += nthreads)
        {
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
    }
    #pragma omp barrier
    
    if (coords[1] == 0 && !(coords[0] == 0 || coords[0] == p - 1)) // left
    {
        for (int j = startY+threadid; j < endY; j += nthreads)
        {
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    #pragma omp barrier
    
    if (coords[1] == p - 1 && !(coords[0] == 0 || coords[0] == p - 1)) // right
    {
        for (int j = startY+threadid; j < endY; j += nthreads)
        {
            vnew[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }
    #pragma omp barrier

}

/// @brief  Calculate the vorticity at the interior nodes with its corresponding mathematical expression
/// @param  startX  Starting x-coordinate (1 or 0 depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param  endX    Ending x-coordinate (Nx_local - 1 or Nx_local depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param  startY  Starting y-coordinate (1 or 0 depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param  endY    Ending y-coordinate (Ny_local - 1 or Ny_local depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
void LidDrivenCavity::InteriorVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads)
{
    // Send and receive the edges of the local domain
    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS); // Initialised this with S not to store more data

    #pragma omp barrier
    // Update the vorticity values using the received data
    for (int j = startY+threadid; j < endY; j += nthreads) {
        for (int i = startX; i < endX; ++i) {
            double leftNeighborValue = (coords[1] > 0 && i == 0) ? receiveBufferLeftS[j] : s[IDX(i - 1, j)];
            double rightNeighborValue = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j] : s[IDX(i + 1, j)];
            double botomNeighborValue = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomS[i] : s[IDX(i, j - 1)];
            double topNeighborValue = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i] : s[IDX(i, j + 1)];
            
            vnew[IDX(i,j)] = dx2i * (2.0 * s[IDX(i,j)] - rightNeighborValue - leftNeighborValue)
                        + 1.0/dy/dy * (2.0 * s[IDX(i,j)] - botomNeighborValue - topNeighborValue);
        }
    }
}

/// @brief  Advance the vorticity in time
/// @param  startX  Starting x-coordinate (1 or 0 depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param  endX    Ending x-coordinate (Nx_local - 1 or Nx_local depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param  startY  Starting y-coordinate (1 or 0 depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
/// @param  endY    Ending y-coordinate (Ny_local - 1 or Ny_local depending on the rank to ensure globally the mathematical expression goes from 1 to < Nx/Ny - 1)
void LidDrivenCavity::TimeAdvanceVorticity(int startX, int endX, int startY, int endY, int threadid, int nthreads)
{   
    // Send and receive the edges of the local domain
    mpiGridCommunicator->SendReceiveEdges(vnew, receiveBufferTopV, receiveBufferBottomV, receiveBufferLeftV, receiveBufferRightV);
    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS);

    #pragma omp barrier
    // Update the vorticity values using the received data
    for (int j = startY+threadid; j < endY; j += nthreads) {
        for (int i = startX; i < endX; ++i) {
            double leftNeighborValueS = (coords[1] > 0 && i == 0) ? receiveBufferLeftS[j] : s[IDX(i - 1, j)];
            double rightNeighborValueS = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j] : s[IDX(i + 1, j)];
            double botomNeighborValueS = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomS[i] : s[IDX(i, j - 1)];
            double topNeighborValueS = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i] : s[IDX(i, j + 1)];

            double leftNeighborValueV = (coords[1] > 0 && i == 0) ? receiveBufferLeftV[j] : vnew[IDX(i - 1, j)];
            double rightNeighborValueV = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightV[j] : vnew[IDX(i + 1, j)];
            double botomNeighborValueV = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomV[i] : vnew[IDX(i, j - 1)];
            double topNeighborValueV = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopV[i] : vnew[IDX(i, j + 1)];

            v[IDX(i,j)] = vnew[IDX(i,j)] + dt*(
                ( (rightNeighborValueS - leftNeighborValueS) * 0.5 * dxi
                *(topNeighborValueV - botomNeighborValueV) * 0.5 * dyi)
            - ( (topNeighborValueS - botomNeighborValueS) * 0.5 * dyi
                *(rightNeighborValueV - leftNeighborValueV) * 0.5 * dxi)
            + nu * (rightNeighborValueV - 2.0 * vnew[IDX(i,j)] + leftNeighborValueV)*dx2i
            + nu * (botomNeighborValueV - 2.0 * vnew[IDX(i,j)] + topNeighborValueV)*dy2i);
        }
    }
}
