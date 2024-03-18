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

LidDrivenCavity::LidDrivenCavity()
{
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
}

void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0 / re;
}

void LidDrivenCavity::SetLocalVariables(int Nx, int Ny, int p, int* coords)
{
    // Calculate starting and ending points of each process
    // Divide number of points over the number of processos and get minimum number of points each needs
    double extra_x = Nx % p;
    double extra_y = Ny % p;
    int min_points_x = (Nx - extra_x) / p;
    int min_points_y = (Ny - extra_y) / p;

    // Calculate the starting and ending points of each process
    if (coords[1] < extra_x)
    {
        min_points_x++;
        int x_first = rank * min_points_x;
        int x_last = (rank + 1) * min_points_x;
        this->Nx_local = x_last - x_first;
    }
    else
    {
        int x_first = (min_points_x + 1) * extra_x + min_points_x * (rank - extra_x);
        int x_last = (min_points_x + 1) * extra_x + min_points_x * (rank - extra_x + 1);
        this->Nx_local = x_last - x_first;
    }

    if (coords[0] < extra_y)
    {
        min_points_y++;
        int y_first = rank * min_points_y;
        int y_last = (rank + 1) * min_points_y;
        this->Ny_local = y_last - y_first;
    }
    else
    {
        int y_first = (min_points_y + 1) * extra_y + min_points_y * (rank - extra_y);
        int y_last = (min_points_y + 1) * extra_y + min_points_y * (rank - extra_y + 1);
        this->Ny_local = y_last - y_first;
    }

    UpdateDxDy();
}

void LidDrivenCavity::Initialise(MPI_Comm comm, int *coords, int p)
{
    CleanUp();
    InitialiseBuffers();

    // Set up starting and end points not to include boundaries (0, Nx/Ny)
    int start_x = coords[1] == 0 ? 1 : 0;
    int end_x = coords[1] == p - 1 ? Nx_local - 1 : Nx_local;
    int start_y = coords[0] == p - 1 ? 1 : 0;
    int end_y = coords[0] == 0 ? Ny_local - 1 : Ny_local;


    mpiGridCommunicator = new MPIGridCommunicator(comm, Nx_local, Ny_local, start_x, end_x, start_y, end_y, coords, p);
    cg = new SolverCG(Nx_local, Ny_local, dx, dy, mpiGridCommunicator);
    this->coords = coords;
    this->p = p;

    v = new double[Npts_local](); // local
    s = new double[Npts_local](); // local
    tmp = new double[Npts_local]();


}

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T / dt);
    for (int t = 0; t < NSteps; ++t)
    {
        // std::cout << "Step: " << setw(8) << t
        //           << "  Time: " << setw(8) << t * dt
        //           << std::endl;
        Advance();
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    double *u0 = new double[Nx * Ny]();
    double *u1 = new double[Nx * Ny]();
    for (int i = 1; i < Nx - 1; ++i)
    {
        for (int j = 1; j < Ny - 1; ++j)
        {
            u0[IDX(i, j)] = (s[IDX(i, j + 1)] - s[IDX(i, j)]) / dy;
            u1[IDX(i, j)] = -(s[IDX(i + 1, j)] - s[IDX(i, j)]) / dx;
        }
    }
    for (int i = 0; i < Nx; ++i)
    {
        u0[IDX(i, Ny - 1)] = U;
    }

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            k = IDX(i, j);
            f << i * dx << " " << j * dy << " " << v[k] << " " << s[k]
              << " " << u0[k] << " " << u1[k] << std::endl;
        }
        f << std::endl;
    }
    f.close();

    delete[] u0;
    delete[] u1;
}

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
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}

void LidDrivenCavity::CleanUp()
{
    if (v)
    {
        delete[] v;
        delete[] s;
        delete[] tmp;
        delete cg;
        delete[] v_whole;
        delete[] s_whole;
        delete[] tmp_whole;
        delete cg_whole;
    }
}

void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx - 1);
    dy = Ly / (Ny - 1);
    Npts_local = Nx_local * Ny_local;
    Npts = Nx * Ny;
}

void LidDrivenCavity::Advance()
{
    this->dxi = 1.0 / dx;
    this->dyi = 1.0 / dy;
    this->dx2i = 1.0 / dx / dx;
    this->dy2i = 1.0 / dy / dy;

    // Boundary node vorticity
    // Cheking if the process is a corner and take that into account for the starting and end points
    // First checking "insider" ranks within the border of the grid
    NodeVorticity();

    // Start and ending points for the interior vorticity so that it goes from 1 to < Nx/Ny - 1
    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    InteriorVorticity(start_x, end_x, start_y, end_y);

    MPI_Barrier(mpiGridCommunicator->cart_comm);
    TimeAdvanceVorticity(start_x, end_x, start_y, end_y);

    // Open the file for writing
    std::ofstream outputFile("testingB.txt", std::ios::app); // Append mode

    // Synchronize the ranks
    MPI_Barrier(mpiGridCommunicator->cart_comm);

    int rank;
    MPI_Comm_rank(mpiGridCommunicator->cart_comm, &rank);
    for (int i = 0; i < mpiGridCommunicator->p*mpiGridCommunicator->p; i++) {
        if (rank == i) {
            outputFile << "Values of rank " << rank << ":" << std::endl;
            for (unsigned int u = 0; u < Nx_local; u++) {
                for (unsigned int j = 0; j < Ny_local; j++) {
                    outputFile << v[IDX(u, j)] << " ";
                }
                outputFile << std::endl;
            }
        }
        // Synchronize the ranks again
        MPI_Barrier(mpiGridCommunicator->cart_comm);
    }

    outputFile.close(); // Close the file

    // cg->Solve(v, s);
    
    // Sinusoidal test case with analytical solution, which can be used to test
    // the Poisson solver
    /*
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    */

    // Solve Poisson problem
    cg->Solve(v, s);
}

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

void LidDrivenCavity::NodeVorticity()
{   
    // int rank;
    // MPI_Comm_rank(mpiGridCommunicator->cart_comm, &rank);

    // cout << "Process " << rank << " coordinates: (" << coords[0] << ", " << coords[1] << ")" << endl;
    // Corners first

    if (coords[0] == 0 && coords[1] == 0)
    {
        for (int i = 1; i < Nx_local; ++i)
        {
            // top
            v[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = 0; j < Ny_local-1; ++j)
        {
            // left
            v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[0] == 0 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local-1; ++i)
        {
            // top
            v[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = 0; j < Ny_local-1; ++j)
        {
            // right
            v[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }
    else if (coords[0] == p - 1 && coords[1] == 0)
    {
        for (int i = 1; i < Nx_local; ++i)
        {
            // bottom
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = 1; j < Ny_local; ++j)
        {
            // left
            v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[0] == p - 1 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local-1; ++i)
        {
            // bottom
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = 1; j < Ny_local; ++j)
        {
            // right
            v[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }

    // Edges
    if (coords[0] == 0 && !(coords[1] == 0 || coords[1] == p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // top
            v[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
    }
    else if (coords[0] == p - 1 && !(coords[1] == 0 || coords[1] == p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // bottom
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
    }
    else if (coords[1] == 0 && !(coords[0] == 0 || coords[0] == p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // left
            v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[1] == p - 1 && !(coords[0] == 0 || coords[0] == p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // right
            v[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }


}

void LidDrivenCavity::InteriorVorticity(int startX, int endX, int startY, int endY)
{
    // Send and receive the edges of the local domain
    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS); // Initialised this with S not to store more data

    // Update the vorticity values using the received data
    for (int i = startX; i < endX; ++i) {
        for (int j = startY; j < endY; ++j) {
            double leftNeighborValue = (coords[1] > 0 && i == 0) ? receiveBufferLeftS[j - startY] : s[IDX(i - 1, j)];
            double rightNeighborValue = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j - startY] : s[IDX(i + 1, j)];
            double botomNeighborValue = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomS[i - startX] : s[IDX(i, j - 1)];
            double topNeighborValue = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i - startX] : s[IDX(i, j + 1)];
            
            v[IDX(i,j)] = dx2i * (2.0 * s[IDX(i,j)] - rightNeighborValue - leftNeighborValue)
                        + 1.0/dy/dy * (2.0 * s[IDX(i,j)] - botomNeighborValue - topNeighborValue);
        }
    }
}

void LidDrivenCavity::TimeAdvanceVorticity(int startX, int endX, int startY, int endY)
{   
    // Generate a new double* v1 that is a copy of v with blas dcopy
    double* v1 = new double[Npts_local];
    cblas_dcopy(Npts_local, v, 1, v1, 1);
    
    // Send and receive the edges of the local domain
    mpiGridCommunicator->SendReceiveEdges(v1, receiveBufferTopV, receiveBufferBottomV, receiveBufferLeftV, receiveBufferRightV);
    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS);

    // Update the vorticity values using the received data
    for (int i = startX; i < endX; ++i)
    {
        for (int j = startY; j < endY; ++j)
        {
            double leftNeighborValueS = (coords[1] > 0 && i == 0) ? receiveBufferLeftS[j - startY] : s[IDX(i - 1, j)];
            double rightNeighborValueS = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j - startY] : s[IDX(i + 1, j)];
            double botomNeighborValueS = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomS[i - startX] : s[IDX(i, j - 1)];
            double topNeighborValueS = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i - startX] : s[IDX(i, j + 1)];

            double leftNeighborValueV = (coords[1] > 0 && i == 0) ? receiveBufferLeftV[j - startY] : v1[IDX(i - 1, j)];
            double rightNeighborValueV = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightV[j - startY] : v1[IDX(i + 1, j)];
            double botomNeighborValueV = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomV[i - startX] : v1[IDX(i, j - 1)];
            double topNeighborValueV = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopV[i - startX] : v1[IDX(i, j + 1)];

            v[IDX(i,j)] = v1[IDX(i,j)] + dt*(
                ( (rightNeighborValueS - leftNeighborValueS) * 0.5 * dxi
                *(botomNeighborValueV - topNeighborValueV) * 0.5 * dyi)
            - ( (botomNeighborValueS - topNeighborValueS) * 0.5 * dyi
                *(rightNeighborValueV - leftNeighborValueV) * 0.5 * dxi)
            + nu * (rightNeighborValueV - 2.0 * v1[IDX(i,j)] + leftNeighborValueV)*dx2i
            + nu * (botomNeighborValueV - 2.0 * v1[IDX(i,j)] + topNeighborValueV)*dy2i);
        }
    }
}
