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

void LidDrivenCavity::SetLocalVariables(int Nx, int Ny, int p, int rank)
{
    // Calculate starting and ending points of each process
    // Divide number of points over the number of processos and get minimum number of points each needs
    double extra_x = Nx % p;
    double extra_y = Ny % p;
    int min_points_x = (Nx - extra_x) / p;
    int min_points_y = (Ny - extra_y) / p;

    // Calculate the starting and ending points of each process
    if (rank < extra_x)
    {
        min_points_x++;
        min_points_y++;
        this->start_x = rank * min_points_x;
        this->end_x = (rank + 1) * min_points_x;
        this->start_y = rank * min_points_y;
        this->end_y = (rank + 1) * min_points_y;
        this->Nx_local = end_x - start_x;
        this->Ny_local = end_y - start_y;
    }
    else
    {
        this->start_x = (min_points_x + 1) * extra_x + min_points_x * (rank - extra_x);
        this->end_x = (min_points_x + 1) * extra_x + min_points_x * (rank - extra_x + 1);
        this->start_y = (min_points_y + 1) * extra_y + min_points_y * (rank - extra_y);
        this->end_y = (min_points_y + 1) * extra_y + min_points_y * (rank - extra_y + 1);
        this->Nx_local = end_x - start_x;
        this->Ny_local = end_y - start_y;
    }

    UpdateDxDy();
}

void LidDrivenCavity::Initialise()
{
    CleanUp();

    v = new double[Npts_local](); // local
    s = new double[Npts_local](); // local
    tmp = new double[Npts_local]();
    cg = new SolverCG(Nx_local, Ny_local, dx, dy);

    v_whole = new double[Npts](); // whole
    s_whole = new double[Npts](); // whole
    tmp_whole = new double[Npts]();
    cg_whole = new SolverCG(Nx, Ny, dx, dy);
}

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T / dt);
    for (int t = 0; t < 1; ++t) // NSteps; ++t)
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
    if (coords[0] == 0 && (coords[1] != 0 || coords[1] != p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // top
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
    }
    else if (coords[0] == p - 1 && (coords[1] != 0 || coords[1] != p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // bottom
            v[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
    }
    else if (coords[1] == 0 && (coords[0] != 0 || coords[0] != p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // left
            v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[1] == p - 1 && (coords[0] != 0 || coords[0] != p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // right
            v[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }
    if (coords[0] == 0 && coords[1] == 0)
    {
        for (int i = 1; i < Nx_local; ++i)
        {
            // top
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = 1; j < Ny_local; ++j)
        {
            // left
            v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[0] == 0 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local - 1; ++i)
        {
            // top
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = 1; j < Ny_local; ++j)
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
            v[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = 0; j < Ny_local - 1; ++j)
        {
            // left
            v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[0] == p - 1 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local - 1; ++i)
        {
            // bottom
            v[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = 1; j < Ny_local - 1; ++j)
        {
            // right
            v[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }


    int x_start = coords[1] == 0 ? 1 : 0;
    int x_end = coords[1] == p - 1 ? Ny_local - 1 : Ny_local;
    int y_start = coords[0] == 0 ? 1 : 0;
    int y_end = coords[0] == p - 1 ? Nx_local - 1 : Nx_local;

    // cout << "Rank " << rank << " x_start: " << x_start << " x_end: " << x_end << " y_start: " << y_start << " y_end: " << y_end << endl;

    // MPI_Barrier(cart_comm);
    InteriorVorticity(x_start, x_end, y_start, y_end);

    MPI_Barrier(cart_comm);
    TimeAdvanceVorticity(x_start, x_end, y_start, y_end);

    // // Paralleslised interior vorticity:
    // // Compute interior vorticity
    // // Interior processes
    // if (!(coords[0] == 0 || coords[0] == p - 1) && !(coords[1] == 0 || coords[1] == p - 1))
    // {
    //     InteriorVorticity(0, Nx_local, 0, Ny_local);
    // }

    // // Edge processes, not corners
    // if (coords[0] == 0 && !(coords[1] == 0 || coords[1] == p - 1))
    // {
    //     InteriorVorticity(0, Nx_local, 1, Ny_local);
    // }
    // else if (coords[0] == p - 1 && !(coords[1] == 0 || coords[1] == p - 1))
    // {
    //     InteriorVorticity(0, Nx_local, 0, Ny_local - 1);
    // }
    // else if (coords[1] == 0 && !(coords[0] == 0 || coords[0] == p - 1))
    // {
    //     InteriorVorticity(1, Nx_local, 0, Ny_local);
    // }
    // else if (coords[1] == p - 1 && !(coords[0] == 0 || coords[0] == p - 1))
    // {
    //     InteriorVorticity(0, Nx_local - 1, 0, Ny_local);
    // }

    // // Corner processes
    // if (coords[0] == 0 && coords[1] == 0) // Top left corner
    // {
    //     InteriorVorticity(1, Nx_local, 1, Ny_local);
    // }
    // else if (coords[0] == 0 && coords[1] == p - 1) // Top right corner
    // {
    //     InteriorVorticity(0, Nx_local - 1, 1, Ny_local);
    // }
    // else if (coords[0] == p - 1 && coords[1] == 0) // Bottom left corner
    // {
    //     InteriorVorticity(1, Nx_local, 0, Ny_local - 1);
    // }
    // else if (coords[0] == p - 1 && coords[1] == p - 1) // Bottom right corner
    // {
    //     InteriorVorticity(0, Nx_local - 1, 0, Ny_local - 1);
    // }

    // // It runs well until here but get deadlock

    // Time advance vorticity
    // Interior processes
    // if (!(coords[0] == 0 || coords[0] == p - 1) && !(coords[1] == 0 || coords[1] == p - 1))
    // {
    //     TimeAdvanceVorticity(0, Nx_local, 0, Ny_local);
    // }

    // // Corners
    // if (coords[0] == 0 && coords[1] == 0) // Top left corner
    // {
    //     TimeAdvanceVorticity(1, Nx_local, 1, Ny_local);
    // }
    // else if (coords[0] == 0 && coords[1] == p - 1) // Top right corner
    // {
    //     TimeAdvanceVorticity(0, Nx_local - 1, 1, Ny_local);
    // }
    // else if (coords[0] == p - 1 && coords[1] == 0) // Bottom left corner
    // {
    //     TimeAdvanceVorticity(1, Nx_local, 0, Ny_local - 1);
    // }
    // else if (coords[0] == p - 1 && coords[1] == p - 1) // Bottom right corner
    // {
    //     TimeAdvanceVorticity(0, Nx_local - 1, 0, Ny_local - 1);
    // }

    cg->Solve(v, s);
    
    // Gather the data in rank 0

    // // Gather the local v arrays from all ranks to rank 0 in the good order
    // MPI_Gather(v, Npts_local, MPI_DOUBLE, v_whole, Npts, MPI_DOUBLE, 0, cart_comm);
    // MPI_Gather(s, Npts_local, MPI_DOUBLE, s_whole, Npts, MPI_DOUBLE, 0, cart_comm);

    // // Only rank 0 will have the whole v array
    // if (rank == 0)
    // {
    //     // Process the whole v array in rank 0
    //     cg_whole->Solve(v_whole, s_whole);
    // }

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
    //cg->Solve(v, s);
}

void LidDrivenCavity::InitialiseBuffers()
{
    if (sendBufferTop)
    {
        CleanUpBuffers();
    }

    sendBufferTop = new double[Nx_local];
    sendBufferBottom = new double[Nx_local];
    sendBufferLeft = new double[Ny_local];
    sendBufferRight = new double[Ny_local];

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
    if (sendBufferTop)
    {
        delete[] sendBufferTop;
        delete[] sendBufferBottom;
        delete[] sendBufferLeft;
        delete[] sendBufferRight;

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

void LidDrivenCavity::SendReceiveEdges(double* varArray, double* receiveBufferTop, double* receiveBufferBottom, double* receiveBufferLeft, double* receiveBufferRight)
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
        // cout << "Rank " << rank << " sending to bottom" << endl;
        MPI_Isend(sendBufferBottom, Nx_local, MPI_DOUBLE, bottomRank, 0, cart_comm, &request);
    }
    if (topRank != MPI_PROC_NULL) {
        // Receive data from top members 
        MPI_Irecv(receiveBufferTop, Nx_local, MPI_DOUBLE, topRank, 0, cart_comm, &request);
    }
    // cout << "Rank " << rank << " received from top" << endl;
    // MPI_Wait(&request, MPI_STATUS_IGNORE);
    
    if (topRank != MPI_PROC_NULL) {
        // Send data to top members - receive from bottom
        for (int i = 0; i < Nx_local; ++i) {
            sendBufferTop[i] = varArray[IDX(i, start_y)];
        }
        // cout << "Rank " << rank << " sending to top" << endl;
        MPI_Isend(sendBufferTop, Nx_local, MPI_DOUBLE, topRank, 0, cart_comm, &request);
    }
    if (bottomRank != MPI_PROC_NULL) {
        // Receive data from bottom members 
        MPI_Irecv(receiveBufferBottom, Nx_local, MPI_DOUBLE, bottomRank, 0, cart_comm, &request);
    }
    // cout << "Rank " << rank << " received from bottom" << endl;
    // MPI_Wait(&request, MPI_STATUS_IGNORE);
    
    if (rightRank != MPI_PROC_NULL) {
        // Send data to right members - receive from left
        for (int j = 0; j < Ny_local; ++j) {
            sendBufferRight[j] = varArray[IDX(end_x, j)];
        }
        MPI_Isend(sendBufferRight, Ny_local, MPI_DOUBLE, rightRank, 0, cart_comm, &request);
    }
    if (leftRank != MPI_PROC_NULL) {
        // Receive data from left members 
        MPI_Irecv(receiveBufferLeft, Ny_local, MPI_DOUBLE, leftRank,0, cart_comm, &request);
    }
    // MPI_Wait(&request, MPI_STATUS_IGNORE);

    if (leftRank != MPI_PROC_NULL) {
        // Send data to left members - receive from right
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

void LidDrivenCavity::InteriorVorticity(int startX, int endX, int startY, int endY)
{
    // Send and receive the edges of the local domain
    SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS);

    // Update the vorticity values using the received data
    for (int i = startX; i < endX; ++i) {
        for (int j = startY; j < endY; ++j) {
            double leftNeighborValue = (coords[0] > 0) ? receiveBufferLeftS[j - startY] : s[IDX(i - 1, j)];
            double rightNeighborValue = (coords[0] < p - 1) ? receiveBufferRightS[j - startY] : s[IDX(i + 1, j)];
            double botomNeighborValue = (coords[1] < p - 1) ? receiveBufferBottomS[i - startX] : s[IDX(i, j + 1)];
            double topNeighborValue = (coords[1] > 0) ? receiveBufferTopS[i - startX] : s[IDX(i, j - 1)];
            
            v[IDX(i,j)] = dx2i * (2.0 * s[IDX(i,j)] - rightNeighborValue - leftNeighborValue)
                        + 1.0/dy/dy * (2.0 * s[IDX(i,j)] - botomNeighborValue - topNeighborValue);
        }
    }
}

void LidDrivenCavity::TimeAdvanceVorticity(int startX, int endX, int startY, int endY)
{
    // Send and receive the edges of the local domain
    SendReceiveEdges(v, receiveBufferTopV, receiveBufferBottomV, receiveBufferLeftV, receiveBufferRightV);
    SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS);

    // Update the vorticity values using the received data
    for (int i = startX; i < endX; ++i)
    {
        for (int j = startY; j < endY; ++j)
        {
            double leftNeighborValueS = (coords[0] > 0) ? receiveBufferLeftS[j - startY] : s[IDX(i - 1, j)];
            double rightNeighborValueS = (coords[0] < p - 1) ? receiveBufferRightS[j - startY] : s[IDX(i + 1, j)];
            double botomNeighborValueS = (coords[1] < p - 1) ? receiveBufferBottomS[i - startX] : s[IDX(i, j + 1)];
            double topNeighborValueS = (coords[1] > 0) ? receiveBufferTopS[i - startX] : s[IDX(i, j - 1)];

            double leftNeighborValueV = (coords[0] > 0) ? receiveBufferLeftV[j - startY] : v[IDX(i - 1, j)];
            double rightNeighborValueV = (coords[0] < p - 1) ? receiveBufferRightV[j - startY] : v[IDX(i + 1, j)];
            double botomNeighborValueV = (coords[1] < p - 1) ? receiveBufferBottomV[i - startX] : v[IDX(i, j + 1)];
            double topNeighborValueV = (coords[1] > 0) ? receiveBufferTopV[i - startX] : v[IDX(i, j - 1)];

            v[IDX(i,j)] = v[IDX(i,j)] + dt*(
                ( (rightNeighborValueS - leftNeighborValueS) * 0.5 * dxi
                *(botomNeighborValueV - topNeighborValueV) * 0.5 * dyi)
            - ( (botomNeighborValueS - topNeighborValueS) * 0.5 * dyi
                *(rightNeighborValueV - leftNeighborValueV) * 0.5 * dxi)
            + nu * (rightNeighborValueV - 2.0 * v[IDX(i,j)] + leftNeighborValueV)*dx2i
            + nu * (botomNeighborValueV - 2.0 * v[IDX(i,j)] + topNeighborValueV)*dy2i);
        }
    }
}

void LidDrivenCavity::GetInfoMPI(MPI_Comm comm, int rank, int size, int *coords, int p)
{
    // Use the comm, rank, size, and coords information here
    this->cart_comm = comm;
    this->rank = rank;
    this->size = size;
    this->coords = coords;
    this->p = p;
}
