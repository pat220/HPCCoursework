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
        this->x_first = rank * min_points_x; // global coordinate
        this->x_last = (rank + 1) * min_points_x; // global coordinate
        this->Nx_local = x_last - x_first;
    }
    else
    {
        this->x_first = (min_points_x + 1) * extra_x + min_points_x * (rank); // global coordinate
        this->x_last = (min_points_x + 1) * extra_x + min_points_x * (rank + 1); // global coordinate
        this->Nx_local = x_last - x_first;
    }

    if (coords[0] < extra_y)
    {
        min_points_y++;
        this->y_first = (rank) * min_points_y; // global coordinate
        this->y_last = (rank + 1) * min_points_y; // global coordinate
        this->Ny_local = y_last - y_first;
    }
    else
    {
        this->y_first = (min_points_y + 1) * extra_y + min_points_y * (rank); // global coordinate
        this->y_last = (min_points_y + 1) * extra_y + min_points_y * (rank + 1); // global coordinate
        this->Ny_local = y_last - y_first;
    }

    UpdateDxDy();
}

void LidDrivenCavity::Initialise(MPI_Comm comm, int *coords, int p)
{
    CleanUp();
    InitialiseBuffers();

    // Set up starting and end points not to include boundaries (0, Nx/Ny - 1)
    int start_x = coords[1] == 0 ? 1 : 0;
    int end_x = coords[1] == p - 1 ? Nx_local - 1 : Nx_local;
    int start_y = coords[0] == p - 1 ? 1 : 0;
    int end_y = coords[0] == 0 ? Ny_local - 1 : Ny_local;
    

    mpiGridCommunicator = new MPIGridCommunicator(comm, Nx_local, Ny_local, start_x, end_x, start_y, end_y, coords, p);
    cg = new SolverCG(Nx_local, Ny_local, dx, dy, mpiGridCommunicator);
    this->coords = coords;
    this->p = p;

    v = new double[Npts_local](); // local
    vnew = new double[Npts_local](); // local
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
    int rank;
    MPI_Comm_rank(mpiGridCommunicator->cart_comm, &rank);

    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    double *u0 = new double[Nx_local * Ny_local]();
    double *u1 = new double[Nx_local * Ny_local]();

    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS); // Initialised this with S not to store more data

    for (int i = start_x; i < end_x; ++i)
    {
        for (int j = start_y; j < end_y; ++j)
        {   
            double rightNeighborValue = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j] : s[IDX(i + 1, j)];
            double topNeighborValue = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i] : s[IDX(i, j + 1)];

            u0[IDX(i, j)] = (topNeighborValue - s[IDX(i, j)]) / dy;
            u1[IDX(i, j)] = -(rightNeighborValue - s[IDX(i, j)]) / dx;
        }
    }

    if (coords[0] == 0){
        for (int i = 0; i < Nx_local; ++i)
        {
            u0[IDX(i, Ny_local - 1)] = U;
        }
    } 

    // // mis velocidades, u1, esta shifted hacia arriba???
    // std::ofstream outputFile("hola.txt", std::ios::app);
    // if (outputFile.is_open()) {
    //     // File opened successfully, you can write to it here
    //     for (int r = 0; r < p*p; r++){
    //         if (rank == r){
    //             outputFile << "rank " << rank << " has:" << endl;
    //                 for (int i = 0; i < Nx_local; ++i)
    //                 {
    //                     for (int j = 0; j < Ny_local; ++j)
    //                     {
    //                         outputFile << u0[IDX(i, j)] << " " << u1[IDX(i, j)]  << std::endl;
    //                     }
    //                     outputFile << std::endl;
    //                 }
    //                 MPI_Barrier(mpiGridCommunicator->cart_comm);
    //         }
    //     }
    // } else {
    //     cout << "Failed to open the file." << endl;
    //     // Failed to open the file, handle the error
    // }
    
    outputFile.close();
    

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

    cout << "rank " << rank << " x_first: " << x_first << " x_last: " << x_last << " y_first: " << y_first << " y_last: " << y_last << endl;
    MPI_Barrier(mpiGridCommunicator->cart_comm);
          
    // MPI_Barrier(mpiGridCommunicator->cart_comm);
    // for (int i = 0; i < p*p; i++){
    //     if (rank == i){
    //         cout << "rank " << rank << " x_first: " << x_first << " x_last: " << x_last << " y_first: " << y_first << " y_last: " << y_last << endl;
    //         for (int j = Ny-1; j >= 0; --j){
    //             for (int i = 0; i < Nx; i++){
    //                 cout << outputArray_u0[IDX_GLOBAL(i, j)] << " ";
    //             } cout << endl;
    //         }
    //     }
    //     MPI_Barrier(mpiGridCommunicator->cart_comm);
    // }
    
    MPI_Allreduce(outputArray_u0, global_u0, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Allreduce(outputArray_u1, global_u1, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Allreduce(outputArray_v, global_v, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Allreduce(outputArray_s, global_s, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    MPI_Barrier(mpiGridCommunicator->cart_comm);
    
    double* u1new = new double[Nx_local * Ny_local]();
    cblas_dcopy(Nx_local * Ny_local, u1, 1, u1new, 1);


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

    // // double* outputArray_v = new double[Npts]();
    // // double* outputArray_s = new double[Npts]();
    // // double* global_v = new double[Npts]();
    // // double* global_s = new double[Npts]();

    // // for (int i = x_first; i < x_last; ++i)
    // // {
    // //     for (int j = y_first; j < y_last; ++j)
    // //     {
    // //         outputArray_v[IDX(i, j)] = v[IDX(i - x_first, j - y_first)];
    // //         outputArray_s[IDX(i, j)] = s[IDX(i - x_first, j - y_first)];
    // //     }
    // // }

    // // MPI_Allreduce(outputArray_v, global_v, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);
    // // MPI_Allreduce(outputArray_s, global_s, Npts, MPI_DOUBLE, MPI_SUM, mpiGridCommunicator->cart_comm);

    // // double* u0 = new double[Nx*Ny]();
    // // double* u1 = new double[Nx*Ny]();
    // // for (int i = 1; i < Nx - 1; ++i) {
    // //     for (int j = 1; j < Ny - 1; ++j) {
    // //         u0[IDX(i,j)] =  (global_s[IDX(i,j+1)] - global_s[IDX(i,j)]) / dy;
    // //         u1[IDX(i,j)] = -(global_s[IDX(i+1,j)] - global_s[IDX(i,j)]) / dx;
    // //     }
    // // }

    // // for (int i = 0; i < Nx; ++i) {
    // //     u0[IDX(i,Ny-1)] = U;
    // // }
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
        delete[] vnew;
        delete[] s;
        delete[] tmp;
        delete cg;
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
    NodeVorticity(); // change vnew

    // Start and ending points for the interior vorticity so that it goes from 1 to < Nx/Ny - 1
    int start_x = mpiGridCommunicator->start_x;
    int end_x = mpiGridCommunicator->end_x;
    int start_y = mpiGridCommunicator->start_y;
    int end_y = mpiGridCommunicator->end_y;

    InteriorVorticity(start_x, end_x, start_y, end_y); // change vnew

    MPI_Barrier(mpiGridCommunicator->cart_comm);
    TimeAdvanceVorticity(start_x, end_x, start_y, end_y); // change v with vnew

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
            vnew[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = 0; j < Ny_local-1; ++j)
        {
            // left
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    
    if (coords[0] == 0 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local-1; ++i)
        {
            // top
            vnew[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
        for (int j = 0; j < Ny_local-1; ++j)
        {
            // right
            vnew[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }
    
    if (coords[0] == p - 1 && coords[1] == 0)
    {
        for (int i = 1; i < Nx_local; ++i)
        {
            // bottom
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = 1; j < Ny_local; ++j)
        {
            // left
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    
    if (coords[0] == p - 1 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local-1; ++i)
        {
            // bottom
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
        for (int j = 1; j < Ny_local; ++j)
        {
            // right
            vnew[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
        }
    }

    // Edges
    if (coords[0] == 0 && !(coords[1] == 0 || coords[1] == p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // top
            vnew[IDX(i, Ny_local - 1)] = 2.0 * dy2i * (s[IDX(i, Ny_local - 1)] - s[IDX(i, Ny_local - 2)]) - 2.0 * dyi * U;
        }
    }
    
    if (coords[0] == p - 1 && !(coords[1] == 0 || coords[1] == p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // bottom
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
    }
    
    if (coords[1] == 0 && !(coords[0] == 0 || coords[0] == p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // left
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    
    if (coords[1] == p - 1 && !(coords[0] == 0 || coords[0] == p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // right
            vnew[IDX(Nx_local - 1, j)] = 2.0 * dx2i * (s[IDX(Nx_local - 1, j)] - s[IDX(Nx_local - 2, j)]);
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
            double leftNeighborValue = (coords[1] > 0 && i == 0) ? receiveBufferLeftS[j] : s[IDX(i - 1, j)];
            double rightNeighborValue = (coords[1] < p - 1 && i == Nx_local-1) ? receiveBufferRightS[j] : s[IDX(i + 1, j)];
            double botomNeighborValue = (coords[0] < p - 1 && j == 0) ? receiveBufferBottomS[i] : s[IDX(i, j - 1)];
            double topNeighborValue = (coords[0] > 0 && j == Ny_local-1) ? receiveBufferTopS[i] : s[IDX(i, j + 1)];
            
            vnew[IDX(i,j)] = dx2i * (2.0 * s[IDX(i,j)] - rightNeighborValue - leftNeighborValue)
                        + 1.0/dy/dy * (2.0 * s[IDX(i,j)] - botomNeighborValue - topNeighborValue);
        }
    }
}

void LidDrivenCavity::TimeAdvanceVorticity(int startX, int endX, int startY, int endY)
{   
    
    // Send and receive the edges of the local domain
    mpiGridCommunicator->SendReceiveEdges(vnew, receiveBufferTopV, receiveBufferBottomV, receiveBufferLeftV, receiveBufferRightV);
    mpiGridCommunicator->SendReceiveEdges(s, receiveBufferTopS, receiveBufferBottomS, receiveBufferLeftS, receiveBufferRightS);

    // Update the vorticity values using the received data
    for (int i = startX; i < endX; ++i)
    {
        for (int j = startY; j < endY; ++j)
        {
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
