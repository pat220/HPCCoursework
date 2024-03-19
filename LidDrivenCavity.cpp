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
        this->x_first = (min_points_x + 1) * extra_x + min_points_x * (rank - extra_x); // global coordinate
        this->x_last = (min_points_x + 1) * extra_x + min_points_x * (rank - extra_x + 1); // global coordinate
        this->Nx_local = x_last - x_first;
    }

    if (coords[0] < extra_y)
    {
        min_points_y++;
        this->y_first = rank * min_points_y; // global coordinate
        this->y_last = (rank + 1) * min_points_y; // global coordinate
        this->Ny_local = y_last - y_first;
    }
    else
    {
        this->y_first = (min_points_y + 1) * extra_y + min_points_y * (rank - extra_y); // global coordinate
        this->y_last = (min_points_y + 1) * extra_y + min_points_y * (rank - extra_y + 1); // global coordinate
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

            u0[IDX(i, j)] = rank; //(topNeighborValue - s[IDX(i, j)]) / dy;
            u1[IDX(i, j)] = rank; //-(rightNeighborValue - s[IDX(i, j)]) / dx;
            v[IDX(i, j)] = rank;
            s[IDX(i, j)] = rank;
        }
    }

    // if (coords[0] == 0){
    //     for (int i = 0; i < Nx_local; ++i)
    //     {
    //         u0[IDX(i, Ny_local - 1)] = U;
    //     }
    // } 


    int* counts = new int[1];
    counts[0] = Nx_local*Ny_local;
    int* counts_global = new int[p*p];

    int* Nx_of_all = new int[p*p];
    int* Ny_of_all = new int[p*p];

    int* displacements = new int[p*p];


    MPI_Gather(counts, 1, MPI_INT, counts_global, 1, MPI_INT, 0, mpiGridCommunicator->cart_comm);
    MPI_Gather(&Nx_local, 1, MPI_INT, Nx_of_all, 1, MPI_INT, 0, mpiGridCommunicator->cart_comm);
    MPI_Gather(&Ny_local, 1, MPI_INT, Ny_of_all, 1, MPI_INT, 0, mpiGridCommunicator->cart_comm);

    if (rank == 0){
        displacements[0] = 0;
        for (int i = 1; i < p*p; ++i)
        { 
            displacements[i] = counts_global[i-1] + displacements[i-1];
        }
    }
    MPI_Bcast(counts_global, p*p, MPI_INT, 0, mpiGridCommunicator->cart_comm);
    MPI_Bcast(displacements, p*p, MPI_INT, 0, mpiGridCommunicator->cart_comm);
    MPI_Bcast(Nx_of_all, p*p, MPI_INT, 0, mpiGridCommunicator->cart_comm);
    MPI_Bcast(Ny_of_all, p*p, MPI_INT, 0, mpiGridCommunicator->cart_comm);

    // for (int i = 0; i < p*p; ++i){
    //     cout << "Rank: " << rank << " counts_global: " << Nx_of_all[i] << " displacements: " << displacements[i] << " Nx_of_all: " << Nx_of_all[i] << " Ny_of_all: " << Ny_of_all[i] << endl;
    // }


    double* outputArray_u0 = new double[Npts]();
    double* outputArray_u1 = new double[Npts]();
    double* outputArray_v = new double[Npts]();
    double* outputArray_s = new double[Npts]();

    MPI_Gatherv(u0, Npts_local, MPI_DOUBLE, outputArray_u0, counts_global, displacements, MPI_DOUBLE, 0, mpiGridCommunicator->cart_comm);
    MPI_Gatherv(u1, Npts_local, MPI_DOUBLE, outputArray_u1, counts_global, displacements, MPI_DOUBLE, 0, mpiGridCommunicator->cart_comm);
    MPI_Gatherv(v, Npts_local, MPI_DOUBLE, outputArray_v, counts_global, displacements, MPI_DOUBLE, 0, mpiGridCommunicator->cart_comm);
    MPI_Gatherv(s, Npts_local, MPI_DOUBLE, outputArray_s, counts_global, displacements, MPI_DOUBLE, 0, mpiGridCommunicator->cart_comm);
    
    if (rank == 1){
        for (int i = 0; i < Npts_local; ++i){
            cout << "Rank: " << rank << " u0: " << u0[i] << " u1: " << u1[i] << " v: " << v[i] << " s: " << s[i] << endl;
        }
        cout << "miau" << endl;
        for (int i = 0; i < Nx_of_all[rank]; ++i){
            for (int j = 0; j < Ny_of_all[rank]; ++j){
                int k = i + j*Nx_of_all[rank] + displacements[rank];
                cout << "Rank: " << rank << " u0: " << outputArray_u0[k] << " u1: " << outputArray_u1[k] << " v: " << outputArray_v[k] << " s: " << outputArray_s[k] << endl;
            }
        }
    }

    // Start in rank bottom left corner = p*(p-1)
    // You start reading the local matrix upwards in j 
    // Once you reach the end of the column Ny_local, you go to the next rank in the same column = p*(p-1) - p
    // Once you reach the end of the column Ny_local, you go to the next rank in the same column = p*(p-1) - 2*p
    // Once you reach the end of the column Ny_local and the rank is p*(p-1) - (p - 1), you start again in p*(p-1) but on the coilumn next
    // You repeat the process until you reach the end of the column Ny_local and the rank is p*p -1
    // Once you have done this you have all the left ranks' values and want to go to the column of ranks to the right: p*(p-1) + 1
    // You repeat the process until you reach the end of the column Ny_local and the rank is p*(p-1) + 1 - (p - 1)
    // You then go into the next column of ranks next: p*(p-1) + 2
    // You repeat the process until you reach the end of the column Ny_local and the rank is p*(p-1) + 2 - (p - 1)
    // If you have reached the end of the row Ny_local and Ny_local == Ny you exit the loop

    int p = mpiGridCommunicator->p;
    int r = p * (p - 1); // starting point at bottom left corner
    
    if (rank == 0){
        std::ofstream f(file.c_str());
        std::cout << "Writing file " << file << std::endl;
        int k = 0;
        int countx = 0;
        int county = 0;

        for (int i = 0; i < Nx_of_all[r]; ++i) {
            for (int j = 0; j < Ny_of_all[r]; ++j) {
                k = i + j*Nx_of_all[r] + displacements[r];

                f << countx * dx << " " << county * dy << " " << outputArray_v[k] << " " << outputArray_s[k]
                << " " << outputArray_u0[k] << " " << outputArray_u1[k] << endl;

                if (j == Ny_of_all[r] - 1 && r >= p){
                    r -= p;
                    j = -1; // start from bottom of next array again
                }
                else if (j == Ny_of_all[r] - 1 && r < p && i == Nx_of_all[r] - 1){
                    // if its smaller than p - 1 but still at the end of the column i want to go back to the 2 previous ranks and advance in x
                    r += p*(p-1) + 1;
                    i = -1;
                }
                else if (j == Ny_of_all[r] - 1 && r < p){
                    // if its smaller than p - 1 but still at the end of the column i want to go back to the 2 previous ranks and advance in x
                    r += p*(p-1);
                }
                
                
                // cout << "county: " << county << " countx: " << countx << endl;
                if (county == Ny - 1 && countx == Nx -1){ 
                    cout << "End of the loop" << endl;
                    break;
                }

                county += 1;
                if (county == Ny){
                    county = 0;
                }
            }
            countx += 1;
            f << std::endl;
        }
        f.close();
    }



    // if (rank == 0){
    //     std::ofstream f(file.c_str());
    //     std::cout << "Writing file " << file << std::endl;
    //     int k = 0;
    //     for (int i = 0; i < Nx; ++i)
    //     {
    //         for (int j = 0; j < Ny; ++j)
    //         {
    //             k = IDX(i, j);
    //             f << i * dx << " " << j * dy << " " << outputArray_v[k] << " " << outputArray_s[k]
    //             << " " << outputArray_u0[k] << " " << outputArray_u1[k] << std::endl;
    //         }
    //         f << std::endl;
    //     }
    //     f.close();

    //     delete[] u0;
    //     delete[] u1;
    // }

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
    else if (coords[0] == 0 && coords[1] == p - 1)
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
    else if (coords[0] == p - 1 && coords[1] == 0)
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
    else if (coords[0] == p - 1 && coords[1] == p - 1)
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
    else if (coords[0] == p - 1 && !(coords[1] == 0 || coords[1] == p - 1))
    {
        for (int i = 0; i < Nx_local; ++i)
        {
            // bottom
            vnew[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
    }
    else if (coords[1] == 0 && !(coords[0] == 0 || coords[0] == p - 1))
    {
        for (int j = 0; j < Ny_local; ++j)
        {
            // left
            vnew[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }
    else if (coords[1] == p - 1 && !(coords[0] == 0 || coords[0] == p - 1))
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
                *(botomNeighborValueV - topNeighborValueV) * 0.5 * dyi)
            - ( (botomNeighborValueS - topNeighborValueS) * 0.5 * dyi
                *(rightNeighborValueV - leftNeighborValueV) * 0.5 * dxi)
            + nu * (rightNeighborValueV - 2.0 * vnew[IDX(i,j)] + leftNeighborValueV)*dx2i
            + nu * (botomNeighborValueV - 2.0 * vnew[IDX(i,j)] + topNeighborValueV)*dy2i);
        }
    }
}
