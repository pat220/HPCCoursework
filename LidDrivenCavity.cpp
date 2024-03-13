#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>

#define IDX(I,J) ((J)*Nx_local + (I))
#define IDX_GLOBAL(I,J) ((J)*Nx + (I))

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
    this->nu = 1.0/re;
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
    if (rank < extra_x) {
        min_points_x++;
        min_points_y++;
        this->start_x = rank * min_points_x;
        this->end_x = (rank  + 1)* min_points_x;
        this->start_y = rank * min_points_y;
        this->end_y = (rank  + 1)* min_points_y;
        this->Nx_local = end_x - start_x;
        this->Ny_local = end_y - start_y;
    } else {
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

    v   = new double[Npts]();
    s   = new double[Npts]();
    tmp = new double[Npts]();
    cg  = new SolverCG(Nx_local, Ny_local, dx, dy);
}

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);
    for (int t = 0; t < 1; ++t) // NSteps; ++t)
    {
        std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*dt
                  << std::endl;
        Advance();
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    double* u0 = new double[Nx*Ny]();
    double* u1 = new double[Nx*Ny]();
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy;
            u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0[IDX(i,Ny-1)] = U;
    }

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            k = IDX(i, j);
            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] 
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
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}


void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] s;
        delete[] tmp;
        delete cg;
    }
}


void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
    Npts = Nx_local * Ny_local; // Nx * Ny;
}


void LidDrivenCavity::Advance()
{
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;


    // Boundary node vorticity
    // Cheking if the process is a corner and take that into account for the starting and end points
    // First checking "insider" ranks within the border of the grid
    if (coords[0] == 0 && (coords[1] != 0 || coords[1] != p -1))
    {
        for (int i = 0; i < Nx_local; ++i) {
            // top
            v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);
        }
    }
    else if (coords[0] == p - 1 && (coords[1] != 0 || coords[1] != p -1))
    {
        for (int i = 0; i < Nx_local; ++i) {
            // bottom
            v[IDX(i,Ny_local-1)] = 2.0 * dy2i * (s[IDX(i,Ny_local-1)] - s[IDX(i,Ny_local-2)])
                           - 2.0 * dyi*U;
        }
    }
    else if (coords[1] == 0 && (coords[0] != 0 || coords[0] != p -1))
    {  
        for (int j = 0; j < Ny_local; ++j) {
            // left
            v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);
        }   
    }
    else if (coords[1] == p - 1 && (coords[0] != 0 || coords[0] != p -1))
    {
        for (int j = 0; j < Ny_local; ++j) {
            // right
            v[IDX(Nx_local-1,j)] = 2.0 * dx2i * (s[IDX(Nx_local-1,j)] - s[IDX(Nx_local-2,j)]);
        }
    }

    if (coords[0] == 0 && coords[1] == 0)
    {
        for (int i = 1; i < Nx_local; ++i) {
            // top
            v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);
        }
        for (int j = 1; j < Ny_local; ++j) {
            // left
            v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);
        }
    }
    else if (coords[0] == 0 && coords[1] == p -1)
    {
        for (int i = 0; i < Nx_local - 1; ++i) {
            // top
            v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);
        }
        for (int j = 1; j < Ny_local; ++j) {
            // right
            v[IDX(Nx_local-1,j)] = 2.0 * dx2i * (s[IDX(Nx_local-1,j)] - s[IDX(Nx_local-2,j)]);
        }
    }
    else if (coords[0] == p - 1 && coords[1] == 0)
    {
        for (int i = 1; i < Nx_local; ++i) {
            // bottom
            v[IDX(i,Ny_local-1)] = 2.0 * dy2i * (s[IDX(i,Ny_local-1)] - s[IDX(i,Ny_local-2)])
                           - 2.0 * dyi*U;
        }
        for (int j = 0; j < Ny_local - 1; ++j) {
            // left
            v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);
        }

    }
    else if (coords[0] == p - 1 && coords[1] == p - 1)
    {
        for (int i = 0; i < Nx_local-1; ++i) {
            // bottom
            v[IDX(i,Ny_local-1)] = 2.0 * dy2i * (s[IDX(i,Ny_local-1)] - s[IDX(i,Ny_local-2)])
                           - 2.0 * dyi*U;
        }
        for (int j = 1; j < Ny_local-1; ++j) {
            // right
            v[IDX(Nx_local-1,j)] = 2.0 * dx2i * (s[IDX(Nx_local-1,j)] - s[IDX(Nx_local-2,j)]);
        }
    }


    // // Compute interior vorticity
    // for (int i = 1; i < Nx - 1; ++i) {
    //     for (int j = 1; j < Ny - 1; ++j) {
    //         v[IDX(i,j)] = dx2i*(
    //                 2.0 * s[IDX(i,j)] - s[IDX(i+1,j)] - s[IDX(i-1,j)])
    //                     + 1.0/dy/dy*(
    //                 2.0 * s[IDX(i,j)] - s[IDX(i,j+1)] - s[IDX(i,j-1)]);
    //     }
    // }

    // // Time advance vorticity
    // for (int i = 1; i < Nx - 1; ++i) {
    //     for (int j = 1; j < Ny - 1; ++j) {
    //         v[IDX(i,j)] = v[IDX(i,j)] + dt*(
    //             ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dxi
    //              *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dyi)
    //           - ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dyi
    //              *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dxi)
    //           + nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dx2i
    //           + nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dy2i);
    //     }
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
    // cg->Solve(v, s);
}


void LidDrivenCavity::GetInfoMPI(MPI_Comm comm, int rank, int size, int* coords, int p)
{
    // Use the comm, rank, size, and coords information here
    this->comm = comm;
    this->rank = rank;
    this->size = size;
    this->coords = coords;
    this->p = p;
}
