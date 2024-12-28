#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void AmrQGD::initData ()
{
    const auto problo = Geom().ProbLoArray();
    const auto dx = Geom().CellSizeArray();
    MultiFab& S_new = get_new_data(State_Type);
    auto const& snew = S_new.arrays();

    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
        //Set Sc number
        snew[bi](i,j,k,0) = rhou;      //rho
        snew[bi](i,j,k,1) = Uu;        //Ux
        snew[bi](i,j,k,2) = Vu;        //Uy
        snew[bi](i,j,k,3) = pu;        //P
        snew[bi](i,j,k,4) = ScQgd;     //Sc
        snew[bi](i,j,k,5) = curl;      //vorticity
        snew[bi](i,j,k,6) = magGradRho;
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0); 
}

