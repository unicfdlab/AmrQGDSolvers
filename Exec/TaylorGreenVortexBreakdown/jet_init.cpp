#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void
AmrQGD::initData ()
{
    const auto problo = Geom().ProbLoArray();
    const auto dx = Geom().CellSizeArray();
    MultiFab& S_new = get_new_data(State_Type);
    auto const& snew = S_new.arrays();

    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
        Real x = problo[0] + (i+0.5)*dx[0];
        Real y = problo[1] + (j+0.5)*dx[1];
//        Real z = problo[2] + (k+0.5)*dx[2];

        double T0 = 273.0;
        double Ma = 0.1;
        double Re = 100.0;
        double L =  6.366197724;

        double U0   = Ma*sqrt(gamma*RGas*T0);
        double rho0 = Re*mutGas/(U0*L);
        double p0   = rho0*RGas*T0;

        //Ux
        snew[bi](i,j,k,1) = U0*sin(x/L)*cos(y/L);
        //Uy
        snew[bi](i,j,k,2) = -U0*cos(x/L)*sin(y/L);
        //p
        snew[bi](i,j,k,3) = p0 + (rho0*pow(U0,2)/8.)*(cos(2.*x/L) + cos(2.*y/L));
        //rho
        snew[bi](i,j,k,0) = snew[bi](i,j,k,3) / (RGas*T0);
        //Sc
        snew[bi](i,j,k,4) =  ScQgd;
        //curl
        snew[bi](i,j,k,5) = (snew[bi](i+1,j,k,2) - snew[bi](i,j,k,2))/dx[0] - (snew[bi](i,j+1,k,1) - snew[bi](i,j,k,1))/dx[1]; 
        snew[bi](i,j,k,6) = 0.0;     //magGradRho
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0);
    amrex::Print() << "Amr QGD solver will start with next params: " << "AlphaQQD = " << alphaQgd << " and ScQGD = " << ScQgd << "\n" 
                   << " varScNumber is " << varScQgd << " grad value is " << gradVal << "\n\n" ;
}

