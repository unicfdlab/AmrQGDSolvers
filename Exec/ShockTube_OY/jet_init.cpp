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

        double R = 0.25;
        double o_x = 1.0;
        double o_y = 1.0;
        double o_z = 1.0;
        double dx = x - o_x;
        double dy = y - o_y;
//        double dz = z - o_z;
        double rr = dx*dx + dy*dy;// + dz*dz;
        double r = sqrt(rr);

        if (y <= 0.5)     //(r <= R)
          {
              snew[bi](i,j,k,0) = 1.0;    //rho
              snew[bi](i,j,k,1) = 0.0;    //Ux
              snew[bi](i,j,k,2) = 1.0;    //Uy
              snew[bi](i,j,k,3) = 1.0;    //p
              snew[bi](i,j,k,4) = ScQgd;  //ScQGD
              snew[bi](i,j,k,5) = 0.0;    //curl
              snew[bi](i,j,k,6) = 0.0;    //magGradRho
          }
      else
          {
              snew[bi](i,j,k,0) = 1.0; //0.125;  //rho
              snew[bi](i,j,k,1) = 0.0;    //Ux
              snew[bi](i,j,k,2) = 0.0;    //Uy
              snew[bi](i,j,k,3) = 1.0;    //p
              snew[bi](i,j,k,4) = ScQgd;  //ScQGD
              snew[bi](i,j,k,5) = 0.0;    //curl
              snew[bi](i,j,k,6) = 0.0;    //magGradRho
          }
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0);
    amrex::Print() << "Amr QGD solver will start with next params: " << "AlphaQQD = " << alphaQgd << " and ScQGD = " << ScQgd << "\n" 
                   << " varScNumber is " << varScQgd << " grad value is " << gradVal << "\n\n" ;
}

