#include "AmrQGD.H"
#include <cmath>

using namespace amrex;

void AmrQGD::initData ()
{
    const auto problo = Geom().ProbLoArray();
    const auto dx = Geom().CellSizeArray();
    MultiFab& S_new = get_new_data(State_Type);
    auto const& snew = S_new.arrays();

    double a = 0.075;
    double b = 0.175;
    double o_x = 0.25;
    double o_y = 0.5;
    double Ms = 1.7;
    double Mv = 1.7;

    amrex::ParmParse pp("vortexShock");
    pp.query("a", a);
    pp.query("b", b);
    pp.query("o_x", o_x);
    pp.query("o_y", o_y);
    pp.query("Ms", Ms);
    pp.query("Mv", Mv);
    //
    amrex::Print() << "Vortex shock case will init with next params: " << "a = " << a << " b = " << b << "\n" 
                   << " Ms = " << Ms << " Mv = " << Mv << "\n\n" ;


    //Upstream condition
    double Uu = Ms*sqrt(gamma);
    double Vu = 0;
    double pu = 1.0;
    double rhou = 1.0;
    double Tu = pu/(rhou*RGas);

    //Downstream condition
    double rhod = rhou*(gamma+1.0)*Ms*Ms/(2.0+(gamma-1.0)*Ms*Ms);
    double Ud = Uu*(2.0+(gamma-1.0)*Ms*Ms)/((gamma+1.0)*Ms*Ms);
    double Vd = Vu;
    double pd = pu*(1.0+(2.0*gamma/(gamma+1.0))*(Ms*Ms-1.0));
    double Td = pd/(rhod*RGas);

    //Vortex
    double Um = Mv*sqrt(gamma);

    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
        Real x = problo[0] + (i+0.5)*dx[0];
        Real y = problo[1] + (j+0.5)*dx[1];
        Real x1 = problo[0] ; //- (i+0.5)*dx[0];
        //shock vortex problem
        double dx = x - o_x;
        double dy = y - o_y;
        double rr = dx*dx + dy*dy;
        double r = sqrt(rr);

        //Set Sc number
        snew[bi](i,j,k,4) = ScQgd;    //Sc

        if (x <= 0.5)
        {
//            double rr = (x-o_x)*(x-o_x) + (y-o_y)*(y-o_y);
//            double r = sqrt(rr);

            snew[bi](i,j,k,0) = rhou; //rho
            snew[bi](i,j,k,1) = Uu;   //Ux
            snew[bi](i,j,k,2) = Vu;   //Uy
            snew[bi](i,j,k,3) = pu;   //P

            if (r <= b)
            {
                double sinTheta = dy/r;
                double cosTheta = dx/r;
                if (r <= a)
                {
                    double magU = Um*r/a;
                    snew[bi](i,j,k,1) = snew[bi](i,j,k,1) - magU*sinTheta;      //Ux
                    snew[bi](i,j,k,2) = snew[bi](i,j,k,2) + magU*cosTheta;      //Uy

                    //temperature
                    double radialTerm = -2.0*b*b*log(b)-(0.5*a*a)+(2.0*b*b*log(a))+(0.5*b*b*b*b/(a*a));
                    double T_ = Tu-(gamma-1.0)*pow(Um*a/(a*a-b*b),2)*radialTerm/(RGas*gamma);
                    radialTerm = 0.5 * (1.0 - r*r/(a*a));
                    double T = T_-(gamma-1.0)*Um*Um*radialTerm/(RGas*gamma);

                    snew[bi](i,j,k,3) = pu*pow(T/Tu,gamma/(gamma-1));
                    snew[bi](i,j,k,0) = snew[bi](i,j,k,3)/(RGas*T);
                }
                else
                {
                    double magU = Um*a*(r-b*b/r)/(a*a-b*b);
                    snew[bi](i,j,k,1) = snew[bi](i,j,k,1) - magU*sinTheta;      //Ux
                    snew[bi](i,j,k,2) = snew[bi](i,j,k,2) + magU*cosTheta;      //Uy
                    //temperature
                    double radialTerm = -2.0*b*b*log(b) - (0.5*r*r)+(2.0*b*b*log(r))+(0.5*b*b*b*b/(r*r));
                    double T = Tu-(gamma-1.0)*pow(Um*a/(a*a-b*b),2)*radialTerm/(RGas*gamma);
                    snew[bi](i,j,k,3) = pu*pow(T/Tu,gamma/(gamma-1));
                    snew[bi](i,j,k,0) = snew[bi](i,j,k,3)/(RGas*T);
                }
            }
        }
        else
        {
            snew[bi](i,j,k,0) = rhod;
            snew[bi](i,j,k,1) = Ud;
            snew[bi](i,j,k,2) = Vd;
            snew[bi](i,j,k,3) = pd; 
        }
    });
    FillPatcherFill(S_new, 0, ncomp, nghost, 0, State_Type, 0); 
    amrex::Print() << "Amr QGD solver will start with next params: " << "AlphaQQD = " << alphaQgd << " and ScQGD = " << ScQgd << "\n" 
                   << " varScNumber is " << varScQgd << " grad value is " << gradVal << "\n\n" ;
}

