#include "AmrQGD.H"
#include <iostream>

using namespace amrex;

Real AmrQGD::advance (Real time, Real dt, int iteration, int ncycle)
{
    // At the beginning of step, we make the new data from previous step the
    // old data of this step.
    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    double mu_T = mutGas;
    
    auto dx = Geom().CellSizeArray();

    MultiFab& S_new = state[0].newData();
    auto const& VectNew = S_new.arrays();
    MultiFab& S_old = state[0].oldData();
    FillPatcherFill(S_old, 0, ncomp, nghost, time, State_Type, 0);

    auto const& VectOld = S_old.arrays();

    amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
        if (varScQgd)
        {
            if (amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i-1,j,k,0)) / dx[0] >= gradVal or 
                amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i,j-1,k,0)) / dx[1] >= gradVal or 
                amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i+1,j,k,0)) / dx[0] >= gradVal or
                amrex::Math::abs(VectOld[bi](i,j,k,0) - VectOld[bi](i,j+1,k,0)) / dx[1] >= gradVal)   
            {
                VectOld[bi](i,j,k,4) = 4.0;
            }
            else
            {
                VectOld[bi](i,j,k,4) = ScQgd;
            }
        }

        double ScQGD = VectOld[bi](i,j,k,4);

        double ROA = 0.5*(VectOld[bi](i,j,k,0) + VectOld[bi](i+1,j,k,0));
        double ROB = 0.5*(VectOld[bi](i,j,k,0) + VectOld[bi](i-1,j,k,0));
        double ROC = 0.5*(VectOld[bi](i,j,k,0) + VectOld[bi](i,j+1,k,0));
        double ROD = 0.5*(VectOld[bi](i,j,k,0) + VectOld[bi](i,j-1,k,0));

        double UxA = 0.5*(VectOld[bi](i,j,k,1) + VectOld[bi](i+1,j,k,1));
        double UxB = 0.5*(VectOld[bi](i,j,k,1) + VectOld[bi](i-1,j,k,1));
        double UxC = 0.5*(VectOld[bi](i,j,k,1) + VectOld[bi](i,j+1,k,1));
        double UxD = 0.5*(VectOld[bi](i,j,k,1) + VectOld[bi](i,j-1,k,1));
    
        double UyA = 0.5*(VectOld[bi](i,j,k,2) + VectOld[bi](i+1,j,k,2));
        double UyB = 0.5*(VectOld[bi](i,j,k,2) + VectOld[bi](i-1,j,k,2)); 
        double UyC = 0.5*(VectOld[bi](i,j,k,2) + VectOld[bi](i,j+1,k,2));
        double UyD = 0.5*(VectOld[bi](i,j,k,2) + VectOld[bi](i,j-1,k,2));

        double PA = 0.5*(VectOld[bi](i,j,k,3) + VectOld[bi](i+1,j,k,3));
        double PB = 0.5*(VectOld[bi](i,j,k,3) + VectOld[bi](i-1,j,k,3));
        double PC = 0.5*(VectOld[bi](i,j,k,3) + VectOld[bi](i,j+1,k,3));
        double PD = 0.5*(VectOld[bi](i,j,k,3) + VectOld[bi](i,j-1,k,3));
    
        double ROE = 0.25*(VectOld[bi](i,j,k,0) + VectOld[bi](i+1,j,k,0) + VectOld[bi](i,j-1,k,0) + VectOld[bi](i+1,j-1,k,0));
        double ROF = 0.25*(VectOld[bi](i,j,k,0) + VectOld[bi](i+1,j,k,0) + VectOld[bi](i,j+1,k,0) + VectOld[bi](i+1,j+1,k,0));
        double ROG = 0.25*(VectOld[bi](i,j,k,0) + VectOld[bi](i-1,j,k,0) + VectOld[bi](i,j+1,k,0) + VectOld[bi](i-1,j+1,k,0));
        double ROH = 0.25*(VectOld[bi](i,j,k,0) + VectOld[bi](i-1,j,k,0) + VectOld[bi](i,j-1,k,0) + VectOld[bi](i-1,j-1,k,0));
        
        double UxE = 0.25*(VectOld[bi](i,j,k,1) + VectOld[bi](i+1,j,k,1) + VectOld[bi](i,j-1,k,1) + VectOld[bi](i+1,j-1,k,1));
        double UxF = 0.25*(VectOld[bi](i,j,k,1) + VectOld[bi](i+1,j,k,1) + VectOld[bi](i,j+1,k,1) + VectOld[bi](i+1,j+1,k,1));
        double UxG = 0.25*(VectOld[bi](i,j,k,1) + VectOld[bi](i-1,j,k,1) + VectOld[bi](i,j+1,k,1) + VectOld[bi](i-1,j+1,k,1));
        double UxH = 0.25*(VectOld[bi](i,j,k,1) + VectOld[bi](i-1,j,k,1) + VectOld[bi](i,j-1,k,1) + VectOld[bi](i-1,j-1,k,1));
          
        double UyE = 0.25*(VectOld[bi](i,j,k,2) + VectOld[bi](i+1,j,k,2) + VectOld[bi](i,j-1,k,2) + VectOld[bi](i+1,j-1,k,2));
        double UyF = 0.25*(VectOld[bi](i,j,k,2) + VectOld[bi](i+1,j,k,2) + VectOld[bi](i,j+1,k,2) + VectOld[bi](i+1,j+1,k,2));
        double UyG = 0.25*(VectOld[bi](i,j,k,2) + VectOld[bi](i-1,j,k,2) + VectOld[bi](i,j+1,k,2) + VectOld[bi](i-1,j+1,k,2));
        double UyH = 0.25*(VectOld[bi](i,j,k,2) + VectOld[bi](i-1,j,k,2) + VectOld[bi](i,j-1,k,2) + VectOld[bi](i-1,j-1,k,2));      

        double PE = 0.25*(VectOld[bi](i,j,k,3) + VectOld[bi](i+1,j,k,3) + VectOld[bi](i,j-1,k,3) + VectOld[bi](i+1,j-1,k,3));
        double PF = 0.25*(VectOld[bi](i,j,k,3) + VectOld[bi](i+1,j,k,3) + VectOld[bi](i,j+1,k,3) + VectOld[bi](i+1,j+1,k,3));
        double PG = 0.25*(VectOld[bi](i,j,k,3) + VectOld[bi](i-1,j,k,3) + VectOld[bi](i,j+1,k,3) + VectOld[bi](i-1,j+1,k,3));
        double PH = 0.25*(VectOld[bi](i,j,k,3) + VectOld[bi](i-1,j,k,3) + VectOld[bi](i,j-1,k,3) + VectOld[bi](i-1,j-1,k,3));
        
        //speed of sound Cs  = sqrt(gamma*VectOld[bi](i,j,k,3) / VectOld[bi](i,j,k,0));
        double CsA = sqrt(gamma*PA / ROA); 
        double CsB = sqrt(gamma*PB / ROB);  
        double CsC = sqrt(gamma*PC / ROC);    
        double CsD = sqrt(gamma*PD / ROD);       

        double hh = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);

        //tau Tau  = alphaQgd*hh/Cs  + mu_T / VectOld[bi](i,j,k,3);
        double TauA = alphaQgd*hh/CsA + mu_T / PA;
        double TauB = alphaQgd*hh/CsB + mu_T / PB;
        double TauC = alphaQgd*hh/CsC + mu_T / PC;
        double TauD = alphaQgd*hh/CsD + mu_T / PD;     

        // mu  = mu_T + VectOld[bi](i,j,k,3)*Tau*Sc_QGD; // mu в (i,j)
        double muA = mu_T + TauA*PA*ScQGD;
        double muB = mu_T + TauB*PB*ScQGD;
        double muC = mu_T + TauC*PC*ScQGD;
        double muD = mu_T + TauD*PD*ScQGD;
       
        double kapA = (mu_T / PrGas + PA*TauA*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);
        double kapB = (mu_T / PrGas + PB*TauB*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);
        double kapC = (mu_T / PrGas + PC*TauC*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);
        double kapD = (mu_T / PrGas + PD*TauD*ScQGD / PrQgd)*gamma*RGas / (gamma - 1.);
        

        // QGD parametr
        double WxA = (TauA / ROA)*((ROF*UyF*UxF - ROE*UyE*UxE) / dx[1] 
                                + (VectOld[bi](i+1,j,k,0)*VectOld[bi](i+1,j,k,1)*VectOld[bi](i+1,j,k,1) - VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,1)*VectOld[bi](i,j,k,1)) / dx[0] 
                                + (VectOld[bi](i+1,j,k,3) - VectOld[bi](i,j,k,3)) / dx[0]);

        double WxB = (TauB / ROB)*((ROG*UyG*UxG - ROH*UyH*UxH) / dx[1] 
                                + (VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,1)*VectOld[bi](i,j,k,1) - VectOld[bi](i-1,j,k,0)*VectOld[bi](i-1,j,k,1)*VectOld[bi](i-1,j,k,1)) / dx[0] 
                                + (VectOld[bi](i,j,k,3) - VectOld[bi](i-1,j,k,3)) / dx[0]);

        double WyC = (TauC / ROC)*((ROF*UyF*UxF - ROG*UyG*UxG) / dx[0]
                                + (VectOld[bi](i,j+1,k,0)*VectOld[bi](i,j+1,k,2)*VectOld[bi](i,j+1,k,2) - VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,2)*VectOld[bi](i,j,k,2)) / dx[1] 
                                + (VectOld[bi](i,j+1,k,3) - VectOld[bi](i,j,k,3)) / dx[1]);

        double WyD = (TauD / ROD)*((ROE*UyE*UxE - ROH*UyH*UxH) / dx[0] 
                                + (VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,2)*VectOld[bi](i,j,k,2) - VectOld[bi](i,j-1,k,0)*VectOld[bi](i,j-1,k,2)*VectOld[bi](i,j-1,k,2)) / dx[1] 
                                + (VectOld[bi](i,j,k,3) - VectOld[bi](i,j-1,k,3)) / dx[1]);
//amrex::Print() << "WxA-B = " << WxA - WxB << " // "<< "WyC-D = " << WyC - WyD  << std::endl;

        // mass flux QGD 
        double JmxA = ROA*(UxA - WxA);
        double JmxB = ROB*(UxB - WxB);
        double JmyC = ROC*(UyC - WyC);
        double JmyD = ROD*(UyD - WyD);

        /*----------------equation of continuity-------------------------*/

        VectNew[bi](i,j,k,0) = VectOld[bi](i,j,k,0) - dt*(JmxA - JmxB)/dx[0] - dt*(JmyC - JmyD)/dx[1];

//amrex::Print() << "JmC-D = " << JmyC - JmyD  << " / ";
//amrex::Print() << "JmJ-I = " << JmxA - JmxB << " /// " << std::endl;
        /*---------------------------------------------------------------*/        
        
        // divU  = (UxC - UxD) / dx[0] + (UyC - UyD) / dx[1] + (UxA - UxB) / dx[0];

        double divuA = (VectOld[bi](i+1,j,k,1) - VectOld[bi](i,j,k,1)) / dx[0] + (UyF - UyE) / dx[1];
        double divuB = (VectOld[bi](i,j,k,1) - VectOld[bi](i-1,j,k,1)) / dx[0] + (UyG - UyH) / dx[1];
        double divuC = (VectOld[bi](i,j+1,k,2) - VectOld[bi](i,j,k,2)) / dx[1] + (UxF - UxG) / dx[0];
        double divuD = (VectOld[bi](i,j,k,2) - VectOld[bi](i,j-1,k,2)) / dx[1] + (UxE - UxH) / dx[0];


         /*--------------------------------------------*/

        // X-component
        double PxxNSA = 2.*muA*(VectOld[bi](i+1,j,k,1) - VectOld[bi](i,j,k,1)) / dx[0] - (2./3.)*muA*divuA;
        double PxxNSB = 2.*muB*(VectOld[bi](i,j,k,1) - VectOld[bi](i-1,j,k,1)) / dx[0] - (2./3.)*muB*divuB;
        double PyxNSC = muC*( (VectOld[bi](i,j+1,k,1) - VectOld[bi](i,j,k,1) ) / dx[1] + (UyF - UyG) / dx[0]);
        double PyxNSD = muD*( (VectOld[bi](i,j,k,1) - VectOld[bi](i,j-1,k,1) ) / dx[1] + (UyE - UyH) / dx[0]); 
        
        // Y-component   
        double PxyNSA = muA*( (VectOld[bi](i+1,j,k,2) - VectOld[bi](i,j,k,2) ) / dx[0] + (UxF - UxE) / dx[1]);
        double PxyNSB = muB*( (VectOld[bi](i,j,k,2) - VectOld[bi](i-1,j,k,2) ) / dx[0] + (UxG - UxH) / dx[1]);             
        double PyyNSC = 2.*muC*(VectOld[bi](i,j+1,k,2) - VectOld[bi](i,j,k,2)) / dx[1] - (2./3.)*muC*divuC;
        double PyyNSD = 2.*muD*(VectOld[bi](i,j,k,2) - VectOld[bi](i,j-1,k,2)) / dx[1] - (2./3.)*muD*divuD;

        
        /*---------------------------------------------------------------*/  
              
        // RG  = Tau*( VectOld[bi](i,j,k,1)*(PA - PB)/dx[0] + VectOld[bi](i,j,k,2)*(PC - PD)/dx[1] + VectOld[bi](i,j,k,1)*(PJ - PI)/dx[0] + gamma*VectOld[bi](i,j,k,3)*divU);

        double RGA = TauA*(UxA*(VectOld[bi](i+1,j,k,3) - VectOld[bi](i,j,k,3)) / dx[0] + UyA*(PF - PE) / dx[1] + gamma*PA*divuA);
        double RGB = TauB*(UxB*(VectOld[bi](i,j,k,3) - VectOld[bi](i-1,j,k,3)) / dx[0] + UyB*(PG - PH) / dx[1] + gamma*PB*divuB);
        double RGC = TauC*(UyC*(VectOld[bi](i,j+1,k,3) - VectOld[bi](i,j,k,3)) / dx[1] + UxC*(PF - PG) / dx[0] + gamma*PC*divuC);
        double RGD = TauD*(UyD*(VectOld[bi](i,j,k,3) - VectOld[bi](i,j-1,k,3)) / dx[1] + UxD*(PE - PH) / dx[0] + gamma*PD*divuD);    


        /*---------------------------------------------------------------*/     
      
        double WWxA = TauA*(UxA*(VectOld[bi](i+1,j,k,1) - VectOld[bi](i,j,k,1)) / dx[0] + UyA*(UxF - UxE) / dx[1] + (1 / ROA)*(VectOld[bi](i+1,j,k,3) - VectOld[bi](i,j,k,3)) / dx[0]);
        double WWxB = TauB*(UxB*(VectOld[bi](i,j,k,1) - VectOld[bi](i-1,j,k,1)) / dx[0] + UyB*(UxG - UxH) / dx[1] + (1 / ROB)*(VectOld[bi](i,j,k,3) - VectOld[bi](i-1,j,k,3)) / dx[0]);
        double WWyC = TauC*(UyC*(VectOld[bi](i,j+1,k,2) - VectOld[bi](i,j,k,2)) / dx[1] + UxC*(UyF - UyG) / dx[0] + (1 / ROC)*(VectOld[bi](i,j+1,k,3) - VectOld[bi](i,j,k,3)) / dx[1]);
        double WWyD = TauD*(UyD*(VectOld[bi](i,j,k,2) - VectOld[bi](i,j-1,k,2)) / dx[1] + UxD*(UyE - UyH) / dx[0] + (1 / ROD)*(VectOld[bi](i,j,k,3) - VectOld[bi](i,j-1,k,3)) / dx[1]);
        
        double WWyA = TauA*(UxA*(VectOld[bi](i+1,j,k,2) - VectOld[bi](i,j,k,2)) / dx[0] + UyA*(UyF - UyE) / dx[1] + (1 / ROA)*(PF - PE) / dx[1]);
        double WWyB = TauB*(UxB*(VectOld[bi](i,j,k,2) - VectOld[bi](i-1,j,k,2)) / dx[0] + UyB*(UyG - UyH) / dx[1] + (1 / ROB)*(PG - PH) / dx[1]);
        double WWxC = TauC*(UyC*(VectOld[bi](i,j+1,k,1) - VectOld[bi](i,j,k,1)) / dx[1] + UxC*(UxF - UxG) / dx[0] + (1 / ROC)*(PF - PG) / dx[0]);
        double WWxD = TauD*(UyD*(VectOld[bi](i,j,k,1) - VectOld[bi](i,j-1,k,1)) / dx[1] + UxD*(UxE - UxH) / dx[0] + (1 / ROD)*(PE - PH) / dx[0]);
          
            
        /*---------------------------------------------------------------*/  

        // X-component
        double PxxA = PxxNSA + ROA*UxA*WWxA + RGA;
        double PxxB = PxxNSB + ROB*UxB*WWxB + RGB;
        double PyxC = PyxNSC + ROC*UyC*WWxC;
        double PyxD = PyxNSD + ROD*UyD*WWxD;
                
        // Y-component
        double PxyA = PxyNSA + ROA*UxA*WWyA;
        double PxyB = PxyNSB + ROB*UxB*WWyB; 
        double PyyC = PyyNSC + ROC*UyC*WWyC + RGC;
        double PyyD = PyyNSD + ROD*UyD*WWyD + RGD;      

//    amrex::Print() << "Pi = " << PxxC << "/" << PyyC << "/" << PxxA;                 
                               
 
         /*-------------momentum equation Z-component---------------------*/
        
        VectNew[bi](i,j,k,1) = VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,1) - 
                               dt*((JmyC*UxC - JmyD*UxD) / dx[1] + (JmxA*UxA - JmxB*UxB) / dx[0] + (PA - PB) / dx[0]) + 
                               dt*((PyxC - PyxD) / dx[1] + (PxxA - PxxB) / dx[0]);   
        
        /*-------------momentum equation Y-component---------------------*/
        
        VectNew[bi](i,j,k,2) = VectOld[bi](i,j,k,0)*VectOld[bi](i,j,k,2) - 
                               dt*((JmyC*UyC - JmyD*UyD) / dx[1] + (JmxA*UyA - JmxB*UyB) / dx[0] + (PC - PD) / dx[1]) + 
                               dt*((PyyC - PyyD) / dx[1] + (PxyA - PxyB) / dx[0]);
                                                                           
                                                                             
        /*------------------energy equation------------------------------*/

        double T0 = VectOld[bi](i,j,k,3)   / (VectOld[bi](i,j,k,0)*RGas);
        double T1 = VectOld[bi](i+1,j,k,3) / (VectOld[bi](i+1,j,k,0)*RGas);
        double T2 = VectOld[bi](i-1,j,k,3) / (VectOld[bi](i-1,j,k,0)*RGas);
        double T3 = VectOld[bi](i,j+1,k,3) / (VectOld[bi](i,j+1,k,0)*RGas);
        double T4 = VectOld[bi](i,j-1,k,3) / (VectOld[bi](i,j-1,k,0)*RGas);

        double eps0 = VectOld[bi](i,j,k,3)   / (VectOld[bi](i,j,k,0)*(gamma - 1.));
        double eps1 = VectOld[bi](i+1,j,k,3) / (VectOld[bi](i+1,j,k,0)*(gamma - 1.));
        double eps2 = VectOld[bi](i-1,j,k,3) / (VectOld[bi](i-1,j,k,0)*(gamma - 1.));
        double eps3 = VectOld[bi](i,j+1,k,3) / (VectOld[bi](i,j+1,k,0)*(gamma - 1.));
        double eps4 = VectOld[bi](i,j-1,k,3) / (VectOld[bi](i,j-1,k,0)*(gamma - 1.));

        double epsA = PA / (ROA*(gamma - 1.));
        double epsB = PB / (ROB*(gamma - 1.));
        double epsC = PC / (ROC*(gamma - 1.));
        double epsD = PD / (ROD*(gamma - 1.));
        
        double epsE = PE / (ROE*(gamma - 1.));
        double epsF = PF / (ROF*(gamma - 1.));
        double epsG = PG / (ROG*(gamma - 1.));
        double epsH = PH / (ROH*(gamma - 1.));

        double HA = pow(UxA,2.)/2. + pow(UyA,2.)/2. + gamma*epsA;
        double HB = pow(UxB,2.)/2. + pow(UyB,2.)/2. + gamma*epsB;
        double HC = pow(UxC,2.)/2. + pow(UyC,2.)/2. + gamma*epsC;
        double HD = pow(UxD,2.)/2. + pow(UyD,2.)/2. + gamma*epsD;


        //NS part of heat flux calculation q = -k(dT/dx), T = p/(rho Rgas) 

        double qxNSA = -kapA*(T1 - T0) / dx[0];
        double qxNSB = -kapB*(T0 - T2) / dx[0];
        double qyNSC = -kapC*(T3 - T0) / dx[1];
        double qyNSD = -kapD*(T0 - T4) / dx[1];
//amrex::Print() << "qxNS = " << qxNSC << "/" << qyNSC << "/" << qxNSA;  

        double qxA = qxNSA - TauA*ROA*UxA*(UxA*(eps1 - eps0) / dx[0] + UyA*(epsF - epsE) / dx[1] + PA*(UxA*(1./VectOld[bi](i+1,j,k,0) - 1./VectOld[bi](i,j,k,0)) / dx[0] + UyA*(1./ROF - 1./ROE) / dx[1]));
        double qxB = qxNSB - TauB*ROB*UxB*(UxB*(eps0 - eps2) / dx[0] + UyB*(epsG - epsH) / dx[1] + PB*(UxB*(1./VectOld[bi](i,j,k,0) - 1./VectOld[bi](i-1,j,k,0)) / dx[0] + UyB*(1./ROG - 1./ROH) / dx[1]));
        double qyC = qyNSC - TauC*ROC*UyC*(UyC*(eps3 - eps0) / dx[1] + UxC*(epsF - epsG) / dx[0] + PC*(UyC*(1./VectOld[bi](i,j+1,k,0) - 1./VectOld[bi](i,j,k,0)) / dx[1] + UxC*(1./ROF - 1./ROG) / dx[0]));
        double qyD = qyNSD - TauD*ROD*UyD*(UyD*(eps0 - eps4) / dx[1] + UxD*(epsE - epsH) / dx[0] + PD*(UyD*(1./VectOld[bi](i,j,k,0) - 1./VectOld[bi](i,j-1,k,0)) / dx[1] + UxD*(1./ROE - 1./ROH) / dx[0]));

        //energy calculation 
        double E = VectOld[bi](i,j,k,0)*(pow(VectOld[bi](i,j,k,1),2.) + pow(VectOld[bi](i,j,k,2),2.))/2. + VectOld[bi](i,j,k,3)/(gamma - 1.);

        //energy equation E_new = ...
       double EN = E - dt*((JmyC*HC - JmyD*HD)   / dx[1] +   (JmxA*HA - JmxB*HB) / dx[0]) 
                     - dt*((qyC - qyD)    / dx[1] + (qxA - qxB) / dx[0])
                     + dt*((PyyC*UyC - PyyD*UyD) / dx[1] + (PyxC*UxC - PyxD*UxD) / dx[1] + 
                           (PxxA*UxA - PxxB*UxB) / dx[0] + (PxyA*UyA - PxyB*UyB) / dx[0]); 
//    amrex::Print() << "EN = " << EN << "/" << E;                              

        /*------------------update parameters----------------------------*/

        VectNew[bi](i,j,k,1) = VectNew[bi](i,j,k,1) / VectNew[bi](i,j,k,0);
        VectNew[bi](i,j,k,2) = VectNew[bi](i,j,k,2) / VectNew[bi](i,j,k,0);
        VectNew[bi](i,j,k,3) = (gamma - 1.)*(EN - 0.5*VectNew[bi](i,j,k,0)*(pow(VectNew[bi](i,j,k,1),2.) + pow(VectNew[bi](i,j,k,2),2.)));
        VectNew[bi](i,j,k,4) = ScQGD;
        //amrex::Print() << "ScQGD = " << Sc_QGD << "\n";
       
       
       
        //solve vorticity (curl)
       VectNew[bi](i,j,k,5) = 0.5*(VectOld[bi](i+1,j,k,2) - VectOld[bi](i-1,j,k,2)) / dx[0] - 0.5*(VectOld[bi](i,j+1,k,1) - VectOld[bi](i,j-1,k,1)) / dx[1];

        // solve magGradRho
       double GradRhoX = 0.5*(VectOld[bi](i,j+1,k,0) - VectOld[bi](i,j-1,k,0)) / dx[0];
       double GradRhoY = 0.5*(VectOld[bi](i+1,j,k,0) - VectOld[bi](i-1,j,k,0)) / dx[1];
       VectNew[bi](i,j,k,6) = sqrt(pow(GradRhoX,2) + pow(GradRhoY,2));
       
        
        
         //limiter
        if (VectNew[bi](i,j,k,3) <= 0)
        {
            if (pressureLimiter)
            {
                 amrex::Print() << "Warning! Pressure less then 0." << "\n";
                 VectNew[bi](i,j,k,3) = 0.01;
                 VectNew[bi](i,j,k,3) = (sqrt(pow(VectNew[bi](i+1,j,k,3),2.)  + pow(VectNew[bi](i-1,j,k,3),2.) )
                                       + sqrt(pow(VectNew[bi](i,j+1,k,3),2.)  + pow(VectNew[bi](i,j-1,k,3),2.) ) 
                                       + sqrt(pow(VectNew[bi](i-1,j+1,k,3),2.)+ pow(VectNew[bi](i+1,j-1,k,3),2.))
                                       + sqrt(pow(VectNew[bi](i+1,j+1,k,3),2.)+ pow(VectNew[bi](i-1,j-1,k,3),2.))) / 4;
                                        //+ (sqrt(pow(VectNew[bi](i-2,j,k,3),2.) + pow(VectNew[bi](i+2,j,k,3),2.)))/2
                                        //+ (sqrt(pow(VectNew[bi](i,j-2,k,3),2.) + pow(VectNew[bi](i,j+2,k,3),2.)))/2) /3;

                 //VectNew[bi](i,j,k,0) = 0.0;
                 VectNew[bi](i,j,k,0) = ((sqrt(pow(VectNew[bi](i+1,j,k,0),2.)  + pow(VectNew[bi](i-1,j,k,0),2.) )
                                       + sqrt(pow(VectNew[bi](i,j+1,k,0),2.)  + pow(VectNew[bi](i,j-1,k,0),2.) ) 
                                       + sqrt(pow(VectNew[bi](i-1,j+1,k,0),2.)+ pow(VectNew[bi](i+1,j-1,k,0),2.))
                                       + sqrt(pow(VectNew[bi](i+1,j+1,k,0),2.)+ pow(VectNew[bi](i-1,j-1,k,0),2.))) / 4 //;
                                        + (sqrt(pow(VectNew[bi](i-2,j,k,0),2.) + pow(VectNew[bi](i+2,j,k,0),2.)))/2
                                        + (sqrt(pow(VectNew[bi](i,j-2,k,0),2.) + pow(VectNew[bi](i,j+2,k,0),2.)))/2) /3;

             }
        }
    });

    Real maxval = S_new.max(0);
    Real minval = S_new.min(0);
    amrex::Print() << "min/max rho = " << minval << "/" << maxval;
    maxval = S_new.max(4);
    minval = S_new.min(4);
    amrex::Print() << "  min/max Sc number = " << minval << "/" << maxval << "\n";

    return dt;
}
