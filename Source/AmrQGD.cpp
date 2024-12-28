#include "AmrQGD.H"

#include <AMReX_ParmParse.H>
#include <numeric>

using namespace amrex;

constexpr int AmrQGD::ncomp;
constexpr int AmrQGD::nghost;
int  AmrQGD::verbose = 0;
Real AmrQGD::cfl = 0.2;
Real AmrQGD::deltaT0 = 0.1;
int AmrQGD::refcond = 0;
Real AmrQGD::refdengrad = 1.2;

Real AmrQGD::gamma = 1.4;
Real AmrQGD::RGas = 287;
Real AmrQGD::PrGas = 0.7;
Real AmrQGD::mutGas = 0.0;

Real AmrQGD::alphaQgd = 0.3;
Real AmrQGD::ScQgd = 1.0;
Real AmrQGD::PrQgd = 0.7;
bool AmrQGD::varScQgd = false;
bool AmrQGD::pressureLimiter = true;
Real AmrQGD::gradVal = 30;

AmrQGD::AmrQGD (Amr& amr, int lev, const Geometry& gm,
                            const BoxArray& ba, const DistributionMapping& dm,
                            Real time)
    : AmrLevel(amr,lev,gm,ba,dm,time)
{}

AmrQGD::~AmrQGD () {}

void
AmrQGD::variableSetUp ()
{
    read_params();

    desc_lst.addDescriptor(State_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, nghost, ncomp,
                           &cell_quartic_interp);


    int lo_bc[BL_SPACEDIM] = {AMREX_D_DECL(BCType::foextrap,
                                           BCType::foextrap,
                                           BCType::foextrap) };
    int hi_bc[BL_SPACEDIM] = {AMREX_D_DECL(BCType::foextrap,
                                           BCType::foextrap,
                                           BCType::foextrap) };

    Vector<BCRec> bcs(ncomp, BCRec(lo_bc, hi_bc));

    StateDescriptor::BndryFunc bndryfunc(bcfill);
    bndryfunc.setRunOnGPU(true);

    desc_lst.setComponent(State_Type, 0, {"rho", "ux", "uy","p", "Sc", "curl", "magGradRho"}, bcs, bndryfunc);
    
}

void
AmrQGD::variableCleanUp ()
{
    desc_lst.clear();
}

void
AmrQGD::init (AmrLevel &old)
{
    Real dt_new    = parent->dtLevel(Level());
    Real cur_time  = old.get_state_data(State_Type).curTime();
    Real prev_time = old.get_state_data(State_Type).prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        MultiFab& S_new = get_new_data(k);
        FillPatch(old, S_new, 0, cur_time, k, 0, ncomp);
    }
}

void
AmrQGD::init ()
{
    Real dt        = parent->dtLevel(Level());
    Real cur_time  = getLevel(Level()-1).state[State_Type].curTime();
    Real prev_time = getLevel(Level()-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/(Real)parent->MaxRefRatio(Level()-1);
    setTimeLevel(cur_time,dt_old,dt);

    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        MultiFab& S_new = get_new_data(k);
        FillCoarsePatch(S_new, 0, cur_time, k, 0, ncomp);
    }
}

void
AmrQGD::computeInitialDt (int finest_level, int /*sub_cycle*/,
                                Vector<int>& n_cycle,
                                const Vector<IntVect>& /*ref_ratio*/,
                                Vector<Real>& dt_level, Real stop_time)
{
    if (Level() > 0) { return; } // Level 0 does this for every level.

    Vector<int> nsteps(n_cycle.size()); // Total number of steps in one level 0 step
    std::partial_sum(n_cycle.begin(), n_cycle.end(), nsteps.begin(),
                     std::multiplies<int>());

    Real dt_0 = deltaT0;//std::numeric_limits<Real>::max();
    for (int ilev = 0; ilev <= finest_level; ++ilev) 
    {
         const auto dx = parent->Geom(ilev).CellSizeArray();
         Real dtlev = cfl * std::min({AMREX_D_DECL(dx[0],dx[1],dx[2])});
         dt_0 = std::min(dt_0, nsteps[ilev] * dtlev);
    }
    // // dt_0 will be the time step on level 0 (unless limited by stop_time).

    if (stop_time > 0) 
    {
         // Limit dt's by the value of stop_time.
         const Real eps = 0.001 * dt_0;
         const Real cur_time = get_state_data(State_Type).curTime();
         if ((cur_time + dt_0) > (stop_time - eps)) {
             dt_0 = stop_time - cur_time;
         }
    }

    for (int ilev = 0; ilev <= finest_level; ++ilev) {
        dt_level[ilev] = dt_0/std::pow(2, ilev);// / Real(nsteps[ilev]);
    }
}

void
AmrQGD::computeNewDt (int finest_level, int sub_cycle,
                            Vector<int>& n_cycle,
                            const Vector<IntVect>& ref_ratio,
                            Vector<Real>& dt_min, Vector<Real>& dt_level,
                            Real stop_time, int post_regrid_flag)
{
    // For this code we can just call computeInitialDt.
    computeInitialDt(finest_level, sub_cycle, n_cycle, ref_ratio, dt_level, stop_time);
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
}

void
AmrQGD::post_timestep (int iteration)
{
    //
    // Integration cycle on fine level grids is complete
    // do post_timestep stuff here.
    //
    int finest_level = parent->finestLevel();

    if (Level() < parent->finestLevel()) {
        auto& fine_level = getLevel(Level()+1);
        MultiFab& S_fine = fine_level.get_new_data(State_Type);
        MultiFab& S_crse =      this->get_new_data(State_Type);
        Real t = get_state_data(State_Type).curTime();

        IntVect ratio = parent->refRatio(Level());
        AMREX_ASSERT(ratio == 2 || ratio == 3);
        if (ratio == 2) 
        {
            // Need to fill one ghost cell for the high-order interpolation below
            // maybe 2 chande to 1 aor ghost cells
            FillPatch(fine_level, S_fine, 2, t, State_Type, 0, ncomp);
        }

        FourthOrderInterpFromFineToCoarse(S_crse, 0, ncomp, S_fine, ratio);
    }

    if (level < finest_level) {
        // fillpatcher on level+1 needs to be reset because data on this
        // level have changed.
        getLevel(level+1).resetFillPatcher();
    }

    AmrLevel::post_timestep(iteration);
}

/**
 * Do work after init().
 */
//void
//AmrQGD::post_init (Real /*stop_time*/)
//{
//    if (level > 0) {
//        return;
//    }
    //
    // Average data down from finer levels
    // so that conserved data is consistent between levels.
    //
//    int finest_level = parent->finestLevel();
//    for (int k = finest_level-1; k>= 0; k--) {
//        getLevel(k).resetFillPatcher();
//    }
//}

void
AmrQGD::errorEst (TagBoxArray& tags, int clearval, int tagval,
                        Real /*time*/, int /*n_error_buf*/, int /*ngrow*/)
{
    const auto problo = Geom().ProbLoArray();
    const auto probhi = Geom().ProbHiArray();
    const auto dx = Geom().CellSizeArray(); 
    auto const& S_new = get_new_data(State_Type);
    //const char tagval = TagBox::SET;
    auto const& a = tags.arrays();
    auto const& s = S_new.const_arrays();
    amrex::ParallelFor(tags, [&] AMREX_GPU_DEVICE (int bi, int i, int j, int k) 
    {
        if (refcond == 0) //grad(Ux)
        {
            if (amrex::Math::abs(s[bi](i,j,k,2)-s[bi](i-1,j,k,2))/dx[0] > refdengrad or 
                amrex::Math::abs(s[bi](i,j,k,2)-s[bi](i,j-1,k,2))/dx[1] > refdengrad or 
                amrex::Math::abs(s[bi](i,j,k,2)-s[bi](i+1,j,k,2))/dx[0] > refdengrad or 
                amrex::Math::abs(s[bi](i,j,k,2)-s[bi](i,j+1,k,2))/dx[1] > refdengrad) 
//             if (sqrt(s[bi](i,j,k,2)*s[bi](i,j,k,2) + s[bi](i,j,k,1)*s[bi](i,j,k,1)) > 2.3) 
            {
                a[bi](i,j,k) = tagval;
            } else {
                a[bi](i,j,k) = clearval;
            }
        }
        else if (refcond == 1) //grad(rho)
        {
            if (amrex::Math::abs(s[bi](i,j,k,0)-s[bi](i-1,j,k,0))/dx[0] > refdengrad or 
                amrex::Math::abs(s[bi](i,j,k,0)-s[bi](i,j-1,k,0))/dx[1] > refdengrad or 
                amrex::Math::abs(s[bi](i,j,k,0)-s[bi](i+1,j,k,0))/dx[0] > refdengrad or 
                amrex::Math::abs(s[bi](i,j,k,0)-s[bi](i,j+1,k,0))/dx[1] > refdengrad) 
            {
                a[bi](i,j,k) = tagval;
            } else {
                a[bi](i,j,k) = clearval;
            }
        }
        else if (refcond == 2) //localRe
        {
            if (mutGas > 0)
            {
                if ((sqrt( s[bi](i,j,k,2)*s[bi](i,j,k,2)+s[bi](i,j,k,1)*s[bi](i,j,k,1))*dx[0]/mutGas) > refdengrad)
                {
                    a[bi](i,j,k) = tagval;
                } else {
                    a[bi](i,j,k) = clearval;
                }
             }
             else
             {
                if ((sqrt( s[bi](i,j,k,2)*s[bi](i,j,k,2)+s[bi](i,j,k,1)*s[bi](i,j,k,1))) > refdengrad)
                {
                    a[bi](i,j,k) = tagval;
                } else {
                    a[bi](i,j,k) = clearval;
                }
             }
        }
        else if (refcond == 3) //
        {
            amrex::Real ax = std::abs(s[bi](i+1,j,k,0) - s[bi](i,j,k,0));
            amrex::Real ay = std::abs(s[bi](i,j+1,k,0) - s[bi](i,j,k,0));
            ax = amrex::max(ax,std::abs(s[bi](i,j,k,0) - s[bi](i-1,j,k,0)));
            ay = amrex::max(ay,std::abs(s[bi](i,j,k,0) - s[bi](i,j-1,k,0)));

            if (amrex::max(ax,ay) >= refdengrad)
            {
                a[bi](i,j,k) = tagval;
            }
            else
            {
                a[bi](i,j,k) = clearval;
            }
        }
        else if (refcond == 4) //vorticity
        {
            amrex::Real gradRhoX  = 0.5*(s[bi](i,j+1,k,0) - s[bi](i,j-1,k,0)) / dx[0];
            amrex::Real gradRhoY  = 0.5*(s[bi](i+1,j,k,0) - s[bi](i-1,j,k,0)) / dx[1];
            amrex::Real gradRho   = sqrt(pow(gradRhoX,2) + pow(gradRhoY,2));
            amrex::Real vorticity = std::abs(0.5*(s[bi](i+1,j,k,2) - s[bi](i-1,j,k,2)) / dx[0] + 0.5*(s[bi](i,j+1,k,1) - s[bi](i,j-1,k,1)) / dx[1]);

            if ((gradRho >= refdengrad) || (vorticity >= 1000*refdengrad))
            {
                a[bi](i,j,k) = tagval;
            }
            else
            {
                a[bi](i,j,k) = clearval;
            }
        }
    });
}

void
AmrQGD::read_params ()
{
    ParmParse pp("qgdSolver");
    pp.query("v", verbose); // Could use this to control verbosity during the run
    pp.query("cfl", cfl);
    pp.query("deltaT0", deltaT0);
    pp.query("refine_dengrad", refdengrad);
    pp.query("refine_condition", refcond);
    if (refcond >= 5)
    {
         amrex::Print() << "Refinement condtion does not exist!" << "\n";
         refcond = 0;
    }
  

    ParmParse pp1("gasProperties");
    pp1.query("gamma", gamma);
    pp1.query("R", RGas);
    pp1.query("Pr", PrGas);
    pp1.query("mut", mutGas);
    
    ParmParse pp2("qgd");
    pp2.query("alphaQgd", alphaQgd);
    pp2.query("ScQgd", ScQgd);
    pp2.query("PrQgd", PrQgd);
    pp2.query("varScQgd", varScQgd);
    pp2.query("dengradVal", gradVal);
    pp2.query("pressure_limiter", pressureLimiter);
    
}