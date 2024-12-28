#ifndef AmrQGD_H_
#define AmrQGD_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_ParmParse.H>

enum StateType { State_Type = 0, NUM_STATE_TYPE};

class AmrQGD 
    :
    public amrex::AmrLevel 
{
public:

    AmrQGD () = default;

    AmrQGD (amrex::Amr& parent, 
            int lev, 
            const amrex::Geometry& gm,
            const amrex::BoxArray& ba, 
            const amrex::DistributionMapping& dm,
                  amrex::Real time);

    virtual ~AmrQGD ();

    //! Define data descriptors.
    static void variableSetUp ();

    //! Cleanup data descriptors at end of run.
    static void variableCleanUp ();

    //! Initialize data at problem start-up.
    virtual void initData () override;

    //! Initialize data on this level from the previous AmrQGD at the
    //! same level during regridding.
    virtual void init (amrex::AmrLevel& old) override;

    //! Initialize data on this level during regridding when this level did
    //! not exist before the regrid.
    virtual void init () override;

    /**
     * \brief Advance this level for one step
     *
     * \param time      time at the beginning of the step
     * \param dt        time step
     * \param iteration iteration number.  For an AMR simulation with
     *                  subcycling and a refinement ratio of 2, the number
     *                  is either 1 or 2.  If this is level 0, it's always
     *                  1.  If it's a fine level, this represents the substep
     *                  number during a coarser level time step.
     * \parame ncycle   number of subcyling steps. It's 1 without subcycling,
     *                  or on level 0.  It's usually 2 or 4 on fine levels when
     *                  subcycling is on.
     */
    virtual amrex::Real advance (amrex::Real time, amrex::Real dt,
                                 int iteration, int ncycle) override;

    //! Compute initial dt.  Sometimes, we might want to start a simulation
    //! with a smaller time step.  The output is dt_level.
    virtual void computeInitialDt (int finest_level, int sub_cycle,
                                   amrex::Vector<int>& n_cycle,
                                   const amrex::Vector<amrex::IntVect>& ref_ratio,
                                   amrex::Vector<amrex::Real>& dt_level,
                                   amrex::Real stop_time) override;

    //! Compute new dt.  The outputs are dt_min and dt_level.
    virtual void computeNewDt (int finest_level, int sub_cycle,
                               amrex::Vector<int>& n_cycle,
                               const amrex::Vector<amrex::IntVect>& ref_ratio,
                               amrex::Vector<amrex::Real>& dt_min,
                               amrex::Vector<amrex::Real>& dt_level,
                               amrex::Real stop_time, int post_regrid_flag) override;

    //! Do work after each time step
    virtual void post_timestep (int iteration) override;

    //! Do work after regrid().
    virtual void post_regrid (int /*lbase*/, int /*new_finest*/) override {}

    //! Do work after init().
    virtual void post_init (amrex::Real /*stop_time*/) override {};

    //! Error estimation for regridding.
    virtual void errorEst (amrex::TagBoxArray& tb, int clearval, int tagval,
                           amrex::Real time, int n_error_buf = 0,
                           int ngrow = 0) override;

    // Need to be public for CUDA
    void computeRHS (amrex::MultiFab& dSdt, amrex::MultiFab const& S);


private:

    static constexpr int ncomp  = 7; //2; // 4 variables: rho & u & v & p & Sc & curl & magGradRho
    static constexpr int nghost = 2; //2; // Two ghost cells needed

    static int verbose;
    static amrex::Real cfl;
    static amrex::Real deltaT0;
    static int refcond;
    static amrex::Real refdengrad;
    
    //Gas properties
    static amrex::Real gamma;
    static amrex::Real RGas;
    static amrex::Real PrGas;
    static amrex::Real mutGas;
    
    //QGD properties
    static amrex::Real alphaQgd;
    static amrex::Real ScQgd;
    static amrex::Real PrQgd;
    static bool varScQgd;
    static bool pressureLimiter;
    static amrex::Real gradVal;

    //! Function to read parameters from input file.
    static void read_params ();

    //! Get AmrQGD
    AmrQGD& getLevel (int lev) {
        return static_cast<AmrQGD&>(parent->getLevel(lev));
    }
};

    /**
    * Boundary condition function that does nothing.
    * See fillBC.cpp for implementation.
    */

   void bcfill (amrex::Box const& bx, amrex::FArrayBox& data,
                  int dcomp, int numcomp,
                  amrex::Geometry const& geom, amrex::Real time,
                  const amrex::Vector<amrex::BCRec>& bcr, int bcomp,int scomp);

#endif /*_AmrQGD_H_*/
