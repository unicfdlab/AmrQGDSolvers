#include "AmrQGD.H"

#include <AMReX_LevelBld.H>

using namespace amrex;

class QGDLevelBld
    : public LevelBld
{
    virtual void variableSetUp () override;
    virtual void variableCleanUp () override;
    virtual AmrLevel *operator() () override;
    virtual AmrLevel *operator() (Amr& amr, int lev, const Geometry& gm,
                                  const BoxArray& ba, const DistributionMapping& dm,
                                  Real time) override;
};

QGDLevelBld QGD_bld;

LevelBld*
getLevelBld ()
{
    return &QGD_bld;
}

void
QGDLevelBld::variableSetUp ()
{
    AmrQGD::variableSetUp();
}

void
QGDLevelBld::variableCleanUp ()
{
    AmrQGD::variableCleanUp();
}

AmrLevel*
QGDLevelBld::operator() ()
{
    return new AmrQGD;
}

AmrLevel*
QGDLevelBld::operator() (Amr& amr, int lev, const Geometry& gm,
                         const BoxArray& ba, const DistributionMapping& dm,
                         Real time)
{
    return new AmrQGD(amr, lev, gm, ba, dm, time);
}
