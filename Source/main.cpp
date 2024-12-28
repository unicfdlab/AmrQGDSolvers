#include <AMReX.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>

#include <chrono>

using namespace amrex;

amrex::LevelBld* getLevelBld ();

int main (int argc, char* argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    amrex::Initialize(argc,argv);

    int  max_step = -1;
    Real strt_time = 0.0;
    Real stop_time = -1.0;
    {
        ParmParse pp;
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);
    }
    if (max_step < 0 && stop_time < 0.0) {
        amrex::Abort("Exiting because neither max_step nor stop_time is non-negative.");
    }

    {
    //    auto amr = std::make_unique<AmrShallowWater>(getLevelBld());
        Amr amr(getLevelBld());


        amr.init(strt_time,stop_time);

        while ( amr.okToContinue() &&
                (amr.levelSteps(0) < max_step || max_step < 0) &&
                (amr.cumTime() < stop_time || stop_time < 0.0) )
        {
            amr.coarseTimeStep(stop_time);
        }

        // Write final checkpoint and plotfile
        if (amr.stepOfLastCheckPoint() < amr.levelSteps(0)) {
            amr.checkPoint();
        }
        if (amr.stepOfLastPlotFile() < amr.levelSteps(0)) {
            amr.writePlotFile();
        }
    }

    amrex::Finalize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << duration.count() << " seconds\n";
}
