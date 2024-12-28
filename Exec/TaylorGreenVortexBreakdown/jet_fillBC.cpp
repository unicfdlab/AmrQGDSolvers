#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct QGDBCFill 
{
        AMREX_GPU_DEVICE
        void operator() (const IntVect& iv, Array4<Real> const& dest,
                         const int /*dcomp*/, const int /*numcomp*/,
                         GeometryData const& geom, const Real /*time*/,
                         const BCRec* /*bcr*/, const int /*bcomp*/,
                         const int /*orig_comp*/) const
        {
            const int ilo = geom.Domain().smallEnd(0);
            const int ihi = geom.Domain().bigEnd(0);
            const int jlo = geom.Domain().smallEnd(1);
            const int jhi = geom.Domain().bigEnd(1);
            const auto problo = geom.ProbLo();//data();
            const auto [i,j,k] = iv.dim3();

             // const auto problo = data1.ProbLo();
            const auto dx = geom.CellSize();
            if (i > ihi) {
                dest(i,j,k,0) = dest(ilo,j,k,0);
                dest(i,j,k,1) = dest(ilo,j,k,1);
                dest(i,j,k,2) = dest(ilo,j,k,2);
                dest(i,j,k,3) = dest(ilo,j,k,3);
                dest(i,j,k,4) = dest(ilo,j,k,4);
                dest(i,j,k,5) = dest(ilo,j,k,5);
                dest(i,j,k,6) = dest(ilo,j,k,6);
            }
            if (i < ilo) {
                dest(i,j,k,0) = dest(ihi,j,k,0);
                dest(i,j,k,1) = dest(ihi,j,k,1);
                dest(i,j,k,2) = dest(ihi,j,k,2);
                dest(i,j,k,3) = dest(ihi,j,k,3);
                dest(i,j,k,4) = dest(ihi,j,k,4);
                dest(i,j,k,5) = dest(ihi,j,k,5);
                dest(i,j,k,6) = dest(ihi,j,k,6);
            }
            if (j > jhi) {
                dest(i,j,k,0) = dest(i,jlo,k,0);
                dest(i,j,k,1) = dest(i,jlo,k,1);
                dest(i,j,k,2) = dest(i,jlo,k,2);
                dest(i,j,k,3) = dest(i,jlo,k,3);
                dest(i,j,k,4) = dest(i,jlo,k,4);
                dest(i,j,k,5) = dest(i,jlo,k,5);
                dest(i,j,k,6) = dest(i,jlo,k,6);
            }
            if (j < jlo) {
                dest(i,j,k,0) = dest(i,jhi,k,0);
                dest(i,j,k,1) = dest(i,jhi,k,1);
                dest(i,j,k,2) = dest(i,jhi,k,2);
                dest(i,j,k,3) = dest(i,jhi,k,3);
                dest(i,j,k,4) = dest(i,jhi,k,4);
                dest(i,j,k,5) = dest(i,jhi,k,5);
                dest(i,j,k,6) = dest(i,jhi,k,6);
            }

            //----------angle cells----------//

            if(i > ihi && j > jhi) {
                dest(i,j,k,0) = dest(ilo,jlo,k,0);
                dest(i,j,k,1) = dest(ilo,jlo,k,1);
                dest(i,j,k,2) = dest(ilo,jlo,k,2);
                dest(i,j,k,3) = dest(ilo,jlo,k,3);
                dest(i,j,k,4) = dest(ilo,jlo,k,4);
                dest(i,j,k,5) = dest(ilo,jlo,k,5);
                dest(i,j,k,6) = dest(ilo,jlo,k,6);
            }

            if(i > ihi && j < jlo) {
                dest(i,j,k,0) = dest(ilo,jhi,k,0);
                dest(i,j,k,1) = dest(ilo,jhi,k,1);
                dest(i,j,k,2) = dest(ilo,jhi,k,2);
                dest(i,j,k,3) = dest(ilo,jhi,k,3);
                dest(i,j,k,4) = dest(ilo,jhi,k,4);
                dest(i,j,k,5) = dest(ilo,jhi,k,5);
                dest(i,j,k,6) = dest(ilo,jhi,k,6);
            }
            if(i < ilo && j > jhi) {
                dest(i,j,k,0) = dest(ihi,jlo,k,0);
                dest(i,j,k,1) = dest(ihi,jlo,k,1);
                dest(i,j,k,2) = dest(ihi,jlo,k,2);
                dest(i,j,k,3) = dest(ihi,jlo,k,3);
                dest(i,j,k,4) = dest(ihi,jlo,k,4);
                dest(i,j,k,5) = dest(ihi,jlo,k,5);
                dest(i,j,k,6) = dest(ihi,jlo,k,6);
            }            
            if(i < ilo && j < jlo) {
                dest(i,j,k,0) = dest(ihi,jhi,k,0);
                dest(i,j,k,1) = dest(ihi,jhi,k,1);
                dest(i,j,k,2) = dest(ihi,jhi,k,2);
                dest(i,j,k,3) = dest(ihi,jhi,k,3);
                dest(i,j,k,4) = dest(ihi,jhi,k,4);
                dest(i,j,k,5) = dest(ihi,jhi,k,5);
                dest(i,j,k,6) = dest(ihi,jhi,k,6);
            }
        }
};

void bcfill (Box const& bx, FArrayBox& data,
             int dcomp, int numcomp,
             Geometry const& geom, Real time,
             const Vector<BCRec>& bcr, int bcomp,int scomp)
{
    GpuBndryFuncFab<QGDBCFill> gpu_bndry_func(QGDBCFill{});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}
