#ifndef telAutoConstantsH
#define telAutoConstantsH
#include <vector>
#include <string>
#include "telProperties.h"
#include "telProperty.h"
#include "tai_exporter.h"
//---------------------------------------------------------------------------

namespace rr
{
class Properties;
}

namespace telauto
{
enum ScanDirection {sdPositive, sdNegative};
class SetupControl;
class AutoTellurimInterface;
using rr::Properties;
using std::vector;
using std::string;

class AutoConstants
{
    public:
        ScanDirection   mScanDirection;
		int				PreSimulation;
		double			PreSimulationStart;
		double			PreSimulationDuration;
		int				PreSimulationSteps;
    public:
        // NDIM: dimension of the system of equations, as specified in the user-supplied subroutine 'func'
        int             NDIM;

        /// IPS: Constant defining the problem type (-2 -> 14) : 1 : stationary states 2 : periodic solutions 4 : BVP
        int             IPS;

        /// Restart label
        int             IRS;

        /// Locate limit points (folds)
        int             ILP;

        /// the number of free parameters
        int             NICP;

        /// the free parameters
        vector<int>     ICP;

        /// the number of mesh intervals
        int             NTST;

        /// the number of collocation points per mesh interval
        int             NCOL;

        /// Mesh adaption:
        ///     0 :fixedmesh
        ///   > 0 : adapt mesh every IAD continuation steps
        int             IAD;

        /// Detection of branch points:
        ///   0 : don’t detect
        ///   1 : detect for algebraic problems
        ///   2 : detect for all problems
        int             ISP;

        /// At a restart solution:
        ///   1 : just follow the branch
        ///  -1 : switch branches (for BPs in BVPs)
        ///   2 : two-parameter continuation (LPs, HBs, PDs, TRs)
        int             ISW;

        /// The ”principal solution measure”
        /// (the second real column in the bifurcation diagram file):
        ///      0 : the L2 norm
        ///   != 0 : other measures (see manual)
        int             IPLT;

        /// the number of boundary conditions (for BVP)
        int             NBC;

        /// the number of integral constraints (for BVP)
        int             NINT;

        /// maximum number of continuation step along any branch
        int             NMX;

        /// bifurcation diagram lower limit for the “first” parameter
        double          RL0;

        /// bifurcation diagram upper limit for the “first” parameter
        double          RL1;

        /// bifurcation diagram lower limit for the “principal solution measure”
        double          A0;

        /// bifurcation diagram upper limit for the “principal solution measure”
        double          A1;

        /// Save the solution in the solution file every NPR continuation steps
        int             NPR;

        /// For algebraic problems: Maximum number of branch switches (if negative only one “leg” will be computed)
        int             MXBF;

        /// Controls the diagnostic file output (normal value is 2; see manual)
        int             IID;

        /// maximum number of iterations for detecting “special” solutions
        int             ITMX;

        /// maximum number of Newton+Chord iterations
        int             ITNW;

        /// maximum number of Newton iterations
        int             NWTN;

        /// Jacobian supplied ?
        int             JAC;

        /// Newton/Chord convergence tolerance for free parameters
        double          EPSL;

        /// Newton/Chord convergence tolerance for solutions
        double          EPSU;

        /// Continuation step size tolerance when locating special solutions
        double          EPSS;

        /// Nominal continuation step size
        double          DS;

        /// Minimum continuation step size
        double          DSMIN;

        /// Maximum continuation step size
        double          DSMAX;

        /// Adapt the continuation step size every IADS steps
        int             IADS;

        /// the number of modified parameter “weights” (for BVP)
        int             NTHL;

        /// <summary>
        /// parameter index, parameter weight (e.g., ICP(11)=0 means PAR(11) is excluded from the step size)
        /// </summary>
        vector<int>     THL;

        /// the number of modified solution component “weights” (for BVP)
        int             NTHU;

        /// component index, component weight
        vector<int>     THU;

        /// the number of “user output points” specified
        int             NUZR;

        /// parameter index, parameter value (if I is negative the continuation stops at the parameter value)
        vector<int>     UZR;


    public:
                        AutoConstants();
        void            populateFrom(Properties* props);
        string          getConstantsAsString();
        string          getICP();
        void            reset();
};

}

//namespace tlp
//{
//template<>
//inline std::string Property< rrauto::ScanDirection >::getType() const
//{
//    return "ScanDirection";
//}
//
//template<>
//inline string Property< rrauto::ScanDirection >::getValueAsString() const
//{
//    return mValue == rrauto::sdPositive ? "Positive" : "Negative";
//}
//
//template<>
//inline void Property< rrauto::ScanDirection >::setValueFromString(const string& val)
//{
//    mValue = compareNoCase(val, "Positive") == 0 ? rrauto::sdPositive : rrauto::sdNegative;
//}
//
//template<>
//inline void Property< rrauto::ScanDirection >::setValue(const rrauto::ScanDirection& val)
//{
//    mValue = (val);
//}
//}

#endif


