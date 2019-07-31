// $Id: auto_api.h,v 1.10 2004/10/28 01:45:39 emery Exp $

#ifndef __AUTO_API_H__
#define __AUTO_API_H__

// make sure this is C++...
#ifndef __cplusplus
#error auto_api.h requires a C++ compiler
#endif

//#include "auto.h"
#include <cmath>

#ifndef __AUTO_API_IMPLEMENTATION__
typedef long integer;
typedef double doublereal;
#endif

#include <vector>
#include <string>
#include <iostream>

namespace autolib {

// declare the AutoApi type for prior use
class AutoApi;

//! to create continuation constants, these constants indicate how
//! to initialize the internal structure best for the given type.
enum ConstantsInitType
{
    DEFAULT                   = 0,
    STEADY_STATE_CONTINUATION = 1
};

//! when inserting things into lists, use these to spec how we do that
//! (this is only currently relevant for PlotSet)
enum InsertType
{
    FORWARD = 0,
    REVERSE = 1
};

////! anything used by the API via a pointer interface should inherit from this.
////! this is so the user can create his own structures and not have the API delete
////! it when the API is closed down. if the API must create something for itself,
////! then it can set the api_owns flag and then remember to delete the construct.
//class AutoApiData
//{
//  friend class AutoApi;
//
//  bool api_owns;
//
//public:
//  AutoApiData(bool o = false) : api_owns(o) {}
//};
//
////! a struct to hold all the relevant user-settable numerical parameters
//struct ContinuationConstants : public AutoApiData
//{
//public:
//  //!< dimension of the system
//  integer ndim;
//  
//  //! defines problem type.
//  //! This constant defines the problem type :
//  //! IPS=0 : An algebraic bifurcation problem. Hopf bifurcations will not
//  //!     be detected and stability properties will not be indicated in the fort.7
//  //!     outputfile.
//  //! IPS=1 : Stationary solutions of ODEs with detection of Hopf bifurcations.
//  //!     The sign of PT, the point number, in fort.7 is used to indicate
//  //!     stability :  - is stable , + is unstable.
//  //!     (Demo ab.)
//  //! IPS=-1 : Fixed points of the discrete dynamical system u^(k+1) = f(u^k,p),
//  //!     with detection of Hopf bifurcations. The sign of PT in
//  //!     fort.7 indicates stability :  - is stable , + is unstable. (Demo dd2.)
//  //!     IPS=2 : Time integration using implicit Euler. The AUTOconstants
//  //!     DS, DSMIN, DSMAX, and ITNW, NWTN control the stepsize. In fact, pseudo
//  //!     arclength is used for continuation in time. Note that the time dis
//  //!     cretization is only first order accurate, so that results should be care
//  //!     fully interpreted. Indeed, this option has been included primarily for
//  //!     the detection of stationary solutions, which can then be entered in the
//  //!     user-supplied subroutine stpnt.
//  //!     (Demo ivp.)
//  //! IPS=2 : Computation of periodic solutions. Starting data can be a Hopf
//  //!     bifurcation point (Run 2 of demo ab), a periodic orbit from a previous
//  //!     run (Run 4 of demo pp2), an analytically known periodic orbit (Run 1
//  //!     of demo frc), or a numerically known periodic orbit (Demo lor). The
//  //!     sign of PT in fort.7 is used to indicate stability : - is stable , + is
//  //!     unstable or unknown.
//  //! IPS=4 : A boundary value problem. Boundary conditions must be
//  //!     specified in the user-supplied subroutine bcnd and integral constraints
//  //!     in icnd. The AUTOconstants NBC and NINT must be given correct
//  //!     values. (Demos exp, int, kar.)
//  //! IPS=5 : Algebraic optimization problems. The objective function must
//  //!     be specified in the user-supplied subroutine fopt. (Demo opt.)
//  //! IPS=7 : A boundary value problem with computation of Floquet multi
//  //!     pliers. This is a very special option; for most boundary value problems
//  //!     one should use IPS=4. Boundary conditions must be specified in the
//  //!     user-supplied subroutine bcnd and integral constraints in icnd. The
//  //!     AUTOconstants NBC and NINT must be given correct values.
//  //! IPS=9 : This option is used in connection with the HomCont algo
//  //!     rithms described in Chapters  for the detection and continuation
//  //!     of homoclinic bifurcations.
//  //!     (Demos san, mtn, kpr, cir, she, rev.)
//  //! IPS=11 : Spatially uniform solutions of a system of parabolic PDEs,
//  //!     with detection of traveling wave bifurcations. The user need only
//  //!     define the nonlinearity (in subroutine func), initialize the wave
//  //!     speed in PAR(10), initialize the diffusion constants in PAR(15,16,...),
//  //!     and set a free equation parameter in ICP(1). (Run 2 of demo wav.)
//  //! IPS=12 : Continuation of traveling wave solutions to a system of parabolic
//  //!     PDEs. Starting data can be a Hopf bifurcation point from a previous
//  //!     run with IPS=11, or a traveling wave from a previous run with IPS=12.
//  //!     (Run 3 and Run 4 of demo wav.)
//  //! IPS=14 : Time evolution for a system of parabolic PDEs subject to
//  //!     periodic boundary conditions. Starting data may be solutions from a
//  //!     previous run with IPS=12 or 14. Starting data can also be specified in
//  //!     stpnt, in which case the wave length must be specified in PAR(11), and
//  //!     the diffusion constants in PAR(15,16,...). AUTO uses PAR(14) for the
//  //!     time variable. DS, DSMIN, and DSMAX govern the pseudoarclength con
//  //!     tinuation in the spacetime variables. Note that the time discretization
//  //!     is only first order accurate, so that results should be carefully inter
//  //!     preted. Indeed, this option is mainly intended for the detection of
//  //!     stationary waves. (Run 5 of demo wav.)
//  //! IPS=15 : Optimization of periodic solutions. The integrand of the
//  //!     objective functional must be specified in the user-supplied subroutine
//  //!     fopt. Only PAR(19) should be used for problem parameters. PAR(10)
//  //!     is the value of the objective functional, PAR(11) the period, PAR(12)
//  //!     the norm of the adjoint variables, PAR(14) and PAR(15) are internal
//  //!     optimality variables. PAR(2129) and PAR(31) are used to monitor
//  //!     the optimality functionals associated with the problem parameters and
//  //!     the period. Computations can be started at a solution computed with
//  //!     IPS=2 or IPS=15. For a detailed example see demo ops.
//  //!     IPS=16 : This option is similar to IPS=14, except that the user sup
//  //!     plies the boundary conditions. Thus this option can be used for time
//  //!     integration of parabolic systems subject to userdefined boundary con
//  //!     ditions. For examples see the first runs of demos pd1, pd2, and bru.
//  //!     Note that the spacederivatives of the initial conditions must also be
//  //!     supplied in the user-supplied subroutine stpnt. The initial conditions
//  //!     must satisfy the boundary conditions. This option is mainly intended
//  //!     for the detecting stationary solutions.
//  //! IPS=17 : This option can be used to continue stationary solutions of
//  //!     parabolic systems obtained from an evolution run with IPS=16. For
//  //!     examples see the second runs of demos pd1 and pd2.
//  integer ips;
//  integer &problem_type; //!< alias for ips
//
//  //! This constant sets the label of the solution where the computation is to
//  //! be restarted.
//  //! IRS=0 : This setting is typically used in the first run of a new problem.
//  //!     In this case a starting solution must be defined in the user-supplied
//  //!     subroutine stpnt; see also Section . For representative examples of
//  //!     analytical starting solutions see demos ab and frc. For starting from
//  //!     unlabeled numerical data see the @fc command (Section ) and demos
//  //!     lor and pen.
//  //! IRS>0 : Restart the computation at the previously computed solution
//  //!     with label IRS. This solution is normally expected to be in the cur
//  //!     rent datafile q.xxx; see also the @r and @R commands in Section .
//  //!     Various AUTO constants can be modified when restarting.
//  integer irs;
//  integer &restart_label; //!< alias for irs.
//
//  //! ILP=0 : No detection of folds. This choice is recommended.
//  //! ILP=1 : Detection of folds. To be used if subsequent fold continuation
//  //!   is intended.
//  integer ilp;
//  integer &fold_detection; //!< alias for ilp.
//
//  //! For each equation type and for each continuation calculation there is a
//  //! typical (generic) number of problem parameters that must be allowed to
//  //! vary, in order for the calculations to be properly posed. The constant NICP
//  //! indicates how many free parameters have been specified, while the array ICP
//  //! actually designates these free parameters. The parameter that appears first
//  //! in the ICP list is called the principal continuation parameter.  Specific
//  //! examples and special cases are described in the manual.
//  integer nicp;
//  integer &num_free_parameters; //!< alias for nicp.
//
//  std::vector<integer> icp; //!< holding space for the continuation parameter indices
//
//  //! The number of mesh intervals used for discretization. NTST remains fixed
//  //! during any particular run, but can be changed when restarting. Recom
//  //! mended value of NTST : As small as possible to maintain convergence.
//  //! (Demos exp, ab, spb.)
//  integer ntst;
//  integer &num_mesh_intervals; //!< alias for ntst.
//
//  //! The number of Gauss collocation points per mesh interval, (2 < NCOL < 7).
//  //! NCOL remains fixed during any given run, but can be changed when
//  //! restarting at a previously computed solution. The choice NCOL=4, used in
//  //! most demos, is recommended. If NDIM is large and the solutions very
//  //! smooth then NCOL=2 may be appropriate.
//  integer ncol;
//  integer &num_collocation_points; //!< alias for ncol.
//
//  //! This constant controls the mesh adaption :
//  //! IAD=0 : Fixed mesh. Normally, this choice should never be used, as it
//  //!     may result in spurious solutions. (Demo ext.)
//  //! IAD>0 : Adapt the mesh every IAD steps along the branch. Most demos
//  //!     use IAD=3, which is the strongly recommended value.
//  integer iad;
//  integer &mesh_adaptation_control; //!< alais for iad.
//
//  //! This constant controls the detection of branch points, period doubling
//  //! bifurcations, and torus bifurcations.
//  //! ISP=0 : This setting disables the detection of branch points, period
//  //!     doubling bifurcations, and torus bifurcations and the computation of
//  //!     Floquet multipliers.
//  //! ISP=1 : Branch points are detected for algebraic equations, but not
//  //!     for periodic solutions and boundary value problems. Perioddoubling
//  //!     bifurcations and torus bifurcations are not located either. However,
//  //!     Floquet multipliers are computed.
//  //! ISP=2 : This setting enables the detection of all special solutions. For
//  //!     periodic solutions and rotations, the choice ISP=2 should be used with
//  //!     care, due to potential inaccuracy in the computation of the linearized
//  //!     Poincare map and possible rapid variation of the Floquet multipliers.
//  //!     The linearized Poincare map always has a multiplier z = 1. If this mul
//  //!     tiplier becomes inaccurate then the automatic detection of secondary
//  //!     periodic bifurcations will be discontinued and a warning message will
//  //!     be printed in fort.9. See also Section .
//  //! ISP=3 : Branch points will be detected, but AUTO will not monitor
//  //!     the Floquet multipliers. Perioddoubling and torus bifurcations will go
//  //!     undetected. This option is useful for certain problems with nongeneric
//  //!     Floquet behavior.
//  integer isp;
//  integer &special_detection_control; //!< alias for isp.
//
//  //! This constant controls branch switching at branch points for the case of
//  //! differential equations. Note that branch switching is automatic for algebraic
//  //! equations.
//  //! ISW=1  : This is the normal value of ISW.
//  //! ISW=-1 : If IRS is the label of a branch point or a perioddoubling
//  //!     bifurcation then branch switching will be done. For period doubling
//  //!     bifurcations it is recommended that NTST be increased. For examples
//  //!     see Run 2 and Run 3 of demo lor, where branch switching is done at
//  //!     perioddoubling bifurcations, and Run 2 and Run 3 of demo bvp, where
//  //!     branch switching is done at a transcritical branch point.
//  //! ISW=2  : If IRS is the label of a fold, a Hopf bifurcation point, or a
//  //!     perioddoubling or torus bifurcation then a locus of such points will
//  //!     be computed. An additional free parameter must be specified for such
//  //!     continuations.
//  integer isw;
//  integer &branch_switch_control; //!< alias for isw
//
//  //! This constant allows redefinition of the principal solution measure, which
//  //! is printed as the second (real) column in the screen output and in the fort.7
//  //! outputfile :
//  //! If IPLT = 0 then the L2 norm is printed. Most demos use this setting.
//  //! For algebraic problems, the standard definition of L2 norm is used. For
//  //! differential equations, the L2 norm is defined as in the manual.
//  //! Note that the interval of integration is [0,1], the standard interval used
//  //! by AUTO. For periodic solutions the independent variable is trans
//  //! formed to range from 0 to 1, before the norm is computed. The AUTO
//  //! constants THL(*) and THU(*) affect
//  //! the definition of the L2 norm.
//  //! If 0 < IPLT < NDIM then the maximum of the IPLTth solution component
//  //! is printed.
//  //! If -NDIM < IPLT< 0 then the minimum of the IPLTth solution component
//  //! is printed. (Demo fsh.)
//  //! If NDIM < IPLT < 2*NDIM then the integral of the (IPLTNDIM)th
//  //! solution component is printed. (Demos exp, lor.)
//  //! If 2*NDIM < IPLT < 3*NDIM then the L2 norm of the (IPLTNDIM)th
//  //! solution component is printed. (Demo frc.)
//  //! Note that for algebraic problems the maximum and the minimum are
//  //! identical. Also, for ODEs the maximum and the minimum of a solution
//  //! component are generally much less accurate than the L2 norm and
//  //! component integrals. Note also that the subroutine pvls provides
//  //! a second, more general way of defining solution measures.
//  integer iplt;
//  integer &pricipal_solution_measure; //!< alias for iplt.
//
//   //! number of boundary conditions (given in bcnd())
//  integer nbc;
//  integer &num_boundary_conditions; //!< alias for nbc.
//
//  //! number of integral conditions (given in icnd())
//  integer nint;
//  integer &num_integral_conditions; //!< alias for nint.
//
//  //! The maximum number of steps to be taken along any branch.
//  integer nmx;
//  integer &max_branch_steps; //!< alias for nmx.
//
//  //! The lower bound on the principal continuation parameter. (This is the
//  //! parameter which appears first in the ICP list;
//  doublereal rl0;
//  
//  //! The upper bound on the principal continuation parameter.
//  doublereal rl1;
//  
//  //! The lower bound on the principal solution measure. (By default, if
//  //! IPLT=0, the principal solution measure is the L2 norm of the state vector or
//  //! state vector function).
//  doublereal a0;
//  
//  //! The upper bound on the principal solution measure.
//  doublereal a1;
//
//  //! This constant can be used to regularly write fort.8 plotting and restart
//  //! data. IF NPR>0 then such output is written every NPR steps. IF NPR=0
//  //! or if NPRNMX then no such output is written. Note that special solutions,
//  //! such as branch points, folds, end points, etc., are always written in fort.8.
//  //! Furthermore, one can specify parameter values where plotting and restart
//  //! data is to be written; see Section . For these reasons, and to limit the
//  //! output volume, it is recommended that NPR output be kept to a minimum.
//  integer npr;
//  integer &solution_storage_increment; //!< alias for npr
//
//  //! This constant, which is effective for algebraic problems only, sets the
//  //! maximum number of bifurcations to be treated. Additional branch points will
//  //! be noted, but the corresponding bifurcating branches will not be computed.
//  //! If MXBF is positive then the bifurcating branches of the first MXBF branch
//  //! points will be traced out in both directions. If MXBF is negative then the
//  //! bifurcating branches of the first |MXBF| branch points will be traced out in
//  //! only one direction.
//  integer mxbf;
//  integer &max_bifurcations; //!< alias for mxbf.
//
//  //! This constant controls the amount of diagnostic output printed in fort.9 :
//  //! the greater IID the more detailed the diagnostic output.
//  //! IID=0 : Minimal diagnostic output. This setting is not recommended.
//  //! IID=2 : Regular diagnostic output. This is the recommended value of IID.
//  //! IID=3 : This setting gives additional diagnostic output for algebraic
//  //!     equations, namely the Jacobian and the residual vector at the starting
//  //!     point. This information, which is printed at the beginning of fort.9,
//  //!     is useful for verifying whether the starting solution in stpnt is indeed
//  //!     a solution.
//  //! IID=4 : This setting gives additional diagnostic output for differen
//  //!     tial equations, namely the reduced system and the associated resid
//  //!     ual vector. This information is printed for every step and for every
//  //!     Newton iteration, and should normally be suppressed. In particular it
//  //!     can be used to verify whether the starting solution is indeed a solu
//  //!     tion. For this purpose, the stepsize DS should be small, and one should
//  //!     look at the residuals printed in the fort.9 outputfile. (Note that the
//  //!     first residual vector printed in fort.9 may be identically zero, as it
//  //!     may correspond to the computation of the starting direction. Look at
//  //!     the second residual vector in such case.) This residual vector has di
//  //!     mension NDIM+NBC+NINT+1, which accounts for the NDIM differential
//  //!     equations, the NBC boundary conditions, the NINT userdefined integral
//  //!     constraints, and the pseudoarclength equation. For proper interpreta
//  //!     tions of these data one may want to refer to the solution algorithm for
//  //!     solving the collocation system, as described in  ().
//  //!     IID=5 : This setting gives very extensive diagnostic output for differ
//  //!     ential equations, namely, debug output from the linear equation solver.
//  //!     This setting should not normally be used as it may result in a huge
//  //!     fort.9 file.
//  integer iid;
//  integer &diagnostic_level; //!< alias for iid.
//
//
//  //! The maximum number of iterations allowed in the accurate location of
//  //! special solutions, such as bifurcations, folds, and user output points, by
//  //! Mullers method with bracketing. The recommended value is ITMX=8, used
//  //! in most demos.
//  integer itmx;
//  integer &max_iterations_special; //!< itmx
//  
//  //! The maximum number of combined NewtonChord iterations. When this
//  //! maximum is reached, the step will be retried with half the stepsize. This
//  //! is repeated until convergence, or until the minimum stepsize is reached. In
//  //! the latter case the computation of the branch is discontinued and a message
//  //! printed in fort.9. The recommended value is ITNW=5, but ITNW=7 may be
//  //! used for difficult problems, for example, demos spb, chu, plp, etc.
//  integer itnw;
//  integer &max_iterations_newton_chord; //!< alias for itnw.
//  
//  //! After NWTN Newton iterations the Jacobian is frozen, i.e., AUTO uses full
//  //! Newton for the first NWTN iterations and the Chord method for iterations
//  //! NWTN+1 to ITNW. The choice NWTN=3 is strongly recommended and used in
//  //! most demos. Note that this constant is only effective for ODEs, i.e., for
//  //! solving the piecewise polynomial collocation equations.
//  //! For algebraic systems AUTO always uses full Newton.
//  integer nwtn;
//  integer &max_iterations_full_newton; //!< alias for nwtn.
//  
//  //! Used to indicate whether derivatives are supplied by the user or to be
//  //! obtained by differencing :
//  //! JAC=0 : No derivatives are given by the user. (Most demos use JAC=0.)
//  //! JAC=1 : Derivatives with respect to state and problemparameters
//  //!     are given in the user-supplied subroutines func, bcnd, icnd and fopt,
//  //!     where applicable. This may be necessary for sensitive problems. It
//  //!     is also recommended for computations in which AUTO generates an
//  //!     extended system, for example, when ISW=2.
//  //!     (Demos int, dd2, obt, plp, ops.)
//  integer jac;
//  integer &jacobian_control; //!< alias for jac.
//
//  //! Relative convergence criterion for equation parameters in the Newton/Chord
//  //! method. Most demos use EPSL=10e-6 or EPSL=10e-7, which is
//  //! the recommended value range.
//  doublereal epsl;
//
//  //! Relative convergence criterion for solution components in the Newton/Chord
//  //! method. Most demos use EPSU=10e-6 or EPSU=10e-7 , which is the recom
//  //! mended value range.
//  doublereal epsu;
//
//  //! Relative arclength convergence criterion for the detection of special
//  //! solutions. Most demos use EPSS=10e-4 or EPSS=10e-5, which is the recommended
//  //! value range. Generally, EPSS should be approximately 100 to 1000 times the
//  //! value of EPSL, EPSU.
//  doublereal epss;
//
//  //! AUTO uses pseudoarclength continuation for following solution branches.
//  //! The pseudoarclength stepsize is the distance between the current solution
//  //! and the next solution on a branch. By default, this distance includes all
//  //! state variables (or state functions) and all free parameters. The constant
//  //! DS defines the pseudoarclength stepsize to be used for the first attempted
//  //! step along any branch. (Note that if IADS > 0 then DS will automatically be
//  //! adapted for subsequent steps and for failed steps.) DS may be chosen positive
//  //! or negative; changing its sign reverses the direction of computation. The re
//  //! lation DSMIN < DS < DSMAX must be satisfied. The precise choice of DS is
//  //! problemdependent; the demos use a value that was found appropriate after
//  //! some experimentation.
//  doublereal ds;
//
//  //! This is minimum allowable absolute value of the pseudoarclength step
//  //! size. DSMIN must be positive. It is only effective if the pseudoarclength
//  //! step is adaptive, i.e., if IADS>0. The choice of DSMIN is highly problem
//  //! dependent; most demos use a value that was found appropriate after some
//  //! experimentation.
//  doublereal dsmin;
//  
//  //! The maximum allowable absolute value of the pseudoarclength stepsize.
//  //! DSMAX must be positive. It is only effective if the pseudoarclength step is
//  //! adaptive, i.e., if IADS>0. The choice of DSMAX is highly problemdependent;
//  //! most demos use a value that was found appropriate after some experimentation.
//  doublereal dsmax;
//  
//  //! This constant controls the frequency of adaption of the pseudoarclength
//  //! stepsize.
//  //! IADS=0 : Use fixed pseudoarclength stepsize, i.e., the stepsize will be
//  //!     equal to the specified value of DS for every step. The computation
//  //!     of a branch will be discontinued as soon as the maximum number of
//  //!     iterations ITNW is reached. This choice is not recommended.
//  //!     (Demo tim.)
//  //! IADS>0 : Adapt the pseudoarclength stepsize after every IADS steps.
//  //!     If the Newton/Chord iteration converges rapidly then |DS| will be
//  //!     increased, but never beyond DSMAX. If a step fails then it
//  //!     will be retried with half the stepsize. This will be done repeatedly
//  //!     until the step is successful or until |DS| reaches DSMIN. In the
//  //!     latter case nonconvergence will be signalled. The strongly
//  //!     recommended value is IADS=1, which is used in almost all demos.
//  integer iads;
//  integer &arc_length_adaptation_increment; //!< alias for iads.
//
//  //! By default, the pseudoarclength stepsize includes all state variables (or
//  //! state functions) and all free parameters. Under certain circumstances one
//  //! may want to modify the weight accorded to individual parameters in the
//  //! definition of stepsize. For this purpose, NTHL defines the number of
//  //! parameters whose weight is to be modified. If NTHL=0 then all weights will
//  //! have default value 1.0 . If NTHL>0 then one must enter NTHL pairs, Parameter
//  //! Index Weight , with each pair on a separate line.
//  //! For example, for the computation of periodic solutions it is recommended
//  //! that the period not be included in the pseudoarclength continuation stepsize,
//  //! in order to avoid periodinduced limitations on the stepsize near orbits of
//  //! infinite period. This exclusion can be accomplished by setting NTHL=1, with,
//  //! on a separate line, the pair 11 0.0 . Most demos that compute periodic
//  //! solutions use this option; see for example demo ab.
//  integer nthl;
//
//  std::vector< std::pair<integer, doublereal> > thl; //!< holding space for thl
//
//  //! Under certain circumstances one may want to modify the weight accorded
//  //! to individual state variables (or state functions) in the definition
//  //! of stepsize.
//  //! For this purpose, NTHU defines the number of states whose weight is to be
//  //! modified. If NTHU=0 then all weights will have default value 1.0 . If NTHU>0
//  //! then one must enter NTHU pairs, State Index Weight , with each pair on a
//  //! separate line. At present none of the demos use this option.
//  integer nthu;
//
//  std::vector< std::pair<integer, doublereal> > thu; //!< holding space for thu
//
//  //! uzr input
//  //! This constant allows the setting of parameter values at which labeled
//  //! plotting and restart information is to be written in the fort.8 outputfile.
//  //! Optionally, it also allows the computation to terminate at such a point.
//  //! Set NUZR=0 if no such output is needed. Many demos use this setting.
//  //! If NUZR>0 then one must enter NUZR pairs:
//  //!
//  //!   ParameterIndex ParameterValue
//  //!
//  //! with each pair on a separate line, to designate the parameters
//  //! and the parameter values at which output is to be written.
//  //! For examples see demos exp, int, and fsh.
//  //! If such a parameter index is preceded by a minus sign then the computation
//  //! will terminate at such a solution point. (Demos pen and bru.)
//  //! Note that fort.8 output can also be written at selected values of over
//  //! specified parameters. For an example see demo pvl. For details on
//  //! overspecified parameters see the manual.
//  integer nuzr;
//
//  std::vector<integer> iuz;    //!< storage for indices in uzr
//  std::vector<doublereal> vuz; //!< storage for values in uzr
//
//public:
//  //! let's make a simple constructor to set "normal" values
//  ContinuationConstants(const std::string &filename);
//
//  //! this is the wizard version of the the constructor. it will set
//  //! things up "basically" correctly for many different types of continuation.
//  ContinuationConstants(int nd, ConstantsInitType w = STEADY_STATE_CONTINUATION);
//
//  //! simple destructor
//  ~ContinuationConstants()
//  {
//      // nothing
//  }
//
//  //! read from file (this assumes the standard format for AUTO constants file
//  void ReadStandardAutoFormat(const std::string &filename);
//
//  //! read from file using more flexible format.
//  //! NOTE: currently identical to ReadStandardAutoFormat
//  void Read(const std::string &filename);
//
//  //! write to file (save to file using standard format for AUTO constants file
//  void WriteStandardAutoFormat(const std::string &filename);
//
//  //! write to file using more flexible format.
//  //! NOTE: currently identical to WriteStandardAutoFormat
//  void Write(const std::string &filename);
//};
//
////! hold the information for a single entry in the fort.8 file.
////! contains complete restart infomration:
//
///* plotting and restart data on unit 8, viz.: */
///* (1) data identifying the corresponding point on unit 7, */
///* (2) the complete solution, */
///* (3) the direction of the branch. */
//
///* Specifically the following is written: */
//
///*  IBR   : The index of the branch. */
///*  NTOT  : The index of the point. */
///*  ITP   : The type of point (see STPLBV above). */
///*  LAB   : The label of the point. */
///*  NFPR : The number of free parameters used in the computation. */
///*  ISW   : The value of ISW used in the computation. */
///*  NTPL  : The number of points in the time interval [0,1] for which */
///*          solution values are written. */
///*  NAR   : The number of values written per point. */
///*          (NAR=NDIM+1, since T and U(i), i=1,..,NDIM are written). */
///*  NROWPR: The number of lines printed following the identifying line */
///*          and before the next data set or the end of the file. */
///*          (Used for quickly skipping a data set when searching). */
///*  NTST  : The number of time intervals used in the discretization. */
///*  NCOL  : The number of collocation points used. */
///*  num_total_pars : The dimension of the par array (and the number of */
///*          number of values in the parameter block).*/
//
///* Following the above described identifying line there are NTPL lines */
///* containing : */
///*     T , U-1(T) , U-2(T) , ... , U-NDIM(T), */
///* where NDIM is the dimension of the system of differential equations. */
//
///*  Following this is a line which lists the active parameters (from ICP) */
//
///* Following this is a line containing */
///*    RL-dot(i) , i=1,NFPR, */
//
///* and following this are NTPL lines each containing */
///*    U-dot-1(T), U-dot-2(T), ... , U-dot-NDIM(T). */
//
///* Finally the parameter values PAR(i) , i=1,num_model_pars, are written. */
//
///*  Above, RL-dot(.) and U-dot(.) specify the direction of the branch. */
//
//struct SolutionEntry
//{
//  integer ibr;    //!< index of branch
//  integer ntot;   //!< index of point
//  integer itp;    //!< type of point
//  integer lab;    //!< label number
//  integer nfpr;   //!< number of free parameters
//  integer isw;    //!< ISW used for computation
//  integer ntpl;   //!< number of times points
//  integer nar;    //!< values written per point
//  integer nrowpr; //!< number of lines before next solution entry (or EOF)
//  integer ntst;   //!< number of time intervals in discretization
//  integer ncol;   //!< number of collocation points used
//  integer npar;   //!< number of parameters values that follow
//
//  //! state struct, can have vector of these guys
//  struct SolutionState
//  {
//      doublereal t;                 //!< time value
//      std::vector<doublereal> x;    //!< state variable data (could be mesh data for ntst and ncol != 0)
//      std::vector<doublereal> xdot; //!< dx/dt values for
//
//      //! simple constructor that sets solution state up
//      SolutionState(doublereal tp = 0, doublereal *xp = 0, doublereal *xdotp = 0, integer xdim = 0)
//          : t(tp)
//      {
//          for(int i = 0; i < xdim; ++i)
//          {
//              if(xp)
//                  x.push_back(xp[i]);
//              if(xdotp)
//                  xdot.push_back(xdotp[i]);
//          }
//      }
//
//      //! op=
//      SolutionState(const SolutionState &se)
//      {
//          this->t = se.t;
//          this->x.insert(this->x.begin(), se.x.begin(), se.x.end());
//          this->x.insert(this->xdot.begin(), se.xdot.begin(), se.xdot.end());
//      }
//  };
//
//  std::vector<SolutionState *> state; //!< state values (size == 1 unless ntst and ncol != 0)
//  std::vector<integer> fpr;           //!< indices of free parameters (empty for ntsts=ncol=0)
//  std::vector<doublereal> fprdot;     //!< branch directions for each free parameter
//
//  std::vector<doublereal> par; //!< parameter data
//
//public:
//  //! simple constructor
//  SolutionEntry();
//
//  //! constructor that fills the state and par values
//  SolutionEntry(doublereal t, doublereal *x, integer xdim, doublereal *p, integer pdim);
//
//  //! constructor that fills the state and par values
//  SolutionEntry(const SolutionEntry &se);
//
//  //! destructor needs to empty vectors of pointers
//  ~SolutionEntry();
//
//  //! we need to be able to read this guy
//  void Read(std::istream &is);
//
//  //! we need to be able to write this guy
//  void Write(std::ostream &os);
//};
//
////! a type for the vector
//typedef std::vector<SolutionEntry *> SolutionEntryVector;
//
////! a type for the interator
//typedef SolutionEntryVector::iterator SolutionEntryIterator;
//
////! holds solution information.
//struct SolutionSet : public AutoApiData
//{
//  //! we need to keep track of what file this is associated with.
//  //! note that there is currently no guarantee that the state of the
//  //! solution is in sync with the file... this might be reasonable later on,
//  //! but it doesn't make sense to restrict control to the data yet...
//  std::string filename;
//
//  //! holds a set of solution data
//  SolutionEntryVector entries;
//
//public:
//  //! basic, simple constructor
//  SolutionSet(const std::string &f = "");
//
//  //! constructor that creates a single point solution set for restarting
//  SolutionSet(doublereal t, doublereal *x, integer xdim, doublereal *pars, integer pdim);
//
//  //! constructor that creates a single point solution set for restarting
//  SolutionSet(const SolutionEntryIterator &it);
//
//  //! must release memory associated with entries
//  ~SolutionSet();
//
//  //! read from file (this assumes the standard format for AUTO constants file
//  void ReadStandardAutoFormat(const std::string &f, bool clean = true);
//
//  //! read from file using more flexible format.
//  //! NOTE: currently identical to ReadStandardAutoFormat
//  void Read(const std::string &f, bool clean = true);
//
//  //! write to file (save to file using standard format for AUTO constants file
//  void WriteStandardAutoFormat(const std::string &f);
//
//  //! write to file using more flexible format.
//  //! NOTE: currently identical to WriteStandardAutoFormat
//  void Write(const std::string &f);
//
//  //! append allows us to take the existing solutions and to
//  //! and in a bunch more. Relabelling is automatic.
//  void Append(const std::string &f);
//
//  //! add a single point solution set for restarting, returns label added
//  int Append(doublereal t, doublereal *x, integer xdim, doublereal *pars, integer pdim);
//
//  //! relabel simply goes though the entries list and changes labels to 1, 2... etc
//  void Relabel();
//
//  ////////////////////////////////////////////////////////////////////////
//  // expose some list functionality
//
//  //! alias for entries.begin()
//  SolutionEntryIterator Begin() { return entries.begin(); }
//
//  //! alias for entries.End()
//  SolutionEntryIterator End() { return entries.end(); }
//
//  //! alias for entries.back()
//  SolutionEntry *Back() { return entries.back(); }
//
//  //! we can use this to traverse the solution set looking for bifurcation points
//  void FindNextBP(SolutionEntryIterator &it); //@@e1 make this more intuitive!!! (SPECIAL ITERATOR CLASS IS PROBABLY APPRORIATE)
//
//  //! we can use this to traverse the solution set looking for a particular label
//  void FindLabel(SolutionEntryIterator &it, int label);
//};
//
////! Plot type will tell us how to read in different data
//enum PlotType
//{
//  ONE_PARAMETER = 0,
//  TWO_PARAMETER = 1
//};
//
////! This will hold a single line from the fort.7 file. Used for plotting results.
//struct PlotEntry
//{
//  integer ibr;               //!< index of branch
//  integer pt;                //!< point
//  integer ty;                //!< point type
//  integer lab;               //!< label of point
//
//  doublereal norm;           //!< prinicpal solution measure
//
//  std::vector<doublereal> p; //!< continuation parameter data
//  std::vector<doublereal> x; //!< state variable data
//
//public:
//  //! we need to be able to read this guy
//  void Read(std::istream &is, PlotType type, int ndim);
//
//  //! we need to be able to write this guy
//  void Write(std::ostream &os, PlotType type, int ndim);
//
//  //! write for bifucation diagram (first column is -1 or 1 for stability,
//  //! then you get all parameter values and state variable values following
//  void WriteBD(std::ostream &os, PlotType type, int ndim);
//};
//
////! holds plot information.
//struct PlotSet : public AutoApiData
//{
//  //! we need to keep track of what file this is associated with.
//  //! note that there is currently no guarantee that the state of the
//  //! solution is in sync with the file... this might be reasonable later on,
//  //! but it doesn't make sense to restrict control to the data yet...
//  std::string filename;
//
//  //! we need to know what type of plot set this is
//  PlotType type;
//
//  //! we need to know how many state variable values to read in
//  int ndim;
//
//  //! holds a set of solution data
//  std::vector<PlotEntry *> entries;
//
//public:
//  //! basic, simple constructor
//  PlotSet(int nd, PlotType w = ONE_PARAMETER);
//
//  //! must release memory associated with entries
//  ~PlotSet();
//
//  //! read from file (this assumes the standard format for AUTO plot file
//  void ReadStandardAutoFormat(const std::string &f, InsertType t = FORWARD, bool clean = true);
//
//  //! read from file using more flexible format.
//  //! NOTE: currently identical to ReadStandardAutoFormat
//  void Read(const std::string &f, InsertType t = FORWARD, bool clean = true);
//
//  //! write to file (save to file using standard format for AUTO plot file
//  void WriteStandardAutoFormat(const std::string &f);
//
//  //! write to file using more flexible format.
//  //! NOTE: currently identical to WriteStandardAutoFormat
//  void Write(const std::string &f);
//
//  //! write bifucation diagram (first column is -1 or 1 for stability,
//  //! then you get all parameter values and state variable values following
//  void WriteBD(const std::string &f);
//
//  //! append allows us to take the existing solutions and to
//  //! and in a bunch more. Relabelling is automatic.
//  //! \param f is filename
//  //! \param branch_index specifies only getting a particular branch, -1 means all
//  //! \param t tells us how to insert points point in the file
//  void Append(const std::string &f, int branch_index = -1, InsertType t = FORWARD);
//};
//
////! main interface to the AUTO library
//class AutoApi
//{
//public:
//  //! simple constructor stub
//  AutoApi()
//      : constants(0), solutions(0)
//  {
//      // nothing
//  }
//  
//  //! simple destructor stub
//  ~AutoApi()
//  {
//      if(constants)
//          if(constants->api_owns)
//              delete constants;
//
//      if(solutions)
//          if(solutions->api_owns)
//              delete solutions;
//  }
//
//  //! we can load constants to prep for a run.
//  ContinuationConstants *LoadConstants(ContinuationConstants *cc);
//
//  //! we can load constants to prep for a run. this version just calls
//  //! the ContinuationConstants version after reading in the constants file
//  ContinuationConstants *LoadConstants(const std::string &f);
//
//  //! we can load solutions from a previous run so we have start labels, etc.
//  SolutionSet *LoadSolutionSet(SolutionSet *sol);
//
//  //! we can load solutions from a previous run from a file.
//  SolutionSet *LoadSolutionSet(const std::string &f);
//
//  //! we can append solutions from a previous run from a file to current solution set.
//  SolutionSet *AppendSolutionSet(const std::string &f);
//
//  //! this is how we run the api
//  //! \param key is a unique key to create files under
//  //! \param append_solutions tells us whether we want to add solutions
//  //! from this run to the solution set we're running from.
//  void Run(const std::string &key, bool append_solutions = true);
//
//private:
//  ContinuationConstants *constants; //!< constants to do next Run() with.
//  SolutionSet *solutions; //!< solution information (fort.8 ingested).
//};
//
}

#endif
