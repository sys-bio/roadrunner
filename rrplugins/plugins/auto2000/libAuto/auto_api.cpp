// $Id: auto_api.cpp,v 1.12 2004/10/28 01:45:39 emery Exp $
#include "auto_f2c.h"
#include "auto_c.h"
#define __AUTO_API_IMPLEMENTATION__
#include "auto_api.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
using std::endl;

#define MY_TEST(XX, YY) do { if(!(XX)) throw std::logic_error(YY); } while(0)

#define NEXT_LINE_AND_CHECK(XX, YY) \
do { \
    XX.getline(YY, 1024); \
    MY_TEST(!XX.fail(), "failed to get to end of line"); \
    MY_TEST(!XX.eof(), "eof reached while trying to get to next line"); \
} while(0)

namespace autolib {

//// empty a pointer vector utility
//template <typename TYPE>
//void EraseVectorOfPointers(std::vector<TYPE *> &v)
//{
//    // delete each
//    for(std::vector<TYPE *>::size_type i = 0; i < v.size(); ++i)
//        delete v[i];
//
//    // empty it out
//    v.erase(v.begin(), v.end());
//}
//
//////////////////////////////////////////////////////////////////////////
////
////  ContinuationConstants
////
//////////////////////////////////////////////////////////////////////////
//
////! let's make a simple constructor to set "normal" values
//ContinuationConstants::ContinuationConstants(const std::string &filename)
//    : ndim(0), ips(1), irs(0), ilp(1),
//      nicp(0), icp(num_total_pars, 0),
//      ntst(50), ncol(4), iad(3), isp(1), isw(1), iplt(0), nbc(0), nint(0),
//        nmx(1000), rl0(0.0), rl1(1000.0), a0(0.0), a1(1000.0),
//        npr(100), mxbf(100), iid(0), itmx(8), itnw(5), nwtn(3), jac(0),
//        epsl(1e-08), epsu(1e-08), epss(1e-06),
//        ds(1e-02), dsmin(1e-06), dsmax(1e-01), iads(1),
//        nthl(0), nthu(0), nuzr(0),
//
//        // we need to set the aliases here as well
//        problem_type(ips),
//        restart_label(irs),
//        fold_detection(ilp),
//        num_free_parameters(nicp),
//        num_mesh_intervals(ntst),
//        num_collocation_points(ncol),
//        mesh_adaptation_control(iad),
//        special_detection_control(isp),
//        branch_switch_control(isw),
//        pricipal_solution_measure(iplt),
//        num_boundary_conditions(nbc),
//        num_integral_conditions(nint),
//        max_branch_steps(nmx),
//        solution_storage_increment(npr),
//        max_bifurcations(mxbf),
//        diagnostic_level(iid),
//        max_iterations_special(itmx),
//        max_iterations_newton_chord(itnw),
//        max_iterations_full_newton(nwtn),
//        jacobian_control(jac),
//        arc_length_adaptation_increment(iads)
//{
//    if(filename != "")
//        this->Read(filename);
//}
//
////! wizard version of constructor
//ContinuationConstants::ContinuationConstants(int nd, ConstantsInitType w)
//    : ndim(nd), icp(num_total_pars, 0),
//
//        // we need to set the aliases here as well
//        problem_type(ips),
//        restart_label(irs),
//        fold_detection(ilp),
//        num_free_parameters(nicp),
//        num_mesh_intervals(ntst),
//        num_collocation_points(ncol),
//        mesh_adaptation_control(iad),
//        special_detection_control(isp),
//        branch_switch_control(isw),
//        pricipal_solution_measure(iplt),
//        num_boundary_conditions(nbc),
//        num_integral_conditions(nint),
//        max_branch_steps(nmx),
//        solution_storage_increment(npr),
//        max_bifurcations(mxbf),
//        diagnostic_level(iid),
//        max_iterations_special(itmx),
//        max_iterations_newton_chord(itnw),
//        max_iterations_full_newton(nwtn),
//        jacobian_control(jac),
//        arc_length_adaptation_increment(iads)
//{
//    switch(w)
//    {
//    case STEADY_STATE_CONTINUATION:
//    default:
//        ips = 1; irs = 0; ilp = 1; nicp = 0;
//      ntst = 50; ncol = 4; iad = 3; isp = 1; isw = 1; iplt = 0; nbc = 0; nint = 0;
//        nmx = 1000; rl0 = 0.0; rl1 = 1000.0; a0 = 0.0; a1 = 1000.0;
//        npr = 100; mxbf = 100; iid = 0; itmx = 8; itnw = 5; nwtn = 3; jac = 0;
//        epsl = 1e-08; epsu = 1e-08; epss = 1e-06;
//        ds = 1e-02; dsmin = 1e-06; dsmax = 1e-01; iads = 1;
//        nthl = 0; nthu = 0; nuzr = 0;
//        break;
//    }
//}
//
//// this code should produce the same results as that found in the init() procedure from
//// autlib1.c, with the obvious changes that we are storing values to member data (not
//// global).
//void ContinuationConstants::ReadStandardAutoFormat(const std::string &filename)
//{
//    char buf[1024];
//    integer tempi;
//    doublereal tempd;
//
//    std::ifstream in(filename.c_str());
//    MY_TEST(in, "could not open constants file for read");
//  
//    // line 1
//  in >> ndim >> ips >> irs >> ilp;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // line 2
//    integer jtmp = num_model_pars;
//  in >> nicp;
//  for(int i = 0; i < jtmp; ++i)
//        icp[i] = i;
//    for(int i = 0; i < nicp; ++i)
//    {
//        in >> tempi;
//    icp[i] = icp[jtmp + i] = tempi;
//    }
//  if(nicp < 1)
//    {
//    nicp = 1;
//    jtmp = num_model_pars;
//    icp[jtmp] = icp[0];
//  }
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // line 3
//    in >> ntst >> ncol >> iad >> isp >> isw >> iplt >> nbc >> nint;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // line 4
//  in >> nmx >> rl0 >> rl1 >> a0 >> a1;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//  // line 5
//    in >> npr >> mxbf >> iid >> itmx >> itnw >> nwtn >> jac;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // line 6
//  in >> epsl >> epsu >> epss;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//  /* Check to make sure these guys are positive... if not give a warning and proceed*/
//  if(epsl < 0.0) {
//    fprintf(fp6, "Warning : EPSL less then 0.0, will use absolute value instead.");
//    epsl = fabs(epsl);
//  }
//  if(epsu < 0.0) {
//    fprintf(fp6, "Warning : EPSU less then 0.0, will use absolute value instead.");
//    epsu = fabs(epsu);
//  }
//  if(epss < 0.0) {
//    fprintf(fp6, "Warning : EPSS less then 0.0, will use absolute value instead.");
//    epss = fabs(epss);
//  }
//
//  // line 7
//    in >> ds >> dsmin >> dsmax >> iads;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // @@e: shouldn't we also make sure that dsmin < abs(ds) < dsmax??
//    
//  if(dsmin < 0.0) {
//    fprintf(fp6, "Warning : DSMIN less then 0.0, will use absolute value instead.");
//    dsmin = fabs(dsmin);
//  }
//
//  if(dsmax < 0.0) {
//    fprintf(fp6, "Warning : DSMAX less then 0.0, will use absolute value instead.");
//    dsmax = fabs(dsmax);
//  }
//
//  // thl input
//    in >> nthl;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // reset and fill
//    thl.erase(thl.begin(), thl.end());
//    for(int i = 0; i < nthl; ++i)
//    {
//        in >> tempi >> tempd;
//        thl.push_back(std::make_pair(tempi, tempd));
//        NEXT_LINE_AND_CHECK(in, buf);
//  }
//    
//  // thu input
//    in >> nthu;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // reset and fill
//    thu.erase(thu.begin(), thu.end());
//  for(int i = 0; i < nthu; ++i)
//    {
//        in >> tempi >> tempd;
//        thu.push_back(std::make_pair(tempi, tempd));
//        NEXT_LINE_AND_CHECK(in, buf);
//    }
//
//    // uzr input
//    in >> nuzr;
//    NEXT_LINE_AND_CHECK(in, buf);
//
//    // empty these guys out
//    iuz.erase(iuz.begin(), iuz.end());
//    vuz.erase(vuz.begin(), vuz.end());
//
//  for(int i = 0; i < nuzr; ++i)
//    {
//    in >> tempi;
//        iuz.push_back(tempi);
//        in >> tempd;
//        vuz.push_back(tempd);
//        
//        if(i < nuzr-1)
//            NEXT_LINE_AND_CHECK(in, buf);
//  }
//
//    // all done
//    in.close();
//}
//
////! read from file using more flexible format.
////! NOTE: currently identical to ReadStandardAutoFormat
//void ContinuationConstants::Read(const std::string &filename)
//{
//    ReadStandardAutoFormat(filename);
//}
//
////! write to file (save to file using standard format for AUTO constants file
//void ContinuationConstants::WriteStandardAutoFormat(const std::string &filename)
//{
//    std::ofstream out(filename.c_str());
//    MY_TEST(out, "could not open constants file for write");
//  
//    // line 1
//  out << ndim << " " << ips << " " << irs << " " << ilp << endl;
//
//    // line 2
//  out << nicp << " ";
//    for(int i = 0; i < nicp; ++i)
//        out << icp[i] << " ";
//    out << endl;
//
//    // line 3
//    out << ntst << " " << ncol << " " << iad << " " << isp << " " << isw << " " << iplt << " " << nbc << " " << nint << endl;
//
//    // line 4
//  out << nmx << " " << rl0 << " " << rl1 << " " << a0 << " " << a1 << endl;
//
//  // line 5
//    out << npr << " " << mxbf << " " << iid << " " << itmx << " " << itnw << " " << nwtn << " " << jac << endl;
//
//    // line 6
//  out << epsl << " " << epsu << " " << epss << endl;
//
//  // line 7
//    out << ds << " " << dsmin << " " << dsmax << " " << iads << endl;
//
//  // thl input
//    out << nthl << endl;
//
//  for(int i = 0; i < nthl; ++i)
//        out << thl[i].first << " " << thl[i].second << endl;
//
//  // thu input
//    out << nthu << endl;
//
//  for(int i = 0; i < nthu; ++i)
//        out << thu[i].first << " " << thu[i].second << endl;
//
//    // uzr input
//    out << nuzr << endl;
//
//  for(int i = 0; i < nuzr; ++i)
//    out << iuz[i] << " " << vuz[i] << endl;
//
//    // all done
//    out.close();
//}
//
////! write to file using more flexible format.
////! NOTE: currently identical to WriteStandardAutoFormat
//void ContinuationConstants::Write(const std::string &filename)
//{
//    WriteStandardAutoFormat(filename);
//}
//
//
//////////////////////////////////////////////////////////////////////////
////
////  SolutionEntry
////
//////////////////////////////////////////////////////////////////////////
//
//// constructor
//SolutionEntry::SolutionEntry()
//    : ibr(1), ntot(0), itp(9), lab(1), nfpr(1), isw(1),
//        ntpl(1), nar(0), nrowpr(0), ntst(0), ncol(0), npar(num_total_pars)
//{
//}
//
//// constructor that stores state/par values
//SolutionEntry::SolutionEntry(doublereal t, doublereal *x, integer xdim, doublereal *p, integer pdim)
//    : ibr(1), ntot(1), itp(9), lab(1), nfpr(1), isw(1),
//        ntpl(1), nar(0), nrowpr(0), ntst(0), ncol(0), npar(num_total_pars)
//{
//    // create state
//    state.push_back(new SolutionState(t, x, 0, xdim));
//
//    // set parameter values
//    for(int i = 0; i < npar; ++i)
//    {
//        if(i < pdim)
//            par.push_back(p[i]);
//        else
//            par.push_back(0.0);
//    }
//}
//
//// constructor
//SolutionEntry::SolutionEntry(const SolutionEntry &se)
//{
//  ibr    = se.ibr;    //!< index of branch
//  ntot   = se.ntot;   //!< index of point
//  itp    = se.itp;    //!< type of point
//  lab    = 1;//se.lab;    //!< label number
//  nfpr   = se.nfpr;   //!< number of free parameters
//  isw    = se.isw;    //!< ISW used for computation
//  ntpl   = se.ntpl;   //!< number of times points
//  nar    = se.nar;    //!< values written per point
//  nrowpr = se.nrowpr; //!< number of lines before next solution entry (or EOF)
//    ntst   = se.ntst;   //!< number of time intervals in discretization
//    ncol   = se.ncol;   //!< number of collocation points used
//  npar   = se.npar;   //!< number of parameters values that follow
//
//    for(std::vector<SolutionState *>::const_iterator i = se.state.begin();
//          i < se.state.end();
//            ++i)
//    {
//        state.push_back(new SolutionState(**i));
//    }
//
//    fpr.insert(fpr.begin(), se.fpr.begin(), se.fpr.end());
//    fprdot.insert(fprdot.begin(), se.fprdot.begin(), se.fprdot.end());
//    par.insert(par.begin(), se.par.begin(), se.par.end());
//}
//
//// destructor
//SolutionEntry::~SolutionEntry()
//{
//    EraseVectorOfPointers(state);
//}
//
//// read data in
//void SolutionEntry::Read(std::istream &is)
//{
//    integer tempi;
//    doublereal tempd;
//
//    // keep track of the number of lines we've read
//    int lines = -1; // first line doesn't count
//
//  is >> ibr >> ntot >> itp >> lab >> nfpr >> isw >> ntpl >> nar >> nrowpr
//         >> ntst >> ncol >> npar;
//
//    // increment lines
//    ++lines;
//
//    // empty state storage
//    EraseVectorOfPointers(state);
//
//    // get new state
//  for(int i = 0; i <= ntst; ++i)
//    {
//    int j = 0;
//        do
//        {            
//            state.push_back(new SolutionState);
//
//            // get time
//            is >> state.back()->t;
//
//            // now get x
//            for(int k = 0; k < nar - 1; ++k)
//            {
//                is >> tempd;
//                state.back()->x.push_back(tempd);
//                
//                // increment lines as we do in the output routine
//                if(((k+1)%7) == 0)
//                    ++lines;
//            }
//
//            // increment lines
//            ++lines;
//    }
//        while((i<ntst) && (++j < ncol));
//  }
//
//    // if we've put out ntst, then we should expect free parameter indices
//    fpr.erase(fpr.begin(), fpr.end());
//    if(ntst)
//    {
//        for(int i = 0; i < nfpr; ++i)
//        {
//            is >> tempi;
//            fpr.push_back(tempi);
//        }
//
//        // increment lines
//        ++lines;
//
//        // get branch direction
//        fprdot.erase(fprdot.begin(), fprdot.end());
//        for(int i = 0; i < nfpr; ++i)
//        {
//            is >> tempd;
//            fprdot.push_back(tempd);
//
//            // increment lines as we do in the output routine
//            if(((i+1)%7) == 0)
//                ++lines;
//        }
//
//        // increment lines
//        ++lines;
//
//        // get xdot into state
//        for(int i = 0, m = 0; i <= ntst; ++i)
//        {
//            int j = 0;
//            do
//            {            
//                // get xdot
//                for(int k = 0; k < nar - 1; ++k)
//                {
//                    is >> tempd;
//                    state[m]->x.push_back(tempd);
//
//                    // increment lines as we do in the output routine
//                    if(((k+1)%7) == 0)
//                        ++lines;
//                }
//
//                // increment lines
//                ++lines;
//
//                // increment state index
//                ++m;
//            }
//            while((i<ntst) && (++j < ncol));
//        }
//    }
//
//    // empty out parameter values
//    par.erase(par.begin(), par.end());
//
//    // get parameter values
//    MY_TEST(npar > 0, "npar must be > 0");
//    for(int i = 0; i < npar; ++i)
//    {
//        is >> tempd;
//        par.push_back(tempd);
//
//        // increment lines as we do in the output routine
//        if(((i+1)%7) == 0)
//            ++lines;
//    }
//
//    ++lines;
//
//    // get any remaining lines before next block
//    char buf[1024];
//    for(int i = lines; (i <= nrowpr) && !is.eof(); ++i)
//        is.getline(buf, 1024);
//}
//
//// write out
//void SolutionEntry::Write(std::ostream &os)
//{
//    integer ndim = (integer) state[0]->x.size();
//
//    // set the number of rows
//    if(ntst == 0)
//    {
//        ntpl = 1;
//        nar = ndim + 1;
//        integer jtmp = num_model_pars;
//      nrowpr = ndim/7 + 1 + (jtmp - 1)/7 + 1;
//    }
//  else
//    {
//        ntpl = ncol * ntst + 1;
//        nar = ndim + 1;
//        integer nrd = ndim / 7 + 2 + (ndim - 1) / 7;
//        integer jtmp = num_model_pars;
//        nrowpr = nrd * (ncol*ntst + 1) + (nfpr - 1)/7 + 1 + (jtmp - 1)/7 +  1 + (nfpr - 1)/20 + 1;
//    }
//
//  os << ibr << " " << ntot << " " << itp << " " << lab << " " << nfpr
//         << " " << isw << " " << ntpl << " " << nar << " " << nrowpr << " "
//         << ntst << " " << ncol << " " << npar << endl;
//
//    // new state
//    os << std::setiosflags(std::ios_base::scientific) << std::setprecision(10);
//
//  for(int i = 0, m = 0; i <= ntst; ++i)
//    {
//    int j = 0;
//        do
//        {            
//            // time
//            os << state[m]->t << " ";
//
//            // now x
//            for(int k = 0; k < ndim; ++k)
//            {
//                os << state[m]->x[k] << " ";
//
//                if(((k+1)%7) == 0)
//                    os << endl;
//            }
//
//            os << endl;
//
//            // increment state index
//            ++m;
//    }
//        while((i<ntst) && (++j < ncol));
//  }
//
//    // if we've put out ntst, then we should see free parameter indices
//    if(ntst)
//    {
//        for(int i = 0; i < nfpr; ++i)
//            os << fpr[i] << " ";
//
//        os << endl;
//
//        // branch direction
//        for(int i = 0; i < nfpr; ++i)
//        {
//            os << fprdot[i] << " ";
//
//            if(((i+1)%7) == 0)
//                os << endl;
//        }
//
//        os << endl;
//
//        // xdot
//        for(int i = 0, m = 0; i <= ntst; ++i)
//        {
//            int j = 0;
//            do
//            {            
//                for(int k = 0; k < ndim; ++k)
//                {
//                    os << state[m]->x[k] << " ";
//
//                    if(((k+1)%7) == 0)
//                        os << endl;
//                }
//
//                os << endl;
//
//                // increment state index
//                ++m;
//            }
//            while((i<ntst) && (++j < ncol));
//        }
//    }
//
//    // parameter values
//    MY_TEST(npar > 0, "npar must be > 0");
//    for(int i = 0; i < npar; ++i)
//    {
//        os << par[i] << " ";
//
//        if(((i+1)%7) == 0)
//            os << endl;
//    }
//
//    os << endl;
//}
//
//
//////////////////////////////////////////////////////////////////////////
////
////  SolutionSet
////
//////////////////////////////////////////////////////////////////////////
//
//// basic, simple constructor
//SolutionSet::SolutionSet(const std::string &f)
//    : filename(f)
//{
//    if(filename != "")
//        this->Read(filename);
//}
//
//// constructor that creates a single point solution set for restarting
//SolutionSet::SolutionSet(doublereal t, doublereal *x, integer xdim, doublereal *p, integer pdim)
//    : filename("")
//{
//    entries.push_back(new SolutionEntry(t, x, xdim, p, pdim));
//}
//
//// constructor that creates a single point solution set for restarting
//SolutionSet::SolutionSet(const SolutionEntryIterator &it)
//    : filename("")
//{
//    entries.push_back(new SolutionEntry(**it));
//}
//
//SolutionSet::~SolutionSet()
//{
//    EraseVectorOfPointers(entries);
//}
//
//void SolutionSet::ReadStandardAutoFormat(const std::string &f, bool clean)
//{
//    // first set the filename
//    filename = f;
//
//    // erase the old
//    if(clean)
//        EraseVectorOfPointers(entries);
//
//    // read in the new
//    std::ifstream in(filename.c_str());
//    MY_TEST(in, "could not open solutions file for read");
//  
//    // read in one SolutionEntry at a time
//    while(!in.eof() && (in.peek() != EOF))
//    {
//        // get new entry into the list and read from file
//        entries.push_back(new SolutionEntry);
//        entries.back()->Read(in);
//    }
//    
//    // all done
//    in.close();
//}
//
////! read from file using more flexible format.
////! NOTE: currently identical to ReadStandardAutoFormat
//void SolutionSet::Read(const std::string &f, bool clean)
//{
//    ReadStandardAutoFormat(f, clean);
//}
//
////! write to standard AUTO formatted fort.8 file
//void SolutionSet::WriteStandardAutoFormat(const std::string &f)
//{
//    // first set filename
//    filename = f;
//
//    std::ofstream out(filename.c_str());
//    MY_TEST(out, "could not open solutions file write");
//  
//    // read in one SolutionEntry at a time
//    for(std::vector<SolutionEntry *>::size_type i = 0; i < entries.size(); ++i)
//        entries[i]->Write(out);
//    
//    // all done
//    out.close();
//}
//
////! write to file using more flexible format.
////! NOTE: currently identical to WriteStandardAutoFormat
//void SolutionSet::Write(const std::string &f)
//{
//    WriteStandardAutoFormat(f);
//}
//
//// append info from file
//void SolutionSet::Append(const std::string &f)
//{
//    // all we need to do is to read in new data, without
//    // cleaning out the old data.
//    Read(f, false);
//
//    // also, let's relabel points after we append to be safe
//    Relabel();
//}
//
//// constructor that creates a single point solution set for restarting
//int SolutionSet::Append(doublereal t, doublereal *x, integer xdim, doublereal *p, integer pdim)
//{
//    // add a new solution entry at the end of the list
//    entries.push_back(new SolutionEntry(t, x, xdim, p, pdim));
//    
//    // for now, let's relabel
//    Relabel();
//
//    // return label of new point
//    return entries.back()->lab;
//}
//
//// relabel solutions
//void SolutionSet::Relabel()
//{
//    // just go through the list and set label numbers to 1, 2...
//    for(std::vector<SolutionEntry *>::size_type i = 0; i < entries.size(); ++i)
//        entries[i]->lab = (integer) i+1;
//}
//
////! we can use this to traverse the solution set looking for bifurcation points
//void SolutionSet::FindNextBP(SolutionEntryIterator &it)
//{
//    while(++it < entries.end())
//    {
//        // check if bp
//        switch((*it)->itp%10)
//        {
//        case 1:
//        case 2:
//        case 3:
//        case 5:
//        case 6:
//        case 7:
//        case 8:
//            return;
//        default:
//            break;
//        }
//    }
//}
//
////! we can use this to traverse the solution set looking for a particular label
//void SolutionSet::FindLabel(SolutionEntryIterator &it, int label)
//{
//    do
//    {
//        // check for label
//        if((*it)->lab == label)
//            break;
//    }
//    while(++it < entries.end());
//}
//
//
//////////////////////////////////////////////////////////////////////////
////
////  PlotSet
////
//////////////////////////////////////////////////////////////////////////
//
//void PlotEntry::Read(std::istream &is, PlotType type, int ndim)
//{
//    // temporary storage
//    doublereal temp;
//    char line[1024];
//
//    // empty things out
//    p.erase(p.begin(), p.end());
//    x.erase(x.begin(), x.end());
//
//    // read in differently depending on type
//    switch(type)
//    {
//    case ONE_PARAMETER:
//        // one par files have something like:
///*
//   0    PT  TY LAB    PAR(3)        L2-NORM         U(1)          U(2)
//   1     1   0   0  1.00000E+000  1.06607E+000  4.35020E-001  9.73279E-001
//   1    -2   0   0  1.01887E+000  1.06169E+000  4.37749E-001  9.67240E-001
//   1    -3   0   0  1.03302E+000  1.05843E+000  4.39812E-001  9.62724E-001
//   1    -4   0   0  1.05426E+000  1.05360E+000  4.42934E-001  9.55971E-001
//*/
//        // get branch index, skipping branches marked as ibr=0 (this is constant info)
//        for(;;)
//        {
//            // get branch idx
//            is >> ibr;
//            
//            // break out if non-zero branch
//            if(ibr != 0)
//                break;
//            
//            // otherwise, suck in whole line
//            is.getline(line, 1024);
//        }
//
//        // get point number, type, and label
//        is >> pt >> ty >> lab;
//
//        // get the single parameter value
//        is >> temp;
//        p.push_back(temp);
//
//        // get norm (technically, pricipal solution measure, but really....
//        is >> norm;
//
//        // now, we get the state variables
//        for(int i = 0; i < ndim; ++i)
//        {
//            is >> temp;
//            x.push_back(temp);
//        }        
//
//        // get to end of line
//        is.getline(line, 1024);
//
//        break;
//
//    case TWO_PARAMETER:
//        // two par files have something like:
///*
//  0    PT  TY LAB    PAR(3)        L2-NORM         U(1)          U(2)        PAR(6)
//  1     1   0   0  2.50000E+001  5.03621E+000  5.03597E+000  4.93570E-002  2.00000E+002
//  1    -2   0   0  2.50197E+001  5.03973E+000  5.03949E+000  4.92944E-002  2.00000E+002
//  1    -3   0   0  2.50345E+001  5.04238E+000  5.04214E+000  4.92475E-002  2.00000E+002
//  1    -4   0   0  2.50566E+001  5.04634E+000  5.04610E+000  4.91774E-002  2.00000E+002
//*/
//        // get branch index, skipping branches marked as ibr=0 (this is constant info)
//        for(;;)
//        {
//            // get branch idx
//            is >> ibr;
//            
//            // break out if non-zero branch
//            if(ibr != 0)
//                break;
//            
//            // otherwise, suck in whole line
//            is.getline(line, 1024);
//        }
//
//        // get point number, type, and label
//        is >> pt >> ty >> lab;
//
//        // get the first parameter value
//        is >> temp;
//        p.push_back(temp);
//
//        // get norm (technically, pricipal solution measure, but really....
//        is >> norm;
//
//        // now, we get the state variables
//        for(int i = 0; i < ndim; ++i)
//        {
//            is >> temp;
//            x.push_back(temp);
//        }        
//
//        // get the second parameter value
//        is >> temp;
//        p.push_back(temp);
//
//        // get to end of line
//        is.getline(line, 1024);
//
//        break;
//
//    default:
//        throw std::logic_error("unknown or un-implemented PlotType in PlotEntry::Read");
//    }
//}
//
//void PlotEntry::Write(std::ostream &os, PlotType type, int ndim)
//{
//    // write differently depending on type
//    switch(type)
//    {
//    case ONE_PARAMETER:
//        // one par files have something like:
///*
//   0    PT  TY LAB    PAR(3)        L2-NORM         U(1)          U(2)
//   1     1   0   0  1.00000E+000  1.06607E+000  4.35020E-001  9.73279E-001
//   1    -2   0   0  1.01887E+000  1.06169E+000  4.37749E-001  9.67240E-001
//   1    -3   0   0  1.03302E+000  1.05843E+000  4.39812E-001  9.62724E-001
//   1    -4   0   0  1.05426E+000  1.05360E+000  4.42934E-001  9.55971E-001
//*/
//        // write branch index, point number, type, and label
//        os << ibr << " " << pt << " " << ty << " " << lab << " ";
//
//        // write the single parameter value
//        os << p[0] << " ";
//
//        // get norm (technically, pricipal solution measure, but really....
//        os << norm << " ";
//
//        // now, we get the state variables
//        for(int i = 0; i < ndim; ++i)
//        {
//            os << x[i];
//            if(i < ndim-1)
//                os << " ";
//        }        
//
//        // end the line
//        os << std::endl;
//
//        break;
//
//    case TWO_PARAMETER:
//        // two par files have something like:
///*
//  0    PT  TY LAB    PAR(3)        L2-NORM         U(1)          U(2)        PAR(6)
//  1     1   0   0  2.50000E+001  5.03621E+000  5.03597E+000  4.93570E-002  2.00000E+002
//  1    -2   0   0  2.50197E+001  5.03973E+000  5.03949E+000  4.92944E-002  2.00000E+002
//  1    -3   0   0  2.50345E+001  5.04238E+000  5.04214E+000  4.92475E-002  2.00000E+002
//  1    -4   0   0  2.50566E+001  5.04634E+000  5.04610E+000  4.91774E-002  2.00000E+002
//*/
//        // write branch index, point number, type, and label
//        os << ibr << " " << pt << " " << ty << " " << lab << " ";
//
//        // write the first parameter value
//        os << p[0] << " ";
//
//        // get norm (technically, pricipal solution measure, but really....
//        os << norm << " ";
//
//        // now, we get the state variables
//        for(int i = 0; i < ndim; ++i)
//        {
//            os << x[i];
//            if(i < ndim-1)
//                os << " ";
//        }        
//
//        // write the second parameter value
//        os << p[1] << " ";
//
//        // end the line
//        os << std::endl;
//
//        break;
//
//    default:
//        throw std::logic_error("unknown or un-implemented PlotType in PlotEntry::Write");
//    }
//}
//
//void PlotEntry::WriteBD(std::ostream &os, PlotType type, int ndim)
//{
//    // write differently depending on type
//    switch(type)
//    {
//    case ONE_PARAMETER:
//        // write stability marker
//        os << ((pt<0) ? -1 : 1) << " ";
//        
//        // write the parameter values
//        for(std::vector<PlotEntry *>::size_type i = 0; i < p.size(); ++i)
//            os << p[i] << " ";
//
//        // now, we write the state variables
//        for(int i = 0; i < ndim; ++i)
//        {
//            os << x[i];
//            if(i < ndim-1)
//                os << " ";
//        }        
//
//        // end the line
//        os << std::endl;
//
//        break;
//
//    case TWO_PARAMETER:
//        // write the parameter values
//        for(std::vector<PlotEntry *>::size_type i = 0; i < p.size(); ++i)
//            os << p[i] << " ";
//
//        // now, we write the state variables
//        for(int i = 0; i < ndim; ++i)
//        {
//            os << x[i];
//            if(i < ndim-1)
//                os << " ";
//        }        
//
//        // end the line
//        os << std::endl;
//
//        break;
//
//    default:
//        throw std::logic_error("unknown or un-implemented PlotType in PlotEntry::WriteBD");
//    }
//}
//
//// simple constructor
//PlotSet::PlotSet(int nd, PlotType t)
//    : ndim(nd), type(t)
//{
//}
//
//// equally simple descructor
//PlotSet::~PlotSet()
//{
//    EraseVectorOfPointers(entries);
//}
//
//void PlotSet::ReadStandardAutoFormat(const std::string &f, InsertType t, bool clean)
//{
//    // first set the filename
//    filename = f;
//
//    // erase the old
//    if(clean)
//        EraseVectorOfPointers(entries);
//
//    // read in the new
//    std::ifstream in(filename.c_str());
//    MY_TEST(in, "could not open plot file");
//  
//    // read in one PlotEntry at a time
//    if(t == FORWARD)
//    {
//        while(!in.eof() && (in.peek() != EOF))
//        {
//            entries.push_back(new PlotEntry);
//            entries.back()->Read(in, type, ndim);
//        }
//    }
//    else if(t == REVERSE)
//    {
//        std::vector<PlotEntry *> temp;
//
//        // read in one PlotEntry at a time
//        while(!in.eof() && (in.peek() != EOF))
//        {
//            temp.push_back(new PlotEntry);
//            temp.back()->Read(in, type, ndim);
//        }
//
//        // insert into entries using reverse iterators
//        entries.insert(entries.end(), temp.rbegin(), temp.rend());
//    }
//
//    // all done
//    in.close();
//}
//
////! read from file using more flexible format.
////! NOTE: currently identical to ReadStandardAutoFormat
//void PlotSet::Read(const std::string &f, InsertType t, bool clean)
//{
//    ReadStandardAutoFormat(f, t, clean);
//}
//
////! write to standard AUTO formatted fort.7 file
//void PlotSet::WriteStandardAutoFormat(const std::string &f)
//{
//    // first set filename
//    filename = f;
//
//    std::ofstream out(filename.c_str());
//    MY_TEST(out, "could not open plot file");
//  
//    // read in one SolutionEntry at a time
//    for(std::vector<PlotEntry *>::size_type i = 0; i < entries.size(); ++i)
//        entries[i]->Write(out, type, ndim);
//    
//    // all done
//    out.close();
//}
//
////! write to file using more flexible format.
////! NOTE: currently identical to WriteStandardAutoFormat
//void PlotSet::Write(const std::string &f)
//{
//    WriteStandardAutoFormat(f);
//}
//
////! write bifucation diagram (first column is -1 or 1 for stability,
////! then you get all parameter values and state variable values following
//void PlotSet::WriteBD(const std::string &f)
//{
//    // first set filename
//    filename = f;
//
//    std::ofstream out(filename.c_str());
//    MY_TEST(out, "could not open bd file");
//  
//    // read in one SolutionEntry at a time
//    for(std::vector<PlotEntry *>::size_type i = 0; i < entries.size(); ++i)
//        entries[i]->WriteBD(out, type, ndim);
//
//    // all done
//    out.close();
//}
//
//// append info from file
//void PlotSet::Append(const std::string &f, int branch_index, InsertType t)
//{
//    // all we need to do is to read in new data, without
//    // cleaning out the old data.
//    Read(f, t, false);
//}
//
//
//
//
//////////////////////////////////////////////////////////////////////////
////
////  AutoApi
////
//////////////////////////////////////////////////////////////////////////
//
////! we can load constants to prep for a run.
//ContinuationConstants *AutoApi::LoadConstants(ContinuationConstants *cc)
//{
//    // first delete constants if non-null
//    if(constants)
//        if(constants->api_owns)
//            delete constants;
//
//    // set constants and return
//    return (constants = cc);
//}
//
////! we can load constants to prep for a run. this version just calls
////! the ContinuationConstants version after reading in the constants file
//ContinuationConstants *AutoApi::LoadConstants(const std::string &f)
//{
//    // delete old
//    if(constants)
//        if(constants->api_owns)
//            delete constants;
//
//    // read in and return constants.
//    constants = new ContinuationConstants(f);
//    return constants;
//}
//
////! we can load solutions from a previous run so we have start labels, etc.
//SolutionSet *AutoApi::LoadSolutionSet(SolutionSet *sol)
//{
//    // sanity check
//    MY_TEST(constants != 0, "can't load solutions without first loading constants");
//
//    // delete old solutions
//    if(solutions)
//        if(solutions->api_owns)
//            delete solutions;
//
//    // read in and return solutions.
//    solutions = sol;
//    return solutions;
//}
//
////! we can load solutions from a previous run from a file.
//SolutionSet *AutoApi::LoadSolutionSet(const std::string &f)
//{
//    // sanity check
//    MY_TEST(constants != 0, "can't load solutions without first loading constants");
//
//    // delete old solutions
//    if(solutions)
//        if(solutions->api_owns)
//            delete solutions;
//
//    // read in and return solutions.
//    solutions = new SolutionSet(f);
//    return solutions;
//}
//
////! we can load solutions from a previous run from a file.
//SolutionSet *AutoApi::AppendSolutionSet(const std::string &f)
//{
//    // sanity checks
//    MY_TEST(constants != 0, "can't append solutions without first loading constants");
//
//    // read in and return solutions with new file stuff appended.
//    if(solutions)
//        solutions->Append(f);
//    else
//        solutions = new SolutionSet(f);
//
//    return solutions;
//}
//
//void AutoApi::Run(const std::string &key, bool append_solutions)
//{
//    // for now, we do the most simple thing, which is to use the NORMAL main proc
//    // from the auto distrubution and just write our continuation constants to the
//    // fort.2 file. the fort.3 input file will be prepared by calls to @@e???
//    
//    // set fort_names
//    SetFortNames(key.c_str());
//    fort_names_are_valid = 1;
//
//    // write numerical constants to fort.2
//    MY_TEST(constants, "oops, run called without constants set!");
//    constants->Write(fort_name[2]);
//
//    // also, check solutions filename
//    if(solutions && (constants->restart_label > 0))
//        solutions->Write(fort_name[3]);
//
//    // run AUTO's main proc
//    char *argv[] = { "auto_api" };
//    AUTO_main(1, argv);
//
//    // get the fort.8 ingested and integrated with the current setup
//    // so that users can continue their runs normally
//    if(append_solutions)
//        AppendSolutionSet(fort_name[8]);
//
//    // go back to the original fort names
//    fort_names_are_valid = 0;
//}
//

}


