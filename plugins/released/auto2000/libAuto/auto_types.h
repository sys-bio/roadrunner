/* This structure contains all of the input data for the setubv routine
   Those values which are arrays and those
   which are input and output are markered as such*/
#ifndef AUTO_TYPES_H
#define AUTO_TYPES_H

namespace autolib
{

typedef struct {
  /* 1 */ integer ndim;
  /* 2 */ integer ips;
  /* 3 */ integer irs;
  /* 4 */ integer ilp;
  /* 5 */ integer ntst;
  /* 6 */ integer ncol;
  /* 7 */ integer iad;
  /* 8 */ integer iads;
  /* 9 */ integer isp;
  /* 10 */ integer isw;
  /* 11 */ integer iplt;
  /* 12 */ integer nbc;
  /* 13 */ integer nint;
#ifdef MANIFOLD
  /* 13a*/ integer nalc;    /* The number of arclength constraints (k) */
#endif
  /* 14 */ integer nmx;
  /* 15 */ integer nuzr;
  /* 16 */ integer npr;
  /* 17 */ integer mxbf;
  /* 18 */ integer iid;
  /* 19 */ integer itmx;
  /* 20 */ integer itnw;
  /* 21 */ integer nwtn;
  /* 22 */ integer jac;
  /* 23 */ integer ndm;
  /* 24 */ integer nbc0;
  /* 25 */ integer nnt0;
  /* 26 */ integer iuzr;
  /* 27 */ integer itp;
  /* 28 */ integer itpst;
  /* 29 */ integer nfpr;
  /* 30 */ integer ibr;
  /* 31 */ integer nit;
  /* 32 */ integer ntot;
  /* 33 */ integer nins;
  /* 34 */ integer istop;
  /* 35 */ integer nbif;
  /* 36 */ integer ipos;
  /* 37 */ integer lab;
  /* 41 */ integer nicp;
  /* The following are not set in init_.  
     They have to do with the old parallel version. */
  /* 38 */ integer mynode;
  /* 39 */ integer numnodes;
  /* 40 */ integer parallel_flag;
} iap_type;

typedef struct {
  /* 1 */ doublereal ds;
  /* 2 */ doublereal dsmin;
  /* 3 */ doublereal dsmax;
  /* There is no 4 */
  /* 5 */ doublereal dsold;
  /* 6 */ doublereal rl0;
  /* 7 */ doublereal rl1;
  /* 8 */ doublereal a0;
  /* 9 */ doublereal a1;
  /* 10 */ doublereal amp;
  /* 11 */ doublereal epsl;
  /* 12 */ doublereal epsu;
  /* 13 */ doublereal epss;
  /* 14 */ doublereal det;
  /* 15 */ doublereal tivp;
  /* 16 */ doublereal fldf;
  /* 17 */ doublereal hbff;
  /* 18 */ doublereal biff;
  /* 19 */ doublereal spbf;
} rap_type;

#define AUTOAE 0
#define AUTOBV 1

/*This is the type for all functions which can be used as "funi" the function
  which evaluates the right hand side of the equations and generates
  the Jacobian*/
#define FUNI_TYPE(X) int X(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp)

/*This is the type for all functions which can be used as "bcni" the function
  which evaluates the boundary conditions */
#define BCNI_TYPE(X) int X(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, const doublereal *u0, const doublereal *u1, doublereal *f, integer ijac, doublereal *dbc)

/*This is the type for all functions which can be used as "icni" the function
  which evaluates kernel of the integral constraints */
#define ICNI_TYPE(X) int X(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *f, integer ijac, doublereal *dint)

/*This is the type for all functions which can be used as additional
  output functions for algebraic problems */
#define PVLI_TYPE_AE(X) int X(iap_type *iap, rap_type *rap, doublereal *u, doublereal *par)

/*This is the type for all functions which can be used as additional
  output functions for BVPs */
#define PVLI_TYPE_BVP(X) int X(iap_type *iap, rap_type *rap, integer *icp, doublereal *dtm, integer *ndxloc, doublereal **ups, integer *ndim, doublereal **p0, doublereal **p1, doublereal *par)

/* This is the type for all functions that can be used at starting points
   for algebraic problems */
#define STPNT_TYPE_AE(X) int X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *u)

/* This is the type for all functions that can be used at starting points
   for BVPs */
#define STPNT_TYPE_BVP(X) int X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsrs, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu)

/*This is the type for all functions which can be used to detect
  special points for algebraic problems */
#define FNCS_TYPE_AE(X) doublereal X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, logical *chng, FUNI_TYPE((*funi)), integer *m1aaloc, doublereal **aa, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *uold, doublereal *udot, doublereal *rhs, doublereal *dfdu, doublereal *dfdp, integer *iuz, doublereal *vuz)

/*This is the type for all functions which can be used to detect
  special points for BVPS */
#define FNCS_TYPE_BVP(X) doublereal X(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, logical *chng, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal **p0, doublereal **p1, doublecomplex *ev, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal **dups, doublereal *tm, doublereal *dtm, doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz)

typedef struct {
  FUNI_TYPE((*funi));
  BCNI_TYPE((*bcni));
  ICNI_TYPE((*icni));
  STPNT_TYPE_BVP((*stpnt));
  PVLI_TYPE_BVP((*pvli));
} autobv_function_list;

typedef struct {
  FUNI_TYPE((*funi));
  STPNT_TYPE_AE((*stpnt));
  PVLI_TYPE_AE((*pvli));
} autoae_function_list;

typedef struct {
  int type;
  autobv_function_list bvlist;
  autoae_function_list aelist;
} function_list;

typedef struct {
  integer ndim, ips, ncol, nbc, nint, ncb, nrc, nra, nca, na; /*scalar input */
#ifdef MANIFOLD
  integer nalc; /* The dimension of the manifold */
#endif

  FUNI_TYPE((*funi)); /*scalar input*/
  ICNI_TYPE((*icni)); /*scalar input*/
  integer ndxloc; /*scalar input*/
  iap_type *iap; /*array input size: NIAP*/
  rap_type *rap; /*array input size: NRAP*/
  doublereal  *par; /*array input size: num_total_pars*/
  integer *icp; /*array input size:  num_total_pars*/
  doublereal ***aa; /*array output (but must be initialized to 0) size: *nca X *nra X *na */
  doublereal ***bb; /*array output (but must be initialized to 0) size: *ncb X *nra X *na */
  doublereal ***cc; /*array output (but must be initialized to 0) size: *nca X *nrc X *na */
  doublereal **dd; /*array output (but must be initialized to 0) size: *ncb X *nrc */
  doublereal **fa; /*array output (but must be initialized to 0) size: *nra X *na */
  doublereal *fc; /*array output (but must be initialized to 0) size: *nrc */
  doublereal **ups; /*array input size: *ndxloc X (*ndim X *ncol) */
  doublereal **uoldps; /*array input size: *ndxloc X (*ndim X *ncol) */
  doublereal **udotps; /*array input size: *ndxloc X (*ndim X *ncol) */
  doublereal **upoldp; /*array input size: *ndxloc X (*ndim X *ncol) */
  doublereal *dtm; /*array input size: *na */
  integer loop_start; /*scalar input*/
  integer loop_end; /*scalar input*/  
  integer loop_offset; /*scalar input*/
  doublereal **wp; /*array input size: MCL2*MCL1 */
  doublereal **wt; /*array input size: MCL2*MCL1 */
  doublereal *wi; /*array input size: MCL2*MCL1??? Not sure of this one yet */
  doublereal *thu; /*array input size: ndim * 8 */
  doublereal *thl; /*array input size: num_model_pars */ 
  doublereal *rldot; /*array input size: num_model_pars */ 
  BCNI_TYPE((*bcni));
} setubv_parallel_arglist;

/* This structure contains all of the input data for the conpar routine
   Those values which are arrays and those
   which are input and output are markered as such*/
typedef struct {
  integer *nov, *nra, *nca; /*scalars input*/
  doublereal ***a; /*array input and output size: nca X nra X na */
  integer *ncb; /*scalar input */
  doublereal ***b; /*array input and output size: ncb X nra X na*/
  integer *nbc, *nrc; /*scalar input */
  doublereal ***c; /*array input and output size: nca X nrc X *na*/
  doublereal **d; /*array input and output size: ncb X nrc*/
  integer *irf; /*array input size: na X nra*/
  integer *icf; /*array input: na X nca*/
  integer loop_start; /*scalar input*/
  integer loop_end; /*scalar output*/  
} conpar_parallel_arglist;

/* This structure contains all of the input data for the reduce routine
   Those values which are arrays and those
   which are input and output are markered as such*/
typedef struct {
  integer *nov, *nra, *nca; /*scalars input*/
  doublereal ***a; /*array input and output size: nca X nra X na */
  integer *ncb; /*scalar input */
  doublereal ***b; /*array input and output size: ncb X nra X na*/
  integer *nbc, *nrc; /*scalar input */
  doublereal ***c; /*array input and output size: nca X nrc X *na*/
  doublereal **d; /*array input and output size: ncb X nrc*/
  integer *irf; /*array input size: na X nra*/
  integer *icf; /*array input: na X nca*/
  integer loop_start; /*scalar input*/
  integer loop_end; /*scalar output*/  
} reduce_parallel_arglist;

}

#endif










