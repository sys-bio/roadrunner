#ifndef __AUTO_C_H__
#define __AUTO_C_H__

#include <stdio.h>
/* @@edc added WIN32 skip of unistd.h and pthreads and mpi stuff */
#ifndef _WIN32
#include <unistd.h>
#else
#include <time.h>
#undef PTHREADS
#undef MPI
#endif

#ifndef _WIN32
#include <sys/times.h>
#endif

#ifdef PTHREADS
#include <pthread.h>
#endif
#ifdef MPI
#include <signal.h>
#include <mpi.h>
#include "auto_mpi.h"
#endif
#include <string.h>

#ifdef __CYGWIN__
#include <sys/time.h>
#endif

#ifdef USAGE
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

//#include "auto.h"
#include "auto_types.h"

namespace autolib {

extern char fort_name[13][512]; //!< names for IO files.
extern int fort_names_are_valid; //!< should AUTO_main set fort_names to the defaults?
extern void SetFortNames(const char *key);

#define NBIFX (20)
#define KREDO (1)  /*get rid of*/
#define HMACH (1.0e-7)
#define RSMALL (1.0e-30)
#define RLARGE (1.0e+30)
#define HMACH1 (HMACH+1.0e0)
#define M1SB (NBIFX) 
#define LEFT (1)
#define RIGHT (2)
#define QZMATZ (.FALSE.)
#define QZEPS1 (0.0E0)
#define HMACHHO (1.0e-13)

extern FILE *fp2;
extern FILE *fp3;
extern FILE *fp6;
extern FILE *fp7;
extern FILE *fp8;
extern FILE *fp9;
extern FILE *fp10;
extern FILE *fp12;

extern int num_model_pars;
extern int num_total_pars;
extern int sysoff;

#define CONPAR_DEFAULT  0
#define CONPAR_PTHREADS 1
#define CONPAR_MPI      2
#define SETUBV_DEFAULT  0
#define SETUBV_PTHREADS 1
#define SETUBV_MPI      2
#define REDUCE_DEFAULT  0
#define REDUCE_PTHREADS 1
#define REDUCE_MPI      2

extern int global_conpar_type;
extern int global_setubv_type;
extern int global_reduce_type;
extern int global_num_procs;
extern int global_verbose_flag;

/* main.c */
int AUTO_main(int argc, char *argv[]);

/* autlib1.c */
/* @@edc added WIN32 version of timing stuff */
#if defined(_WIN32) || defined(__WIN32__)
void time_start(clock_t *);
void time_end(clock_t, char *, FILE *fp);
#else
void time_start(clock_t *);
void time_end(clock_t, char *, FILE *fp);
#endif

#ifdef USAGE
void usage_start(struct rusage **);
void usage_end(struct rusage *, char *);
#endif

void allocate_global_memory(const iap_type);
int init(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *thl, doublereal **thu_pointer, integer **iuz_pointer, doublereal **vuz_pointer, logical *eof);
int chdim(iap_type *iap);
int autoae(iap_type *iap, rap_type *rap, doublereal *par, 
integer *icp, 
FUNI_TYPE((*funi)), 
STPNT_TYPE_AE((*stpnt)), 
PVLI_TYPE_AE((*pvli)), 
doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);

int autobv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), STPNT_TYPE_BVP((*stpnt)), PVLI_TYPE_BVP((*pvli)), doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
int init1(iap_type *iap, rap_type *rap, integer *icp, doublereal *par);
int cnrlae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), STPNT_TYPE_AE((*stpnt)), PVLI_TYPE_AE((*pvli)), doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
STPNT_TYPE_AE(stpnus);
STPNT_TYPE_AE(stpnae);
int stprae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rds, integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *thl, doublereal *thu);
int contae(iap_type *iap, rap_type *rap, doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *uold, doublereal *udot);
int solvae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rds, integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *thl, doublereal *thu);
int lcspae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FNCS_TYPE_AE((*fncs)), FUNI_TYPE((*funi)), integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *q, doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
int mueller(doublereal *q0, doublereal *q1, doublereal *q, doublereal *s0, doublereal *s1, doublereal *s, doublereal *rds);
FNCS_TYPE_AE(fnbpae);
FNCS_TYPE_AE(fnlpae);
FNCS_TYPE_AE(fnhbae);
FNCS_TYPE_AE(fnuzae);
int stbif(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *m1aaloc, doublereal **aa, integer m1sbloc, doublereal **stud, doublereal **stu, doublereal *stla, doublereal *stld, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *udot, doublereal *dfdu, doublereal *dfdp, doublereal *thl, doublereal *thu);
int swpnt(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, integer m1sbloc, doublereal **stud, doublereal **stu, doublereal *stla, doublereal *stld, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *udot);
int swprc(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), integer *m1aaloc, doublereal **aa, doublereal *rhs, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal *u, doublereal *du, doublereal *uold, doublereal *udot, doublereal *f, doublereal *dfdu, doublereal *dfdp, doublereal *rds, doublereal *thl, doublereal *thu);
int sthd(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *thl, doublereal *thu);
int headng(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer iunit, integer *n1, integer *n2);
int stplae(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rlcur, doublereal *u);
int wrline(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *icu, integer *ibr, integer *ntot, integer *lab, doublereal *vaxis, doublereal *u);
int wrtsp8(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *lab, doublereal *rlcur, doublereal *u);
int wrjac(iap_type *iap, integer *n, integer *m1aaloc, doublereal **aa, doublereal *rhs);
int msh(const iap_type *iap, const rap_type *rap, doublereal *tm);
int genwts(const integer ncol, const integer n1, doublereal **wt, doublereal **wp);
int cpnts(const integer ncol, doublereal *zm);
int cntdif(integer *n, doublereal *d);
int wint(const integer n, doublereal *wi);
int adptds(iap_type *iap, rap_type *rap, doublereal *rds);
int adapt(iap_type *iap, rap_type *rap, integer *nold, integer *ncold, integer *nnew, integer *ncnew, doublereal *tm, doublereal *dtm, integer *ndxloc, doublereal **ups, doublereal **vps);
int interp(iap_type *iap, rap_type *rap, integer *ndim, integer *n, integer *nc, doublereal *tm, integer *ndxloc, doublereal **ups, integer *n1, integer *nc1, doublereal *tm1, doublereal **ups1, doublereal *tm2, integer *itm1);
int newmsh(iap_type *iap, rap_type *rap, integer *ndxloc, doublereal **ups, integer *nold, integer *ncold, doublereal *tmold, doublereal *dtmold, integer *nnew, doublereal *tmnew, integer *iper);
int ordr(iap_type *iap, rap_type *rap, integer *n, doublereal *tm, integer *n1, doublereal *tm1, integer *itm1);
int intwts(iap_type *iap, rap_type *rap, integer *n, doublereal *z__, doublereal *x, doublereal *wts);
int eqdf(iap_type *iap, rap_type *rap, integer *ntst, integer *ndim, integer *ncol, doublereal *dtm, integer *ndxloc, doublereal **ups, doublereal *eqf, integer *iper);
int eig(iap_type *iap, integer *ndim, integer *m1a, doublereal *a, doublecomplex *ev, integer *ier);
int nlvc(integer n, integer m, integer k, doublereal **a, doublereal *u);
int nrmlz(integer *ndim, doublereal *v);
doublereal pi(doublereal r__);
int ge(integer n, integer m1a, doublereal *a, integer nrhs, integer ndxloc, doublereal *u, integer m1f, doublereal *f, doublereal *det);
int newlab(iap_type *iap, rap_type *rap);
int findlb(iap_type *iap, const rap_type *rap, integer irs, integer *nfpr, logical *found);
int readlb(const iap_type *iap, const rap_type *rap, doublereal *u, doublereal *par);
int skip3(integer *nskip, logical *eof3);
doublereal rinpr(iap_type *iap, integer *ndim1, integer *ndxloc, doublereal **ups, doublereal **vps, doublereal *dtm, doublereal *thu);
doublereal rnrmsq(iap_type *iap, integer *ndim1, integer *ndxloc, doublereal **ups, doublereal *dtm, doublereal *thu);
doublereal rintg(iap_type *iap, integer *ndxloc, integer ic, doublereal **ups, doublereal *dtm);
doublereal rnrm2(iap_type *iap, integer *ndxloc, integer ic, doublereal **ups, doublereal *dtm);
doublereal rmxups(iap_type *iap, integer *ndxloc, integer i, doublereal **ups);
doublereal rmnups(iap_type *iap, integer *ndxloc, integer i, doublereal **ups);
int scaleb(iap_type *iap, integer *icp, integer *ndxloc, doublereal **dvps, doublereal *rld, doublereal *dtm, doublereal *thl, doublereal *thu);
int cnrlbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), STPNT_TYPE_BVP((*stpnt)), PVLI_TYPE_BVP((*pvli)), doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
int contbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal *dtm, doublereal *thl, doublereal *thu);
int extrbv(iap_type *iap, rap_type *rap, FUNI_TYPE((*funi)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps);
int stupbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **upoldp);
int stepbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), PVLI_TYPE_BVP((*pvli)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal *tm, doublereal *dtm, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu);
int rsptbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), STPNT_TYPE_BVP((*stpnt)), doublereal *rds, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *tm, doublereal *dtm, doublecomplex *ev, integer *nodir, doublereal *thl, doublereal *thu);
STPNT_TYPE_BVP(stpnbv);
STPNT_TYPE_BVP(stpnub);
int setrtn(iap_type *iap, integer *ntst, integer *ndxloc, doublereal **ups, doublereal *par);
int stdrbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal *dtm, integer iperp, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu);
int lcspbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FNCS_TYPE_BVP((*fncs)), FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), PVLI_TYPE_BVP((*pvli)), doublereal *q, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **fa, doublereal *fc, doublereal *tm, doublereal *dtm, doublereal **p0, doublereal **p1, doublecomplex *ev, doublereal *thl, doublereal *thu, integer *iuz, doublereal *vuz);
FNCS_TYPE_BVP(fnlpbv);
FNCS_TYPE_BVP(fnbpbv);
FNCS_TYPE_BVP(fnspbv);
FNCS_TYPE_BVP(fnuzbv);
int tpspbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublecomplex *ev);
int stplbv(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *dtm, doublereal *thl, doublereal *thu);
int wrtbv8(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *dtm);
int wrtbv9(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rlcur, integer *ndxloc, doublereal **ups, doublereal *tm, doublereal *dtm, doublereal *thl, doublereal *thu);
PVLI_TYPE_AE(pvlsae);
PVLI_TYPE_BVP(pvlsbv);
int setpae(iap_type *iap, rap_type *rap);
int setpbv(iap_type *iap, rap_type *rap, doublereal *dtm);
int autim0(doublereal *t);
int autim1(doublereal *t);
/* sometimes here ups is just one vector .... *double, and sometimes a **double */
doublereal getp(char *code, integer ic, void *u_or_ups);
int set_function_pointers(const iap_type,function_list *);
/* autlib2.c */
int solvbv(integer *ifst, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), doublereal *rds, integer *nllv, doublereal *rlcur, doublereal *rlold, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **dups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal *dtm, doublereal **fa, doublereal *fc, doublereal **p0, doublereal **p1, doublereal *thl, doublereal *thu);
int setfcdd(integer *ifst, doublereal **dd, doublereal *fc, integer *ncb, integer *nrc);
int faft(doublereal **ff, doublereal **fa, integer *ntst, integer *nrow, integer *ndxloc);
int partition(integer *n, integer *kwt, integer *m);
integer mypart(integer *iam, integer *np);
#ifndef MANIFOLD
int setrhs(integer *ndim, integer *ips, integer *na, integer *ntst, integer *np, integer *ncol, integer *nbc, integer *nint, integer *ncb, integer *nrc, integer *nra, integer *nca, integer *iam, integer *kwt, logical *ipar, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer *ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, doublereal **fa, doublereal *fc, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#else
int setrhs(integer *ndim, integer *ips, integer *na, integer *ntst, integer *np, integer *ncol, integer *nbc, integer *nint, integer *nalc, integer *ncb, integer *nrc, integer *nra, integer *nca, integer *iam, integer *kwt, logical *ipar, FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer *ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, doublereal *rds, doublereal **fa, doublereal *fc, doublereal *rlcur, doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#endif
int brbd(doublereal ***a, doublereal ***b, doublereal ***c, doublereal **d, doublereal **fa, doublereal *fc, doublereal **p0, doublereal **p1, integer *ifst, integer *idb, integer *nllv, doublereal *det, integer *nov, integer *na, integer *nbc, integer *nra, integer *nca, integer *ncb, integer *nrc, integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, doublereal **faa, doublereal ***ca1, doublereal ***s1, doublereal ***s2, integer *icf11, integer *ipr, integer *icf1, integer *icf2, integer *irf, integer *icf);
int setzero(doublereal **fa, doublereal *fc, integer *na, integer *nra, integer *nrc);
int conrhs(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *nbc, integer *nrc, doublereal ***c, doublereal **fa, doublereal *fc, integer *irf, integer *icf, integer *iam);
int copycp(integer na, integer nov, integer nra, integer nca, doublereal ***a, integer ncb, doublereal ***b, integer nrc, doublereal ***c, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, integer *irf);
int cpyrhs(integer na, integer nov, integer nra, doublereal **faa, doublereal **fa, integer *irf);
int redrhs(integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***cc, doublereal **faa, doublereal *fc, integer *na, integer *nov, integer *ncb, integer *nrc, doublereal ***ca1, integer *icf1, integer *icf2, integer *icf11, integer *ipr, integer *nbc);
int dimrge(integer *iam, integer *kwt, logical *par, doublereal **e, doublereal ***cc, doublereal **d, doublereal *fc, integer *ifst, integer *na, integer *nrc, integer *nov, integer *ncb, integer *idb, integer *nllv, doublereal *fcc, doublereal **p0, doublereal **p1, doublereal *det, doublereal ***s, doublereal ***a2, doublereal **faa, doublereal ***bb);
int bcksub(integer *iam, integer *kwt, logical *par, doublereal ***s1, doublereal ***s2, doublereal ***a2, doublereal ***bb, doublereal **faa, doublereal *fc, doublereal *fcc, doublereal *sol1, doublereal *sol2, doublereal *sol3, integer *na, integer *nov, integer *ncb, integer *icf2);
int infpar(integer *iam, logical *par, doublereal ***a, doublereal ***b, doublereal **fa, doublereal *sol1, doublereal *sol2, doublereal *fc, integer *na, integer *nov, integer *nra, integer *nca, integer *ncb, integer *irf, integer *icf);
int rd0(integer *iam, integer *kwt, doublereal *d, integer *nrc);
int print1(integer *nov, integer *na, integer *nra, integer *nca, integer *ncb, integer *nrc, doublereal ***a, doublereal ***b, doublereal ***c, doublereal **d, doublereal **fa, doublereal *fc);
integer mynode(void);
integer numnodes(void);
int gsync(void);
doublereal dclock(void);
int csend(void);
int crecv(void);
int gdsum(void);
int gsendx(void);
int gcol(void);
int led(void);
int setiomode(void);
/* autlib3.c */
FUNI_TYPE(fnlp);
int fflp(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnlp);
FUNI_TYPE(fnc1);
STPNT_TYPE_AE(stpnc1);
FUNI_TYPE(fnc2);
int ffc2(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnc2);
FUNI_TYPE(fnds);
FUNI_TYPE(fnti);
FUNI_TYPE(fnhd);
int ffhd(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnhd);
FUNI_TYPE(fnhb);
int ffhb(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnhb);
FUNI_TYPE(fnhw);
int ffhw(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_AE(stpnhw);
FUNI_TYPE(fnps);
BCNI_TYPE(bcps);
ICNI_TYPE(icps);
int pdble(const iap_type *iap, const rap_type *rap, integer *ndim, integer *ntst, integer *ncol, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *par);
STPNT_TYPE_BVP(stpnps);
FUNI_TYPE(fnws);
int ffws(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp, integer ndm, doublereal *dfu, doublereal *dfp);
FUNI_TYPE(fnwp);
int stpnwp(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fnsp);
int ffsp(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp, integer ndm, doublereal *dfu, doublereal *dfp);
FUNI_TYPE(fnpe);
int ffpe(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp, integer ndm, doublereal *dfu, doublereal *dfp);
ICNI_TYPE(icpe);
FUNI_TYPE(fnpl);
int ffpl(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcpl);
ICNI_TYPE(icpl);
int stpnpl(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fnpd);
int ffpd(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcpd);
ICNI_TYPE(icpd);
int stpnpd(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fntr);
int fftr(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
int bctr(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, const doublereal *u0,const  doublereal *u1, doublereal *f, integer ijac, doublereal *dbc);
ICNI_TYPE(ictr);
int stpntr(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
FUNI_TYPE(fnpo);
int ffpo(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const doublereal *upold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcpo);
ICNI_TYPE(icpo);
int fipo(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *fi, doublereal *dint, integer ndmt, doublereal *dfdu, doublereal *dfdp);
STPNT_TYPE_BVP(stpnpo);
FUNI_TYPE(fnbl);
int ffbl(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu, doublereal *dfdp);
BCNI_TYPE(bcbl);
int fbbl(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nbc, integer nbc0, const doublereal *u0, const doublereal *u1, doublereal *f, doublereal *dbc);
ICNI_TYPE(icbl);
int fibl(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *f, doublereal *dint);
STPNT_TYPE_BVP(stpnbl);
FUNI_TYPE(funi);
BCNI_TYPE(bcni);
ICNI_TYPE(icni);
int fopi(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const integer *icp, doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp);
/* autlib4.c */
int flowkm(integer ndim, doublereal **c0, doublereal **c1, integer iid, doublecomplex *ev);
int dhhpr(integer k, integer j, integer n, doublereal *x, integer incx, doublereal *beta, doublereal *v);
int dhhap(integer k, integer j, integer n, integer q, doublereal *beta, doublereal *v, integer job, doublereal **a, integer lda);
/* autlib5.c */
FUNI_TYPE(fnho);
int ffho(const iap_type *iap, const rap_type *rap, integer ndim, const doublereal *u, const doublereal *uold, const integer *icp, doublereal *par, doublereal *f, integer ndm, doublereal *dfdu);
BCNI_TYPE(bcho);
int fbho(const iap_type *iap, integer ndim, doublereal *par, const integer *icp, integer nbc, integer nbc0, const doublereal *u0, const doublereal *u1, doublereal *fb);
ICNI_TYPE(icho);
int fiho(const iap_type *iap, const rap_type *rap, integer ndim, doublereal *par, const integer *icp, integer nint, integer nnt0, const doublereal *u, const doublereal *uold, const doublereal *udot, const doublereal *upold, doublereal *fi);
int inho(iap_type *iap, integer *icp, doublereal *par);
int preho(iap_type *iap, rap_type *rap, doublereal *par, const integer *icp, integer ndx, integer *ntsr, integer *nar, integer ncolrs, doublereal **ups, doublereal **udotps, doublereal *tm, doublereal *dtm);
int stpnho(iap_type *iap, rap_type *rap, doublereal *par, integer *icp, integer *ntsr, integer *ncolrs, doublereal *rlcur, doublereal *rldot, integer *ndxloc, doublereal **ups, doublereal **udotps, doublereal **upoldp, doublereal *tm, doublereal *dtm, integer *nodir, doublereal *thl, doublereal *thu);
int stpho(iap_type *iap, integer *icp, doublereal *u, doublereal *par, doublereal t);
PVLI_TYPE_BVP(pvlsho);
doublereal psiho(const iap_type *iap, integer is, doublereal **rr, doublereal **ri, doublereal ***v, doublereal ***vt, const integer *icp, doublereal *par, const doublereal *pu0, const doublereal *pu1);
int eigho(integer isign, integer itrans, doublereal *rr, doublereal *ri, doublereal **vret, const doublereal *xequib, const integer *icp, doublereal *par, integer ndm);
int prjctn(doublereal **bound, const doublereal *xequib, const integer *icp, doublereal *par, integer imfd, integer is, integer itrans, integer ndm);
/* eispack.c */
int rg(integer nm, integer n, doublereal *a, doublereal *wr, doublereal *wi, integer matz, doublereal *z__, integer *iv1, doublereal *fv1, integer *ierr);
int hqr(integer *nm, integer *n, integer *low, integer *igh, doublereal *h__, doublereal *wr, doublereal *wi, integer *ierr);
int hqr2(integer *nm, integer *n, integer *low, integer *igh, doublereal *h__, doublereal *wr, doublereal *wi, doublereal *z__, integer *ierr);
int cdiv(doublereal *ar, doublereal *ai, doublereal *br, doublereal *bi, doublereal *cr, doublereal *ci);
int balanc(integer *nm, integer *n, doublereal *a, integer *low, integer *igh, doublereal *scale);
int balbak(integer *nm, integer *n, integer *low, integer *igh, doublereal *scale, integer *m, doublereal *z__);
int elmhes(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, integer *int__);
int eltran(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, integer *int__, doublereal *z__);
int qzhes(integer nm, integer n, doublereal *a, doublereal *b, logical matz, doublereal *z__);
int qzit(integer nm, integer n, doublereal *a, doublereal *b, doublereal eps1, logical matz, doublereal *z__, integer *ierr);
int qzval(integer nm, integer n, doublereal *a, doublereal *b, doublereal *alfr, doublereal *alfi, doublereal *beta, logical matz, doublereal *z__);
doublereal epslon(doublereal x);
doublereal dnrm2(integer *n, doublereal *dx, integer *incx);
doublereal ddot(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy);
int dscal(integer *n, doublereal *da, doublereal *dx, integer *incx);
integer idamax(integer *n, doublereal *dx, integer *incx);
int daxpy(integer *n, doublereal *da, doublereal *dx, integer *incx, doublereal *dy, integer *incy);
int drot(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy, doublereal *c, doublereal *s);
int dswap(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy);
int dgemc(integer *m, integer *n, doublereal *a, integer *lda, doublereal *b, integer *ldb, logical *trans);
int xerbla(char *srname, integer *info, integer srname_len);
logical lsame(char *ca, char *cb, integer ca_len, integer cb_len);
int dgemm(char *transa, char *transb, integer *m, integer *n, integer *k, doublereal *alpha, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *beta, doublereal *c, integer *ldc, integer transa_len, integer transb_len);
int ezsvd(doublereal *x, integer *ldx, integer *n, integer *p, doublereal *s, doublereal *e, doublereal *u, integer *ldu, doublereal *v, integer *ldv, doublereal *work, integer *job, integer *info, doublereal *tol);
int ndrotg(doublereal *f, doublereal *g, doublereal *cs, doublereal *sn);
int ndsvd(doublereal *x, integer *ldx, integer *n, integer *p, doublereal *s, doublereal *e, doublereal *u, integer *ldu, doublereal *v, integer *ldv, doublereal *work, integer *job, integer *info, integer *maxitr, doublereal *tol, integer *idbg, integer *ifull, integer *kount, integer *kount1, integer *kount2, integer *skip, integer *limshf, doublereal *maxsin, integer *iidir);
int prse(integer *ll, integer *m, integer *nrow, integer *ncol, doublereal *s, doublereal *e);
int sig22(doublereal *a, doublereal *b, doublereal *c, doublereal *sigmin, doublereal *sigmax, doublereal *snr, doublereal *csr, doublereal *snl, doublereal *csl);
doublereal sigmin(doublereal *a, doublereal *b, doublereal *c);
int sndrtg(doublereal *f, doublereal *g, doublereal *cs, doublereal *sn);
int hqr3lc(doublereal *a, doublereal *v, integer *n, integer *nlow, integer *nup, doublereal *eps, doublereal *er, doublereal *ei, integer *type__, integer *na, integer *nv, integer *imfd);
int split(doublereal *a, doublereal *v, integer *n, integer *l, doublereal *e1, doublereal *e2, integer *na, integer *nv);
int exchng(doublereal *a, doublereal *v, integer *n, integer *l, integer *b1, integer *b2, doublereal *eps, logical *fail, integer *na, integer *nv);
int qrstep(doublereal *a, doublereal *v, doublereal *p, doublereal *q, doublereal *r__, integer *nl, integer *nu, integer *n, integer *na, integer *nv);
int orthes(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, doublereal *ort);
int ortran(integer *nm, integer *n, integer *low, integer *igh, doublereal *a, doublereal *ort, doublereal *z__);


/* problem defined functions*/
int func(integer ndim, const doublereal *u, const integer *icp, 
     const doublereal *par, integer ijac, 
     doublereal *f, doublereal *dfdu, doublereal *dfdp);
int stpnt(integer ndim, doublereal t, 
      doublereal *u, doublereal *par);
int bcnd(integer ndim, const doublereal *par, const integer *icp, integer nbc, 
     const doublereal *u0, const doublereal *u1, integer ijac,
     doublereal *f, doublereal *dbc);
int icnd(integer ndim, const doublereal *par, const integer *icp, integer nint, 
     const doublereal *u, const doublereal *uold, const doublereal *udot, 
     const doublereal *upold, integer ijac,
     doublereal *fi, doublereal *dint);
int fopt(integer ndim, const doublereal *u, const integer *icp, 
     const doublereal *par, integer ijac, 
     doublereal *fs, doublereal *dfdu, doublereal *dfdp);
/* This is a dirty trick with mismatching prototypes -
   sometimes u has to be **double and sometimes *double */
int pvls(integer ndim, const void *u, doublereal *par);
/* conpar.c */
void *conpar_process(void *);
int conpar(integer *nov, integer *na, integer *nra, integer *nca, doublereal ***a, integer *ncb, doublereal ***b, integer *nbc, integer *nrc, doublereal ***c, doublereal **d, integer *irf, integer *icf);


/* reduce.c */
int reduce(integer *iam, integer *kwt, logical *par, doublereal ***a1, doublereal ***a2, doublereal ***bb, doublereal ***cc, doublereal **dd, integer *na, integer *nov, integer *ncb, integer *nrc, doublereal ***s1, doublereal ***s2, doublereal ***ca1, integer *icf1, integer *icf2, integer *icf11, integer *ipr, integer *nbc);

/*setubv.c */
void *setubv_make_aa_bb_cc(void *);
#ifndef MANIFOLD
int setubv(integer ndim, integer ips, integer na, integer ncol, integer nbc, integer nint, integer ncb, integer nrc, integer nra, integer nca, 
       FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, 
       doublereal rds, doublereal ***aa, doublereal ***bb, doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal *rlcur, 
       doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, 
       doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#else
int setubv(integer ndim, integer ips, integer na, integer ncol, integer nbc, integer nint, integer nalc, integer ncb, integer nrc, integer nra, integer nca, 
       FUNI_TYPE((*funi)), BCNI_TYPE((*bcni)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, integer *icp, 
       doublereal *rds, doublereal ***aa, doublereal ***bb, doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal *rlcur, 
       doublereal *rlold, doublereal *rldot, doublereal **ups, doublereal **uoldps, doublereal **udotps, doublereal **upoldp, doublereal **dups, 
       doublereal *dtm, doublereal *thl, doublereal *thu, doublereal **p0, doublereal **p1);
#endif
void setubv_parallel_arglist_copy(setubv_parallel_arglist *output, const setubv_parallel_arglist input);
#ifndef MANIFOLD
void setubv_parallel_arglist_constructor(integer ndim, integer ips, integer na, integer ncol, 
                     integer nbc, integer nint, integer ncb, integer nrc, integer nra, integer nca, 
                     FUNI_TYPE((*funi)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, 
                     integer *icp, doublereal ***aa, doublereal ***bb, 
                     doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal **ups, 
                     doublereal **uoldps, doublereal **udotps, 
                     doublereal **upoldp, doublereal *dtm, 
                     doublereal **wp, doublereal **wt, doublereal *wi,
                     doublereal *thu, doublereal *thl, doublereal *rldot, BCNI_TYPE((*bcni)),
                     setubv_parallel_arglist *data);
#else
void setubv_parallel_arglist_constructor(integer ndim, integer ips, integer na, integer ncol, 
                     integer nbc, integer nint, integer nalc, integer ncb, integer nrc, integer nra, integer nca, 
                     FUNI_TYPE((*funi)), ICNI_TYPE((*icni)), integer ndxloc, iap_type *iap, rap_type *rap, doublereal *par, 
                     integer *icp, doublereal ***aa, doublereal ***bb, 
                     doublereal ***cc, doublereal **dd, doublereal **fa, doublereal *fc, doublereal **ups, 
                     doublereal **uoldps, doublereal **udotps, 
                     doublereal **upoldp, doublereal *dtm, 
                     doublereal **wp, doublereal **wt, doublereal *wi,
                     doublereal *thu, doublereal *thl, doublereal *rldot, BCNI_TYPE((*bcni)),
                     setubv_parallel_arglist *data);
#endif
void setubv_make_fa(setubv_parallel_arglist larg);
#ifndef MANIFOLD
void setubv_make_fc_dd(setubv_parallel_arglist larg,doublereal **dups, doublereal *rlcur, 
               doublereal *rlold, doublereal rds);
#else
void setubv_make_fc_dd(setubv_parallel_arglist larg,doublereal **dups, doublereal *rlcur, 
               doublereal *rlold, doublereal *rds);
#endif


/*worker.c*/
int mpi_worker();
int mpi_setubv_worker();
int mpi_conpar_worker();
int set_funi_and_icni(iap_type *,setubv_parallel_arglist *);

#ifdef AUTO_CONSTRUCT_DESCTRUCT
int user_construct(int argc, char **argv);
int user_destruct();
#endif

/*dmatrix.c*/
doublereal **dmatrix(integer n_rows, integer n_cols);
doublereal ***dmatrix_3d(integer n_levels, integer n_rows, integer n_cols);
void free_dmatrix(doublereal **m);
void free_dmatrix_3d(doublereal ***m);

/* slower allocation routines to facilitate debugging - don't work with MPI! */
/* mess up eispack interfacing as well - can mess up Floquet multipliers!    */

doublereal **dmatrix_debug(integer n_rows, integer n_cols);
doublereal ***dmatrix_3d_debug(integer n_levels, integer n_rows, integer n_cols);
void free_dmatrix_debug(doublereal **m);
void free_dmatrix_3d_debug(doublereal ***m);

#ifndef DMATRIX_C
#ifdef MALLOC_DEBUG
#define dmatrix dmatrix_debug
#define dmatrix_3d dmatrix_3d_debug
#define free_dmatrix free_dmatrix_debug
#define free_dmatrix_3d free_dmatrix_3d_debug
#endif
#endif

}

#endif





