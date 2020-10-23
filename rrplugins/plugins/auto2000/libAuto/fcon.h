#if 0
#define NDIMX (12)
#define NCOLX (4)
#define NTSTX (10000)
#define NBCX (NDIMX+3)
#define NINTX (6)
//#define NUM_USER_PARS (128)
//#define NBIFX (20)
#define NUZRX (20)
#define NODES (1)
#define KREDO (1)
#define NIAP (41)
#define NRAP (19)
#define NDX (NTSTX+1)
#define M2U (NDIMX*NCOLX)
#define M1T (NTSTX+1)
#define M2T (NDIMX*NCOLX)
#define MCL1 (NCOLX)
#define MCL2 (MCL1+1)
#define HMACH (1.0e-7)
#define RSMALL (1.0e-30)
#define RLARGE (1.0e+30)
#endif

/* cabs.c */
double f__cabs(double real, double imag);
/* d_imag.c */
double d_imag(doublecomplex *z);
/* d_lg10.c */
double d_lg10(doublereal *x);
/* d_sign.c */
double d_sign(doublereal a, doublereal b);
/* etime_.c */
double etime(float *tarray);
/* i_dnnt.c */
integer i_dnnt(doublereal *x);
/* i_nint.c */
integer i_nint(real *x);
/* pow_dd.c */
double pow_dd(doublereal *ap, doublereal *bp);
/* pow_di.c */
double pow_di(doublereal *ap, integer *bp);
/* pow_ii.c */
integer pow_ii(integer ap, integer bp);
/* r_lg10.c */
double r_lg10(real x);
/* z_abs.c */
double z_abs(doublecomplex *z);
/* z_exp.c */
void z_exp(doublecomplex *r, doublecomplex *z);
/* z_log.c */
void z_log(doublecomplex *r, doublecomplex *z);

