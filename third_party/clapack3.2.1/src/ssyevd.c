/* ssyevd.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"
#include "blaswrap.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__0 = 0;
static real c_b17 = 1.f;

/* Subroutine */ int ssyevd_(char *jobz, char *uplo, integer *n, real *a, 
	integer *lda, real *w, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    real eps;
    integer inde;
    real anrm, rmin, rmax;
    integer lopt;
    real sigma;
    extern logical lsame_(char *, char *);
    integer iinfo;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    integer lwmin, liopt;
    logical lower, wantz;
    integer indwk2, llwrk2, iscale;
    extern doublereal slamch_(char *);
    real safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    real bignum;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *, 
	    real *, integer *, integer *, real *, integer *, integer *);
    integer indtau;
    extern /* Subroutine */ int sstedc_(char *, integer *, real *, real *, 
	    real *, integer *, real *, integer *, integer *, integer *, 
	    integer *), slacpy_(char *, integer *, integer *, real *, 
	    integer *, real *, integer *);
    integer indwrk, liwmin;
    extern /* Subroutine */ int ssterf_(integer *, real *, real *, integer *);
    extern doublereal slansy_(char *, char *, integer *, real *, integer *, 
	    real *);
    integer llwork;
    real smlnum;
    logical lquery;
    extern /* Subroutine */ int sormtr_(char *, char *, char *, integer *, 
	    integer *, real *, integer *, real *, real *, integer *, real *, 
	    integer *, integer *), ssytrd_(char *, 
	    integer *, real *, integer *, real *, real *, real *, real *, 
	    integer *, integer *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SSYEVD computes all eigenvalues and, optionally, eigenvectors of a */
/*  real symmetric matrix A. If eigenvectors are desired, it uses a */
/*  divide and conquer algorithm. */

/*  The divide and conquer algorithm makes very mild assumptions about */
/*  floating point arithmetic. It will work on machines with a guard */
/*  digit in add/subtract, or on those binary machines without guard */
/*  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/*  Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Because of large use of blas of level 3, SSYEVD needs N**2 more */
/*  workspace than SSYEVX. */

/*  Arguments */
/*  ========= */

/*  JOBZ    (input) CHARACTER*1 */
/*          = 'N':  Compute eigenvalues only; */
/*          = 'V':  Compute eigenvalues and eigenvectors. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  A       (input/output) REAL array, dimension (LDA, N) */
/*          On entry, the symmetric matrix A.  If UPLO = 'U', the */
/*          leading N-by-N upper triangular part of A contains the */
/*          upper triangular part of the matrix A.  If UPLO = 'L', */
/*          the leading N-by-N lower triangular part of A contains */
/*          the lower triangular part of the matrix A. */
/*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
/*          orthonormal eigenvectors of the matrix A. */
/*          If JOBZ = 'N', then on exit the lower triangle (if UPLO='L') */
/*          or the upper triangle (if UPLO='U') of A, including the */
/*          diagonal, is destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  W       (output) REAL array, dimension (N) */
/*          If INFO = 0, the eigenvalues in ascending order. */

/*  WORK    (workspace/output) REAL array, */
/*                                         dimension (LWORK) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. */
/*          If N <= 1,               LWORK must be at least 1. */
/*          If JOBZ = 'N' and N > 1, LWORK must be at least 2*N+1. */
/*          If JOBZ = 'V' and N > 1, LWORK must be at least */
/*                                                1 + 6*N + 2*N**2. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal sizes of the WORK and IWORK */
/*          arrays, returns these values as the first entries of the WORK */
/*          and IWORK arrays, and no error message related to LWORK or */
/*          LIWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of the array IWORK. */
/*          If N <= 1,                LIWORK must be at least 1. */
/*          If JOBZ  = 'N' and N > 1, LIWORK must be at least 1. */
/*          If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N. */

/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal sizes of the WORK and */
/*          IWORK arrays, returns these values as the first entries of */
/*          the WORK and IWORK arrays, and no error message related to */
/*          LWORK or LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i and JOBZ = 'N', then the algorithm failed */
/*                to converge; i off-diagonal elements of an intermediate */
/*                tridiagonal form did not converge to zero; */
/*                if INFO = i and JOBZ = 'V', then the algorithm failed */
/*                to compute an eigenvalue while working on the submatrix */
/*                lying in rows and columns INFO/(N+1) through */
/*                mod(INFO,N+1). */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Jeff Rutter, Computer Science Division, University of California */
/*     at Berkeley, USA */
/*  Modified by Francoise Tisseur, University of Tennessee. */

/*  Modified description of INFO. Sven, 16 Feb 05. */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */

/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --work;
    --iwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    lower = lsame_(uplo, "L");
    lquery = *lwork == -1 || *liwork == -1;

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (lower || lsame_(uplo, "U"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }

    if (*info == 0) {
	if (*n <= 1) {
	    liwmin = 1;
	    lwmin = 1;
	    lopt = lwmin;
	    liopt = liwmin;
	} else {
	    if (wantz) {
		liwmin = *n * 5 + 3;
/* Computing 2nd power */
		i__1 = *n;
		lwmin = *n * 6 + 1 + (i__1 * i__1 << 1);
	    } else {
		liwmin = 1;
		lwmin = (*n << 1) + 1;
	    }
/* Computing MAX */
	    i__1 = lwmin, i__2 = (*n << 1) + ilaenv_(&c__1, "SSYTRD", uplo, n, 
		     &c_n1, &c_n1, &c_n1);
	    lopt = max(i__1,i__2);
	    liopt = liwmin;
	}
	work[1] = (real) lopt;
	iwork[1] = liopt;

	if (*lwork < lwmin && ! lquery) {
	    *info = -8;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -10;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYEVD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	w[1] = a[a_dim1 + 1];
	if (wantz) {
	    a[a_dim1 + 1] = 1.f;
	}
	return 0;
    }

/*     Get machine constants. */

    safmin = slamch_("Safe minimum");
    eps = slamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1.f / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

/*     Scale matrix to allowable range, if necessary. */

    anrm = slansy_("M", uplo, n, &a[a_offset], lda, &work[1]);
    iscale = 0;
    if (anrm > 0.f && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	slascl_(uplo, &c__0, &c__0, &c_b17, &sigma, n, n, &a[a_offset], lda, 
		info);
    }

/*     Call SSYTRD to reduce symmetric matrix to tridiagonal form. */

    inde = 1;
    indtau = inde + *n;
    indwrk = indtau + *n;
    llwork = *lwork - indwrk + 1;
    indwk2 = indwrk + *n * *n;
    llwrk2 = *lwork - indwk2 + 1;

    ssytrd_(uplo, n, &a[a_offset], lda, &w[1], &work[inde], &work[indtau], &
	    work[indwrk], &llwork, &iinfo);

/*     For eigenvalues only, call SSTERF.  For eigenvectors, first call */
/*     SSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the */
/*     tridiagonal matrix, then call SORMTR to multiply it by the */
/*     Householder transformations stored in A. */

    if (! wantz) {
	ssterf_(n, &w[1], &work[inde], info);
    } else {
	sstedc_("I", n, &w[1], &work[inde], &work[indwrk], n, &work[indwk2], &
		llwrk2, &iwork[1], liwork, info);
	sormtr_("L", uplo, "N", n, n, &a[a_offset], lda, &work[indtau], &work[
		indwrk], n, &work[indwk2], &llwrk2, &iinfo);
	slacpy_("A", n, n, &work[indwrk], n, &a[a_offset], lda);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
	r__1 = 1.f / sigma;
	sscal_(n, &r__1, &w[1], &c__1);
    }

    work[1] = (real) lopt;
    iwork[1] = liopt;

    return 0;

/*     End of SSYEVD */

} /* ssyevd_ */