/* HW3CRT.f -- translated by f2c (version 20060506).
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

/* Subroutine */ int hw3crt_(doublereal *xs, doublereal *xf, integer *l, integer *lbdcnd, 
	doublereal *bdxs, doublereal *bdxf, doublereal *ys, doublereal *yf, integer *m, integer *
	mbdcnd, doublereal *bdys, doublereal *bdyf, doublereal *zs, doublereal *zf, integer *n, 
	integer *nbdcnd, doublereal *bdzs, doublereal *bdzf, doublereal *elmbda, integer *ldimf,
	 integer *mdimf, doublereal *f, doublereal *pertrb, integer *ierror, doublereal *w)
{
    /* System generated locals */
    integer bdxs_dim1, bdxs_offset, bdxf_dim1, bdxf_offset, bdys_dim1, 
	    bdys_offset, bdyf_dim1, bdyf_offset, bdzs_dim1, bdzs_offset, 
	    bdzf_dim1, bdzf_offset, f_dim1, f_dim2, f_offset, i__1, i__2, 
	    i__3;
    doublereal r__1;

    /* Local variables */
    static integer i__, j, k;
    static doublereal s, c1, c2, c3, s1, s2;
    static integer ir;
    static doublereal dx, dy;
    static integer mp;
    static doublereal dz;
    static integer np, lp, lp1, mp1, np1, iwb, iwc;
    static doublereal xlp, ylp, zlp;
    static integer iww, lunk, munk, nunk, lstop, mstop, nstop;
    extern /* Subroutine */ int pois3d_(integer *, integer *, doublereal *, integer 
	    *, integer *, doublereal *, integer *, integer *, doublereal *, doublereal *, doublereal *
	    , integer *, integer *, doublereal *, integer *, doublereal *);
    static integer lstpm1, mstpm1, nstpm1, nperod, lstart, mstart, nstart;
    static doublereal twbydx, twbydy, twbydz;

/* ***BEGIN PROLOGUE  HW3CRT */
/* ***PURPOSE  Solve the standard seven-point finite difference */
/*            approximation to the Helmholtz equation in Cartesian */
/*            coordinates. */
/* ***LIBRARY   SLATEC (FISHPACK) */
/* ***CATEGORY  I2B1A1A */
/* ***TYPE      SINGLE PRECISION (HW3CRT-S) */
/* ***KEYWORDS  CARTESIAN, ELLIPTIC, FISHPACK, HELMHOLTZ, PDE */
/* ***AUTHOR  ADAMS, J., (NCAR) */
/*           SWARZTRAUBER, P., (NCAR) */
/*           SWEET, R., (NCAR) */
/* ***DESCRIPTION */

/*     Subroutine HW3CRT solves the standard seven-point finite */
/*     difference approximation to the Helmholtz equation in Cartesian */
/*     coordinates: */

/*         (d/dX)(dU/dX) + (d/dY)(dU/dY) + (d/dZ)(dU/dZ) */

/*                    + LAMBDA*U = F(X,Y,Z) . */

/*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


/*    * * * * * * * *    Parameter Description     * * * * * * * * * * */


/*            * * * * * *   On Input    * * * * * * */

/*     XS,XF */
/*        The range of X, i.e. XS .LE. X .LE. XF . */
/*        XS must be less than XF. */

/*     L */
/*        The number of panels into which the interval (XS,XF) is */
/*        subdivided.  Hence, there will be L+1 grid points in the */
/*        X-direction given by X(I) = XS+(I-1)DX for I=1,2,...,L+1, */
/*        where DX = (XF-XS)/L is the panel width.  L must be at */
/*        least 5 . */

/*     LBDCND */
/*        Indicates the type of boundary conditions at X = XS and X = XF. */

/*        = 0  If the solution is periodic in X, i.e. */
/*             U(L+I,J,K) = U(I,J,K). */
/*        = 1  If the solution is specified at X = XS and X = XF. */
/*        = 2  If the solution is specified at X = XS and the derivative */
/*             of the solution with respect to X is specified at X = XF. */
/*        = 3  If the derivative of the solution with respect to X is */
/*             specified at X = XS and X = XF. */
/*        = 4  If the derivative of the solution with respect to X is */
/*             specified at X = XS and the solution is specified at X=XF. */

/*     BDXS */
/*        A two-dimensional array that specifies the values of the */
/*        derivative of the solution with respect to X at X = XS. */
/*        when LBDCND = 3 or 4, */

/*             BDXS(J,K) = (d/dX)U(XS,Y(J),Z(K)), J=1,2,...,M+1, */
/*                                                K=1,2,...,N+1. */

/*        When LBDCND has any other value, BDXS is a dummy variable. */
/*        BDXS must be dimensioned at least (M+1)*(N+1). */

/*     BDXF */
/*        A two-dimensional array that specifies the values of the */
/*        derivative of the solution with respect to X at X = XF. */
/*        When LBDCND = 2 or 3, */

/*             BDXF(J,K) = (d/dX)U(XF,Y(J),Z(K)), J=1,2,...,M+1, */
/*                                                K=1,2,...,N+1. */

/*        When LBDCND has any other value, BDXF is a dummy variable. */
/*        BDXF must be dimensioned at least (M+1)*(N+1). */

/*     YS,YF */
/*        The range of Y, i.e. YS .LE. Y .LE. YF. */
/*        YS must be less than YF. */

/*     M */
/*        The number of panels into which the interval (YS,YF) is */
/*        subdivided.  Hence, there will be M+1 grid points in the */
/*        Y-direction given by Y(J) = YS+(J-1)DY for J=1,2,...,M+1, */
/*        where DY = (YF-YS)/M is the panel width.  M must be at */
/*        least 5 . */

/*     MBDCND */
/*        Indicates the type of boundary conditions at Y = YS and Y = YF. */

/*        = 0  If the solution is periodic in Y, i.e. */
/*             U(I,M+J,K) = U(I,J,K). */
/*        = 1  If the solution is specified at Y = YS and Y = YF. */
/*        = 2  If the solution is specified at Y = YS and the derivative */
/*             of the solution with respect to Y is specified at Y = YF. */
/*        = 3  If the derivative of the solution with respect to Y is */
/*             specified at Y = YS and Y = YF. */
/*        = 4  If the derivative of the solution with respect to Y is */
/*             specified at Y = YS and the solution is specified at Y=YF. */

/*     BDYS */
/*        A two-dimensional array that specifies the values of the */
/*        derivative of the solution with respect to Y at Y = YS. */
/*        When MBDCND = 3 or 4, */

/*             BDYS(I,K) = (d/dY)U(X(I),YS,Z(K)), I=1,2,...,L+1, */
/*                                                K=1,2,...,N+1. */

/*        When MBDCND has any other value, BDYS is a dummy variable. */
/*        BDYS must be dimensioned at least (L+1)*(N+1). */

/*     BDYF */
/*        A two-dimensional array that specifies the values of the */
/*        derivative of the solution with respect to Y at Y = YF. */
/*        When MBDCND = 2 or 3, */

/*             BDYF(I,K) = (d/dY)U(X(I),YF,Z(K)), I=1,2,...,L+1, */
/*                                                K=1,2,...,N+1. */

/*        When MBDCND has any other value, BDYF is a dummy variable. */
/*        BDYF must be dimensioned at least (L+1)*(N+1). */

/*     ZS,ZF */
/*        The range of Z, i.e. ZS .LE. Z .LE. ZF. */
/*        ZS must be less than ZF. */

/*     N */
/*        The number of panels into which the interval (ZS,ZF) is */
/*        subdivided.  Hence, there will be N+1 grid points in the */
/*        Z-direction given by Z(K) = ZS+(K-1)DZ for K=1,2,...,N+1, */
/*        where DZ = (ZF-ZS)/N is the panel width.  N must be at least 5. */

/*     NBDCND */
/*        Indicates the type of boundary conditions at Z = ZS and Z = ZF. */

/*        = 0  If the solution is periodic in Z, i.e. */
/*             U(I,J,N+K) = U(I,J,K). */
/*        = 1  If the solution is specified at Z = ZS and Z = ZF. */
/*        = 2  If the solution is specified at Z = ZS and the derivative */
/*             of the solution with respect to Z is specified at Z = ZF. */
/*        = 3  If the derivative of the solution with respect to Z is */
/*             specified at Z = ZS and Z = ZF. */
/*        = 4  If the derivative of the solution with respect to Z is */
/*             specified at Z = ZS and the solution is specified at Z=ZF. */

/*     BDZS */
/*        A two-dimensional array that specifies the values of the */
/*        derivative of the solution with respect to Z at Z = ZS. */
/*        When NBDCND = 3 or 4, */

/*             BDZS(I,J) = (d/dZ)U(X(I),Y(J),ZS), I=1,2,...,L+1, */
/*                                                J=1,2,...,M+1. */

/*        When NBDCND has any other value, BDZS is a dummy variable. */
/*        BDZS must be dimensioned at least (L+1)*(M+1). */

/*     BDZF */
/*        A two-dimensional array that specifies the values of the */
/*        derivative of the solution with respect to Z at Z = ZF. */
/*        When NBDCND = 2 or 3, */

/*             BDZF(I,J) = (d/dZ)U(X(I),Y(J),ZF), I=1,2,...,L+1, */
/*                                                J=1,2,...,M+1. */

/*        When NBDCND has any other value, BDZF is a dummy variable. */
/*        BDZF must be dimensioned at least (L+1)*(M+1). */

/*     ELMBDA */
/*        The constant LAMBDA in the Helmholtz equation. If */
/*        LAMBDA .GT. 0, a solution may not exist.  However, HW3CRT will */
/*        attempt to find a solution. */

/*     F */
/*        A three-dimensional array that specifies the values of the */
/*        right side of the Helmholtz equation and boundary values (if */
/*        any).  For I=2,3,...,L, J=2,3,...,M, and K=2,3,...,N */

/*                   F(I,J,K) = F(X(I),Y(J),Z(K)). */

/*        On the boundaries F is defined by */

/*        LBDCND      F(1,J,K)         F(L+1,J,K) */
/*        ------   ---------------   --------------- */

/*          0      F(XS,Y(J),Z(K))   F(XS,Y(J),Z(K)) */
/*          1      U(XS,Y(J),Z(K))   U(XF,Y(J),Z(K)) */
/*          2      U(XS,Y(J),Z(K))   F(XF,Y(J),Z(K))   J=1,2,...,M+1 */
/*          3      F(XS,Y(J),Z(K))   F(XF,Y(J),Z(K))   K=1,2,...,N+1 */
/*          4      F(XS,Y(J),Z(K))   U(XF,Y(J),Z(K)) */

/*        MBDCND      F(I,1,K)         F(I,M+1,K) */
/*        ------   ---------------   --------------- */

/*          0      F(X(I),YS,Z(K))   F(X(I),YS,Z(K)) */
/*          1      U(X(I),YS,Z(K))   U(X(I),YF,Z(K)) */
/*          2      U(X(I),YS,Z(K))   F(X(I),YF,Z(K))   I=1,2,...,L+1 */
/*          3      F(X(I),YS,Z(K))   F(X(I),YF,Z(K))   K=1,2,...,N+1 */
/*          4      F(X(I),YS,Z(K))   U(X(I),YF,Z(K)) */

/*        NBDCND      F(I,J,1)         F(I,J,N+1) */
/*        ------   ---------------   --------------- */

/*          0      F(X(I),Y(J),ZS)   F(X(I),Y(J),ZS) */
/*          1      U(X(I),Y(J),ZS)   U(X(I),Y(J),ZF) */
/*          2      U(X(I),Y(J),ZS)   F(X(I),Y(J),ZF)   I=1,2,...,L+1 */
/*          3      F(X(I),Y(J),ZS)   F(X(I),Y(J),ZF)   J=1,2,...,M+1 */
/*          4      F(X(I),Y(J),ZS)   U(X(I),Y(J),ZF) */

/*        F must be dimensioned at least (L+1)*(M+1)*(N+1). */

/*        NOTE: */

/*        If the table calls for both the solution U and the right side F */
/*        on a boundary, then the solution must be specified. */

/*     LDIMF */
/*        The row (or first) dimension of the arrays F,BDYS,BDYF,BDZS, */
/*        and BDZF as it appears in the program calling HW3CRT. this */
/*        parameter is used to specify the variable dimension of these */
/*        arrays.  LDIMF must be at least L+1. */

/*     MDIMF */
/*        The column (or second) dimension of the array F and the row (or */
/*        first) dimension of the arrays BDXS and BDXF as it appears in */
/*        the program calling HW3CRT.  This parameter is used to specify */
/*        the variable dimension of these arrays. */
/*        MDIMF must be at least M+1. */

/*     W */
/*        A one-dimensional array that must be provided by the user for */
/*        work space.  The length of W must be at least 30 + L + M + 5*N */
/*        + MAX(L,M,N) + 7*(INT((L+1)/2) + INT((M+1)/2)) */


/*            * * * * * *   On Output   * * * * * * */

/*     F */
/*        Contains the solution U(I,J,K) of the finite difference */
/*        approximation for the grid point (X(I),Y(J),Z(K)) for */
/*        I=1,2,...,L+1, J=1,2,...,M+1, and K=1,2,...,N+1. */

/*     PERTRB */
/*        If a combination of periodic or derivative boundary conditions */
/*        is specified for a Poisson equation (LAMBDA = 0), a solution */
/*        may not exist.  PERTRB is a constant, calculated and subtracted */
/*        from F, which ensures that a solution exists.  pwscrt then */
/*        computes this solution, which is a least squares solution to */
/*        the original approximation.  This solution is not unique and is */
/*        unnormalized.  The value of PERTRB should be small compared to */
/*        the right side F.  Otherwise, a solution is obtained to an */
/*        essentially different problem.  This comparison should always */
/*        be made to insure that a meaningful solution has been obtained. */

/*     IERROR */
/*        An error flag that indicates invalid input parameters.  Except */
/*        for numbers 0 and 12, a solution is not attempted. */

/*        =  0  No error */
/*        =  1  XS .GE. XF */
/*        =  2  L .LT. 5 */
/*        =  3  LBDCND .LT. 0 .OR. LBDCND .GT. 4 */
/*        =  4  YS .GE. YF */
/*        =  5  M .LT. 5 */
/*        =  6  MBDCND .LT. 0 .OR. MBDCND .GT. 4 */
/*        =  7  ZS .GE. ZF */
/*        =  8  N .LT. 5 */
/*        =  9  NBDCND .LT. 0 .OR. NBDCND .GT. 4 */
/*        = 10  LDIMF .LT. L+1 */
/*        = 11  MDIMF .LT. M+1 */
/*        = 12  LAMBDA .GT. 0 */

/*        Since this is the only means of indicating a possibly incorrect */
/*        call to HW3CRT, the user should test IERROR after the call. */
/* *Long Description: */

/*    * * * * * * *   Program Specifications    * * * * * * * * * * * * */

/*     Dimension of   BDXS(MDIMF,N+1),BDXF(MDIMF,N+1),BDYS(LDIMF,N+1), */
/*     Arguments      BDYF(LDIMF,N+1),BDZS(LDIMF,M+1),BDZF(LDIMF,M+1), */
/*                    F(LDIMF,MDIMF,N+1),W(see argument list) */

/*     Latest         December 1, 1978 */
/*     Revision */

/*     Subprograms    HW3CRT,POIS3D,POS3D1,TRIDQ,RFFTI,RFFTF,RFFTF1, */
/*     Required       RFFTB,RFFTB1,COSTI,COST,SINTI,SINT,COSQI,COSQF, */
/*                    COSQF1,COSQB,COSQB1,SINQI,SINQF,SINQB,CFFTI, */
/*                    CFFTI1,CFFTB,CFFTB1,PASSB2,PASSB3,PASSB4,PASSB, */
/*                    CFFTF,CFFTF1,PASSF1,PASSF2,PASSF3,PASSF4,PASSF, */
/*                    PIMACH */

/*     Special        NONE */
/*     Conditions */

/*     Common         NONE */
/*     Blocks */

/*     I/O            NONE */

/*     Precision      Single */

/*     Specialist     Roland Sweet */

/*     Language       FORTRAN */

/*     History        Written by Roland Sweet at NCAR in July,1977 */

/*     Algorithm      This subroutine defines the finite difference */
/*                    equations, incorporates boundary data, and */
/*                    adjusts the right side of singular systems and */
/*                    then calls POIS3D to solve the system. */

/*     Space          7862(decimal) = 17300(octal) locations on the */
/*     Required       NCAR Control Data 7600 */

/*     Timing and        The execution time T on the NCAR Control Data */
/*     Accuracy       7600 for subroutine HW3CRT is roughly proportional */
/*                    to L*M*N*(log2(L)+log2(M)+5), but also depends on */
/*                    input parameters LBDCND and MBDCND.  Some typical */
/*                    values are listed in the table below. */
/*                       The solution process employed results in a loss */
/*                    of no more than three significant digits for L,M */
/*                    and N as large as 32.  More detailed information */
/*                    about accuracy can be found in the documentation */
/*                    for subroutine POIS3D which is the routine that */
/*                    actually solves the finite difference equations. */


/*                       L(=M=N)     LBDCND(=MBDCND=NBDCND)      T(MSECS) */
/*                       -------     ----------------------      -------- */

/*                         16                  0                    300 */
/*                         16                  1                    302 */
/*                         16                  3                    348 */
/*                         32                  0                   1925 */
/*                         32                  1                   1929 */
/*                         32                  3                   2109 */

/*     Portability    American National Standards Institute FORTRAN. */
/*                    The machine dependent constant PI is defined in */
/*                    function PIMACH. */

/*     Required       COS,SIN,ATAN */
/*     Resident */
/*     Routines */

/*     Reference      NONE */

/*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  POIS3D */
/* ***REVISION HISTORY  (YYMMDD) */
/*   801001  DATE WRITTEN */
/*   890531  Changed all specific intrinsics to generic.  (WRB) */
/*   890531  REVISION DATE from Version 3.2 */
/*   891214  Prologue converted to Version 4.0 format.  (BAB) */
/* ***END PROLOGUE  HW3CRT */


/* ***FIRST EXECUTABLE STATEMENT  HW3CRT */
    /* Parameter adjustments */
    bdzf_dim1 = *ldimf;
    bdzf_offset = 1 + bdzf_dim1;
    bdzf -= bdzf_offset;
    bdzs_dim1 = *ldimf;
    bdzs_offset = 1 + bdzs_dim1;
    bdzs -= bdzs_offset;
    bdyf_dim1 = *ldimf;
    bdyf_offset = 1 + bdyf_dim1;
    bdyf -= bdyf_offset;
    bdys_dim1 = *ldimf;
    bdys_offset = 1 + bdys_dim1;
    bdys -= bdys_offset;
    f_dim1 = *ldimf;
    f_dim2 = *mdimf;
    f_offset = 1 + f_dim1 * (1 + f_dim2);
    f -= f_offset;
    bdxf_dim1 = *mdimf;
    bdxf_offset = 1 + bdxf_dim1;
    bdxf -= bdxf_offset;
    bdxs_dim1 = *mdimf;
    bdxs_offset = 1 + bdxs_dim1;
    bdxs -= bdxs_offset;
    --w;

    /* Function Body */
    *ierror = 0;
    if (*xf <= *xs) {
	*ierror = 1;
    }
    if (*l < 5) {
	*ierror = 2;
    }
    if (*lbdcnd < 0 || *lbdcnd > 4) {
	*ierror = 3;
    }
    if (*yf <= *ys) {
	*ierror = 4;
    }
    if (*m < 5) {
	*ierror = 5;
    }
    if (*mbdcnd < 0 || *mbdcnd > 4) {
	*ierror = 6;
    }
    if (*zf <= *zs) {
	*ierror = 7;
    }
    if (*n < 5) {
	*ierror = 8;
    }
    if (*nbdcnd < 0 || *nbdcnd > 4) {
	*ierror = 9;
    }
    if (*ldimf < *l + 1) {
	*ierror = 10;
    }
    if (*mdimf < *m + 1) {
	*ierror = 11;
    }
    if (*ierror != 0) {
	goto L188;
    }
    dy = (*yf - *ys) / *m;
    twbydy = 2.f / dy;
/* Computing 2nd power */
    r__1 = dy;
    c2 = 1.f / (r__1 * r__1);
    mstart = 1;
    mstop = *m;
    mp1 = *m + 1;
    mp = *mbdcnd + 1;
    switch (mp) {
	case 1:  goto L104;
	case 2:  goto L101;
	case 3:  goto L101;
	case 4:  goto L102;
	case 5:  goto L102;
    }
L101:
    mstart = 2;
L102:
    switch (mp) {
	case 1:  goto L104;
	case 2:  goto L104;
	case 3:  goto L103;
	case 4:  goto L103;
	case 5:  goto L104;
    }
L103:
    mstop = mp1;
L104:
    munk = mstop - mstart + 1;
    dz = (*zf - *zs) / *n;
    twbydz = 2.f / dz;
    np = *nbdcnd + 1;
/* Computing 2nd power */
    r__1 = dz;
    c3 = 1.f / (r__1 * r__1);
    np1 = *n + 1;
    nstart = 1;
    nstop = *n;
    switch (np) {
	case 1:  goto L108;
	case 2:  goto L105;
	case 3:  goto L105;
	case 4:  goto L106;
	case 5:  goto L106;
    }
L105:
    nstart = 2;
L106:
    switch (np) {
	case 1:  goto L108;
	case 2:  goto L108;
	case 3:  goto L107;
	case 4:  goto L107;
	case 5:  goto L108;
    }
L107:
    nstop = np1;
L108:
    nunk = nstop - nstart + 1;
    lp1 = *l + 1;
    dx = (*xf - *xs) / *l;
/* Computing 2nd power */
    r__1 = dx;
    c1 = 1.f / (r__1 * r__1);
    twbydx = 2.f / dx;
    lp = *lbdcnd + 1;
    lstart = 1;
    lstop = *l;

/*     ENTER BOUNDARY DATA FOR X-BOUNDARIES. */

    switch (lp) {
	case 1:  goto L122;
	case 2:  goto L109;
	case 3:  goto L109;
	case 4:  goto L112;
	case 5:  goto L112;
    }
L109:
    lstart = 2;
    i__1 = mstop;
    for (j = mstart; j <= i__1; ++j) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[(j + k * f_dim2) * f_dim1 + 2] -= c1 * f[(j + k * f_dim2) * 
		    f_dim1 + 1];
/* L110: */
	}
/* L111: */
    }
    goto L115;
L112:
    i__1 = mstop;
    for (j = mstart; j <= i__1; ++j) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[(j + k * f_dim2) * f_dim1 + 1] += twbydx * bdxs[j + k * 
		    bdxs_dim1];
/* L113: */
	}
/* L114: */
    }
L115:
    switch (lp) {
	case 1:  goto L122;
	case 2:  goto L116;
	case 3:  goto L119;
	case 4:  goto L119;
	case 5:  goto L116;
    }
L116:
    i__1 = mstop;
    for (j = mstart; j <= i__1; ++j) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[*l + (j + k * f_dim2) * f_dim1] -= c1 * f[lp1 + (j + k * f_dim2)
		     * f_dim1];
/* L117: */
	}
/* L118: */
    }
    goto L122;
L119:
    lstop = lp1;
    i__1 = mstop;
    for (j = mstart; j <= i__1; ++j) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[lp1 + (j + k * f_dim2) * f_dim1] -= twbydx * bdxf[j + k * 
		    bdxf_dim1];
/* L120: */
	}
/* L121: */
    }
L122:
    lunk = lstop - lstart + 1;

/*     ENTER BOUNDARY DATA FOR Y-BOUNDARIES. */

    switch (mp) {
	case 1:  goto L136;
	case 2:  goto L123;
	case 3:  goto L123;
	case 4:  goto L126;
	case 5:  goto L126;
    }
L123:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[i__ + (k * f_dim2 + 2) * f_dim1] -= c2 * f[i__ + (k * f_dim2 + 
		    1) * f_dim1];
/* L124: */
	}
/* L125: */
    }
    goto L129;
L126:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[i__ + (k * f_dim2 + 1) * f_dim1] += twbydy * bdys[i__ + k * 
		    bdys_dim1];
/* L127: */
	}
/* L128: */
    }
L129:
    switch (mp) {
	case 1:  goto L136;
	case 2:  goto L130;
	case 3:  goto L133;
	case 4:  goto L133;
	case 5:  goto L130;
    }
L130:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[i__ + (*m + k * f_dim2) * f_dim1] -= c2 * f[i__ + (mp1 + k * 
		    f_dim2) * f_dim1];
/* L131: */
	}
/* L132: */
    }
    goto L136;
L133:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[i__ + (mp1 + k * f_dim2) * f_dim1] -= twbydy * bdyf[i__ + k * 
		    bdyf_dim1];
/* L134: */
	}
/* L135: */
    }
L136:

/*     ENTER BOUNDARY DATA FOR Z-BOUNDARIES. */

    switch (np) {
	case 1:  goto L150;
	case 2:  goto L137;
	case 3:  goto L137;
	case 4:  goto L140;
	case 5:  goto L140;
    }
L137:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = mstop;
	for (j = mstart; j <= i__2; ++j) {
	    f[i__ + (j + (f_dim2 << 1)) * f_dim1] -= c3 * f[i__ + (j + f_dim2)
		     * f_dim1];
/* L138: */
	}
/* L139: */
    }
    goto L143;
L140:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = mstop;
	for (j = mstart; j <= i__2; ++j) {
	    f[i__ + (j + f_dim2) * f_dim1] += twbydz * bdzs[i__ + j * 
		    bdzs_dim1];
/* L141: */
	}
/* L142: */
    }
L143:
    switch (np) {
	case 1:  goto L150;
	case 2:  goto L144;
	case 3:  goto L147;
	case 4:  goto L147;
	case 5:  goto L144;
    }
L144:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = mstop;
	for (j = mstart; j <= i__2; ++j) {
	    f[i__ + (j + *n * f_dim2) * f_dim1] -= c3 * f[i__ + (j + np1 * 
		    f_dim2) * f_dim1];
/* L145: */
	}
/* L146: */
    }
    goto L150;
L147:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = mstop;
	for (j = mstart; j <= i__2; ++j) {
	    f[i__ + (j + np1 * f_dim2) * f_dim1] -= twbydz * bdzf[i__ + j * 
		    bdzf_dim1];
/* L148: */
	}
/* L149: */
    }

/*     DEFINE A,B,C COEFFICIENTS IN W-ARRAY. */

L150:
    iwb = nunk + 1;
    iwc = iwb + nunk;
    iww = iwc + nunk;
    i__1 = nunk;
    for (k = 1; k <= i__1; ++k) {
	i__ = iwc + k - 1;
	w[k] = c3;
	w[i__] = c3;
	i__ = iwb + k - 1;
	w[i__] = c3 * -2.f + *elmbda;
/* L151: */
    }
    switch (np) {
	case 1:  goto L155;
	case 2:  goto L155;
	case 3:  goto L153;
	case 4:  goto L152;
	case 5:  goto L152;
    }
L152:
    w[iwc] = c3 * 2.f;
L153:
    switch (np) {
	case 1:  goto L155;
	case 2:  goto L155;
	case 3:  goto L154;
	case 4:  goto L154;
	case 5:  goto L155;
    }
L154:
    w[iwb - 1] = c3 * 2.f;
L155:
    *pertrb = 0.f;

/*     FOR SINGULAR PROBLEMS ADJUST DATA TO INSURE A SOLUTION WILL EXIST. */

    switch (lp) {
	case 1:  goto L156;
	case 2:  goto L172;
	case 3:  goto L172;
	case 4:  goto L156;
	case 5:  goto L172;
    }
L156:
    switch (mp) {
	case 1:  goto L157;
	case 2:  goto L172;
	case 3:  goto L172;
	case 4:  goto L157;
	case 5:  goto L172;
    }
L157:
    switch (np) {
	case 1:  goto L158;
	case 2:  goto L172;
	case 3:  goto L172;
	case 4:  goto L158;
	case 5:  goto L172;
    }
L158:
    if (*elmbda < 0.f) {
	goto L172;
    } else if (*elmbda == 0) {
	goto L160;
    } else {
	goto L159;
    }
L159:
    *ierror = 12;
    goto L172;
L160:
    mstpm1 = mstop - 1;
    lstpm1 = lstop - 1;
    nstpm1 = nstop - 1;
    xlp = (doublereal) ((lp + 2) / 3);
    ylp = (doublereal) ((mp + 2) / 3);
    zlp = (doublereal) ((np + 2) / 3);
    s1 = 0.f;
    i__1 = nstpm1;
    for (k = 2; k <= i__1; ++k) {
	i__2 = mstpm1;
	for (j = 2; j <= i__2; ++j) {
	    i__3 = lstpm1;
	    for (i__ = 2; i__ <= i__3; ++i__) {
		s1 += f[i__ + (j + k * f_dim2) * f_dim1];
/* L161: */
	    }
	    s1 += (f[(j + k * f_dim2) * f_dim1 + 1] + f[lstop + (j + k * 
		    f_dim2) * f_dim1]) / xlp;
/* L162: */
	}
	s2 = 0.f;
	i__2 = lstpm1;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    s2 = s2 + f[i__ + (k * f_dim2 + 1) * f_dim1] + f[i__ + (mstop + k 
		    * f_dim2) * f_dim1];
/* L163: */
	}
	s2 = (s2 + (f[(k * f_dim2 + 1) * f_dim1 + 1] + f[(mstop + k * f_dim2) 
		* f_dim1 + 1] + f[lstop + (k * f_dim2 + 1) * f_dim1] + f[
		lstop + (mstop + k * f_dim2) * f_dim1]) / xlp) / ylp;
	s1 += s2;
/* L164: */
    }
    s = (f[(f_dim2 + 1) * f_dim1 + 1] + f[lstop + (f_dim2 + 1) * f_dim1] + f[(
	    nstop * f_dim2 + 1) * f_dim1 + 1] + f[lstop + (nstop * f_dim2 + 1)
	     * f_dim1] + f[(mstop + f_dim2) * f_dim1 + 1] + f[lstop + (mstop 
	    + f_dim2) * f_dim1] + f[(mstop + nstop * f_dim2) * f_dim1 + 1] + 
	    f[lstop + (mstop + nstop * f_dim2) * f_dim1]) / (xlp * ylp);
    i__1 = mstpm1;
    for (j = 2; j <= i__1; ++j) {
	i__2 = lstpm1;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    s = s + f[i__ + (j + f_dim2) * f_dim1] + f[i__ + (j + nstop * 
		    f_dim2) * f_dim1];
/* L165: */
	}
/* L166: */
    }
    s2 = 0.f;
    i__1 = lstpm1;
    for (i__ = 2; i__ <= i__1; ++i__) {
	s2 = s2 + f[i__ + (f_dim2 + 1) * f_dim1] + f[i__ + (nstop * f_dim2 + 
		1) * f_dim1] + f[i__ + (mstop + f_dim2) * f_dim1] + f[i__ + (
		mstop + nstop * f_dim2) * f_dim1];
/* L167: */
    }
    s = s2 / ylp + s;
    s2 = 0.f;
    i__1 = mstpm1;
    for (j = 2; j <= i__1; ++j) {
	s2 = s2 + f[(j + f_dim2) * f_dim1 + 1] + f[(j + nstop * f_dim2) * 
		f_dim1 + 1] + f[lstop + (j + f_dim2) * f_dim1] + f[lstop + (j 
		+ nstop * f_dim2) * f_dim1];
/* L168: */
    }
    s = s2 / xlp + s;
    *pertrb = (s / zlp + s1) / ((lunk + 1.f - xlp) * (munk + 1.f - ylp) * (
	    nunk + 1.f - zlp));
    i__1 = lunk;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = munk;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = nunk;
	    for (k = 1; k <= i__3; ++k) {
		f[i__ + (j + k * f_dim2) * f_dim1] -= *pertrb;
/* L169: */
	    }
/* L170: */
	}
/* L171: */
    }
L172:
    nperod = 0;
    if (*nbdcnd == 0) {
	goto L173;
    }
    nperod = 1;
    w[1] = 0.f;
    w[iww - 1] = 0.f;
L173:
    pois3d_(lbdcnd, &lunk, &c1, mbdcnd, &munk, &c2, &nperod, &nunk, &w[1], &w[
	    iwb], &w[iwc], ldimf, mdimf, &f[lstart + (mstart + nstart * 
	    f_dim2) * f_dim1], &ir, &w[iww]);

/*     FILL IN SIDES FOR PERIODIC BOUNDARY CONDITIONS. */

    if (lp != 1) {
	goto L180;
    }
    if (mp != 1) {
	goto L175;
    }
    i__1 = nstop;
    for (k = nstart; k <= i__1; ++k) {
	f[(mp1 + k * f_dim2) * f_dim1 + 1] = f[(k * f_dim2 + 1) * f_dim1 + 1];
/* L174: */
    }
    mstop = mp1;
L175:
    if (np != 1) {
	goto L177;
    }
    i__1 = mstop;
    for (j = mstart; j <= i__1; ++j) {
	f[(j + np1 * f_dim2) * f_dim1 + 1] = f[(j + f_dim2) * f_dim1 + 1];
/* L176: */
    }
    nstop = np1;
L177:
    i__1 = mstop;
    for (j = mstart; j <= i__1; ++j) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[lp1 + (j + k * f_dim2) * f_dim1] = f[(j + k * f_dim2) * f_dim1 
		    + 1];
/* L178: */
	}
/* L179: */
    }
L180:
    if (mp != 1) {
	goto L185;
    }
    if (np != 1) {
	goto L182;
    }
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	f[i__ + (np1 * f_dim2 + 1) * f_dim1] = f[i__ + (f_dim2 + 1) * f_dim1];
/* L181: */
    }
    nstop = np1;
L182:
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = nstop;
	for (k = nstart; k <= i__2; ++k) {
	    f[i__ + (mp1 + k * f_dim2) * f_dim1] = f[i__ + (k * f_dim2 + 1) * 
		    f_dim1];
/* L183: */
	}
/* L184: */
    }
L185:
    if (np != 1) {
	goto L188;
    }
    i__1 = lstop;
    for (i__ = lstart; i__ <= i__1; ++i__) {
	i__2 = mstop;
	for (j = mstart; j <= i__2; ++j) {
	    f[i__ + (j + np1 * f_dim2) * f_dim1] = f[i__ + (j + f_dim2) * 
		    f_dim1];
/* L186: */
	}
/* L187: */
    }
L188:
    return 0;
} /* hw3crt_ */

/* Subroutine */ int pois3d_(integer *lperod, integer *l, doublereal *c1, integer *
	mperod, integer *m, doublereal *c2, integer *nperod, integer *n, doublereal *a, 
	doublereal *b, doublereal *c__, integer *ldimf, integer *mdimf, doublereal *f, integer *
	ierror, doublereal *w)
{
    /* System generated locals */
    integer f_dim1, f_dim2, f_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, k, nh, lp, mp, np, iwd, iwt, iwx, iwy, nhm1, iwbb, 
	    nodd, nhmk;
    static doublereal save[6];
    static integer nhpk;
    extern /* Subroutine */ int pos3d1_(integer *, integer *, integer *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, integer *, integer *
	    , doublereal *, doublereal *, doublereal *, doublereal *, doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *);
    static integer iwyrt;

/* ***BEGIN PROLOGUE  POIS3D */
/* ***PURPOSE  Solve a three-dimensional block tridiagonal linear system */
/*            which arises from a finite difference approximation to a */
/*            three-dimensional Poisson equation using the Fourier */
/*            transform package FFTPAK written by Paul Swarztrauber. */
/* ***LIBRARY   SLATEC (FISHPACK) */
/* ***CATEGORY  I2B4B */
/* ***TYPE      SINGLE PRECISION (POIS3D-S) */
/* ***KEYWORDS  ELLIPTIC PDE, FISHPACK, HELMHOLTZ, PDE, POISSON */
/* ***AUTHOR  ADAMS, J., (NCAR) */
/*           SWARZTRAUBER, P., (NCAR) */
/*           SWEET, R., (NCAR) */
/* ***DESCRIPTION */

/*     Subroutine POIS3D solves the linear system of equations */

/*       C1*(X(I-1,J,K)-2.*X(I,J,K)+X(I+1,J,K)) */
/*     + C2*(X(I,J-1,K)-2.*X(I,J,K)+X(I,J+1,K)) */
/*     + A(K)*X(I,J,K-1)+B(K)*X(I,J,K)+C(K)*X(I,J,K+1) = F(I,J,K) */

/*     for  I=1,2,...,L , J=1,2,...,M , and K=1,2,...,N . */

/*     The indices K-1 and K+1 are evaluated modulo N, i.e. */
/*     X(I,J,0) = X(I,J,N) and X(I,J,N+1) = X(I,J,1). The unknowns */
/*     X(0,J,K), X(L+1,J,K), X(I,0,K), and X(I,M+1,K) are assumed to take */
/*     on certain prescribed values described below. */

/*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


/*    * * * * * * * *    Parameter Description     * * * * * * * * * * */


/*            * * * * * *   On Input    * * * * * * */

/*     LPEROD   Indicates the values that X(0,J,K) and X(L+1,J,K) are */
/*              assumed to have. */

/*              = 0  If X(0,J,K) = X(L,J,K) and X(L+1,J,K) = X(1,J,K). */
/*              = 1  If X(0,J,K) = X(L+1,J,K) = 0. */
/*              = 2  If X(0,J,K) = 0  and X(L+1,J,K) = X(L-1,J,K). */
/*              = 3  If X(0,J,K) = X(2,J,K) and X(L+1,J,K) = X(L-1,J,K). */
/*              = 4  If X(0,J,K) = X(2,J,K) and X(L+1,J,K) = 0. */

/*     L        The number of unknowns in the I-direction. L must be at */
/*              least 3. */

/*     C1       The doublereal constant that appears in the above equation. */

/*     MPEROD   Indicates the values that X(I,0,K) and X(I,M+1,K) are */
/*              assumed to have. */

/*              = 0  If X(I,0,K) = X(I,M,K) and X(I,M+1,K) = X(I,1,K). */
/*              = 1  If X(I,0,K) = X(I,M+1,K) = 0. */
/*              = 2  If X(I,0,K) = 0 and X(I,M+1,K) = X(I,M-1,K). */
/*              = 3  If X(I,0,K) = X(I,2,K) and X(I,M+1,K) = X(I,M-1,K). */
/*              = 4  If X(I,0,K) = X(I,2,K) and X(I,M+1,K) = 0. */

/*     M        The number of unknowns in the J-direction. M must be at */
/*              least 3. */

/*     C2       The doublereal constant which appears in the above equation. */

/*     NPEROD   = 0  If A(1) and C(N) are not zero. */
/*              = 1  If A(1) = C(N) = 0. */

/*     N        The number of unknowns in the K-direction. N must be at */
/*              least 3. */


/*     A,B,C    One-dimensional arrays of length N that specify the */
/*              coefficients in the linear equations given above. */

/*              If NPEROD = 0 the array elements must not depend upon the */
/*              index K, but must be constant.  Specifically,the */
/*              subroutine checks the following condition */

/*                          A(K) = C(1) */
/*                          C(K) = C(1) */
/*                          B(K) = B(1) */

/*                  for K=1,2,...,N. */

/*     LDIMF    The row (or first) dimension of the three-dimensional */
/*              array F as it appears in the program calling POIS3D. */
/*              This parameter is used to specify the variable dimension */
/*              of F.  LDIMF must be at least L. */

/*     MDIMF    The column (or second) dimension of the three-dimensional */
/*              array F as it appears in the program calling POIS3D. */
/*              This parameter is used to specify the variable dimension */
/*              of F.  MDIMF must be at least M. */

/*     F        A three-dimensional array that specifies the values of */
/*              the right side of the linear system of equations given */
/*              above.  F must be dimensioned at least L x M x N. */

/*     W        A one-dimensional array that must be provided by the */
/*              user for work space.  The length of W must be at least */
/*              30 + L + M + 2*N + MAX(L,M,N) + */
/*              7*(INT((L+1)/2) + INT((M+1)/2)). */


/*            * * * * * *   On Output   * * * * * * */

/*     F        Contains the solution X. */

/*     IERROR   An error flag that indicates invalid input parameters. */
/*              Except for number zero, a solution is not attempted. */
/*              = 0  No error */
/*              = 1  If LPEROD .LT. 0 or .GT. 4 */
/*              = 2  If L .LT. 3 */
/*              = 3  If MPEROD .LT. 0 or .GT. 4 */
/*              = 4  If M .LT. 3 */
/*              = 5  If NPEROD .LT. 0 or .GT. 1 */
/*              = 6  If N .LT. 3 */
/*              = 7  If LDIMF .LT. L */
/*              = 8  If MDIMF .LT. M */
/*              = 9  If A(K) .NE. C(1) or C(K) .NE. C(1) or B(I) .NE.B(1) */
/*                      for some K=1,2,...,N. */
/*              = 10 If NPEROD = 1 and A(1) .NE. 0 or C(N) .NE. 0 */

/*              Since this is the only means of indicating a possibly */
/*              incorrect call to POIS3D, the user should test IERROR */
/*              after the call. */
/* *Long Description: */

/*    * * * * * * *   Program Specifications    * * * * * * * * * * * * */

/*     Dimension of   A(N),B(N),C(N),F(LDIMF,MDIMF,N), */
/*     Arguments      W(see argument list) */

/*     Latest         December 1, 1978 */
/*     Revision */

/*     Subprograms    POIS3D,POS3D1,TRIDQ,RFFTI,RFFTF,RFFTF1,RFFTB, */
/*     Required       RFFTB1,COSTI,COST,SINTI,SINT,COSQI,COSQF,COSQF1 */
/*                    COSQB,COSQB1,SINQI,SINQF,SINQB,CFFTI,CFFTI1, */
/*                    CFFTB,CFFTB1,PASSB2,PASSB3,PASSB4,PASSB,CFFTF, */
/*                    CFFTF1,PASSF1,PASSF2,PASSF3,PASSF4,PASSF,PIMACH, */

/*     Special        NONE */
/*     Conditions */

/*     Common         NONE */
/*     Blocks */

/*     I/O            NONE */

/*     Precision      Single */

/*     Specialist     Roland Sweet */

/*     Language       FORTRAN */

/*     History        Written by Roland Sweet at NCAR in July,1977 */

/*     Algorithm      This subroutine solves three-dimensional block */
/*                    tridiagonal linear systems arising from finite */
/*                    difference approximations to three-dimensional */
/*                    Poisson equations using the Fourier transform */
/*                    package SCLRFFTPAK written by Paul Swarztrauber. */

/*     Space          6561(decimal) = 14641(octal) locations on the */
/*     Required       NCAR Control Data 7600 */

/*     Timing and        The execution time T on the NCAR Control Data */
/*     Accuracy       7600 for subroutine POIS3D is roughly proportional */
/*                    to L*M*N*(log2(L)+log2(M)+5), but also depends on */
/*                    input parameters LPEROD and MPEROD.  Some typical */
/*                    values are listed in the table below when NPEROD=0. */
/*                       To measure the accuracy of the algorithm a */
/*                    uniform random number generator was used to create */
/*                    a solution array X for the system given in the */
/*                    'PURPOSE' with */

/*                       A(K) = C(K) = -0.5*B(K) = 1,       K=1,2,...,N */

/*                    and, when NPEROD = 1 */

/*                       A(1) = C(N) = 0 */
/*                       A(N) = C(1) = 2. */

/*                    The solution X was substituted into the given sys- */
/*                    tem and, using double precision, a right side Y was */
/*                    computed.  Using this array Y subroutine POIS3D was */
/*                    called to produce an approximate solution Z.  Then */
/*                    the relative error, defined as */

/*                    E = MAX(ABS(Z(I,J,K)-X(I,J,K)))/MAX(ABS(X(I,J,K))) */

/*                    where the two maxima are taken over I=1,2,...,L, */
/*                    J=1,2,...,M and K=1,2,...,N, was computed.  The */
/*                    value of E is given in the table below for some */
/*                    typical values of L,M and N. */


/*                       L(=M=N)   LPEROD    MPEROD    T(MSECS)    E */
/*                       ------    ------    ------    --------  ------ */

/*                         16        0         0         272     1.E-13 */
/*                         15        1         1         287     4.E-13 */
/*                         17        3         3         338     2.E-13 */
/*                         32        0         0        1755     2.E-13 */
/*                         31        1         1        1894     2.E-12 */
/*                         33        3         3        2042     7.E-13 */


/*     Portability    American National Standards Institute FORTRAN. */
/*                    The machine dependent constant PI is defined in */
/*                    function PIMACH. */

/*     Required       COS,SIN,ATAN */
/*     Resident */
/*     Routines */

/*     Reference      NONE */

/*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  POS3D1 */
/* ***REVISION HISTORY  (YYMMDD) */
/*   801001  DATE WRITTEN */
/*   890531  Changed all specific intrinsics to generic.  (WRB) */
/*   890531  REVISION DATE from Version 3.2 */
/*   891214  Prologue converted to Version 4.0 format.  (BAB) */
/* ***END PROLOGUE  POIS3D */


/* ***FIRST EXECUTABLE STATEMENT  POIS3D */
    /* Parameter adjustments */
    --a;
    --b;
    --c__;
    f_dim1 = *ldimf;
    f_dim2 = *mdimf;
    f_offset = 1 + f_dim1 * (1 + f_dim2);
    f -= f_offset;
    --w;

    /* Function Body */
    lp = *lperod + 1;
    mp = *mperod + 1;
    np = *nperod + 1;

/*     CHECK FOR INVALID INPUT. */

    *ierror = 0;
    if (lp < 1 || lp > 5) {
	*ierror = 1;
    }
    if (*l < 3) {
	*ierror = 2;
    }
    if (mp < 1 || mp > 5) {
	*ierror = 3;
    }
    if (*m < 3) {
	*ierror = 4;
    }
    if (np < 1 || np > 2) {
	*ierror = 5;
    }
    if (*n < 3) {
	*ierror = 6;
    }
    if (*ldimf < *l) {
	*ierror = 7;
    }
    if (*mdimf < *m) {
	*ierror = 8;
    }
    if (np != 1) {
	goto L103;
    }
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (a[k] != c__[1]) {
	    goto L102;
	}
	if (c__[k] != c__[1]) {
	    goto L102;
	}
	if (b[k] != b[1]) {
	    goto L102;
	}
/* L101: */
    }
    goto L104;
L102:
    *ierror = 9;
L103:
    if (*nperod == 1 && (a[1] != 0.f || c__[*n] != 0.f)) {
	*ierror = 10;
    }
L104:
    if (*ierror != 0) {
	goto L122;
    }
    iwyrt = *l + 1;
    iwt = iwyrt + *m;
/* Computing MAX */
    i__1 = max(*l,*m);
    iwd = iwt + max(i__1,*n) + 1;
    iwbb = iwd + *n;
    iwx = iwbb + *n;
    iwy = iwx + (*l + 1) / 2 * 7 + 15;
    switch (np) {
	case 1:  goto L105;
	case 2:  goto L114;
    }

/*     REORDER UNKNOWNS WHEN NPEROD = 0. */

L105:
    nh = (*n + 1) / 2;
    nhm1 = nh - 1;
    nodd = 1;
    if (nh << 1 == *n) {
	nodd = 2;
    }
    i__1 = *l;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *m;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = nhm1;
	    for (k = 1; k <= i__3; ++k) {
		nhpk = nh + k;
		nhmk = nh - k;
		w[k] = f[i__ + (j + nhmk * f_dim2) * f_dim1] - f[i__ + (j + 
			nhpk * f_dim2) * f_dim1];
		w[nhpk] = f[i__ + (j + nhmk * f_dim2) * f_dim1] + f[i__ + (j 
			+ nhpk * f_dim2) * f_dim1];
/* L106: */
	    }
	    w[nh] = f[i__ + (j + nh * f_dim2) * f_dim1] * 2.f;
	    switch (nodd) {
		case 1:  goto L108;
		case 2:  goto L107;
	    }
L107:
	    w[*n] = f[i__ + (j + *n * f_dim2) * f_dim1] * 2.f;
L108:
	    i__3 = *n;
	    for (k = 1; k <= i__3; ++k) {
		f[i__ + (j + k * f_dim2) * f_dim1] = w[k];
/* L109: */
	    }
/* L110: */
	}
/* L111: */
    }
    save[0] = c__[nhm1];
    save[1] = a[nh];
    save[2] = c__[nh];
    save[3] = b[nhm1];
    save[4] = b[*n];
    save[5] = a[*n];
    c__[nhm1] = 0.f;
    a[nh] = 0.f;
    c__[nh] *= 2.f;
    switch (nodd) {
	case 1:  goto L112;
	case 2:  goto L113;
    }
L112:
    b[nhm1] -= a[nh - 1];
    b[*n] += a[*n];
    goto L114;
L113:
    a[*n] = c__[nh];
L114:
    pos3d1_(&lp, l, &mp, m, n, &a[1], &b[1], &c__[1], ldimf, mdimf, &f[
	    f_offset], &w[1], &w[iwyrt], &w[iwt], &w[iwd], &w[iwx], &w[iwy], 
	    c1, c2, &w[iwbb]);
    switch (np) {
	case 1:  goto L115;
	case 2:  goto L122;
    }
L115:
    i__1 = *l;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *m;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = nhm1;
	    for (k = 1; k <= i__3; ++k) {
		nhmk = nh - k;
		nhpk = nh + k;
		w[nhmk] = (f[i__ + (j + nhpk * f_dim2) * f_dim1] + f[i__ + (j 
			+ k * f_dim2) * f_dim1]) * .5f;
		w[nhpk] = (f[i__ + (j + nhpk * f_dim2) * f_dim1] - f[i__ + (j 
			+ k * f_dim2) * f_dim1]) * .5f;
/* L116: */
	    }
	    w[nh] = f[i__ + (j + nh * f_dim2) * f_dim1] * .5f;
	    switch (nodd) {
		case 1:  goto L118;
		case 2:  goto L117;
	    }
L117:
	    w[*n] = f[i__ + (j + *n * f_dim2) * f_dim1] * .5f;
L118:
	    i__3 = *n;
	    for (k = 1; k <= i__3; ++k) {
		f[i__ + (j + k * f_dim2) * f_dim1] = w[k];
/* L119: */
	    }
/* L120: */
	}
/* L121: */
    }
    c__[nhm1] = save[0];
    a[nh] = save[1];
    c__[nh] = save[2];
    b[nhm1] = save[3];
    b[*n] = save[4];
    a[*n] = save[5];
L122:
    return 0;
} /* pois3d_ */

/* Subroutine */ int pos3d1_(integer *lp, integer *l, integer *mp, integer *m,
	 integer *n, doublereal *a, doublereal *b, doublereal *c__, integer *ldimf, integer *
	mdimf, doublereal *f, doublereal *xrt, doublereal *yrt, doublereal *t, doublereal *d__, doublereal *wx, 
	doublereal *wy, doublereal *c1, doublereal *c2, doublereal *bb)
{
    /* System generated locals */
    integer f_dim1, f_dim2, f_offset, i__1, i__2, i__3;
    doublereal r__1;

    /* Builtin functions */
    double sin(doublereal);

    /* Local variables */
    static integer i__, j, k;
    static doublereal di, dj, pi;
    static integer lr, mr, nr;
    static doublereal dx, dy, dum;
    extern /* Subroutine */ int cost_(integer *, doublereal *, doublereal *), sint_(
	    integer *, doublereal *, doublereal *);
    static integer lrdel, mrdel;
    extern /* Subroutine */ int rfftb_(integer *, doublereal *, doublereal *), rfftf_(
	    integer *, doublereal *, doublereal *), cosqb_(integer *, doublereal *, doublereal *);
    static doublereal scalx;
    extern /* Subroutine */ int rffti_(integer *, doublereal *);
    static doublereal scaly;
    static integer ifwrd;
    extern /* Subroutine */ int cosqi_(integer *, doublereal *), sinqb_(integer *, 
	    doublereal *, doublereal *), sinqf_(integer *, doublereal *, doublereal *), costi_(
	    integer *, doublereal *), cosqf_(integer *, doublereal *, doublereal *), sinqi_(
	    integer *, doublereal *), tridq_(integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *), sinti_(integer *, doublereal *);
    extern doublereal pimach_(doublereal *);

/* ***BEGIN PROLOGUE  POS3D1 */
/* ***SUBSIDIARY */
/* ***PURPOSE  Subsidiary to POIS3D */
/* ***LIBRARY   SLATEC */
/* ***TYPE      SINGLE PRECISION (POS3D1-S) */
/* ***AUTHOR  (UNKNOWN) */
/* ***SEE ALSO  POIS3D */
/* ***ROUTINES CALLED  COSQB, COSQF, COSQI, COST, COSTI, PIMACH, RFFTB, */
/*                    RFFTF, RFFTI, SINQB, SINQF, SINQI, SINT, SINTI, */
/*                    TRIDQ */
/* ***REVISION HISTORY  (YYMMDD) */
/*   801001  DATE WRITTEN */
/*   890531  Changed all specific intrinsics to generic.  (WRB) */
/*   891009  Removed unreferenced variable.  (WRB) */
/*   891214  Prologue converted to Version 4.0 format.  (BAB) */
/*   900308  Changed call to TRID to call to TRIDQ.  (WRB) */
/*   900402  Added TYPE section.  (WRB) */
/* ***END PROLOGUE  POS3D1 */
/* ***FIRST EXECUTABLE STATEMENT  POS3D1 */
    /* Parameter adjustments */
    --a;
    --b;
    --c__;
    f_dim1 = *ldimf;
    f_dim2 = *mdimf;
    f_offset = 1 + f_dim1 * (1 + f_dim2);
    f -= f_offset;
    --xrt;
    --yrt;
    --t;
    --d__;
    --wx;
    --wy;
    --bb;

    /* Function Body */
    pi = pimach_(&dum);
    lr = *l;
    mr = *m;
    nr = *n;

/*     GENERATE TRANSFORM ROOTS */

    lrdel = (*lp - 1) * (*lp - 3) * (*lp - 5) / 3;
    scalx = (doublereal) (lr + lrdel);
    dx = pi / (scalx * 2.f);
    switch (*lp) {
	case 1:  goto L108;
	case 2:  goto L103;
	case 3:  goto L101;
	case 4:  goto L102;
	case 5:  goto L101;
    }
L101:
    di = .5f;
    scalx *= 2.f;
    goto L104;
L102:
    di = 1.f;
    goto L104;
L103:
    di = 0.f;
L104:
    i__1 = lr;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	r__1 = sin((i__ - di) * dx);
	xrt[i__] = *c1 * -4.f * (r__1 * r__1);
/* L105: */
    }
    scalx *= 2.f;
    switch (*lp) {
	case 1:  goto L112;
	case 2:  goto L106;
	case 3:  goto L110;
	case 4:  goto L107;
	case 5:  goto L111;
    }
L106:
    sinti_(&lr, &wx[1]);
    goto L112;
L107:
    costi_(&lr, &wx[1]);
    goto L112;
L108:
    xrt[1] = 0.f;
    xrt[lr] = *c1 * -4.f;
    i__1 = lr;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
/* Computing 2nd power */
	r__1 = sin((i__ - 1) * dx);
	xrt[i__ - 1] = *c1 * -4.f * (r__1 * r__1);
	xrt[i__] = xrt[i__ - 1];
/* L109: */
    }
    rffti_(&lr, &wx[1]);
    goto L112;
L110:
    sinqi_(&lr, &wx[1]);
    goto L112;
L111:
    cosqi_(&lr, &wx[1]);
L112:
    mrdel = (*mp - 1) * (*mp - 3) * (*mp - 5) / 3;
    scaly = (doublereal) (mr + mrdel);
    dy = pi / (scaly * 2.f);
    switch (*mp) {
	case 1:  goto L120;
	case 2:  goto L115;
	case 3:  goto L113;
	case 4:  goto L114;
	case 5:  goto L113;
    }
L113:
    dj = .5f;
    scaly *= 2.f;
    goto L116;
L114:
    dj = 1.f;
    goto L116;
L115:
    dj = 0.f;
L116:
    i__1 = mr;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	r__1 = sin((j - dj) * dy);
	yrt[j] = *c2 * -4.f * (r__1 * r__1);
/* L117: */
    }
    scaly *= 2.f;
    switch (*mp) {
	case 1:  goto L124;
	case 2:  goto L118;
	case 3:  goto L122;
	case 4:  goto L119;
	case 5:  goto L123;
    }
L118:
    sinti_(&mr, &wy[1]);
    goto L124;
L119:
    costi_(&mr, &wy[1]);
    goto L124;
L120:
    yrt[1] = 0.f;
    yrt[mr] = *c2 * -4.f;
    i__1 = mr;
    for (j = 3; j <= i__1; j += 2) {
/* Computing 2nd power */
	r__1 = sin((j - 1) * dy);
	yrt[j - 1] = *c2 * -4.f * (r__1 * r__1);
	yrt[j] = yrt[j - 1];
/* L121: */
    }
    rffti_(&mr, &wy[1]);
    goto L124;
L122:
    sinqi_(&mr, &wy[1]);
    goto L124;
L123:
    cosqi_(&mr, &wy[1]);
L124:
    ifwrd = 1;
L125:

/*     TRANSFORM X */

    i__1 = mr;
    for (j = 1; j <= i__1; ++j) {
	i__2 = nr;
	for (k = 1; k <= i__2; ++k) {
	    i__3 = lr;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		t[i__] = f[i__ + (j + k * f_dim2) * f_dim1];
/* L126: */
	    }
	    switch (*lp) {
		case 1:  goto L127;
		case 2:  goto L130;
		case 3:  goto L131;
		case 4:  goto L134;
		case 5:  goto L135;
	    }
L127:
	    switch (ifwrd) {
		case 1:  goto L128;
		case 2:  goto L129;
	    }
L128:
	    rfftf_(&lr, &t[1], &wx[1]);
	    goto L138;
L129:
	    rfftb_(&lr, &t[1], &wx[1]);
	    goto L138;
L130:
	    sint_(&lr, &t[1], &wx[1]);
	    goto L138;
L131:
	    switch (ifwrd) {
		case 1:  goto L132;
		case 2:  goto L133;
	    }
L132:
	    sinqf_(&lr, &t[1], &wx[1]);
	    goto L138;
L133:
	    sinqb_(&lr, &t[1], &wx[1]);
	    goto L138;
L134:
	    cost_(&lr, &t[1], &wx[1]);
	    goto L138;
L135:
	    switch (ifwrd) {
		case 1:  goto L136;
		case 2:  goto L137;
	    }
L136:
	    cosqf_(&lr, &t[1], &wx[1]);
	    goto L138;
L137:
	    cosqb_(&lr, &t[1], &wx[1]);
L138:
	    i__3 = lr;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		f[i__ + (j + k * f_dim2) * f_dim1] = t[i__];
/* L139: */
	    }
/* L140: */
	}
/* L141: */
    }
    switch (ifwrd) {
	case 1:  goto L142;
	case 2:  goto L164;
    }

/*     TRANSFORM Y */

L142:
    i__1 = lr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = nr;
	for (k = 1; k <= i__2; ++k) {
	    i__3 = mr;
	    for (j = 1; j <= i__3; ++j) {
		t[j] = f[i__ + (j + k * f_dim2) * f_dim1];
/* L143: */
	    }
	    switch (*mp) {
		case 1:  goto L144;
		case 2:  goto L147;
		case 3:  goto L148;
		case 4:  goto L151;
		case 5:  goto L152;
	    }
L144:
	    switch (ifwrd) {
		case 1:  goto L145;
		case 2:  goto L146;
	    }
L145:
	    rfftf_(&mr, &t[1], &wy[1]);
	    goto L155;
L146:
	    rfftb_(&mr, &t[1], &wy[1]);
	    goto L155;
L147:
	    sint_(&mr, &t[1], &wy[1]);
	    goto L155;
L148:
	    switch (ifwrd) {
		case 1:  goto L149;
		case 2:  goto L150;
	    }
L149:
	    sinqf_(&mr, &t[1], &wy[1]);
	    goto L155;
L150:
	    sinqb_(&mr, &t[1], &wy[1]);
	    goto L155;
L151:
	    cost_(&mr, &t[1], &wy[1]);
	    goto L155;
L152:
	    switch (ifwrd) {
		case 1:  goto L153;
		case 2:  goto L154;
	    }
L153:
	    cosqf_(&mr, &t[1], &wy[1]);
	    goto L155;
L154:
	    cosqb_(&mr, &t[1], &wy[1]);
L155:
	    i__3 = mr;
	    for (j = 1; j <= i__3; ++j) {
		f[i__ + (j + k * f_dim2) * f_dim1] = t[j];
/* L156: */
	    }
/* L157: */
	}
/* L158: */
    }
    switch (ifwrd) {
	case 1:  goto L159;
	case 2:  goto L125;
    }
L159:

/*     SOLVE TRIDIAGONAL SYSTEMS IN Z */

    i__1 = lr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = mr;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = nr;
	    for (k = 1; k <= i__3; ++k) {
		bb[k] = b[k] + xrt[i__] + yrt[j];
		t[k] = f[i__ + (j + k * f_dim2) * f_dim1];
/* L160: */
	    }
	    tridq_(&nr, &a[1], &bb[1], &c__[1], &t[1], &d__[1]);
	    i__3 = nr;
	    for (k = 1; k <= i__3; ++k) {
		f[i__ + (j + k * f_dim2) * f_dim1] = t[k];
/* L161: */
	    }
/* L162: */
	}
/* L163: */
    }
    ifwrd = 2;
    goto L142;
L164:
    i__1 = lr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = mr;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = nr;
	    for (k = 1; k <= i__3; ++k) {
		f[i__ + (j + k * f_dim2) * f_dim1] /= scalx * scaly;
/* L165: */
	    }
/* L166: */
	}
/* L167: */
    }
    return 0;
} /* pos3d1_ */

/* Subroutine */ int tridq_(integer *mr, doublereal *a, doublereal *b, doublereal *c__, doublereal *y,
	 doublereal *d__)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m;
    static doublereal z__;
    static integer ip, mm1;

/* ***BEGIN PROLOGUE  TRIDQ */
/* ***SUBSIDIARY */
/* ***PURPOSE  Subsidiary to POIS3D */
/* ***LIBRARY   SLATEC */
/* ***TYPE      SINGLE PRECISION (TRIDQ-S) */
/* ***AUTHOR  (UNKNOWN) */
/* ***SEE ALSO  POIS3D */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   801001  DATE WRITTEN */
/*   891214  Prologue converted to Version 4.0 format.  (BAB) */
/*   900308  Renamed routine from TRID to TRIDQ.  (WRB) */
/*   900402  Added TYPE section.  (WRB) */
/* ***END PROLOGUE  TRIDQ */
/* ***FIRST EXECUTABLE STATEMENT  TRIDQ */
    /* Parameter adjustments */
    --d__;
    --y;
    --c__;
    --b;
    --a;

    /* Function Body */
    m = *mr;
    mm1 = m - 1;
    z__ = 1.f / b[1];
    d__[1] = c__[1] * z__;
    y[1] *= z__;
    i__1 = mm1;
    for (i__ = 2; i__ <= i__1; ++i__) {
	z__ = 1.f / (b[i__] - a[i__] * d__[i__ - 1]);
	d__[i__] = c__[i__] * z__;
	y[i__] = (y[i__] - a[i__] * y[i__ - 1]) * z__;
/* L101: */
    }
    z__ = b[m] - a[m] * d__[mm1];
    if (z__ != 0.f) {
	goto L102;
    }
    y[m] = 0.f;
    goto L103;
L102:
    y[m] = (y[m] - a[m] * y[mm1]) / z__;
L103:
    i__1 = mm1;
    for (ip = 1; ip <= i__1; ++ip) {
	i__ = m - ip;
	y[i__] -= d__[i__] * y[i__ + 1];
/* L104: */
    }
    return 0;
} /* tridq_ */

/* Subroutine */ int cosqb_(integer *n, doublereal *x, doublereal *wsave)
{
    /* Initialized data */

    static doublereal tsqrt2 = 2.82842712474619f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    static doublereal x1;
    extern /* Subroutine */ int cosqb1_(integer *, doublereal *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  COSQB */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Unnormalized inverse of COSQF. */
/* ***DESCRIPTION */

/*  Subroutine COSQB computes the fast Fourier transform of quarter */
/*  wave data. That is, COSQB computes a sequence from its */
/*  representation in terms of a cosine series with odd wave numbers. */
/*  The transform is defined below at output parameter X. */

/*  COSQB is the unnormalized inverse of COSQF since a call of COSQB */
/*  followed by a call of COSQF will multiply the input sequence X */
/*  by 4*N. */

/*  The array WSAVE which is used by subroutine COSQB must be */
/*  initialized by calling subroutine COSQI(N,WSAVE). */


/*  Input Parameters */

/*  N       the length of the array X to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */

/*  X       an array which contains the sequence to be transformed */

/*  WSAVE   a work array that must be dimensioned at least 3*N+15 */
/*          in the program that calls COSQB.  The WSAVE array must be */
/*          initialized by calling subroutine COSQI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */

/*  Output Parameters */

/*  X       For I=1,...,N */

/*               X(I)= the sum from K=1 to K=N of */

/*                 4*X(K)*COS((2*K-1)*(I-1)*PI/(2*N)) */

/*               A call of COSQB followed by a call of */
/*               COSQF will multiply the sequence X by 4*N. */
/*               Therefore COSQF is the unnormalized inverse */
/*               of COSQB. */

/*  WSAVE   contains initialization calculations which must not */
/*          be destroyed between calls of COSQB or COSQF. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  COSQB1 */
/* ***END PROLOGUE  COSQB */
    /* Parameter adjustments */
    --wsave;
    --x;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  COSQB */
    if ((i__1 = *n - 2) < 0) {
	goto L101;
    } else if (i__1 == 0) {
	goto L102;
    } else {
	goto L103;
    }
L101:
    x[1] *= 4.f;
    return 0;
L102:
    x1 = (x[1] + x[2]) * 4.f;
    x[2] = tsqrt2 * (x[1] - x[2]);
    x[1] = x1;
    return 0;
L103:
    cosqb1_(n, &x[1], &wsave[1], &wsave[*n + 1]);
    return 0;
} /* cosqb_ */

/* Subroutine */ int cosqb1_(integer *n, doublereal *x, doublereal *w, doublereal *xh)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, k, kc, np2, ns2;
    static doublereal xim1;
    static integer modn;
    extern /* Subroutine */ int rfftb_(integer *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  COSQB1 */
/* ***REFER TO  COSQB */
/* ***ROUTINES CALLED  RFFTB */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  COSQB1 */
/* ***FIRST EXECUTABLE STATEMENT  COSQB1 */
    /* Parameter adjustments */
    --xh;
    --w;
    --x;

    /* Function Body */
    ns2 = (*n + 1) / 2;
    np2 = *n + 2;
    i__1 = *n;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	xim1 = x[i__ - 1] + x[i__];
	x[i__] -= x[i__ - 1];
	x[i__ - 1] = xim1;
/* L101: */
    }
    x[1] += x[1];
    modn = *n % 2;
    if (modn == 0) {
	x[*n] += x[*n];
    }
    rfftb_(n, &x[1], &xh[1]);
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np2 - k;
	xh[k] = w[k - 1] * x[kc] + w[kc - 1] * x[k];
	xh[kc] = w[k - 1] * x[k] - w[kc - 1] * x[kc];
/* L102: */
    }
    if (modn == 0) {
	x[ns2 + 1] = w[ns2] * (x[ns2 + 1] + x[ns2 + 1]);
    }
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np2 - k;
	x[k] = xh[k] + xh[kc];
	x[kc] = xh[k] - xh[kc];
/* L103: */
    }
    x[1] += x[1];
    return 0;
} /* cosqb1_ */

/* Subroutine */ int cosqf_(integer *n, doublereal *x, doublereal *wsave)
{
    /* Initialized data */

    static doublereal sqrt2 = 1.4142135623731f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    static doublereal tsqx;
    extern /* Subroutine */ int cosqf1_(integer *, doublereal *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  COSQF */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Forward cosine transform with odd wave numbers. */
/* ***DESCRIPTION */

/*  Subroutine COSQF computes the fast Fourier transform of quarter */
/*  wave data. That is, COSQF computes the coefficients in a cosine */
/*  series representation with only odd wave numbers.  The transform */
/*  is defined below at Output Parameter X */

/*  COSQF is the unnormalized inverse of COSQB since a call of COSQF */
/*  followed by a call of COSQB will multiply the input sequence X */
/*  by 4*N. */

/*  The array WSAVE which is used by subroutine COSQF must be */
/*  initialized by calling subroutine COSQI(N,WSAVE). */


/*  Input Parameters */

/*  N       the length of the array X to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */

/*  X       an array which contains the sequence to be transformed */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15 */
/*          in the program that calls COSQF.  The WSAVE array must be */
/*          initialized by calling subroutine COSQI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */

/*  Output Parameters */

/*  X       For I=1,...,N */

/*               X(I) = X(1) plus the sum from K=2 to K=N of */

/*                  2*X(K)*COS((2*I-1)*(K-1)*PI/(2*N)) */

/*               A call of COSQF followed by a call of */
/*               COSQB will multiply the sequence X by 4*N. */
/*               Therefore COSQB is the unnormalized inverse */
/*               of COSQF. */

/*  WSAVE   contains initialization calculations which must not */
/*          be destroyed between calls of COSQF or COSQB. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  COSQF1 */
/* ***END PROLOGUE  COSQF */
    /* Parameter adjustments */
    --wsave;
    --x;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  COSQF */
    if ((i__1 = *n - 2) < 0) {
	goto L102;
    } else if (i__1 == 0) {
	goto L101;
    } else {
	goto L103;
    }
L101:
    tsqx = sqrt2 * x[2];
    x[2] = x[1] - tsqx;
    x[1] += tsqx;
L102:
    return 0;
L103:
    cosqf1_(n, &x[1], &wsave[1], &wsave[*n + 1]);
    return 0;
} /* cosqf_ */

/* Subroutine */ int cosqf1_(integer *n, doublereal *x, doublereal *w, doublereal *xh)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, k, kc, np2, ns2;
    static doublereal xim1;
    static integer modn;
    extern /* Subroutine */ int rfftf_(integer *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  COSQF1 */
/* ***REFER TO  COSQF */
/* ***ROUTINES CALLED  RFFTF */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  COSQF1 */
/* ***FIRST EXECUTABLE STATEMENT  COSQF1 */
    /* Parameter adjustments */
    --xh;
    --w;
    --x;

    /* Function Body */
    ns2 = (*n + 1) / 2;
    np2 = *n + 2;
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np2 - k;
	xh[k] = x[k] + x[kc];
	xh[kc] = x[k] - x[kc];
/* L101: */
    }
    modn = *n % 2;
    if (modn == 0) {
	xh[ns2 + 1] = x[ns2 + 1] + x[ns2 + 1];
    }
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np2 - k;
	x[k] = w[k - 1] * xh[kc] + w[kc - 1] * xh[k];
	x[kc] = w[k - 1] * xh[k] - w[kc - 1] * xh[kc];
/* L102: */
    }
    if (modn == 0) {
	x[ns2 + 1] = w[ns2] * xh[ns2 + 1];
    }
    rfftf_(n, &x[1], &xh[1]);
    i__1 = *n;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	xim1 = x[i__ - 1] - x[i__];
	x[i__] = x[i__ - 1] + x[i__];
	x[i__ - 1] = xim1;
/* L103: */
    }
    return 0;
} /* cosqf1_ */

/* Subroutine */ int cosqi_(integer *n, doublereal *wsave)
{
    /* Initialized data */

    static doublereal pih = 1.57079632679491f;

    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    double cos(doublereal);

    /* Local variables */
    static integer k;
    static doublereal fk, dt;
    extern /* Subroutine */ int rffti_(integer *, doublereal *);

/* ***BEGIN PROLOGUE  COSQI */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Initialize for COSQF and COSQB. */
/* ***DESCRIPTION */

/*  Subroutine COSQI initializes the array WSAVE which is used in */
/*  both COSQF and COSQB.  The prime factorization of N together with */
/*  a tabulation of the trigonometric functions are computed and */
/*  stored in WSAVE. */

/*  Input Parameter */

/*  N       the length of the array to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */

/*  Output Parameter */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15. */
/*          The same work array can be used for both COSQF and COSQB */
/*          as long as N remains unchanged.  Different WSAVE arrays */
/*          are required for different values of N.  The contents of */
/*          WSAVE must not be changed between calls of COSQF or COSQB. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTI */
/* ***END PROLOGUE  COSQI */
    /* Parameter adjustments */
    --wsave;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  COSQI */
    dt = pih / (doublereal) (*n);
    fk = 0.f;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	fk += 1.f;
	wsave[k] = cos(fk * dt);
/* L101: */
    }
    rffti_(n, &wsave[*n + 1]);
    return 0;
} /* cosqi_ */

/* Subroutine */ int cost_(integer *n, doublereal *x, doublereal *wsave)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, k;
    static doublereal c1, t1, t2;
    static integer kc;
    static doublereal xi;
    static integer nm1, np1;
    static doublereal x1h;
    static integer ns2;
    static doublereal tx2, x1p3, xim2;
    static integer modn;
    extern /* Subroutine */ int rfftf_(integer *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  COST */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Cosine transform of a doublereal, even sequence. */
/* ***DESCRIPTION */

/*  Subroutine COST computes the discrete Fourier cosine transform */
/*  of an even sequence X(I).  The transform is defined below at output */
/*  parameter X. */

/*  COST is the unnormalized inverse of itself since a call of COST */
/*  followed by another call of COST will multiply the input sequence */
/*  X by 2*(N-1).  The transform is defined below at output parameter X. */

/*  The array WSAVE which is used by subroutine COST must be */
/*  initialized by calling subroutine COSTI(N,WSAVE). */

/*  Input Parameters */

/*  N       the length of the sequence X.  N must be greater than 1. */
/*          The method is most efficient when N-1 is a product of */
/*          small primes. */

/*  X       an array which contains the sequence to be transformed */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15 */
/*          in the program that calls COST.  The WSAVE array must be */
/*          initialized by calling subroutine COSTI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */

/*  Output Parameters */

/*  X       For I=1,...,N */

/*             X(I) = X(1)+(-1)**(I-1)*X(N) */

/*               + the sum from K=2 to K=N-1 */

/*                   X(K)*COS((K-1)*(I-1)*PI/(N-1)) */

/*               A call of COST followed by another call of */
/*               COST will multiply the sequence X by 2*(N-1). */
/*               Hence COST is the unnormalized inverse */
/*               of itself. */

/*  WSAVE   contains initialization calculations which must not be */
/*          destroyed between calls of COST. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTF */
/* ***END PROLOGUE  COST */
/* ***FIRST EXECUTABLE STATEMENT  COST */
    /* Parameter adjustments */
    --wsave;
    --x;

    /* Function Body */
    nm1 = *n - 1;
    np1 = *n + 1;
    ns2 = *n / 2;
    if ((i__1 = *n - 2) < 0) {
	goto L106;
    } else if (i__1 == 0) {
	goto L101;
    } else {
	goto L102;
    }
L101:
    x1h = x[1] + x[2];
    x[2] = x[1] - x[2];
    x[1] = x1h;
    return 0;
L102:
    if (*n > 3) {
	goto L103;
    }
    x1p3 = x[1] + x[3];
    tx2 = x[2] + x[2];
    x[2] = x[1] - x[3];
    x[1] = x1p3 + tx2;
    x[3] = x1p3 - tx2;
    return 0;
L103:
    c1 = x[1] - x[*n];
    x[1] += x[*n];
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np1 - k;
	t1 = x[k] + x[kc];
	t2 = x[k] - x[kc];
	c1 += wsave[kc] * t2;
	t2 = wsave[k] * t2;
	x[k] = t1 - t2;
	x[kc] = t1 + t2;
/* L104: */
    }
    modn = *n % 2;
    if (modn != 0) {
	x[ns2 + 1] += x[ns2 + 1];
    }
    rfftf_(&nm1, &x[1], &wsave[*n + 1]);
    xim2 = x[2];
    x[2] = c1;
    i__1 = *n;
    for (i__ = 4; i__ <= i__1; i__ += 2) {
	xi = x[i__];
	x[i__] = x[i__ - 2] - x[i__ - 1];
	x[i__ - 1] = xim2;
	xim2 = xi;
/* L105: */
    }
    if (modn != 0) {
	x[*n] = xim2;
    }
L106:
    return 0;
} /* cost_ */

/* Subroutine */ int costi_(integer *n, doublereal *wsave)
{
    /* Initialized data */

    static doublereal pi = 3.14159265358979f;

    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    double sin(doublereal), cos(doublereal);

    /* Local variables */
    static integer k, kc;
    static doublereal fk, dt;
    static integer nm1, np1, ns2;
    extern /* Subroutine */ int rffti_(integer *, doublereal *);

/* ***BEGIN PROLOGUE  COSTI */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Initialize for COST. */
/* ***DESCRIPTION */

/*  Subroutine COSTI initializes the array WSAVE which is used in */
/*  subroutine COST.  The prime factorization of N together with */
/*  a tabulation of the trigonometric functions are computed and */
/*  stored in WSAVE. */

/*  Input Parameter */

/*  N       the length of the sequence to be transformed.  The method */
/*          is most efficient when N-1 is a product of small primes. */

/*  Output Parameter */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15. */
/*          Different WSAVE arrays are required for different values */
/*          of N.  The contents of WSAVE must not be changed between */
/*          calls of COST. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTI */
/* ***END PROLOGUE  COSTI */
    /* Parameter adjustments */
    --wsave;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  COSTI */
    if (*n <= 3) {
	return 0;
    }
    nm1 = *n - 1;
    np1 = *n + 1;
    ns2 = *n / 2;
    dt = pi / (doublereal) nm1;
    fk = 0.f;
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np1 - k;
	fk += 1.f;
	wsave[k] = sin(fk * dt) * 2.f;
	wsave[kc] = cos(fk * dt) * 2.f;
/* L101: */
    }
    rffti_(&nm1, &wsave[*n + 1]);
    return 0;
} /* costi_ */

/* Subroutine */ int rfftb1_(integer *n, doublereal *c__, doublereal *ch, doublereal *wa,
	integer *ifac)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, k1, l1, l2, na, nf, ip, iw, ix2, ix3, ix4, ido, idl1;
    extern /* Subroutine */ int radb2_(integer *, integer *, doublereal *, doublereal *,
	    doublereal *), radb3_(integer *, integer *, doublereal *, doublereal *, doublereal *,
	    doublereal *), radb4_(integer *, integer *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *), radb5_(integer *, integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *), radbg_(integer *, integer *,
	    integer *, integer *, doublereal *, doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *);

/* ***BEGIN PROLOGUE  RFFTB1 */
/* ***REFER TO  RFFTB */
/* ***ROUTINES CALLED  RADB2,RADB3,RADB4,RADB5,RADBG */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RFFTB1 */
/* ***FIRST EXECUTABLE STATEMENT  RFFTB1 */
    /* Parameter adjustments */
    --ifac;
    --wa;
    --ch;
    --c__;

    /* Function Body */
    nf = ifac[2];
    na = 0;
    l1 = 1;
    iw = 1;
    i__1 = nf;
    for (k1 = 1; k1 <= i__1; ++k1) {
	ip = ifac[k1 + 2];
	l2 = ip * l1;
	ido = *n / l2;
	idl1 = ido * l1;
	if (ip != 4) {
	    goto L103;
	}
	ix2 = iw + ido;
	ix3 = ix2 + ido;
	if (na != 0) {
	    goto L101;
	}
	radb4_(&ido, &l1, &c__[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3]);
	goto L102;
L101:
	radb4_(&ido, &l1, &ch[1], &c__[1], &wa[iw], &wa[ix2], &wa[ix3]);
L102:
	na = 1 - na;
	goto L115;
L103:
	if (ip != 2) {
	    goto L106;
	}
	if (na != 0) {
	    goto L104;
	}
	radb2_(&ido, &l1, &c__[1], &ch[1], &wa[iw]);
	goto L105;
L104:
	radb2_(&ido, &l1, &ch[1], &c__[1], &wa[iw]);
L105:
	na = 1 - na;
	goto L115;
L106:
	if (ip != 3) {
	    goto L109;
	}
	ix2 = iw + ido;
	if (na != 0) {
	    goto L107;
	}
	radb3_(&ido, &l1, &c__[1], &ch[1], &wa[iw], &wa[ix2]);
	goto L108;
L107:
	radb3_(&ido, &l1, &ch[1], &c__[1], &wa[iw], &wa[ix2]);
L108:
	na = 1 - na;
	goto L115;
L109:
	if (ip != 5) {
	    goto L112;
	}
	ix2 = iw + ido;
	ix3 = ix2 + ido;
	ix4 = ix3 + ido;
	if (na != 0) {
	    goto L110;
	}
	radb5_(&ido, &l1, &c__[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
	goto L111;
L110:
	radb5_(&ido, &l1, &ch[1], &c__[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
L111:
	na = 1 - na;
	goto L115;
L112:
	if (na != 0) {
	    goto L113;
	}
	radbg_(&ido, &ip, &l1, &idl1, &c__[1], &c__[1], &c__[1], &ch[1], &ch[
		1], &wa[iw]);
	goto L114;
L113:
	radbg_(&ido, &ip, &l1, &idl1, &ch[1], &ch[1], &ch[1], &c__[1], &c__[1]
		, &wa[iw]);
L114:
	if (ido == 1) {
	    na = 1 - na;
	}
L115:
	l1 = l2;
	iw += (ip - 1) * ido;
/* L116: */
    }
    if (na == 0) {
	return 0;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	c__[i__] = ch[i__];
/* L117: */
    }
    return 0;
} /* rfftb1_ */

/* Subroutine */ int rfftb_(integer *n, doublereal *r__, doublereal *wsave)
{
//     extern /* Subroutine */ int rfftb1_(integer *, doublereal *, doublereal *, doublereal *, 
// 	    doublereal *);

/* ***BEGIN PROLOGUE  RFFTB */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A1 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Backward transform of a doublereal coefficient array. */
/* ***DESCRIPTION */

/*  Subroutine RFFTB computes the doublereal perodic sequence from its */
/*  Fourier coefficients (Fourier synthesis).  The transform is defined */
/*  below at output parameter R. */

/*  Input Parameters */

/*  N       the length of the array R to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */
/*          N may change so long as different work arrays are provided. */

/*  R       a doublereal array of length N which contains the sequence */
/*          to be transformed */

/*  WSAVE   a work array which must be dimensioned at least 2*N+15 */
/*          in the program that calls RFFTB.  The WSAVE array must be */
/*          initialized by calling subroutine RFFTI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */
/*          The same WSAVE array can be used by RFFTF and RFFTB. */


/*  Output Parameters */

/*  R       For N even and For I = 1,...,N */

/*               R(I) = R(1)+(-1)**(I-1)*R(N) */

/*                    plus the sum from K=2 to K=N/2 of */

/*                     2.*R(2*K-2)*COS((K-1)*(I-1)*2*PI/N) */

/*                    -2.*R(2*K-1)*SIN((K-1)*(I-1)*2*PI/N) */

/*          For N odd and For I = 1,...,N */

/*               R(I) = R(1) plus the sum from K=2 to K=(N+1)/2 of */

/*                    2.*R(2*K-2)*COS((K-1)*(I-1)*2*PI/N) */

/*                   -2.*R(2*K-1)*SIN((K-1)*(I-1)*2*PI/N) */

/*   *****  Note: */
/*               This transform is unnormalized since a call of RFFTF */
/*               followed by a call of RFFTB will multiply the input */
/*               sequence by N. */

/*  WSAVE   contains results which must not be destroyed between */
/*          calls of RFFTB or RFFTF. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTB1 */
/* ***END PROLOGUE  RFFTB */
/* ***FIRST EXECUTABLE STATEMENT  RFFTB */
    /* Parameter adjustments */
    --wsave;
    --r__;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    rfftb1_(n, &r__[1], &wsave[1], &wsave[*n + 1], (integer *)&wsave[(*n << 1) + 1]);
    return 0;
} /* rfftb_ */


/* Subroutine */ int rfftf1_(integer *n, doublereal *c__, doublereal *ch, doublereal *wa,
	integer *ifac)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, k1, l1, l2, na, kh, nf, ip, iw, ix2, ix3, ix4, ido,
	    idl1;
    extern /* Subroutine */ int radf2_(integer *, integer *, doublereal *, doublereal *,
	    doublereal *), radf3_(integer *, integer *, doublereal *, doublereal *, doublereal *,
	    doublereal *), radf4_(integer *, integer *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *), radf5_(integer *, integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *), radfg_(integer *, integer *,
	    integer *, integer *, doublereal *, doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *);

/* ***BEGIN PROLOGUE  RFFTF1 */
/* ***REFER TO  RFFTF */
/* ***ROUTINES CALLED  RADF2,RADF3,RADF4,RADF5,RADFG */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RFFTF1 */
/* ***FIRST EXECUTABLE STATEMENT  RFFTF1 */
    /* Parameter adjustments */
    --ifac;
    --wa;
    --ch;
    --c__;

    /* Function Body */
    nf = ifac[2];
    na = 1;
    l2 = *n;
    iw = *n;
    i__1 = nf;
    for (k1 = 1; k1 <= i__1; ++k1) {
	kh = nf - k1;
	ip = ifac[kh + 3];
	l1 = l2 / ip;
	ido = *n / l2;
	idl1 = ido * l1;
	iw -= (ip - 1) * ido;
	na = 1 - na;
	if (ip != 4) {
	    goto L102;
	}
	ix2 = iw + ido;
	ix3 = ix2 + ido;
	if (na != 0) {
	    goto L101;
	}
	radf4_(&ido, &l1, &c__[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3]);
	goto L110;
L101:
	radf4_(&ido, &l1, &ch[1], &c__[1], &wa[iw], &wa[ix2], &wa[ix3]);
	goto L110;
L102:
	if (ip != 2) {
	    goto L104;
	}
	if (na != 0) {
	    goto L103;
	}
	radf2_(&ido, &l1, &c__[1], &ch[1], &wa[iw]);
	goto L110;
L103:
	radf2_(&ido, &l1, &ch[1], &c__[1], &wa[iw]);
	goto L110;
L104:
	if (ip != 3) {
	    goto L106;
	}
	ix2 = iw + ido;
	if (na != 0) {
	    goto L105;
	}
	radf3_(&ido, &l1, &c__[1], &ch[1], &wa[iw], &wa[ix2]);
	goto L110;
L105:
	radf3_(&ido, &l1, &ch[1], &c__[1], &wa[iw], &wa[ix2]);
	goto L110;
L106:
	if (ip != 5) {
	    goto L108;
	}
	ix2 = iw + ido;
	ix3 = ix2 + ido;
	ix4 = ix3 + ido;
	if (na != 0) {
	    goto L107;
	}
	radf5_(&ido, &l1, &c__[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
	goto L110;
L107:
	radf5_(&ido, &l1, &ch[1], &c__[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
	goto L110;
L108:
	if (ido == 1) {
	    na = 1 - na;
	}
	if (na != 0) {
	    goto L109;
	}
	radfg_(&ido, &ip, &l1, &idl1, &c__[1], &c__[1], &c__[1], &ch[1], &ch[
		1], &wa[iw]);
	na = 1;
	goto L110;
L109:
	radfg_(&ido, &ip, &l1, &idl1, &ch[1], &ch[1], &ch[1], &c__[1], &c__[1]
		, &wa[iw]);
	na = 0;
L110:
	l2 = l1;
/* L111: */
    }
    if (na == 1) {
	return 0;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	c__[i__] = ch[i__];
/* L112: */
    }
    return 0;
} /* rfftf1_ */

/* Subroutine */ int rfftf_(integer *n, doublereal *r__, doublereal *wsave)
{
//     extern /* Subroutine */ int rfftf1_(integer *, doublereal *, doublereal *, doublereal *, 
// 	    doublereal *);

/* ***BEGIN PROLOGUE  RFFTF */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***CATEGORY NO.  J1A1 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Forward transform of a doublereal, periodic sequence. */
/* ***DESCRIPTION */

/*  Subroutine RFFTF computes the Fourier coefficients of a doublereal */
/*  perodic sequence (Fourier analysis).  The transform is defined */
/*  below at output parameter R. */

/*  Input Parameters */

/*  N       the length of the array R to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */
/*          N may change so long as different work arrays are provided */

/*  R       a doublereal array of length N which contains the sequence */
/*          to be transformed */

/*  WSAVE   a work array which must be dimensioned at least 2*N+15 */
/*          in the program that calls RFFTF.  The WSAVE array must be */
/*          initialized by calling subroutine RFFTI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */
/*          the same WSAVE array can be used by RFFTF and RFFTB. */


/*  Output Parameters */

/*  R       R(1) = the sum from I=1 to I=N of R(I) */

/*          If N is even set L = N/2; if N is odd set L = (N+1)/2 */

/*            then for K = 2,...,L */

/*               R(2*K-2) = the sum from I = 1 to I = N of */

/*                    R(I)*COS((K-1)*(I-1)*2*PI/N) */

/*               R(2*K-1) = the sum from I = 1 to I = N of */

/*                   -R(I)*SIN((K-1)*(I-1)*2*PI/N) */

/*          If N is even */

/*               R(N) = the sum from I = 1 to I = N of */

/*                    (-1)**(I-1)*R(I) */

/*   *****  Note: */
/*               This transform is unnormalized since a call of RFFTF */
/*               followed by a call of RFFTB will multiply the input */
/*               sequence by N. */

/*  WSAVE   contains results which must not be destroyed between */
/*          calls of RFFTF or RFFTB. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTF1 */
/* ***END PROLOGUE  RFFTF */
/* ***FIRST EXECUTABLE STATEMENT  RFFTF */
    /* Parameter adjustments */
    --wsave;
    --r__;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    rfftf1_(n, &r__[1], &wsave[1], &wsave[*n + 1], (integer *)&wsave[(*n << 1) + 1]);
    return 0;
} /* rfftf_ */


/* Subroutine */ int rffti1_(integer *n, doublereal *wa, integer *ifac)
{
    /* Initialized data */

    static integer ntryh[4] = { 4,2,3,5 };

    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Builtin functions */
    double cos(doublereal), sin(doublereal);

    /* Local variables */
    static integer i__, j, k1, l1, l2, ib;
    static doublereal fi;
    static integer ld, ii, nf, ip, nl, is, nq, nr;
    static doublereal arg;
    static integer ido, ipm;
    static doublereal tpi;
    static integer nfm1;
    static doublereal argh;
    static integer ntry;
    static doublereal argld;

/* ***BEGIN PROLOGUE  RFFTI1 */
/* ***REFER TO  RFFTI */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RFFTI1 */
    /* Parameter adjustments */
    --ifac;
    --wa;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RFFTI1 */
    nl = *n;
    nf = 0;
    j = 0;
L101:
    ++j;
    if (j - 4 <= 0) {
	goto L102;
    } else {
	goto L103;
    }
L102:
    ntry = ntryh[j - 1];
    goto L104;
L103:
    ntry += 2;
L104:
    nq = nl / ntry;
    nr = nl - ntry * nq;
    if (nr != 0) {
	goto L101;
    } else {
	goto L105;
    }
L105:
    ++nf;
    ifac[nf + 2] = ntry;
    nl = nq;
    if (ntry != 2) {
	goto L107;
    }
    if (nf == 1) {
	goto L107;
    }
    i__1 = nf;
    for (i__ = 2; i__ <= i__1; ++i__) {
	ib = nf - i__ + 2;
	ifac[ib + 2] = ifac[ib + 1];
/* L106: */
    }
    ifac[3] = 2;
L107:
    if (nl != 1) {
	goto L104;
    }
    ifac[1] = *n;
    ifac[2] = nf;
    tpi = 6.28318530717959f;
    argh = tpi / (doublereal) (*n);
    is = 0;
    nfm1 = nf - 1;
    l1 = 1;
    if (nfm1 == 0) {
	return 0;
    }
    i__1 = nfm1;
    for (k1 = 1; k1 <= i__1; ++k1) {
	ip = ifac[k1 + 2];
	ld = 0;
	l2 = l1 * ip;
	ido = *n / l2;
	ipm = ip - 1;
	i__2 = ipm;
	for (j = 1; j <= i__2; ++j) {
	    ld += l1;
	    i__ = is;
	    argld = (doublereal) ld * argh;
	    fi = 0.f;
	    i__3 = ido;
	    for (ii = 3; ii <= i__3; ii += 2) {
		i__ += 2;
		fi += 1.f;
		arg = fi * argld;
		wa[i__ - 1] = cos(arg);
		wa[i__] = sin(arg);
/* L108: */
	    }
	    is += ido;
/* L109: */
	}
	l1 = l2;
/* L110: */
    }
    return 0;
} /* rffti1_ */

/* Subroutine */ int rffti_(integer *n, doublereal *wsave)
{
//     extern /* Subroutine */ int rffti1_(integer *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  RFFTI */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */
/* ***CATEGORY NO.  J1A1 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Initialize for RFFTF and RFFTB. */
/* ***DESCRIPTION */

/*  Subroutine RFFTI initializes the array WSAVE which is used in */
/*  both RFFTF and RFFTB.  The prime factorization of N together with */
/*  a tabulation of the trigonometric functions are computed and */
/*  stored in WSAVE. */

/*  Input Parameter */

/*  N       the length of the sequence to be transformed. */

/*  Output Parameter */

/*  WSAVE   a work array which must be dimensioned at least 2*N+15. */
/*          The same work array can be used for both RFFTF and RFFTB */
/*          as long as N remains unchanged.  Different WSAVE arrays */
/*          are required for different values of N.  The contents of */
/*          WSAVE must not be changed between calls of RFFTF or RFFTB. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTI1 */
/* ***END PROLOGUE  RFFTI */
/* ***FIRST EXECUTABLE STATEMENT  RFFTI */
    /* Parameter adjustments */
    --wsave;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    rffti1_(n, &wsave[*n + 1], (integer *)&wsave[(*n << 1) + 1]);
    return 0;
} /* rffti_ */



/* Subroutine */ int sinqb_(integer *n, doublereal *x, doublereal *wsave)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer k, kc, ns2;
    extern /* Subroutine */ int cosqb_(integer *, doublereal *, doublereal *);
    static doublereal xhold;

/* ***BEGIN PROLOGUE  SINQB */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */
/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Unnormalized inverse of SINQF. */
/* ***DESCRIPTION */

/*  Subroutine SINQB computes the fast Fourier transform of quarter */
/*  wave data.  That is, SINQB computes a sequence from its */
/*  representation in terms of a sine series with odd wave numbers. */
/*  the transform is defined below at output parameter X. */

/*  SINQF is the unnormalized inverse of SINQB since a call of SINQB */
/*  followed by a call of SINQF will multiply the input sequence X */
/*  by 4*N. */

/*  The array WSAVE which is used by subroutine SINQB must be */
/*  initialized by calling subroutine SINQI(N,WSAVE). */


/*  Input Parameters */

/*  N       the length of the array X to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */

/*  X       an array which contains the sequence to be transformed */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15 */
/*          in the program that calls SINQB.  The WSAVE array must be */
/*          initialized by calling subroutine SINQI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */

/*  Output Parameters */

/*  X       for I=1,...,N */

/*               X(I)= the sum from K=1 to K=N of */

/*                 4*X(K)*SIN((2k-1)*I*PI/(2*N)) */

/*               a call of SINQB followed by a call of */
/*               SINQF will multiply the sequence X by 4*N. */
/*               Therefore SINQF is the unnormalized inverse */
/*               of SINQB. */

/*  WSAVE   contains initialization calculations which must not */
/*          be destroyed between calls of SINQB or SINQF. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  COSQB */
/* ***END PROLOGUE  SINQB */
/* ***FIRST EXECUTABLE STATEMENT  SINQB */
    /* Parameter adjustments */
    --wsave;
    --x;

    /* Function Body */
    if (*n > 1) {
	goto L101;
    }
    x[1] *= 4.f;
    return 0;
L101:
    ns2 = *n / 2;
    i__1 = *n;
    for (k = 2; k <= i__1; k += 2) {
	x[k] = -x[k];
/* L102: */
    }
    cosqb_(n, &x[1], &wsave[1]);
    i__1 = ns2;
    for (k = 1; k <= i__1; ++k) {
	kc = *n - k;
	xhold = x[k];
	x[k] = x[kc + 1];
	x[kc + 1] = xhold;
/* L103: */
    }
    return 0;
} /* sinqb_ */

/* Subroutine */ int sinqf_(integer *n, doublereal *x, doublereal *wsave)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer k, kc, ns2;
    extern /* Subroutine */ int cosqf_(integer *, doublereal *, doublereal *);
    static doublereal xhold;

/* ***BEGIN PROLOGUE  SINQF */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */
/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Forward sine transform with odd wave numbers. */
/* ***DESCRIPTION */

/*  Subroutine SINQF computes the fast Fourier transform of quarter */
/*  wave data.  That is, SINQF computes the coefficients in a sine */
/*  series representation with only odd wave numbers.  The transform */
/*  is defined below at output parameter X. */

/*  SINQB is the unnormalized inverse of SINQF since a call of SINQF */
/*  followed by a call of SINQB will multiply the input sequence X */
/*  by 4*N. */

/*  The array WSAVE which is used by subroutine SINQF must be */
/*  initialized by calling subroutine SINQI(N,WSAVE). */


/*  Input Parameters */

/*  N       the length of the array X to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */

/*  X       an array which contains the sequence to be transformed */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15 */
/*          in the program that calls SINQF.  The WSAVE array must be */
/*          initialized by calling subroutine SINQI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */

/*  Output Parameters */

/*  X       For I=1,...,N */

/*               X(I) = (-1)**(I-1)*X(N) */

/*                  + the sum from K=1 to K=N-1 of */

/*                  2*X(K)*SIN((2*I-1)*K*PI/(2*N)) */

/*               A call of SINQF followed by a call of */
/*               SINQB will multiply the sequence X by 4*N. */
/*               Therefore SINQB is the unnormalized inverse */
/*               of SINQF. */

/*  WSAVE   contains initialization calculations which must not */
/*          be destroyed between calls of SINQF or SINQB. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  COSQF */
/* ***END PROLOGUE  SINQF */
/* ***FIRST EXECUTABLE STATEMENT  SINQF */
    /* Parameter adjustments */
    --wsave;
    --x;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    ns2 = *n / 2;
    i__1 = ns2;
    for (k = 1; k <= i__1; ++k) {
	kc = *n - k;
	xhold = x[k];
	x[k] = x[kc + 1];
	x[kc + 1] = xhold;
/* L101: */
    }
    cosqf_(n, &x[1], &wsave[1]);
    i__1 = *n;
    for (k = 2; k <= i__1; k += 2) {
	x[k] = -x[k];
/* L102: */
    }
    return 0;
} /* sinqf_ */

/* Subroutine */ int sinqi_(integer *n, doublereal *wsave)
{
    extern /* Subroutine */ int cosqi_(integer *, doublereal *);

/* ***BEGIN PROLOGUE  SINQI */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */
/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Initialize for SINQF and SINQB. */
/* ***DESCRIPTION */

/*  Subroutine SINQI initializes the array WSAVE which is used in */
/*  both SINQF and SINQB.  The prime factorization of N together with */
/*  a tabulation of the trigonometric functions are computed and */
/*  stored in WSAVE. */

/*  Input Parameter */

/*  N       the length of the sequence to be transformed.  The method */
/*          is most efficient when N is a product of small primes. */

/*  Output Parameter */

/*  WSAVE   a work array which must be dimensioned at least 3*N+15. */
/*          The same work array can be used for both SINQF and SINQB */
/*          as long as N remains unchanged.  Different WSAVE arrays */
/*          are required for different values of N.  The contents of */
/*          WSAVE must not be changed between calls of SINQF or SINQB. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  COSQI */
/* ***END PROLOGUE  SINQI */
/* ***FIRST EXECUTABLE STATEMENT  SINQI */
    /* Parameter adjustments */
    --wsave;

    /* Function Body */
    cosqi_(n, &wsave[1]);
    return 0;
} /* sinqi_ */

/* Subroutine */ int sint_(integer *n, doublereal *x, doublereal *wsave)
{
    /* Initialized data */

    static doublereal sqrt3 = 1.73205080756888f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, k;
    static doublereal t1, t2;
    static integer kc, nf;
    static doublereal xh;
    static integer kw, np1, ns2, modn;
    extern /* Subroutine */ int rfftf_(integer *, doublereal *, doublereal *);

/* ***BEGIN PROLOGUE  SINT */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */
/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Sine transform of a doublereal, odd sequence. */
/* ***DESCRIPTION */

/*  Subroutine SINT computes the discrete Fourier sine transform */
/*  of an odd sequence X(I).  The transform is defined below at */
/*  output parameter X. */

/*  SINT is the unnormalized inverse of itself since a call of SINT */
/*  followed by another call of SINT will multiply the input sequence */
/*  X by 2*(N+1). */

/*  The array WSAVE which is used by subroutine SINT must be */
/*  initialized by calling subroutine SINTI(N,WSAVE). */

/*  Input Parameters */

/*  N       the length of the sequence to be transformed.  The method */
/*          is most efficient when N+1 is the product of small primes. */

/*  X       an array which contains the sequence to be transformed */


/*  WSAVE   a work array with dimension at least INT(3.5*N+16) */
/*          in the program that calls SINT.  The WSAVE array must be */
/*          initialized by calling subroutine SINTI(N,WSAVE), and a */
/*          different WSAVE array must be used for each different */
/*          value of N.  This initialization does not have to be */
/*          repeated so long as N remains unchanged.  Thus subsequent */
/*          transforms can be obtained faster than the first. */

/*  Output Parameters */

/*  X       for I=1,...,N */

/*               X(I)= the sum from k=1 to k=N */

/*                    2*X(K)*SIN(K*I*PI/(N+1)) */

/*               A call of SINT followed by another call of */
/*               SINT will multiply the sequence X by 2*(N+1). */
/*               Hence SINT is the unnormalized inverse */
/*               of itself. */

/*  WSAVE   contains initialization calculations which must not be */
/*          destroyed between calls of SINT. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTF */
/* ***END PROLOGUE  SINT */
    /* Parameter adjustments */
    --wsave;
    --x;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  SINT */
    if ((i__1 = *n - 2) < 0) {
	goto L101;
    } else if (i__1 == 0) {
	goto L102;
    } else {
	goto L103;
    }
L101:
    x[1] += x[1];
    return 0;
L102:
    xh = sqrt3 * (x[1] + x[2]);
    x[2] = sqrt3 * (x[1] - x[2]);
    x[1] = xh;
    return 0;
L103:
    np1 = *n + 1;
    ns2 = *n / 2;
    wsave[1] = 0.f;
    kw = np1;
    i__1 = ns2;
    for (k = 1; k <= i__1; ++k) {
/* L1: */
	++kw;
	kc = np1 - k;
	t1 = x[k] - x[kc];
	t2 = wsave[kw] * (x[k] + x[kc]);
	wsave[k + 1] = t1 + t2;
	wsave[kc + 1] = t2 - t1;
/* L104: */
    }
    modn = *n % 2;
    if (modn != 0) {
	wsave[ns2 + 2] = x[ns2 + 1] * 4.f;
    }
    nf = np1 + ns2 + 1;
    rfftf_(&np1, &wsave[1], &wsave[nf]);
    x[1] = wsave[1] * .5f;
    i__1 = *n;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	x[i__ - 1] = -wsave[i__];
	x[i__] = x[i__ - 2] + wsave[i__ - 1];
/* L105: */
    }
    if (modn != 0) {
	return 0;
    }
    x[*n] = -wsave[*n + 1];
    return 0;
} /* sint_ */

/* Subroutine */ int sinti_(integer *n, doublereal *wsave)
{
    /* Initialized data */

    static doublereal pi = 3.14159265358979f;

    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    double sin(doublereal);

    /* Local variables */
    static integer k, kf;
    static doublereal fk, dt;
    static integer ks, np1, ns2;
    extern /* Subroutine */ int rffti_(integer *, doublereal *);

/* ***BEGIN PROLOGUE  SINTI */
/* ***DATE WRITTEN   790601   (YYMMDD) */
/* ***REVISION DATE  830401   (YYMMDD) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */
/* ***CATEGORY NO.  J1A3 */
/* ***KEYWORDS  FOURIER TRANSFORM */
/* ***AUTHOR  SWARZTRAUBER, P. N., (NCAR) */
/* ***PURPOSE  Initialize for SINT. */
/* ***DESCRIPTION */

/*  Subroutine SINTI initializes the array WSAVE which is used in */
/*  subroutine SINT.  The prime factorization of N together with */
/*  a tabulation of the trigonometric functions are computed and */
/*  stored in WSAVE. */

/*  Input Parameter */

/*  N       the length of the sequence to be transformed.  The method */
/*          is most efficient when N+1 is a product of small primes. */

/*  Output Parameter */

/*  WSAVE   a work array with at least INT(3.5*N+16) locations. */
/*          Different WSAVE arrays are required for different values */
/*          of N.  The contents of WSAVE must not be changed between */
/*          calls of SINT. */
/* ***REFERENCES  (NONE) */
/* ***ROUTINES CALLED  RFFTI */
/* ***END PROLOGUE  SINTI */
    /* Parameter adjustments */
    --wsave;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  SINTI */
    if (*n <= 1) {
	return 0;
    }
    np1 = *n + 1;
    ns2 = *n / 2;
    dt = pi / (doublereal) np1;
    ks = *n + 2;
    kf = ks + ns2 - 1;
    fk = 0.f;
    i__1 = kf;
    for (k = ks; k <= i__1; ++k) {
	fk += 1.f;
	wsave[k] = sin(fk * dt) * 2.f;
/* L101: */
    }
    rffti_(&np1, &wsave[kf + 1]);
    return 0;
} /* sinti_ */

//doublereal pimach_(doublereal *dum)
//{
//    /* System generated locals */
//    doublereal ret_val;
//
///* ***BEGIN PROLOGUE  PIMACH */
///* ***SUBSIDIARY */
///* ***PURPOSE  Subsidiary to HSTCSP, HSTSSP and HWSCSP */
///* ***LIBRARY   SLATEC */
///* ***TYPE      SINGLE PRECISION (PIMACH-S) */
///* ***AUTHOR  (UNKNOWN) */
///* ***DESCRIPTION */
//
///*     This subprogram supplies the value of the constant PI correct to */
///*     machine precision where */
//
///*     PI=3.1415926535897932384626433832795028841971693993751058209749446 */
//
///* ***SEE ALSO  HSTCSP, HSTSSP, HWSCSP */
///* ***ROUTINES CALLED  (NONE) */
///* ***REVISION HISTORY  (YYMMDD) */
///*   801001  DATE WRITTEN */
///*   891214  Prologue converted to Version 4.0 format.  (BAB) */
///*   900402  Added TYPE section.  (WRB) */
///* ***END PROLOGUE  PIMACH */
//
///* ***FIRST EXECUTABLE STATEMENT  PIMACH */
//    ret_val = 3.1415926535897932f;
//    return ret_val;
//} /* pimach_ */

/* Subroutine */ int radb2_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1)
{
    /* System generated locals */
    integer cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ti2, tr2;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADB2 */
/* ***REFER TO  RFFTB */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADB2 */
/* ***FIRST EXECUTABLE STATEMENT  RADB2 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2);
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_offset = 1 + cc_dim1 * 3;
    cc -= cc_offset;
    --wa1;

    /* Function Body */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[((k << 1) + 1) * cc_dim1 + 1] + 
		cc[*ido + ((k << 1) + 2) * cc_dim1];
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cc[((k << 1) + 1) * cc_dim1 
		+ 1] - cc[*ido + ((k << 1) + 2) * cc_dim1];
/* L101: */
    }
    if ((i__1 = *ido - 2) < 0) {
	goto L107;
    } else if (i__1 == 0) {
	goto L105;
    } else {
	goto L102;
    }
L102:
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L108;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = cc[i__ - 1 + ((k << 1) + 
		    1) * cc_dim1] + cc[ic - 1 + ((k << 1) + 2) * cc_dim1];
	    tr2 = cc[i__ - 1 + ((k << 1) + 1) * cc_dim1] - cc[ic - 1 + ((k << 
		    1) + 2) * cc_dim1];
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + ((k << 1) + 1) * 
		    cc_dim1] - cc[ic + ((k << 1) + 2) * cc_dim1];
	    ti2 = cc[i__ + ((k << 1) + 1) * cc_dim1] + cc[ic + ((k << 1) + 2) 
		    * cc_dim1];
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * tr2 
		    - wa1[i__ - 1] * ti2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * ti2 + 
		    wa1[i__ - 1] * tr2;
/* L103: */
	}
/* L104: */
    }
    goto L111;
L108:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = cc[i__ - 1 + ((k << 1) + 
		    1) * cc_dim1] + cc[ic - 1 + ((k << 1) + 2) * cc_dim1];
	    tr2 = cc[i__ - 1 + ((k << 1) + 1) * cc_dim1] - cc[ic - 1 + ((k << 
		    1) + 2) * cc_dim1];
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + ((k << 1) + 1) * 
		    cc_dim1] - cc[ic + ((k << 1) + 2) * cc_dim1];
	    ti2 = cc[i__ + ((k << 1) + 1) * cc_dim1] + cc[ic + ((k << 1) + 2) 
		    * cc_dim1];
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * tr2 
		    - wa1[i__ - 1] * ti2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * ti2 + 
		    wa1[i__ - 1] * tr2;
/* L109: */
	}
/* L110: */
    }
L111:
    if (*ido % 2 == 1) {
	return 0;
    }
L105:
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ch[*ido + (k + ch_dim2) * ch_dim1] = cc[*ido + ((k << 1) + 1) * 
		cc_dim1] + cc[*ido + ((k << 1) + 1) * cc_dim1];
	ch[*ido + (k + (ch_dim2 << 1)) * ch_dim1] = -(cc[((k << 1) + 2) * 
		cc_dim1 + 1] + cc[((k << 1) + 2) * cc_dim1 + 1]);
/* L106: */
    }
L107:
    return 0;
} /* radb2_ */

/* Subroutine */ int radb3_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1, doublereal *wa2)
{
    /* Initialized data */

    static doublereal taur = -.5f;
    static doublereal taui = .866025403784439f;

    /* System generated locals */
    integer cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADB3 */
/* ***REFER TO  RFFTB */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADB3 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2);
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_offset = 1 + (cc_dim1 << 2);
    cc -= cc_offset;
    --wa1;
    --wa2;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADB3 */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	tr2 = cc[*ido + (k * 3 + 2) * cc_dim1] + cc[*ido + (k * 3 + 2) * 
		cc_dim1];
	cr2 = cc[(k * 3 + 1) * cc_dim1 + 1] + taur * tr2;
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[(k * 3 + 1) * cc_dim1 + 1] + tr2;
	ci3 = taui * (cc[(k * 3 + 3) * cc_dim1 + 1] + cc[(k * 3 + 3) * 
		cc_dim1 + 1]);
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cr2 - ci3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = cr2 + ci3;
/* L101: */
    }
    if (*ido == 1) {
	return 0;
    }
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L104;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    tr2 = cc[i__ - 1 + (k * 3 + 3) * cc_dim1] + cc[ic - 1 + (k * 3 + 
		    2) * cc_dim1];
	    cr2 = cc[i__ - 1 + (k * 3 + 1) * cc_dim1] + taur * tr2;
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = cc[i__ - 1 + (k * 3 + 1) *
		     cc_dim1] + tr2;
	    ti2 = cc[i__ + (k * 3 + 3) * cc_dim1] - cc[ic + (k * 3 + 2) * 
		    cc_dim1];
	    ci2 = cc[i__ + (k * 3 + 1) * cc_dim1] + taur * ti2;
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + (k * 3 + 1) * 
		    cc_dim1] + ti2;
	    cr3 = taui * (cc[i__ - 1 + (k * 3 + 3) * cc_dim1] - cc[ic - 1 + (
		    k * 3 + 2) * cc_dim1]);
	    ci3 = taui * (cc[i__ + (k * 3 + 3) * cc_dim1] + cc[ic + (k * 3 + 
		    2) * cc_dim1]);
	    dr2 = cr2 - ci3;
	    dr3 = cr2 + ci3;
	    di2 = ci2 + cr3;
	    di3 = ci2 - cr3;
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * dr2 
		    - wa1[i__ - 1] * di2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * di2 + 
		    wa1[i__ - 1] * dr2;
	    ch[i__ - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * dr3 - 
		    wa2[i__ - 1] * di3;
	    ch[i__ + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * di3 + wa2[
		    i__ - 1] * dr3;
/* L102: */
	}
/* L103: */
    }
    return 0;
L104:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    tr2 = cc[i__ - 1 + (k * 3 + 3) * cc_dim1] + cc[ic - 1 + (k * 3 + 
		    2) * cc_dim1];
	    cr2 = cc[i__ - 1 + (k * 3 + 1) * cc_dim1] + taur * tr2;
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = cc[i__ - 1 + (k * 3 + 1) *
		     cc_dim1] + tr2;
	    ti2 = cc[i__ + (k * 3 + 3) * cc_dim1] - cc[ic + (k * 3 + 2) * 
		    cc_dim1];
	    ci2 = cc[i__ + (k * 3 + 1) * cc_dim1] + taur * ti2;
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + (k * 3 + 1) * 
		    cc_dim1] + ti2;
	    cr3 = taui * (cc[i__ - 1 + (k * 3 + 3) * cc_dim1] - cc[ic - 1 + (
		    k * 3 + 2) * cc_dim1]);
	    ci3 = taui * (cc[i__ + (k * 3 + 3) * cc_dim1] + cc[ic + (k * 3 + 
		    2) * cc_dim1]);
	    dr2 = cr2 - ci3;
	    dr3 = cr2 + ci3;
	    di2 = ci2 + cr3;
	    di3 = ci2 - cr3;
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * dr2 
		    - wa1[i__ - 1] * di2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * di2 + 
		    wa1[i__ - 1] * dr2;
	    ch[i__ - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * dr3 - 
		    wa2[i__ - 1] * di3;
	    ch[i__ + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * di3 + wa2[
		    i__ - 1] * dr3;
/* L105: */
	}
/* L106: */
    }
    return 0;
} /* radb3_ */

/* Subroutine */ int radb4_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1, doublereal *wa2, doublereal *wa3)
{
    /* Initialized data */

    static doublereal sqrt2 = 1.414213562373095f;

    /* System generated locals */
    integer cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, 
	    tr3, tr4;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADB4 */
/* ***REFER TO  RFFTB */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADB4 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2);
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_offset = 1 + cc_dim1 * 5;
    cc -= cc_offset;
    --wa1;
    --wa2;
    --wa3;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADB4 */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	tr1 = cc[((k << 2) + 1) * cc_dim1 + 1] - cc[*ido + ((k << 2) + 4) * 
		cc_dim1];
	tr2 = cc[((k << 2) + 1) * cc_dim1 + 1] + cc[*ido + ((k << 2) + 4) * 
		cc_dim1];
	tr3 = cc[*ido + ((k << 2) + 2) * cc_dim1] + cc[*ido + ((k << 2) + 2) *
		 cc_dim1];
	tr4 = cc[((k << 2) + 3) * cc_dim1 + 1] + cc[((k << 2) + 3) * cc_dim1 
		+ 1];
	ch[(k + ch_dim2) * ch_dim1 + 1] = tr2 + tr3;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = tr1 - tr4;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = tr2 - tr3;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 1] = tr1 + tr4;
/* L101: */
    }
    if ((i__1 = *ido - 2) < 0) {
	goto L107;
    } else if (i__1 == 0) {
	goto L105;
    } else {
	goto L102;
    }
L102:
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L108;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    ti1 = cc[i__ + ((k << 2) + 1) * cc_dim1] + cc[ic + ((k << 2) + 4) 
		    * cc_dim1];
	    ti2 = cc[i__ + ((k << 2) + 1) * cc_dim1] - cc[ic + ((k << 2) + 4) 
		    * cc_dim1];
	    ti3 = cc[i__ + ((k << 2) + 3) * cc_dim1] - cc[ic + ((k << 2) + 2) 
		    * cc_dim1];
	    tr4 = cc[i__ + ((k << 2) + 3) * cc_dim1] + cc[ic + ((k << 2) + 2) 
		    * cc_dim1];
	    tr1 = cc[i__ - 1 + ((k << 2) + 1) * cc_dim1] - cc[ic - 1 + ((k << 
		    2) + 4) * cc_dim1];
	    tr2 = cc[i__ - 1 + ((k << 2) + 1) * cc_dim1] + cc[ic - 1 + ((k << 
		    2) + 4) * cc_dim1];
	    ti4 = cc[i__ - 1 + ((k << 2) + 3) * cc_dim1] - cc[ic - 1 + ((k << 
		    2) + 2) * cc_dim1];
	    tr3 = cc[i__ - 1 + ((k << 2) + 3) * cc_dim1] + cc[ic - 1 + ((k << 
		    2) + 2) * cc_dim1];
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = tr2 + tr3;
	    cr3 = tr2 - tr3;
	    ch[i__ + (k + ch_dim2) * ch_dim1] = ti2 + ti3;
	    ci3 = ti2 - ti3;
	    cr2 = tr1 - tr4;
	    cr4 = tr1 + tr4;
	    ci2 = ti1 + ti4;
	    ci4 = ti1 - ti4;
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * cr2 
		    - wa1[i__ - 1] * ci2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * ci2 + 
		    wa1[i__ - 1] * cr2;
	    ch[i__ - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * cr3 - 
		    wa2[i__ - 1] * ci3;
	    ch[i__ + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * ci3 + wa2[
		    i__ - 1] * cr3;
	    ch[i__ - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * cr4 
		    - wa3[i__ - 1] * ci4;
	    ch[i__ + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * ci4 + 
		    wa3[i__ - 1] * cr4;
/* L103: */
	}
/* L104: */
    }
    goto L111;
L108:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ti1 = cc[i__ + ((k << 2) + 1) * cc_dim1] + cc[ic + ((k << 2) + 4) 
		    * cc_dim1];
	    ti2 = cc[i__ + ((k << 2) + 1) * cc_dim1] - cc[ic + ((k << 2) + 4) 
		    * cc_dim1];
	    ti3 = cc[i__ + ((k << 2) + 3) * cc_dim1] - cc[ic + ((k << 2) + 2) 
		    * cc_dim1];
	    tr4 = cc[i__ + ((k << 2) + 3) * cc_dim1] + cc[ic + ((k << 2) + 2) 
		    * cc_dim1];
	    tr1 = cc[i__ - 1 + ((k << 2) + 1) * cc_dim1] - cc[ic - 1 + ((k << 
		    2) + 4) * cc_dim1];
	    tr2 = cc[i__ - 1 + ((k << 2) + 1) * cc_dim1] + cc[ic - 1 + ((k << 
		    2) + 4) * cc_dim1];
	    ti4 = cc[i__ - 1 + ((k << 2) + 3) * cc_dim1] - cc[ic - 1 + ((k << 
		    2) + 2) * cc_dim1];
	    tr3 = cc[i__ - 1 + ((k << 2) + 3) * cc_dim1] + cc[ic - 1 + ((k << 
		    2) + 2) * cc_dim1];
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = tr2 + tr3;
	    cr3 = tr2 - tr3;
	    ch[i__ + (k + ch_dim2) * ch_dim1] = ti2 + ti3;
	    ci3 = ti2 - ti3;
	    cr2 = tr1 - tr4;
	    cr4 = tr1 + tr4;
	    ci2 = ti1 + ti4;
	    ci4 = ti1 - ti4;
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * cr2 
		    - wa1[i__ - 1] * ci2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * ci2 + 
		    wa1[i__ - 1] * cr2;
	    ch[i__ - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * cr3 - 
		    wa2[i__ - 1] * ci3;
	    ch[i__ + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * ci3 + wa2[
		    i__ - 1] * cr3;
	    ch[i__ - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * cr4 
		    - wa3[i__ - 1] * ci4;
	    ch[i__ + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * ci4 + 
		    wa3[i__ - 1] * cr4;
/* L109: */
	}
/* L110: */
    }
L111:
    if (*ido % 2 == 1) {
	return 0;
    }
L105:
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ti1 = cc[((k << 2) + 2) * cc_dim1 + 1] + cc[((k << 2) + 4) * cc_dim1 
		+ 1];
	ti2 = cc[((k << 2) + 4) * cc_dim1 + 1] - cc[((k << 2) + 2) * cc_dim1 
		+ 1];
	tr1 = cc[*ido + ((k << 2) + 1) * cc_dim1] - cc[*ido + ((k << 2) + 3) *
		 cc_dim1];
	tr2 = cc[*ido + ((k << 2) + 1) * cc_dim1] + cc[*ido + ((k << 2) + 3) *
		 cc_dim1];
	ch[*ido + (k + ch_dim2) * ch_dim1] = tr2 + tr2;
	ch[*ido + (k + (ch_dim2 << 1)) * ch_dim1] = sqrt2 * (tr1 - ti1);
	ch[*ido + (k + ch_dim2 * 3) * ch_dim1] = ti2 + ti2;
	ch[*ido + (k + (ch_dim2 << 2)) * ch_dim1] = -sqrt2 * (tr1 + ti1);
/* L106: */
    }
L107:
    return 0;
} /* radb4_ */

/* Subroutine */ int radb5_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1, doublereal *wa2, doublereal *wa3, doublereal *wa4)
{
    /* Initialized data */

    static doublereal tr11 = .309016994374947f;
    static doublereal ti11 = .951056516295154f;
    static doublereal tr12 = -.809016994374947f;
    static doublereal ti12 = .587785252292473f;

    /* System generated locals */
    integer cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, 
	    ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADB5 */
/* ***REFER TO  RFFTB */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADB5 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2);
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_offset = 1 + cc_dim1 * 6;
    cc -= cc_offset;
    --wa1;
    --wa2;
    --wa3;
    --wa4;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADB5 */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ti5 = cc[(k * 5 + 3) * cc_dim1 + 1] + cc[(k * 5 + 3) * cc_dim1 + 1];
	ti4 = cc[(k * 5 + 5) * cc_dim1 + 1] + cc[(k * 5 + 5) * cc_dim1 + 1];
	tr2 = cc[*ido + (k * 5 + 2) * cc_dim1] + cc[*ido + (k * 5 + 2) * 
		cc_dim1];
	tr3 = cc[*ido + (k * 5 + 4) * cc_dim1] + cc[*ido + (k * 5 + 4) * 
		cc_dim1];
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[(k * 5 + 1) * cc_dim1 + 1] + tr2 
		+ tr3;
	cr2 = cc[(k * 5 + 1) * cc_dim1 + 1] + tr11 * tr2 + tr12 * tr3;
	cr3 = cc[(k * 5 + 1) * cc_dim1 + 1] + tr12 * tr2 + tr11 * tr3;
	ci5 = ti11 * ti5 + ti12 * ti4;
	ci4 = ti12 * ti5 - ti11 * ti4;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cr2 - ci5;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = cr3 - ci4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 1] = cr3 + ci4;
	ch[(k + ch_dim2 * 5) * ch_dim1 + 1] = cr2 + ci5;
/* L101: */
    }
    if (*ido == 1) {
	return 0;
    }
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L104;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    ti5 = cc[i__ + (k * 5 + 3) * cc_dim1] + cc[ic + (k * 5 + 2) * 
		    cc_dim1];
	    ti2 = cc[i__ + (k * 5 + 3) * cc_dim1] - cc[ic + (k * 5 + 2) * 
		    cc_dim1];
	    ti4 = cc[i__ + (k * 5 + 5) * cc_dim1] + cc[ic + (k * 5 + 4) * 
		    cc_dim1];
	    ti3 = cc[i__ + (k * 5 + 5) * cc_dim1] - cc[ic + (k * 5 + 4) * 
		    cc_dim1];
	    tr5 = cc[i__ - 1 + (k * 5 + 3) * cc_dim1] - cc[ic - 1 + (k * 5 + 
		    2) * cc_dim1];
	    tr2 = cc[i__ - 1 + (k * 5 + 3) * cc_dim1] + cc[ic - 1 + (k * 5 + 
		    2) * cc_dim1];
	    tr4 = cc[i__ - 1 + (k * 5 + 5) * cc_dim1] - cc[ic - 1 + (k * 5 + 
		    4) * cc_dim1];
	    tr3 = cc[i__ - 1 + (k * 5 + 5) * cc_dim1] + cc[ic - 1 + (k * 5 + 
		    4) * cc_dim1];
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = cc[i__ - 1 + (k * 5 + 1) *
		     cc_dim1] + tr2 + tr3;
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + (k * 5 + 1) * 
		    cc_dim1] + ti2 + ti3;
	    cr2 = cc[i__ - 1 + (k * 5 + 1) * cc_dim1] + tr11 * tr2 + tr12 * 
		    tr3;
	    ci2 = cc[i__ + (k * 5 + 1) * cc_dim1] + tr11 * ti2 + tr12 * ti3;
	    cr3 = cc[i__ - 1 + (k * 5 + 1) * cc_dim1] + tr12 * tr2 + tr11 * 
		    tr3;
	    ci3 = cc[i__ + (k * 5 + 1) * cc_dim1] + tr12 * ti2 + tr11 * ti3;
	    cr5 = ti11 * tr5 + ti12 * tr4;
	    ci5 = ti11 * ti5 + ti12 * ti4;
	    cr4 = ti12 * tr5 - ti11 * tr4;
	    ci4 = ti12 * ti5 - ti11 * ti4;
	    dr3 = cr3 - ci4;
	    dr4 = cr3 + ci4;
	    di3 = ci3 + cr4;
	    di4 = ci3 - cr4;
	    dr5 = cr2 + ci5;
	    dr2 = cr2 - ci5;
	    di5 = ci2 - cr5;
	    di2 = ci2 + cr5;
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * dr2 
		    - wa1[i__ - 1] * di2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * di2 + 
		    wa1[i__ - 1] * dr2;
	    ch[i__ - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * dr3 - 
		    wa2[i__ - 1] * di3;
	    ch[i__ + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * di3 + wa2[
		    i__ - 1] * dr3;
	    ch[i__ - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * dr4 
		    - wa3[i__ - 1] * di4;
	    ch[i__ + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * di4 + 
		    wa3[i__ - 1] * dr4;
	    ch[i__ - 1 + (k + ch_dim2 * 5) * ch_dim1] = wa4[i__ - 2] * dr5 - 
		    wa4[i__ - 1] * di5;
	    ch[i__ + (k + ch_dim2 * 5) * ch_dim1] = wa4[i__ - 2] * di5 + wa4[
		    i__ - 1] * dr5;
/* L102: */
	}
/* L103: */
    }
    return 0;
L104:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ti5 = cc[i__ + (k * 5 + 3) * cc_dim1] + cc[ic + (k * 5 + 2) * 
		    cc_dim1];
	    ti2 = cc[i__ + (k * 5 + 3) * cc_dim1] - cc[ic + (k * 5 + 2) * 
		    cc_dim1];
	    ti4 = cc[i__ + (k * 5 + 5) * cc_dim1] + cc[ic + (k * 5 + 4) * 
		    cc_dim1];
	    ti3 = cc[i__ + (k * 5 + 5) * cc_dim1] - cc[ic + (k * 5 + 4) * 
		    cc_dim1];
	    tr5 = cc[i__ - 1 + (k * 5 + 3) * cc_dim1] - cc[ic - 1 + (k * 5 + 
		    2) * cc_dim1];
	    tr2 = cc[i__ - 1 + (k * 5 + 3) * cc_dim1] + cc[ic - 1 + (k * 5 + 
		    2) * cc_dim1];
	    tr4 = cc[i__ - 1 + (k * 5 + 5) * cc_dim1] - cc[ic - 1 + (k * 5 + 
		    4) * cc_dim1];
	    tr3 = cc[i__ - 1 + (k * 5 + 5) * cc_dim1] + cc[ic - 1 + (k * 5 + 
		    4) * cc_dim1];
	    ch[i__ - 1 + (k + ch_dim2) * ch_dim1] = cc[i__ - 1 + (k * 5 + 1) *
		     cc_dim1] + tr2 + tr3;
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + (k * 5 + 1) * 
		    cc_dim1] + ti2 + ti3;
	    cr2 = cc[i__ - 1 + (k * 5 + 1) * cc_dim1] + tr11 * tr2 + tr12 * 
		    tr3;
	    ci2 = cc[i__ + (k * 5 + 1) * cc_dim1] + tr11 * ti2 + tr12 * ti3;
	    cr3 = cc[i__ - 1 + (k * 5 + 1) * cc_dim1] + tr12 * tr2 + tr11 * 
		    tr3;
	    ci3 = cc[i__ + (k * 5 + 1) * cc_dim1] + tr12 * ti2 + tr11 * ti3;
	    cr5 = ti11 * tr5 + ti12 * tr4;
	    ci5 = ti11 * ti5 + ti12 * ti4;
	    cr4 = ti12 * tr5 - ti11 * tr4;
	    ci4 = ti12 * ti5 - ti11 * ti4;
	    dr3 = cr3 - ci4;
	    dr4 = cr3 + ci4;
	    di3 = ci3 + cr4;
	    di4 = ci3 - cr4;
	    dr5 = cr2 + ci5;
	    dr2 = cr2 - ci5;
	    di5 = ci2 - cr5;
	    di2 = ci2 + cr5;
	    ch[i__ - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * dr2 
		    - wa1[i__ - 1] * di2;
	    ch[i__ + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i__ - 2] * di2 + 
		    wa1[i__ - 1] * dr2;
	    ch[i__ - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * dr3 - 
		    wa2[i__ - 1] * di3;
	    ch[i__ + (k + ch_dim2 * 3) * ch_dim1] = wa2[i__ - 2] * di3 + wa2[
		    i__ - 1] * dr3;
	    ch[i__ - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * dr4 
		    - wa3[i__ - 1] * di4;
	    ch[i__ + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i__ - 2] * di4 + 
		    wa3[i__ - 1] * dr4;
	    ch[i__ - 1 + (k + ch_dim2 * 5) * ch_dim1] = wa4[i__ - 2] * dr5 - 
		    wa4[i__ - 1] * di5;
	    ch[i__ + (k + ch_dim2 * 5) * ch_dim1] = wa4[i__ - 2] * di5 + wa4[
		    i__ - 1] * dr5;
/* L105: */
	}
/* L106: */
    }
    return 0;
} /* radb5_ */

/* Subroutine */ int radbg_(integer *ido, integer *ip, integer *l1, integer *
	idl1, doublereal *cc, doublereal *c1, doublereal *c2, doublereal *ch, doublereal *ch2, doublereal *wa)
{
    /* Initialized data */

    static doublereal tpi = 6.28318530717959f;

    /* System generated locals */
    integer ch_dim1, ch_dim2, ch_offset, cc_dim1, cc_dim2, cc_offset, c1_dim1,
	     c1_dim2, c1_offset, c2_dim1, c2_offset, ch2_dim1, ch2_offset, 
	    i__1, i__2, i__3;

    /* Builtin functions */
    double cos(doublereal), sin(doublereal);

    /* Local variables */
    static integer i__, j, k, l, j2, ic, jc, lc, ik, is;
    static doublereal dc2, ai1, ai2, ar1, ar2, ds2;
    static integer nbd;
    static doublereal dcp, arg, dsp, ar1h, ar2h;
    static integer idp2, ipp2, idij, ipph;

/* ***BEGIN PROLOGUE  RADBG */
/* ***REFER TO  RFFTB */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADBG */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2);
    ch -= ch_offset;
    c1_dim1 = *ido;
    c1_dim2 = *l1;
    c1_offset = 1 + c1_dim1 * (1 + c1_dim2);
    c1 -= c1_offset;
    cc_dim1 = *ido;
    cc_dim2 = *ip;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2);
    cc -= cc_offset;
    ch2_dim1 = *idl1;
    ch2_offset = 1 + ch2_dim1;
    ch2 -= ch2_offset;
    c2_dim1 = *idl1;
    c2_offset = 1 + c2_dim1;
    c2 -= c2_offset;
    --wa;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADBG */
    arg = tpi / (doublereal) (*ip);
    dcp = cos(arg);
    dsp = sin(arg);
    idp2 = *ido + 2;
    nbd = (*ido - 1) / 2;
    ipp2 = *ip + 2;
    ipph = (*ip + 1) / 2;
    if (*ido < *l1) {
	goto L103;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = *ido;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + (k * cc_dim2 + 1) * 
		    cc_dim1];
/* L101: */
	}
/* L102: */
    }
    goto L106;
L103:
    i__1 = *ido;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ch[i__ + (k + ch_dim2) * ch_dim1] = cc[i__ + (k * cc_dim2 + 1) * 
		    cc_dim1];
/* L104: */
	}
/* L105: */
    }
L106:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	j2 = j + j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ch[(k + j * ch_dim2) * ch_dim1 + 1] = cc[*ido + (j2 - 2 + k * 
		    cc_dim2) * cc_dim1] + cc[*ido + (j2 - 2 + k * cc_dim2) * 
		    cc_dim1];
	    ch[(k + jc * ch_dim2) * ch_dim1 + 1] = cc[(j2 - 1 + k * cc_dim2) *
		     cc_dim1 + 1] + cc[(j2 - 1 + k * cc_dim2) * cc_dim1 + 1];
/* L107: */
	}
/* L108: */
    }
    if (*ido == 1) {
	goto L116;
    }
    if (nbd < *l1) {
	goto L112;
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
/* DIR$ IVDEP */
	    i__3 = *ido;
	    for (i__ = 3; i__ <= i__3; i__ += 2) {
		ic = idp2 - i__;
		ch[i__ - 1 + (k + j * ch_dim2) * ch_dim1] = cc[i__ - 1 + ((j 
			<< 1) - 1 + k * cc_dim2) * cc_dim1] + cc[ic - 1 + ((j 
			<< 1) - 2 + k * cc_dim2) * cc_dim1];
		ch[i__ - 1 + (k + jc * ch_dim2) * ch_dim1] = cc[i__ - 1 + ((j 
			<< 1) - 1 + k * cc_dim2) * cc_dim1] - cc[ic - 1 + ((j 
			<< 1) - 2 + k * cc_dim2) * cc_dim1];
		ch[i__ + (k + j * ch_dim2) * ch_dim1] = cc[i__ + ((j << 1) - 
			1 + k * cc_dim2) * cc_dim1] - cc[ic + ((j << 1) - 2 + 
			k * cc_dim2) * cc_dim1];
		ch[i__ + (k + jc * ch_dim2) * ch_dim1] = cc[i__ + ((j << 1) - 
			1 + k * cc_dim2) * cc_dim1] + cc[ic + ((j << 1) - 2 + 
			k * cc_dim2) * cc_dim1];
/* L109: */
	    }
/* L110: */
	}
/* L111: */
    }
    goto L116;
L112:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    i__3 = *l1;
	    for (k = 1; k <= i__3; ++k) {
		ch[i__ - 1 + (k + j * ch_dim2) * ch_dim1] = cc[i__ - 1 + ((j 
			<< 1) - 1 + k * cc_dim2) * cc_dim1] + cc[ic - 1 + ((j 
			<< 1) - 2 + k * cc_dim2) * cc_dim1];
		ch[i__ - 1 + (k + jc * ch_dim2) * ch_dim1] = cc[i__ - 1 + ((j 
			<< 1) - 1 + k * cc_dim2) * cc_dim1] - cc[ic - 1 + ((j 
			<< 1) - 2 + k * cc_dim2) * cc_dim1];
		ch[i__ + (k + j * ch_dim2) * ch_dim1] = cc[i__ + ((j << 1) - 
			1 + k * cc_dim2) * cc_dim1] - cc[ic + ((j << 1) - 2 + 
			k * cc_dim2) * cc_dim1];
		ch[i__ + (k + jc * ch_dim2) * ch_dim1] = cc[i__ + ((j << 1) - 
			1 + k * cc_dim2) * cc_dim1] + cc[ic + ((j << 1) - 2 + 
			k * cc_dim2) * cc_dim1];
/* L113: */
	    }
/* L114: */
	}
/* L115: */
    }
L116:
    ar1 = 1.f;
    ai1 = 0.f;
    i__1 = ipph;
    for (l = 2; l <= i__1; ++l) {
	lc = ipp2 - l;
	ar1h = dcp * ar1 - dsp * ai1;
	ai1 = dcp * ai1 + dsp * ar1;
	ar1 = ar1h;
	i__2 = *idl1;
	for (ik = 1; ik <= i__2; ++ik) {
	    c2[ik + l * c2_dim1] = ch2[ik + ch2_dim1] + ar1 * ch2[ik + (
		    ch2_dim1 << 1)];
	    c2[ik + lc * c2_dim1] = ai1 * ch2[ik + *ip * ch2_dim1];
/* L117: */
	}
	dc2 = ar1;
	ds2 = ai1;
	ar2 = ar1;
	ai2 = ai1;
	i__2 = ipph;
	for (j = 3; j <= i__2; ++j) {
	    jc = ipp2 - j;
	    ar2h = dc2 * ar2 - ds2 * ai2;
	    ai2 = dc2 * ai2 + ds2 * ar2;
	    ar2 = ar2h;
	    i__3 = *idl1;
	    for (ik = 1; ik <= i__3; ++ik) {
		c2[ik + l * c2_dim1] += ar2 * ch2[ik + j * ch2_dim1];
		c2[ik + lc * c2_dim1] += ai2 * ch2[ik + jc * ch2_dim1];
/* L118: */
	    }
/* L119: */
	}
/* L120: */
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	i__2 = *idl1;
	for (ik = 1; ik <= i__2; ++ik) {
	    ch2[ik + ch2_dim1] += ch2[ik + j * ch2_dim1];
/* L121: */
	}
/* L122: */
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ch[(k + j * ch_dim2) * ch_dim1 + 1] = c1[(k + j * c1_dim2) * 
		    c1_dim1 + 1] - c1[(k + jc * c1_dim2) * c1_dim1 + 1];
	    ch[(k + jc * ch_dim2) * ch_dim1 + 1] = c1[(k + j * c1_dim2) * 
		    c1_dim1 + 1] + c1[(k + jc * c1_dim2) * c1_dim1 + 1];
/* L123: */
	}
/* L124: */
    }
    if (*ido == 1) {
	goto L132;
    }
    if (nbd < *l1) {
	goto L128;
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
/* DIR$ IVDEP */
	    i__3 = *ido;
	    for (i__ = 3; i__ <= i__3; i__ += 2) {
		ch[i__ - 1 + (k + j * ch_dim2) * ch_dim1] = c1[i__ - 1 + (k + 
			j * c1_dim2) * c1_dim1] - c1[i__ + (k + jc * c1_dim2) 
			* c1_dim1];
		ch[i__ - 1 + (k + jc * ch_dim2) * ch_dim1] = c1[i__ - 1 + (k 
			+ j * c1_dim2) * c1_dim1] + c1[i__ + (k + jc * 
			c1_dim2) * c1_dim1];
		ch[i__ + (k + j * ch_dim2) * ch_dim1] = c1[i__ + (k + j * 
			c1_dim2) * c1_dim1] + c1[i__ - 1 + (k + jc * c1_dim2) 
			* c1_dim1];
		ch[i__ + (k + jc * ch_dim2) * ch_dim1] = c1[i__ + (k + j * 
			c1_dim2) * c1_dim1] - c1[i__ - 1 + (k + jc * c1_dim2) 
			* c1_dim1];
/* L125: */
	    }
/* L126: */
	}
/* L127: */
    }
    goto L132;
L128:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    i__3 = *l1;
	    for (k = 1; k <= i__3; ++k) {
		ch[i__ - 1 + (k + j * ch_dim2) * ch_dim1] = c1[i__ - 1 + (k + 
			j * c1_dim2) * c1_dim1] - c1[i__ + (k + jc * c1_dim2) 
			* c1_dim1];
		ch[i__ - 1 + (k + jc * ch_dim2) * ch_dim1] = c1[i__ - 1 + (k 
			+ j * c1_dim2) * c1_dim1] + c1[i__ + (k + jc * 
			c1_dim2) * c1_dim1];
		ch[i__ + (k + j * ch_dim2) * ch_dim1] = c1[i__ + (k + j * 
			c1_dim2) * c1_dim1] + c1[i__ - 1 + (k + jc * c1_dim2) 
			* c1_dim1];
		ch[i__ + (k + jc * ch_dim2) * ch_dim1] = c1[i__ + (k + j * 
			c1_dim2) * c1_dim1] - c1[i__ - 1 + (k + jc * c1_dim2) 
			* c1_dim1];
/* L129: */
	    }
/* L130: */
	}
/* L131: */
    }
L132:
    if (*ido == 1) {
	return 0;
    }
    i__1 = *idl1;
    for (ik = 1; ik <= i__1; ++ik) {
	c2[ik + c2_dim1] = ch2[ik + ch2_dim1];
/* L133: */
    }
    i__1 = *ip;
    for (j = 2; j <= i__1; ++j) {
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    c1[(k + j * c1_dim2) * c1_dim1 + 1] = ch[(k + j * ch_dim2) * 
		    ch_dim1 + 1];
/* L134: */
	}
/* L135: */
    }
    if (nbd > *l1) {
	goto L139;
    }
    is = -(*ido);
    i__1 = *ip;
    for (j = 2; j <= i__1; ++j) {
	is += *ido;
	idij = is;
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    idij += 2;
	    i__3 = *l1;
	    for (k = 1; k <= i__3; ++k) {
		c1[i__ - 1 + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[
			i__ - 1 + (k + j * ch_dim2) * ch_dim1] - wa[idij] * 
			ch[i__ + (k + j * ch_dim2) * ch_dim1];
		c1[i__ + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i__ 
			+ (k + j * ch_dim2) * ch_dim1] + wa[idij] * ch[i__ - 
			1 + (k + j * ch_dim2) * ch_dim1];
/* L136: */
	    }
/* L137: */
	}
/* L138: */
    }
    goto L143;
L139:
    is = -(*ido);
    i__1 = *ip;
    for (j = 2; j <= i__1; ++j) {
	is += *ido;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    idij = is;
/* DIR$ IVDEP */
	    i__3 = *ido;
	    for (i__ = 3; i__ <= i__3; i__ += 2) {
		idij += 2;
		c1[i__ - 1 + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[
			i__ - 1 + (k + j * ch_dim2) * ch_dim1] - wa[idij] * 
			ch[i__ + (k + j * ch_dim2) * ch_dim1];
		c1[i__ + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i__ 
			+ (k + j * ch_dim2) * ch_dim1] + wa[idij] * ch[i__ - 
			1 + (k + j * ch_dim2) * ch_dim1];
/* L140: */
	    }
/* L141: */
	}
/* L142: */
    }
L143:
    return 0;
} /* radbg_ */

/* Subroutine */ int radf2_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1)
{
    /* System generated locals */
    integer ch_dim1, ch_offset, cc_dim1, cc_dim2, cc_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ti2, tr2;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADF2 */
/* ***REFER TO  RFFTF */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADF2 */
/* ***FIRST EXECUTABLE STATEMENT  RADF2 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_offset = 1 + ch_dim1 * 3;
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_dim2 = *l1;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2);
    cc -= cc_offset;
    --wa1;

    /* Function Body */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ch[((k << 1) + 1) * ch_dim1 + 1] = cc[(k + cc_dim2) * cc_dim1 + 1] + 
		cc[(k + (cc_dim2 << 1)) * cc_dim1 + 1];
	ch[*ido + ((k << 1) + 2) * ch_dim1] = cc[(k + cc_dim2) * cc_dim1 + 1] 
		- cc[(k + (cc_dim2 << 1)) * cc_dim1 + 1];
/* L101: */
    }
    if ((i__1 = *ido - 2) < 0) {
	goto L107;
    } else if (i__1 == 0) {
	goto L105;
    } else {
	goto L102;
    }
L102:
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L108;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    tr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    ti2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    ch[i__ + ((k << 1) + 1) * ch_dim1] = cc[i__ + (k + cc_dim2) * 
		    cc_dim1] + ti2;
	    ch[ic + ((k << 1) + 2) * ch_dim1] = ti2 - cc[i__ + (k + cc_dim2) *
		     cc_dim1];
	    ch[i__ - 1 + ((k << 1) + 1) * ch_dim1] = cc[i__ - 1 + (k + 
		    cc_dim2) * cc_dim1] + tr2;
	    ch[ic - 1 + ((k << 1) + 2) * ch_dim1] = cc[i__ - 1 + (k + cc_dim2)
		     * cc_dim1] - tr2;
/* L103: */
	}
/* L104: */
    }
    goto L111;
L108:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    tr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    ti2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    ch[i__ + ((k << 1) + 1) * ch_dim1] = cc[i__ + (k + cc_dim2) * 
		    cc_dim1] + ti2;
	    ch[ic + ((k << 1) + 2) * ch_dim1] = ti2 - cc[i__ + (k + cc_dim2) *
		     cc_dim1];
	    ch[i__ - 1 + ((k << 1) + 1) * ch_dim1] = cc[i__ - 1 + (k + 
		    cc_dim2) * cc_dim1] + tr2;
	    ch[ic - 1 + ((k << 1) + 2) * ch_dim1] = cc[i__ - 1 + (k + cc_dim2)
		     * cc_dim1] - tr2;
/* L109: */
	}
/* L110: */
    }
L111:
    if (*ido % 2 == 1) {
	return 0;
    }
L105:
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ch[((k << 1) + 2) * ch_dim1 + 1] = -cc[*ido + (k + (cc_dim2 << 1)) * 
		cc_dim1];
	ch[*ido + ((k << 1) + 1) * ch_dim1] = cc[*ido + (k + cc_dim2) * 
		cc_dim1];
/* L106: */
    }
L107:
    return 0;
} /* radf2_ */

/* Subroutine */ int radf3_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1, doublereal *wa2)
{
    /* Initialized data */

    static doublereal taur = -.5f;
    static doublereal taui = .866025403784439f;

    /* System generated locals */
    integer ch_dim1, ch_offset, cc_dim1, cc_dim2, cc_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADF3 */
/* ***REFER TO  RFFTF */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADF3 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_offset = 1 + (ch_dim1 << 2);
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_dim2 = *l1;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2);
    cc -= cc_offset;
    --wa1;
    --wa2;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADF3 */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	cr2 = cc[(k + (cc_dim2 << 1)) * cc_dim1 + 1] + cc[(k + cc_dim2 * 3) * 
		cc_dim1 + 1];
	ch[(k * 3 + 1) * ch_dim1 + 1] = cc[(k + cc_dim2) * cc_dim1 + 1] + cr2;
	ch[(k * 3 + 3) * ch_dim1 + 1] = taui * (cc[(k + cc_dim2 * 3) * 
		cc_dim1 + 1] - cc[(k + (cc_dim2 << 1)) * cc_dim1 + 1]);
	ch[*ido + (k * 3 + 2) * ch_dim1] = cc[(k + cc_dim2) * cc_dim1 + 1] + 
		taur * cr2;
/* L101: */
    }
    if (*ido == 1) {
	return 0;
    }
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L104;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    dr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    di2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    dr3 = wa2[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1] + 
		    wa2[i__ - 1] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1];
	    di3 = wa2[i__ - 2] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1] - wa2[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1];
	    cr2 = dr2 + dr3;
	    ci2 = di2 + di3;
	    ch[i__ - 1 + (k * 3 + 1) * ch_dim1] = cc[i__ - 1 + (k + cc_dim2) *
		     cc_dim1] + cr2;
	    ch[i__ + (k * 3 + 1) * ch_dim1] = cc[i__ + (k + cc_dim2) * 
		    cc_dim1] + ci2;
	    tr2 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + taur * cr2;
	    ti2 = cc[i__ + (k + cc_dim2) * cc_dim1] + taur * ci2;
	    tr3 = taui * (di2 - di3);
	    ti3 = taui * (dr3 - dr2);
	    ch[i__ - 1 + (k * 3 + 3) * ch_dim1] = tr2 + tr3;
	    ch[ic - 1 + (k * 3 + 2) * ch_dim1] = tr2 - tr3;
	    ch[i__ + (k * 3 + 3) * ch_dim1] = ti2 + ti3;
	    ch[ic + (k * 3 + 2) * ch_dim1] = ti3 - ti2;
/* L102: */
	}
/* L103: */
    }
    return 0;
L104:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    dr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    di2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    dr3 = wa2[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1] + 
		    wa2[i__ - 1] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1];
	    di3 = wa2[i__ - 2] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1] - wa2[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1];
	    cr2 = dr2 + dr3;
	    ci2 = di2 + di3;
	    ch[i__ - 1 + (k * 3 + 1) * ch_dim1] = cc[i__ - 1 + (k + cc_dim2) *
		     cc_dim1] + cr2;
	    ch[i__ + (k * 3 + 1) * ch_dim1] = cc[i__ + (k + cc_dim2) * 
		    cc_dim1] + ci2;
	    tr2 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + taur * cr2;
	    ti2 = cc[i__ + (k + cc_dim2) * cc_dim1] + taur * ci2;
	    tr3 = taui * (di2 - di3);
	    ti3 = taui * (dr3 - dr2);
	    ch[i__ - 1 + (k * 3 + 3) * ch_dim1] = tr2 + tr3;
	    ch[ic - 1 + (k * 3 + 2) * ch_dim1] = tr2 - tr3;
	    ch[i__ + (k * 3 + 3) * ch_dim1] = ti2 + ti3;
	    ch[ic + (k * 3 + 2) * ch_dim1] = ti3 - ti2;
/* L105: */
	}
/* L106: */
    }
    return 0;
} /* radf3_ */

/* Subroutine */ int radf4_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1, doublereal *wa2, doublereal *wa3)
{
    /* Initialized data */

    static doublereal hsqt2 = .7071067811865475f;

    /* System generated locals */
    integer cc_dim1, cc_dim2, cc_offset, ch_dim1, ch_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, 
	    tr3, tr4;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADF4 */
/* ***REFER TO  RFFTF */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADF4 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_offset = 1 + ch_dim1 * 5;
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_dim2 = *l1;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2);
    cc -= cc_offset;
    --wa1;
    --wa2;
    --wa3;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADF4 */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	tr1 = cc[(k + (cc_dim2 << 1)) * cc_dim1 + 1] + cc[(k + (cc_dim2 << 2))
		 * cc_dim1 + 1];
	tr2 = cc[(k + cc_dim2) * cc_dim1 + 1] + cc[(k + cc_dim2 * 3) * 
		cc_dim1 + 1];
	ch[((k << 2) + 1) * ch_dim1 + 1] = tr1 + tr2;
	ch[*ido + ((k << 2) + 4) * ch_dim1] = tr2 - tr1;
	ch[*ido + ((k << 2) + 2) * ch_dim1] = cc[(k + cc_dim2) * cc_dim1 + 1] 
		- cc[(k + cc_dim2 * 3) * cc_dim1 + 1];
	ch[((k << 2) + 3) * ch_dim1 + 1] = cc[(k + (cc_dim2 << 2)) * cc_dim1 
		+ 1] - cc[(k + (cc_dim2 << 1)) * cc_dim1 + 1];
/* L101: */
    }
    if ((i__1 = *ido - 2) < 0) {
	goto L107;
    } else if (i__1 == 0) {
	goto L105;
    } else {
	goto L102;
    }
L102:
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L111;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    cr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    ci2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    cr3 = wa2[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1] + 
		    wa2[i__ - 1] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1];
	    ci3 = wa2[i__ - 2] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1] - wa2[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1];
	    cr4 = wa3[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * cc_dim1] 
		    + wa3[i__ - 1] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1];
	    ci4 = wa3[i__ - 2] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1] - 
		    wa3[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * 
		    cc_dim1];
	    tr1 = cr2 + cr4;
	    tr4 = cr4 - cr2;
	    ti1 = ci2 + ci4;
	    ti4 = ci2 - ci4;
	    ti2 = cc[i__ + (k + cc_dim2) * cc_dim1] + ci3;
	    ti3 = cc[i__ + (k + cc_dim2) * cc_dim1] - ci3;
	    tr2 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + cr3;
	    tr3 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] - cr3;
	    ch[i__ - 1 + ((k << 2) + 1) * ch_dim1] = tr1 + tr2;
	    ch[ic - 1 + ((k << 2) + 4) * ch_dim1] = tr2 - tr1;
	    ch[i__ + ((k << 2) + 1) * ch_dim1] = ti1 + ti2;
	    ch[ic + ((k << 2) + 4) * ch_dim1] = ti1 - ti2;
	    ch[i__ - 1 + ((k << 2) + 3) * ch_dim1] = ti4 + tr3;
	    ch[ic - 1 + ((k << 2) + 2) * ch_dim1] = tr3 - ti4;
	    ch[i__ + ((k << 2) + 3) * ch_dim1] = tr4 + ti3;
	    ch[ic + ((k << 2) + 2) * ch_dim1] = tr4 - ti3;
/* L103: */
	}
/* L104: */
    }
    goto L110;
L111:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    cr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    ci2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    cr3 = wa2[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1] + 
		    wa2[i__ - 1] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1];
	    ci3 = wa2[i__ - 2] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1] - wa2[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1];
	    cr4 = wa3[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * cc_dim1] 
		    + wa3[i__ - 1] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1];
	    ci4 = wa3[i__ - 2] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1] - 
		    wa3[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * 
		    cc_dim1];
	    tr1 = cr2 + cr4;
	    tr4 = cr4 - cr2;
	    ti1 = ci2 + ci4;
	    ti4 = ci2 - ci4;
	    ti2 = cc[i__ + (k + cc_dim2) * cc_dim1] + ci3;
	    ti3 = cc[i__ + (k + cc_dim2) * cc_dim1] - ci3;
	    tr2 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + cr3;
	    tr3 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] - cr3;
	    ch[i__ - 1 + ((k << 2) + 1) * ch_dim1] = tr1 + tr2;
	    ch[ic - 1 + ((k << 2) + 4) * ch_dim1] = tr2 - tr1;
	    ch[i__ + ((k << 2) + 1) * ch_dim1] = ti1 + ti2;
	    ch[ic + ((k << 2) + 4) * ch_dim1] = ti1 - ti2;
	    ch[i__ - 1 + ((k << 2) + 3) * ch_dim1] = ti4 + tr3;
	    ch[ic - 1 + ((k << 2) + 2) * ch_dim1] = tr3 - ti4;
	    ch[i__ + ((k << 2) + 3) * ch_dim1] = tr4 + ti3;
	    ch[ic + ((k << 2) + 2) * ch_dim1] = tr4 - ti3;
/* L108: */
	}
/* L109: */
    }
L110:
    if (*ido % 2 == 1) {
	return 0;
    }
L105:
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ti1 = -hsqt2 * (cc[*ido + (k + (cc_dim2 << 1)) * cc_dim1] + cc[*ido + 
		(k + (cc_dim2 << 2)) * cc_dim1]);
	tr1 = hsqt2 * (cc[*ido + (k + (cc_dim2 << 1)) * cc_dim1] - cc[*ido + (
		k + (cc_dim2 << 2)) * cc_dim1]);
	ch[*ido + ((k << 2) + 1) * ch_dim1] = tr1 + cc[*ido + (k + cc_dim2) * 
		cc_dim1];
	ch[*ido + ((k << 2) + 3) * ch_dim1] = cc[*ido + (k + cc_dim2) * 
		cc_dim1] - tr1;
	ch[((k << 2) + 2) * ch_dim1 + 1] = ti1 - cc[*ido + (k + cc_dim2 * 3) *
		 cc_dim1];
	ch[((k << 2) + 4) * ch_dim1 + 1] = ti1 + cc[*ido + (k + cc_dim2 * 3) *
		 cc_dim1];
/* L106: */
    }
L107:
    return 0;
} /* radf4_ */

/* Subroutine */ int radf5_(integer *ido, integer *l1, doublereal *cc, doublereal *ch, 
	doublereal *wa1, doublereal *wa2, doublereal *wa3, doublereal *wa4)
{
    /* Initialized data */

    static doublereal tr11 = .309016994374947f;
    static doublereal ti11 = .951056516295154f;
    static doublereal tr12 = -.809016994374947f;
    static doublereal ti12 = .587785252292473f;

    /* System generated locals */
    integer cc_dim1, cc_dim2, cc_offset, ch_dim1, ch_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, ic;
    static doublereal ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, 
	    dr4, dr5, cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
    static integer idp2;

/* ***BEGIN PROLOGUE  RADF5 */
/* ***REFER TO  RFFTF */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADF5 */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_offset = 1 + ch_dim1 * 6;
    ch -= ch_offset;
    cc_dim1 = *ido;
    cc_dim2 = *l1;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2);
    cc -= cc_offset;
    --wa1;
    --wa2;
    --wa3;
    --wa4;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADF5 */
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	cr2 = cc[(k + cc_dim2 * 5) * cc_dim1 + 1] + cc[(k + (cc_dim2 << 1)) * 
		cc_dim1 + 1];
	ci5 = cc[(k + cc_dim2 * 5) * cc_dim1 + 1] - cc[(k + (cc_dim2 << 1)) * 
		cc_dim1 + 1];
	cr3 = cc[(k + (cc_dim2 << 2)) * cc_dim1 + 1] + cc[(k + cc_dim2 * 3) * 
		cc_dim1 + 1];
	ci4 = cc[(k + (cc_dim2 << 2)) * cc_dim1 + 1] - cc[(k + cc_dim2 * 3) * 
		cc_dim1 + 1];
	ch[(k * 5 + 1) * ch_dim1 + 1] = cc[(k + cc_dim2) * cc_dim1 + 1] + cr2 
		+ cr3;
	ch[*ido + (k * 5 + 2) * ch_dim1] = cc[(k + cc_dim2) * cc_dim1 + 1] + 
		tr11 * cr2 + tr12 * cr3;
	ch[(k * 5 + 3) * ch_dim1 + 1] = ti11 * ci5 + ti12 * ci4;
	ch[*ido + (k * 5 + 4) * ch_dim1] = cc[(k + cc_dim2) * cc_dim1 + 1] + 
		tr12 * cr2 + tr11 * cr3;
	ch[(k * 5 + 5) * ch_dim1 + 1] = ti12 * ci5 - ti11 * ci4;
/* L101: */
    }
    if (*ido == 1) {
	return 0;
    }
    idp2 = *ido + 2;
    if ((*ido - 1) / 2 < *l1) {
	goto L104;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
/* DIR$ IVDEP */
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    dr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    di2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    dr3 = wa2[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1] + 
		    wa2[i__ - 1] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1];
	    di3 = wa2[i__ - 2] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1] - wa2[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1];
	    dr4 = wa3[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * cc_dim1] 
		    + wa3[i__ - 1] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1];
	    di4 = wa3[i__ - 2] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1] - 
		    wa3[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * 
		    cc_dim1];
	    dr5 = wa4[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 5) * cc_dim1] + 
		    wa4[i__ - 1] * cc[i__ + (k + cc_dim2 * 5) * cc_dim1];
	    di5 = wa4[i__ - 2] * cc[i__ + (k + cc_dim2 * 5) * cc_dim1] - wa4[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 5) * cc_dim1];
	    cr2 = dr2 + dr5;
	    ci5 = dr5 - dr2;
	    cr5 = di2 - di5;
	    ci2 = di2 + di5;
	    cr3 = dr3 + dr4;
	    ci4 = dr4 - dr3;
	    cr4 = di3 - di4;
	    ci3 = di3 + di4;
	    ch[i__ - 1 + (k * 5 + 1) * ch_dim1] = cc[i__ - 1 + (k + cc_dim2) *
		     cc_dim1] + cr2 + cr3;
	    ch[i__ + (k * 5 + 1) * ch_dim1] = cc[i__ + (k + cc_dim2) * 
		    cc_dim1] + ci2 + ci3;
	    tr2 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + tr11 * cr2 + tr12 * 
		    cr3;
	    ti2 = cc[i__ + (k + cc_dim2) * cc_dim1] + tr11 * ci2 + tr12 * ci3;
	    tr3 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + tr12 * cr2 + tr11 * 
		    cr3;
	    ti3 = cc[i__ + (k + cc_dim2) * cc_dim1] + tr12 * ci2 + tr11 * ci3;
	    tr5 = ti11 * cr5 + ti12 * cr4;
	    ti5 = ti11 * ci5 + ti12 * ci4;
	    tr4 = ti12 * cr5 - ti11 * cr4;
	    ti4 = ti12 * ci5 - ti11 * ci4;
	    ch[i__ - 1 + (k * 5 + 3) * ch_dim1] = tr2 + tr5;
	    ch[ic - 1 + (k * 5 + 2) * ch_dim1] = tr2 - tr5;
	    ch[i__ + (k * 5 + 3) * ch_dim1] = ti2 + ti5;
	    ch[ic + (k * 5 + 2) * ch_dim1] = ti5 - ti2;
	    ch[i__ - 1 + (k * 5 + 5) * ch_dim1] = tr3 + tr4;
	    ch[ic - 1 + (k * 5 + 4) * ch_dim1] = tr3 - tr4;
	    ch[i__ + (k * 5 + 5) * ch_dim1] = ti3 + ti4;
	    ch[ic + (k * 5 + 4) * ch_dim1] = ti4 - ti3;
/* L102: */
	}
/* L103: */
    }
    return 0;
L104:
    i__1 = *ido;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	ic = idp2 - i__;
/* DIR$ IVDEP */
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    dr2 = wa1[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * cc_dim1] 
		    + wa1[i__ - 1] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1];
	    di2 = wa1[i__ - 2] * cc[i__ + (k + (cc_dim2 << 1)) * cc_dim1] - 
		    wa1[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 1)) * 
		    cc_dim1];
	    dr3 = wa2[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1] + 
		    wa2[i__ - 1] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1];
	    di3 = wa2[i__ - 2] * cc[i__ + (k + cc_dim2 * 3) * cc_dim1] - wa2[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 3) * cc_dim1];
	    dr4 = wa3[i__ - 2] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * cc_dim1] 
		    + wa3[i__ - 1] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1];
	    di4 = wa3[i__ - 2] * cc[i__ + (k + (cc_dim2 << 2)) * cc_dim1] - 
		    wa3[i__ - 1] * cc[i__ - 1 + (k + (cc_dim2 << 2)) * 
		    cc_dim1];
	    dr5 = wa4[i__ - 2] * cc[i__ - 1 + (k + cc_dim2 * 5) * cc_dim1] + 
		    wa4[i__ - 1] * cc[i__ + (k + cc_dim2 * 5) * cc_dim1];
	    di5 = wa4[i__ - 2] * cc[i__ + (k + cc_dim2 * 5) * cc_dim1] - wa4[
		    i__ - 1] * cc[i__ - 1 + (k + cc_dim2 * 5) * cc_dim1];
	    cr2 = dr2 + dr5;
	    ci5 = dr5 - dr2;
	    cr5 = di2 - di5;
	    ci2 = di2 + di5;
	    cr3 = dr3 + dr4;
	    ci4 = dr4 - dr3;
	    cr4 = di3 - di4;
	    ci3 = di3 + di4;
	    ch[i__ - 1 + (k * 5 + 1) * ch_dim1] = cc[i__ - 1 + (k + cc_dim2) *
		     cc_dim1] + cr2 + cr3;
	    ch[i__ + (k * 5 + 1) * ch_dim1] = cc[i__ + (k + cc_dim2) * 
		    cc_dim1] + ci2 + ci3;
	    tr2 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + tr11 * cr2 + tr12 * 
		    cr3;
	    ti2 = cc[i__ + (k + cc_dim2) * cc_dim1] + tr11 * ci2 + tr12 * ci3;
	    tr3 = cc[i__ - 1 + (k + cc_dim2) * cc_dim1] + tr12 * cr2 + tr11 * 
		    cr3;
	    ti3 = cc[i__ + (k + cc_dim2) * cc_dim1] + tr12 * ci2 + tr11 * ci3;
	    tr5 = ti11 * cr5 + ti12 * cr4;
	    ti5 = ti11 * ci5 + ti12 * ci4;
	    tr4 = ti12 * cr5 - ti11 * cr4;
	    ti4 = ti12 * ci5 - ti11 * ci4;
	    ch[i__ - 1 + (k * 5 + 3) * ch_dim1] = tr2 + tr5;
	    ch[ic - 1 + (k * 5 + 2) * ch_dim1] = tr2 - tr5;
	    ch[i__ + (k * 5 + 3) * ch_dim1] = ti2 + ti5;
	    ch[ic + (k * 5 + 2) * ch_dim1] = ti5 - ti2;
	    ch[i__ - 1 + (k * 5 + 5) * ch_dim1] = tr3 + tr4;
	    ch[ic - 1 + (k * 5 + 4) * ch_dim1] = tr3 - tr4;
	    ch[i__ + (k * 5 + 5) * ch_dim1] = ti3 + ti4;
	    ch[ic + (k * 5 + 4) * ch_dim1] = ti4 - ti3;
/* L105: */
	}
/* L106: */
    }
    return 0;
} /* radf5_ */

/* Subroutine */ int radfg_(integer *ido, integer *ip, integer *l1, integer *
	idl1, doublereal *cc, doublereal *c1, doublereal *c2, doublereal *ch, doublereal *ch2, doublereal *wa)
{
    /* Initialized data */

    static doublereal tpi = 6.28318530717959f;

    /* System generated locals */
    integer ch_dim1, ch_dim2, ch_offset, cc_dim1, cc_dim2, cc_offset, c1_dim1,
	     c1_dim2, c1_offset, c2_dim1, c2_offset, ch2_dim1, ch2_offset, 
	    i__1, i__2, i__3;

    /* Builtin functions */
    double cos(doublereal), sin(doublereal);

    /* Local variables */
    static integer i__, j, k, l, j2, ic, jc, lc, ik, is;
    static doublereal dc2, ai1, ai2, ar1, ar2, ds2;
    static integer nbd;
    static doublereal dcp, arg, dsp, ar1h, ar2h;
    static integer idp2, ipp2, idij, ipph;

/* ***BEGIN PROLOGUE  RADFG */
/* ***REFER TO  RFFTF */
/* ***ROUTINES CALLED  (NONE) */
/* ***REVISION HISTORY  (YYMMDD) */
/*   000330  Modified array declarations.  (JEC) */

/* ***END PROLOGUE  RADFG */
    /* Parameter adjustments */
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2);
    ch -= ch_offset;
    c1_dim1 = *ido;
    c1_dim2 = *l1;
    c1_offset = 1 + c1_dim1 * (1 + c1_dim2);
    c1 -= c1_offset;
    cc_dim1 = *ido;
    cc_dim2 = *ip;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2);
    cc -= cc_offset;
    ch2_dim1 = *idl1;
    ch2_offset = 1 + ch2_dim1;
    ch2 -= ch2_offset;
    c2_dim1 = *idl1;
    c2_offset = 1 + c2_dim1;
    c2 -= c2_offset;
    --wa;

    /* Function Body */
/* ***FIRST EXECUTABLE STATEMENT  RADFG */
    arg = tpi / (doublereal) (*ip);
    dcp = cos(arg);
    dsp = sin(arg);
    ipph = (*ip + 1) / 2;
    ipp2 = *ip + 2;
    idp2 = *ido + 2;
    nbd = (*ido - 1) / 2;
    if (*ido == 1) {
	goto L119;
    }
    i__1 = *idl1;
    for (ik = 1; ik <= i__1; ++ik) {
	ch2[ik + ch2_dim1] = c2[ik + c2_dim1];
/* L101: */
    }
    i__1 = *ip;
    for (j = 2; j <= i__1; ++j) {
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    ch[(k + j * ch_dim2) * ch_dim1 + 1] = c1[(k + j * c1_dim2) * 
		    c1_dim1 + 1];
/* L102: */
	}
/* L103: */
    }
    if (nbd > *l1) {
	goto L107;
    }
    is = -(*ido);
    i__1 = *ip;
    for (j = 2; j <= i__1; ++j) {
	is += *ido;
	idij = is;
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    idij += 2;
	    i__3 = *l1;
	    for (k = 1; k <= i__3; ++k) {
		ch[i__ - 1 + (k + j * ch_dim2) * ch_dim1] = wa[idij - 1] * c1[
			i__ - 1 + (k + j * c1_dim2) * c1_dim1] + wa[idij] * 
			c1[i__ + (k + j * c1_dim2) * c1_dim1];
		ch[i__ + (k + j * ch_dim2) * ch_dim1] = wa[idij - 1] * c1[i__ 
			+ (k + j * c1_dim2) * c1_dim1] - wa[idij] * c1[i__ - 
			1 + (k + j * c1_dim2) * c1_dim1];
/* L104: */
	    }
/* L105: */
	}
/* L106: */
    }
    goto L111;
L107:
    is = -(*ido);
    i__1 = *ip;
    for (j = 2; j <= i__1; ++j) {
	is += *ido;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    idij = is;
/* DIR$ IVDEP */
	    i__3 = *ido;
	    for (i__ = 3; i__ <= i__3; i__ += 2) {
		idij += 2;
		ch[i__ - 1 + (k + j * ch_dim2) * ch_dim1] = wa[idij - 1] * c1[
			i__ - 1 + (k + j * c1_dim2) * c1_dim1] + wa[idij] * 
			c1[i__ + (k + j * c1_dim2) * c1_dim1];
		ch[i__ + (k + j * ch_dim2) * ch_dim1] = wa[idij - 1] * c1[i__ 
			+ (k + j * c1_dim2) * c1_dim1] - wa[idij] * c1[i__ - 
			1 + (k + j * c1_dim2) * c1_dim1];
/* L108: */
	    }
/* L109: */
	}
/* L110: */
    }
L111:
    if (nbd < *l1) {
	goto L115;
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
/* DIR$ IVDEP */
	    i__3 = *ido;
	    for (i__ = 3; i__ <= i__3; i__ += 2) {
		c1[i__ - 1 + (k + j * c1_dim2) * c1_dim1] = ch[i__ - 1 + (k + 
			j * ch_dim2) * ch_dim1] + ch[i__ - 1 + (k + jc * 
			ch_dim2) * ch_dim1];
		c1[i__ - 1 + (k + jc * c1_dim2) * c1_dim1] = ch[i__ + (k + j *
			 ch_dim2) * ch_dim1] - ch[i__ + (k + jc * ch_dim2) * 
			ch_dim1];
		c1[i__ + (k + j * c1_dim2) * c1_dim1] = ch[i__ + (k + j * 
			ch_dim2) * ch_dim1] + ch[i__ + (k + jc * ch_dim2) * 
			ch_dim1];
		c1[i__ + (k + jc * c1_dim2) * c1_dim1] = ch[i__ - 1 + (k + jc 
			* ch_dim2) * ch_dim1] - ch[i__ - 1 + (k + j * ch_dim2)
			 * ch_dim1];
/* L112: */
	    }
/* L113: */
	}
/* L114: */
    }
    goto L121;
L115:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    i__3 = *l1;
	    for (k = 1; k <= i__3; ++k) {
		c1[i__ - 1 + (k + j * c1_dim2) * c1_dim1] = ch[i__ - 1 + (k + 
			j * ch_dim2) * ch_dim1] + ch[i__ - 1 + (k + jc * 
			ch_dim2) * ch_dim1];
		c1[i__ - 1 + (k + jc * c1_dim2) * c1_dim1] = ch[i__ + (k + j *
			 ch_dim2) * ch_dim1] - ch[i__ + (k + jc * ch_dim2) * 
			ch_dim1];
		c1[i__ + (k + j * c1_dim2) * c1_dim1] = ch[i__ + (k + j * 
			ch_dim2) * ch_dim1] + ch[i__ + (k + jc * ch_dim2) * 
			ch_dim1];
		c1[i__ + (k + jc * c1_dim2) * c1_dim1] = ch[i__ - 1 + (k + jc 
			* ch_dim2) * ch_dim1] - ch[i__ - 1 + (k + j * ch_dim2)
			 * ch_dim1];
/* L116: */
	    }
/* L117: */
	}
/* L118: */
    }
    goto L121;
L119:
    i__1 = *idl1;
    for (ik = 1; ik <= i__1; ++ik) {
	c2[ik + c2_dim1] = ch2[ik + ch2_dim1];
/* L120: */
    }
L121:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    c1[(k + j * c1_dim2) * c1_dim1 + 1] = ch[(k + j * ch_dim2) * 
		    ch_dim1 + 1] + ch[(k + jc * ch_dim2) * ch_dim1 + 1];
	    c1[(k + jc * c1_dim2) * c1_dim1 + 1] = ch[(k + jc * ch_dim2) * 
		    ch_dim1 + 1] - ch[(k + j * ch_dim2) * ch_dim1 + 1];
/* L122: */
	}
/* L123: */
    }

    ar1 = 1.f;
    ai1 = 0.f;
    i__1 = ipph;
    for (l = 2; l <= i__1; ++l) {
	lc = ipp2 - l;
	ar1h = dcp * ar1 - dsp * ai1;
	ai1 = dcp * ai1 + dsp * ar1;
	ar1 = ar1h;
	i__2 = *idl1;
	for (ik = 1; ik <= i__2; ++ik) {
	    ch2[ik + l * ch2_dim1] = c2[ik + c2_dim1] + ar1 * c2[ik + (
		    c2_dim1 << 1)];
	    ch2[ik + lc * ch2_dim1] = ai1 * c2[ik + *ip * c2_dim1];
/* L124: */
	}
	dc2 = ar1;
	ds2 = ai1;
	ar2 = ar1;
	ai2 = ai1;
	i__2 = ipph;
	for (j = 3; j <= i__2; ++j) {
	    jc = ipp2 - j;
	    ar2h = dc2 * ar2 - ds2 * ai2;
	    ai2 = dc2 * ai2 + ds2 * ar2;
	    ar2 = ar2h;
	    i__3 = *idl1;
	    for (ik = 1; ik <= i__3; ++ik) {
		ch2[ik + l * ch2_dim1] += ar2 * c2[ik + j * c2_dim1];
		ch2[ik + lc * ch2_dim1] += ai2 * c2[ik + jc * c2_dim1];
/* L125: */
	    }
/* L126: */
	}
/* L127: */
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	i__2 = *idl1;
	for (ik = 1; ik <= i__2; ++ik) {
	    ch2[ik + ch2_dim1] += c2[ik + j * c2_dim1];
/* L128: */
	}
/* L129: */
    }

    if (*ido < *l1) {
	goto L132;
    }
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = *ido;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    cc[i__ + (k * cc_dim2 + 1) * cc_dim1] = ch[i__ + (k + ch_dim2) * 
		    ch_dim1];
/* L130: */
	}
/* L131: */
    }
    goto L135;
L132:
    i__1 = *ido;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    cc[i__ + (k * cc_dim2 + 1) * cc_dim1] = ch[i__ + (k + ch_dim2) * 
		    ch_dim1];
/* L133: */
	}
/* L134: */
    }
L135:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	j2 = j + j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
	    cc[*ido + (j2 - 2 + k * cc_dim2) * cc_dim1] = ch[(k + j * ch_dim2)
		     * ch_dim1 + 1];
	    cc[(j2 - 1 + k * cc_dim2) * cc_dim1 + 1] = ch[(k + jc * ch_dim2) *
		     ch_dim1 + 1];
/* L136: */
	}
/* L137: */
    }
    if (*ido == 1) {
	return 0;
    }
    if (nbd < *l1) {
	goto L141;
    }
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	j2 = j + j;
	i__2 = *l1;
	for (k = 1; k <= i__2; ++k) {
/* DIR$ IVDEP */
	    i__3 = *ido;
	    for (i__ = 3; i__ <= i__3; i__ += 2) {
		ic = idp2 - i__;
		cc[i__ - 1 + (j2 - 1 + k * cc_dim2) * cc_dim1] = ch[i__ - 1 + 
			(k + j * ch_dim2) * ch_dim1] + ch[i__ - 1 + (k + jc * 
			ch_dim2) * ch_dim1];
		cc[ic - 1 + (j2 - 2 + k * cc_dim2) * cc_dim1] = ch[i__ - 1 + (
			k + j * ch_dim2) * ch_dim1] - ch[i__ - 1 + (k + jc * 
			ch_dim2) * ch_dim1];
		cc[i__ + (j2 - 1 + k * cc_dim2) * cc_dim1] = ch[i__ + (k + j *
			 ch_dim2) * ch_dim1] + ch[i__ + (k + jc * ch_dim2) * 
			ch_dim1];
		cc[ic + (j2 - 2 + k * cc_dim2) * cc_dim1] = ch[i__ + (k + jc *
			 ch_dim2) * ch_dim1] - ch[i__ + (k + j * ch_dim2) * 
			ch_dim1];
/* L138: */
	    }
/* L139: */
	}
/* L140: */
    }
    return 0;
L141:
    i__1 = ipph;
    for (j = 2; j <= i__1; ++j) {
	jc = ipp2 - j;
	j2 = j + j;
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    i__3 = *l1;
	    for (k = 1; k <= i__3; ++k) {
		cc[i__ - 1 + (j2 - 1 + k * cc_dim2) * cc_dim1] = ch[i__ - 1 + 
			(k + j * ch_dim2) * ch_dim1] + ch[i__ - 1 + (k + jc * 
			ch_dim2) * ch_dim1];
		cc[ic - 1 + (j2 - 2 + k * cc_dim2) * cc_dim1] = ch[i__ - 1 + (
			k + j * ch_dim2) * ch_dim1] - ch[i__ - 1 + (k + jc * 
			ch_dim2) * ch_dim1];
		cc[i__ + (j2 - 1 + k * cc_dim2) * cc_dim1] = ch[i__ + (k + j *
			 ch_dim2) * ch_dim1] + ch[i__ + (k + jc * ch_dim2) * 
			ch_dim1];
		cc[ic + (j2 - 2 + k * cc_dim2) * cc_dim1] = ch[i__ + (k + jc *
			 ch_dim2) * ch_dim1] - ch[i__ + (k + j * ch_dim2) * 
			ch_dim1];
/* L142: */
	    }
/* L143: */
	}
/* L144: */
    }
    return 0;
} /* radfg_ */

