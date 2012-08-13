/* hwscrt.f -- translated by f2c (version 20090411).
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

/* Table of constant values */

static integer c__1 = 1;
static integer c__2 = 2;
static integer c__0 = 0;
static doublereal c_b105 = .5;
static doublereal c_b106 = 0.;
static doublereal c_b314 = 1.;

/* Subroutine */ int hwscrt_(doublereal *a, doublereal *b, integer *m, 
	integer *mbdcnd, doublereal *bda, doublereal *bdb, doublereal *c__, 
	doublereal *d__, integer *n, integer *nbdcnd, doublereal *bdc, 
	doublereal *bdd, doublereal *elmbda, doublereal *f, integer *idimf, 
	doublereal *pertrb, integer *ierror, doublereal *w)
{
    /* System generated locals */
    integer f_dim1, f_offset, i__1, i__2;
    doublereal d__1;

    /* Local variables */
    static integer i__, j;
    static doublereal s, a1, a2, s1;
    static integer mp, np, id2, id3, id4, mp1, np1;
    static doublereal st2;
    static integer msp1, nsp1, munk, nunk, ierr1, mstm1, nstm1, mskip, nskip, 
	    mstop, nstop;
    extern /* Subroutine */ int genbun_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *);
    static doublereal deltax, deltay;
    static integer mperod, nperod;
    static doublereal delxsq, delysq, twdelx, twdely;
    static integer nstart, mstart;



/*       DOUBLE PRECISION VERSION  - LOCAL MODIFICATION */






/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*     *                                                               * */
/*     *                        f i s h p a k                          * */
/*     *                                                               * */
/*     *                                                               * */
/*     *     a package of fortran subprograms for the solution of      * */
/*     *                                                               * */
/*     *      separable elliptic partial differential equations        * */
/*     *                                                               * */
/*     *                  (version 3.1 , october 1980)                 * */
/*     *                                                               * */
/*     *                             by                                * */
/*     *                                                               * */
/*     *        john adams, paul swarztrauber and roland sweet         * */
/*     *                                                               * */
/*     *                             of                                * */
/*     *                                                               * */
/*     *         the national center for atmospheric research          * */
/*     *                                                               * */
/*     *                boulder, colorado  (80307)  u.s.a.             * */
/*     *                                                               * */
/*     *                   which is sponsored by                       * */
/*     *                                                               * */
/*     *              the national science foundation                  * */
/*     *                                                               * */
/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


/*     * * * * * * * * *  purpose    * * * * * * * * * * * * * * * * * * */

/*          subroutine hwscrt solves the standard five-point finite */
/*     difference approximation to the helmholtz equation in cartesian */
/*     coordinates: */

/*          (d/dx)(du/dx) + (d/dy)(du/dy) + lambda*u = f(x,y). */



/*     * * * * * * * *    parameter description     * * * * * * * * * * */

/*             * * * * * *   on input    * * * * * * */

/*     a,b */
/*       the range of x, i.e., a .le. x .le. b.  a must be less than b. */

/*     m */
/*       the number of panels into which the interval (a,b) is */
/*       subdivided.  hence, there will be m+1 grid points in the */
/*       x-direction given by x(i) = a+(i-1)dx for i = 1,2,...,m+1, */
/*       where dx = (b-a)/m is the panel width. m must be greater than 3. */

/*     mbdcnd */
/*       indicates the type of boundary conditions at x = a and x = b. */

/*       = 0  if the solution is periodic in x, i.e., u(i,j) = u(m+i,j). */
/*       = 1  if the solution is specified at x = a and x = b. */
/*       = 2  if the solution is specified at x = a and the derivative of */
/*            the solution with respect to x is specified at x = b. */
/*       = 3  if the derivative of the solution with respect to x is */
/*            specified at x = a and x = b. */
/*       = 4  if the derivative of the solution with respect to x is */
/*            specified at x = a and the solution is specified at x = b. */

/*     bda */
/*       a one-dimensional array of length n+1 that specifies the values */
/*       of the derivative of the solution with respect to x at x = a. */
/*       when mbdcnd = 3 or 4, */

/*            bda(j) = (d/dx)u(a,y(j)), j = 1,2,...,n+1  . */

/*       when mbdcnd has any other value, bda is a dummy variable. */

/*     bdb */
/*       a one-dimensional array of length n+1 that specifies the values */
/*       of the derivative of the solution with respect to x at x = b. */
/*       when mbdcnd = 2 or 3, */

/*            bdb(j) = (d/dx)u(b,y(j)), j = 1,2,...,n+1  . */

/*       when mbdcnd has any other value bdb is a dummy variable. */

/*     c,d */
/*       the range of y, i.e., c .le. y .le. d.  c must be less than d. */

/*     n */
/*       the number of panels into which the interval (c,d) is */
/*       subdivided.  hence, there will be n+1 grid points in the */
/*       y-direction given by y(j) = c+(j-1)dy for j = 1,2,...,n+1, where */
/*       dy = (d-c)/n is the panel width.  n must be greater than 3. */

/*     nbdcnd */
/*       indicates the type of boundary conditions at y = c and y = d. */

/*       = 0  if the solution is periodic in y, i.e., u(i,j) = u(i,n+j). */
/*       = 1  if the solution is specified at y = c and y = d. */
/*       = 2  if the solution is specified at y = c and the derivative of */
/*            the solution with respect to y is specified at y = d. */
/*       = 3  if the derivative of the solution with respect to y is */
/*            specified at y = c and y = d. */
/*       = 4  if the derivative of the solution with respect to y is */
/*            specified at y = c and the solution is specified at y = d. */

/*     bdc */
/*       a one-dimensional array of length m+1 that specifies the values */
/*       of the derivative of the solution with respect to y at y = c. */
/*       when nbdcnd = 3 or 4, */

/*            bdc(i) = (d/dy)u(x(i),c), i = 1,2,...,m+1  . */

/*       when nbdcnd has any other value, bdc is a dummy variable. */

/*     bdd */
/*       a one-dimensional array of length m+1 that specifies the values */
/*       of the derivative of the solution with respect to y at y = d. */
/*       when nbdcnd = 2 or 3, */

/*            bdd(i) = (d/dy)u(x(i),d), i = 1,2,...,m+1  . */

/*       when nbdcnd has any other value, bdd is a dummy variable. */

/*     elmbda */
/*       the constant lambda in the helmholtz equation.  if */
/*       lambda .gt. 0, a solution may not exist.  however, hwscrt will */
/*       attempt to find a solution. */

/*     f */
/*       a two-dimensional array which specifies the values of the right */
/*       side of the helmholtz equation and boundary values (if any). */
/*       for i = 2,3,...,m and j = 2,3,...,n */

/*            f(i,j) = f(x(i),y(j)). */

/*       on the boundaries f is defined by */

/*            mbdcnd     f(1,j)        f(m+1,j) */
/*            ------     ---------     -------- */

/*              0        f(a,y(j))     f(a,y(j)) */
/*              1        u(a,y(j))     u(b,y(j)) */
/*              2        u(a,y(j))     f(b,y(j))     j = 1,2,...,n+1 */
/*              3        f(a,y(j))     f(b,y(j)) */
/*              4        f(a,y(j))     u(b,y(j)) */


/*            nbdcnd     f(i,1)        f(i,n+1) */
/*            ------     ---------     -------- */

/*              0        f(x(i),c)     f(x(i),c) */
/*              1        u(x(i),c)     u(x(i),d) */
/*              2        u(x(i),c)     f(x(i),d)     i = 1,2,...,m+1 */
/*              3        f(x(i),c)     f(x(i),d) */
/*              4        f(x(i),c)     u(x(i),d) */

/*       f must be dimensioned at least (m+1)*(n+1). */

/*       note */

/*       if the table calls for both the solution u and the right side f */
/*       at  a corner then the solution must be specified. */

/*     idimf */
/*       the row (or first) dimension of the array f as it appears in the */
/*       program calling hwscrt.  this parameter is used to specify the */
/*       variable dimension of f.  idimf must be at least m+1  . */

/*     w */
/*       a one-dimensional array that must be provided by the user for */
/*       work space.  w may require up to 4*(n+1) + */
/*       (13 + int(log2(n+1)))*(m+1) locations.  the actual number of */
/*       locations used is computed by hwscrt and is returned in location */
/*       w(1). */


/*             * * * * * *   on output     * * * * * * */

/*     f */
/*       contains the solution u(i,j) of the finite difference */
/*       approximation for the grid point (x(i),y(j)), i = 1,2,...,m+1, */
/*       j = 1,2,...,n+1  . */

/*     pertrb */
/*       if a combination of periodic or derivative boundary conditions */
/*       is specified for a poisson equation (lambda = 0), a solution may */
/*       not exist.  pertrb is a constant, calculated and subtracted from */
/*       f, which ensures that a solution exists.  hwscrt then computes */
/*       this solution, which is a least squares solution to the original */
/*       approximation.  this solution plus any constant is also a */
/*       solution.  hence, the solution is not unique.  the value of */
/*       pertrb should be small compared to the right side f.  otherwise, */
/*       a solution is obtained to an essentially different problem. */
/*       this comparison should always be made to insure that a */
/*       meaningful solution has been obtained. */

/*     ierror */
/*       an error flag that indicates invalid input parameters.  except */
/*       for numbers 0 and 6, a solution is not attempted. */

/*       = 0  no error. */
/*       = 1  a .ge. b. */
/*       = 2  mbdcnd .lt. 0 or mbdcnd .gt. 4  . */
/*       = 3  c .ge. d. */
/*       = 4  n .le. 3 */
/*       = 5  nbdcnd .lt. 0 or nbdcnd .gt. 4  . */
/*       = 6  lambda .gt. 0  . */
/*       = 7  idimf .lt. m+1  . */
/*       = 8  m .le. 3 */

/*       since this is the only means of indicating a possibly incorrect */
/*       call to hwscrt, the user should test ierror after the call. */

/*     w */
/*       w(1) contains the required length of w. */


/*     * * * * * * *   program specifications    * * * * * * * * * * * * */


/*     dimension of   bda(n+1),bdb(n+1),bdc(m+1),bdd(m+1),f(idimf,n+1), */
/*     arguments      w(see argument list) */

/*     latest         june 1, 1976 */
/*     revision */

/*     subprograms    hwscrt,genbun,poisd2,poisn2,poisp2,cosgen,merge, */
/*     required       trix,tri3,pimach */

/*     special        none */
/*     conditions */

/*     common         none */
/*     blocks */

/*     i/o            none */

/*     precision      single */

/*     specialist     roland sweet */

/*     language       fortran */

/*     history        standardized september 1, 1973 */
/*                    revised april 1, 1976 */

/*     algorithm      the routine defines the finite difference */
/*                    equations, incorporates boundary data, and adjusts */
/*                    the right side of singular systems and then calls */
/*                    genbun to solve the system. */

/*     space          13110(octal) = 5704(decimal) locations on the ncar */
/*     required       control data 7600 */

/*     timing and        the execution time t on the ncar control data */
/*     accuracy       7600 for subroutine hwscrt is roughly proportional */
/*                    to m*n*log2(n), but also depends on the input */
/*                    parameters nbdcnd and mbdcnd.  some typical values */
/*                    are listed in the table below. */
/*                       the solution process employed results in a loss */
/*                    of no more than three significant digits for n and */
/*                    m as large as 64.  more detailed information about */
/*                    accuracy can be found in the documentation for */
/*                    subroutine genbun which is the routine that */
/*                    solves the finite difference equations. */


/*                       m(=n)    mbdcnd    nbdcnd    t(msecs) */
/*                       -----    ------    ------    -------- */

/*                        32        0         0          31 */
/*                        32        1         1          23 */
/*                        32        3         3          36 */
/*                        64        0         0         128 */
/*                        64        1         1          96 */
/*                        64        3         3         142 */

/*     portability    american national standards institute fortran. */
/*                    all machine dependent constants are located in the */
/*                    function pimach. */

/*     reference      swarztrauber,p. and r. sweet, 'efficient fortran */
/*                    subprograms for the solution of elliptic equations' */
/*                    ncar tn/ia-109, july, 1975, 138 pp. */

/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */



/*     check for invalid parameters. */

    /* Parameter adjustments */
    --bda;
    --bdb;
    --bdc;
    --bdd;
    f_dim1 = *idimf;
    f_offset = 1 + f_dim1;
    f -= f_offset;
    --w;

    /* Function Body */
    *ierror = 0;
    if (*a >= *b) {
	*ierror = 1;
    }
    if (*mbdcnd < 0 || *mbdcnd > 4) {
	*ierror = 2;
    }
    if (*c__ >= *d__) {
	*ierror = 3;
    }
    if (*n <= 3) {
	*ierror = 4;
    }
    if (*nbdcnd < 0 || *nbdcnd > 4) {
	*ierror = 5;
    }
    if (*idimf < *m + 1) {
	*ierror = 7;
    }
    if (*m <= 3) {
	*ierror = 8;
    }
    if (*ierror != 0) {
	return 0;
    }
    nperod = *nbdcnd;
    mperod = 0;
    if (*mbdcnd > 0) {
	mperod = 1;
    }
    deltax = (*b - *a) / (doublereal) (*m);
    twdelx = 2. / deltax;
/* Computing 2nd power */
    d__1 = deltax;
    delxsq = 1. / (d__1 * d__1);
    deltay = (*d__ - *c__) / (doublereal) (*n);
    twdely = 2. / deltay;
/* Computing 2nd power */
    d__1 = deltay;
    delysq = 1. / (d__1 * d__1);
    np = *nbdcnd + 1;
    np1 = *n + 1;
    mp = *mbdcnd + 1;
    mp1 = *m + 1;
    nstart = 1;
    nstop = *n;
    nskip = 1;
    switch (np) {
	case 1:  goto L104;
	case 2:  goto L101;
	case 3:  goto L102;
	case 4:  goto L103;
	case 5:  goto L104;
    }
L101:
    nstart = 2;
    goto L104;
L102:
    nstart = 2;
L103:
    nstop = np1;
    nskip = 2;
L104:
    nunk = nstop - nstart + 1;

/*     enter boundary data for x-boundaries. */

    mstart = 1;
    mstop = *m;
    mskip = 1;
    switch (mp) {
	case 1:  goto L117;
	case 2:  goto L105;
	case 3:  goto L106;
	case 4:  goto L109;
	case 5:  goto L110;
    }
L105:
    mstart = 2;
    goto L107;
L106:
    mstart = 2;
    mstop = mp1;
    mskip = 2;
L107:
    i__1 = nstop;
    for (j = nstart; j <= i__1; ++j) {
	f[j * f_dim1 + 2] -= f[j * f_dim1 + 1] * delxsq;
/* L108: */
    }
    goto L112;
L109:
    mstop = mp1;
    mskip = 2;
L110:
    i__1 = nstop;
    for (j = nstart; j <= i__1; ++j) {
	f[j * f_dim1 + 1] += bda[j] * twdelx;
/* L111: */
    }
L112:
    switch (mskip) {
	case 1:  goto L113;
	case 2:  goto L115;
    }
L113:
    i__1 = nstop;
    for (j = nstart; j <= i__1; ++j) {
	f[*m + j * f_dim1] -= f[mp1 + j * f_dim1] * delxsq;
/* L114: */
    }
    goto L117;
L115:
    i__1 = nstop;
    for (j = nstart; j <= i__1; ++j) {
	f[mp1 + j * f_dim1] -= bdb[j] * twdelx;
/* L116: */
    }
L117:
    munk = mstop - mstart + 1;

/*     enter boundary data for y-boundaries. */

    switch (np) {
	case 1:  goto L127;
	case 2:  goto L118;
	case 3:  goto L118;
	case 4:  goto L120;
	case 5:  goto L120;
    }
L118:
    i__1 = mstop;
    for (i__ = mstart; i__ <= i__1; ++i__) {
	f[i__ + (f_dim1 << 1)] -= f[i__ + f_dim1] * delysq;
/* L119: */
    }
    goto L122;
L120:
    i__1 = mstop;
    for (i__ = mstart; i__ <= i__1; ++i__) {
	f[i__ + f_dim1] += bdc[i__] * twdely;
/* L121: */
    }
L122:
    switch (nskip) {
	case 1:  goto L123;
	case 2:  goto L125;
    }
L123:
    i__1 = mstop;
    for (i__ = mstart; i__ <= i__1; ++i__) {
	f[i__ + *n * f_dim1] -= f[i__ + np1 * f_dim1] * delysq;
/* L124: */
    }
    goto L127;
L125:
    i__1 = mstop;
    for (i__ = mstart; i__ <= i__1; ++i__) {
	f[i__ + np1 * f_dim1] -= bdd[i__] * twdely;
/* L126: */
    }

/*    multiply right side by deltay**2. */

L127:
    delysq = deltay * deltay;
    i__1 = mstop;
    for (i__ = mstart; i__ <= i__1; ++i__) {
	i__2 = nstop;
	for (j = nstart; j <= i__2; ++j) {
	    f[i__ + j * f_dim1] *= delysq;
/* L128: */
	}
/* L129: */
    }

/*     define the a,b,c coefficients in w-array. */

    id2 = munk;
    id3 = id2 + munk;
    id4 = id3 + munk;
    s = delysq * delxsq;
    st2 = s * 2.;
    i__1 = munk;
    for (i__ = 1; i__ <= i__1; ++i__) {
	w[i__] = s;
	j = id2 + i__;
	w[j] = -st2 + *elmbda * delysq;
	j = id3 + i__;
	w[j] = s;
/* L130: */
    }
    if (mp == 1) {
	goto L131;
    }
    w[1] = 0.;
    w[id4] = 0.;
L131:
    switch (mp) {
	case 1:  goto L135;
	case 2:  goto L135;
	case 3:  goto L132;
	case 4:  goto L133;
	case 5:  goto L134;
    }
L132:
    w[id2] = st2;
    goto L135;
L133:
    w[id2] = st2;
L134:
    w[id3 + 1] = st2;
L135:
    *pertrb = 0.;
    if (*elmbda < 0.) {
	goto L144;
    } else if (*elmbda == 0) {
	goto L137;
    } else {
	goto L136;
    }
L136:
    *ierror = 6;
    goto L144;
L137:
    if ((*nbdcnd == 0 || *nbdcnd == 3) && (*mbdcnd == 0 || *mbdcnd == 3)) {
	goto L138;
    }
    goto L144;

/*     for singular problems must adjust data to insure that a solution */
/*     will exist. */

L138:
    a1 = 1.f;
    a2 = 1.f;
    if (*nbdcnd == 3) {
	a2 = 2.f;
    }
    if (*mbdcnd == 3) {
	a1 = 2.f;
    }
    s1 = 0.;
    msp1 = mstart + 1;
    mstm1 = mstop - 1;
    nsp1 = nstart + 1;
    nstm1 = nstop - 1;
    i__1 = nstm1;
    for (j = nsp1; j <= i__1; ++j) {
	s = 0.;
	i__2 = mstm1;
	for (i__ = msp1; i__ <= i__2; ++i__) {
	    s += f[i__ + j * f_dim1];
/* L139: */
	}
	s1 = s1 + s * a1 + f[mstart + j * f_dim1] + f[mstop + j * f_dim1];
/* L140: */
    }
    s1 = a2 * s1;
    s = 0.;
    i__1 = mstm1;
    for (i__ = msp1; i__ <= i__1; ++i__) {
	s = s + f[i__ + nstart * f_dim1] + f[i__ + nstop * f_dim1];
/* L141: */
    }
    s1 = s1 + s * a1 + f[mstart + nstart * f_dim1] + f[mstart + nstop * 
	    f_dim1] + f[mstop + nstart * f_dim1] + f[mstop + nstop * f_dim1];
    s = ((doublereal) (nunk - 2) * a2 + 2.) * ((doublereal) (munk - 2) * a1 + 
	    2.);
    *pertrb = s1 / s;
    i__1 = nstop;
    for (j = nstart; j <= i__1; ++j) {
	i__2 = mstop;
	for (i__ = mstart; i__ <= i__2; ++i__) {
	    f[i__ + j * f_dim1] -= *pertrb;
/* L142: */
	}
/* L143: */
    }
    *pertrb /= delysq;

/*     solve the equation. */

L144:
    genbun_(&nperod, &nunk, &mperod, &munk, &w[1], &w[id2 + 1], &w[id3 + 1], 
	    idimf, &f[mstart + nstart * f_dim1], &ierr1, &w[id4 + 1]);
    w[1] = w[id4 + 1] + (doublereal) munk * 3.;

/*     fill in identical values when have periodic boundary conditions. */

    if (*nbdcnd != 0) {
	goto L146;
    }
    i__1 = mstop;
    for (i__ = mstart; i__ <= i__1; ++i__) {
	f[i__ + np1 * f_dim1] = f[i__ + f_dim1];
/* L145: */
    }
L146:
    if (*mbdcnd != 0) {
	goto L148;
    }
    i__1 = nstop;
    for (j = nstart; j <= i__1; ++j) {
	f[mp1 + j * f_dim1] = f[j * f_dim1 + 1];
/* L147: */
    }
    if (*nbdcnd == 0) {
	f[mp1 + np1 * f_dim1] = f[np1 * f_dim1 + 1];
    }
L148:
    return 0;
} /* hwscrt_ */

/* Subroutine */ int genbun_(integer *nperod, integer *n, integer *mperod, 
	integer *m, doublereal *a, doublereal *b, doublereal *c__, integer *
	idimy, doublereal *y, integer *ierror, doublereal *w)
{
    /* System generated locals */
    integer y_dim1, y_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, k;
    static doublereal a1;
    static integer mh, mp, np, mp1, iwd, iwp, mhm1, iwb2, iwb3, nby2, iww1, 
	    iww2, iww3, iwba, iwbb, iwbc, modd, mhmi, mhpi, irev, mskip;
    extern /* Subroutine */ int poisd2_(integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *,
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), poisn2_(integer *, integer *, integer *, integer *,
	     doublereal *, doublereal *, doublereal *, doublereal *, integer *
	    , doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), poisp2_(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *);
    static integer iwtcos, ipstor;





/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*     *                                                               * */
/*     *                        f i s h p a k                          * */
/*     *                                                               * */
/*     *                                                               * */
/*     *     a package of fortran subprograms for the solution of      * */
/*     *                                                               * */
/*     *      separable elliptic partial differential equations        * */
/*     *                                                               * */
/*     *                  (version 3.1 , october 1980)                  * */
/*     *                                                               * */
/*     *                             by                                * */
/*     *                                                               * */
/*     *        john adams, paul swarztrauber and roland sweet         * */
/*     *                                                               * */
/*     *                             of                                * */
/*     *                                                               * */
/*     *         the national center for atmospheric research          * */
/*     *                                                               * */
/*     *                boulder, colorado  (80307)  u.s.a.             * */
/*     *                                                               * */
/*     *                   which is sponsored by                       * */
/*     *                                                               * */
/*     *              the national science foundation                  * */
/*     *                                                               * */
/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


/*     * * * * * * * * *  purpose    * * * * * * * * * * * * * * * * * * */


/*     subroutine genbun solves the linear system of equations */

/*          a(i)*x(i-1,j) + b(i)*x(i,j) + c(i)*x(i+1,j) */

/*          + x(i,j-1) - 2.*x(i,j) + x(i,j+1) = y(i,j) */

/*               for i = 1,2,...,m  and  j = 1,2,...,n. */

/*     the indices i+1 and i-1 are evaluated modulo m, i.e., */
/*     x(0,j) = x(m,j) and x(m+1,j) = x(1,j), and x(i,0) may be equal to */
/*     0, x(i,2), or x(i,n) and x(i,n+1) may be equal to 0, x(i,n-1), or */
/*     x(i,1) depending on an input parameter. */


/*     * * * * * * * *    parameter description     * * * * * * * * * * */

/*             * * * * * *   on input    * * * * * * */

/*     nperod */
/*       indicates the values that x(i,0) and x(i,n+1) are assumed to */
/*       have. */

/*       = 0  if x(i,0) = x(i,n) and x(i,n+1) = x(i,1). */
/*       = 1  if x(i,0) = x(i,n+1) = 0  . */
/*       = 2  if x(i,0) = 0 and x(i,n+1) = x(i,n-1). */
/*       = 3  if x(i,0) = x(i,2) and x(i,n+1) = x(i,n-1). */
/*       = 4  if x(i,0) = x(i,2) and x(i,n+1) = 0. */

/*     n */
/*       the number of unknowns in the j-direction.  n must be greater */
/*       than 2. */

/*     mperod */
/*       = 0 if a(1) and c(m) are not zero */
/*       = 1 if a(1) = c(m) = 0 */

/*     m */
/*       the number of unknowns in the i-direction.  m must be greater */
/*       than 2. */

/*     a,b,c */
/*       one-dimensional arrays of length m that specify the */
/*       coefficients in the linear equations given above.  if mperod = 0 */
/*       the array elements must not depend upon the index i, but must be */
/*       constant.  specifically, the subroutine checks the following */
/*       condition */

/*             a(i) = c(1) */
/*             c(i) = c(1) */
/*             b(i) = b(1) */

/*       for i=1,2,...,m. */

/*     idimy */
/*       the row (or first) dimension of the two-dimensional array y as */
/*       it appears in the program calling genbun.  this parameter is */
/*       used to specify the variable dimension of y.  idimy must be at */
/*       least m. */

/*     y */
/*       a two-dimensional array that specifies the values of the right */
/*       side of the linear system of equations given above.  y must be */
/*       dimensioned at least m*n. */

/*     w */
/*       a one-dimensional array that must be provided by the user for */
/*       work space.  w may require up to 4*n + (10 + int(log2(n)))*m */
/*       locations.  the actual number of locations used is computed by */
/*       genbun and is returned in location w(1). */


/*             * * * * * *   on output     * * * * * * */

/*     y */
/*       contains the solution x. */

/*     ierror */
/*       an error flag that indicates invalid input parameters  except */
/*       for number zero, a solution is not attempted. */

/*       = 0  no error. */
/*       = 1  m .le. 2  . */
/*       = 2  n .le. 2 */
/*       = 3  idimy .lt. m */
/*       = 4  nperod .lt. 0 or nperod .gt. 4 */
/*       = 5  mperod .lt. 0 or mperod .gt. 1 */
/*       = 6  a(i) .ne. c(1) or c(i) .ne. c(1) or b(i) .ne. b(1) for */
/*            some i=1,2,...,m. */
/*       = 7  a(1) .ne. 0 or c(m) .ne. 0 and mperod = 1 */

/*     w */
/*       w(1) contains the required length of w. */

/*     * * * * * * *   program specifications    * * * * * * * * * * * * */

/*     dimension of   a(m),b(m),c(m),y(idimy,n),w(see parameter list) */
/*     arguments */

/*     latest         june 1, 1976 */
/*     revision */

/*     subprograms    genbun,poisd2,poisn2,poisp2,cosgen,merge,trix,tri3, */
/*     required       pimach */

/*     special        none */
/*     conditions */

/*     common         none */
/*     blocks */

/*     i/o            none */

/*     precision      single */

/*     specialist     roland sweet */

/*     language       fortran */

/*     history        standardized april 1, 1973 */
/*                    revised august 20,1973 */
/*                    revised january 1, 1976 */

/*     algorithm      the linear system is solved by a cyclic reduction */
/*                    algorithm described in the reference. */

/*     space          4944(decimal) = 11520(octal) locations on the ncar */
/*     required       control data 7600 */

/*     timing and        the execution time t on the ncar control data */
/*     accuracy       7600 for subroutine genbun is roughly proportional */
/*                    to m*n*log2(n), but also depends on the input */
/*                    parameter nperod.  some typical values are listed */
/*                    in the table below.  more comprehensive timing */
/*                    charts may be found in the reference. */
/*                       to measure the accuracy of the algorithm a */
/*                    uniform random number generator was used to create */
/*                    a solution array x for the system given in the */
/*                    'purpose' with */

/*                       a(i) = c(i) = -0.5*b(i) = 1,       i=1,2,...,m */

/*                    and, when mperod = 1 */

/*                       a(1) = c(m) = 0 */
/*                       a(m) = c(1) = 2. */

/*                    the solution x was substituted into the given sys- */
/*                    tem and, using double precision, a right side y was */
/*                    computed.  using this array y subroutine genbun was */
/*                    called to produce an approximate solution z.  then */
/*                    the relative error, defined as */

/*                       e = max(abs(z(i,j)-x(i,j)))/max(abs(x(i,j))) */

/*                    where the two maxima are taken over all i=1,2,...,m */
/*                    and j=1,2,...,n, was computed.  the value of e is */
/*                    given in the table below for some typical values of */
/*                    m and n. */


/*                       m (=n)    mperod    nperod    t(msecs)    e */
/*                       ------    ------    ------    --------  ------ */

/*                         31        0         0          36     6.e-14 */
/*                         31        1         1          21     4.e-13 */
/*                         31        1         3          41     3.e-13 */
/*                         32        0         0          29     9.e-14 */
/*                         32        1         1          32     3.e-13 */
/*                         32        1         3          48     1.e-13 */
/*                         33        0         0          36     9.e-14 */
/*                         33        1         1          30     4.e-13 */
/*                         33        1         3          34     1.e-13 */
/*                         63        0         0         150     1.e-13 */
/*                         63        1         1          91     1.e-12 */
/*                         63        1         3         173     2.e-13 */
/*                         64        0         0         122     1.e-13 */
/*                         64        1         1         128     1.e-12 */
/*                         64        1         3         199     6.e-13 */
/*                         65        0         0         143     2.e-13 */
/*                         65        1         1         120     1.e-12 */
/*                         65        1         3         138     4.e-13 */

/*     portability    american national standards institue fortran. */
/*                    all machine dependent constants are located in the */
/*                    function pimach. */

/*     required       cos */
/*     resident */
/*     routines */

/*     reference      sweet, r., 'a cyclic reduction algorithm for */
/*                    solving block tridiagonal systems of arbitrary */
/*                    dimensions,' siam j. on numer. anal., */
/*                    14(sept., 1977), pp. 706-720. */

/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


    /* Parameter adjustments */
    --a;
    --b;
    --c__;
    y_dim1 = *idimy;
    y_offset = 1 + y_dim1;
    y -= y_offset;
    --w;

    /* Function Body */
    *ierror = 0;
    if (*m <= 2) {
	*ierror = 1;
    }
    if (*n <= 2) {
	*ierror = 2;
    }
    if (*idimy < *m) {
	*ierror = 3;
    }
    if (*nperod < 0 || *nperod > 4) {
	*ierror = 4;
    }
    if (*mperod < 0 || *mperod > 1) {
	*ierror = 5;
    }
    if (*mperod == 1) {
	goto L102;
    }
    i__1 = *m;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if (a[i__] != c__[1]) {
	    goto L103;
	}
	if (c__[i__] != c__[1]) {
	    goto L103;
	}
	if (b[i__] != b[1]) {
	    goto L103;
	}
/* L101: */
    }
    goto L104;
L102:
    if (a[1] != 0.f || c__[*m] != 0.f) {
	*ierror = 7;
    }
    goto L104;
L103:
    *ierror = 6;
L104:
    if (*ierror != 0) {
	return 0;
    }
    mp1 = *m + 1;
    iwba = mp1;
    iwbb = iwba + *m;
    iwbc = iwbb + *m;
    iwb2 = iwbc + *m;
    iwb3 = iwb2 + *m;
    iww1 = iwb3 + *m;
    iww2 = iww1 + *m;
    iww3 = iww2 + *m;
    iwd = iww3 + *m;
    iwtcos = iwd + *m;
    iwp = iwtcos + (*n << 2);
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	k = iwba + i__ - 1;
	w[k] = -a[i__];
	k = iwbc + i__ - 1;
	w[k] = -c__[i__];
	k = iwbb + i__ - 1;
	w[k] = 2.f - b[i__];
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    y[i__ + j * y_dim1] = -y[i__ + j * y_dim1];
/* L105: */
	}
/* L106: */
    }
    mp = *mperod + 1;
    np = *nperod + 1;
    switch (mp) {
	case 1:  goto L114;
	case 2:  goto L107;
    }
L107:
    switch (np) {
	case 1:  goto L108;
	case 2:  goto L109;
	case 3:  goto L110;
	case 4:  goto L111;
	case 5:  goto L123;
    }
L108:
    poisp2_(m, n, &w[iwba], &w[iwbb], &w[iwbc], &y[y_offset], idimy, &w[1], &
	    w[iwb2], &w[iwb3], &w[iww1], &w[iww2], &w[iww3], &w[iwd], &w[
	    iwtcos], &w[iwp]);
    goto L112;
L109:
    poisd2_(m, n, &c__1, &w[iwba], &w[iwbb], &w[iwbc], &y[y_offset], idimy, &
	    w[1], &w[iww1], &w[iwd], &w[iwtcos], &w[iwp]);
    goto L112;
L110:
    poisn2_(m, n, &c__1, &c__2, &w[iwba], &w[iwbb], &w[iwbc], &y[y_offset], 
	    idimy, &w[1], &w[iwb2], &w[iwb3], &w[iww1], &w[iww2], &w[iww3], &
	    w[iwd], &w[iwtcos], &w[iwp]);
    goto L112;
L111:
    poisn2_(m, n, &c__1, &c__1, &w[iwba], &w[iwbb], &w[iwbc], &y[y_offset], 
	    idimy, &w[1], &w[iwb2], &w[iwb3], &w[iww1], &w[iww2], &w[iww3], &
	    w[iwd], &w[iwtcos], &w[iwp]);
L112:
    ipstor = (integer) w[iww1];
    irev = 2;
    if (*nperod == 4) {
	goto L124;
    }
L113:
    switch (mp) {
	case 1:  goto L127;
	case 2:  goto L133;
    }
L114:

/*     reorder unknowns when mp =0 */

    mh = (*m + 1) / 2;
    mhm1 = mh - 1;
    modd = 1;
    if (mh << 1 == *m) {
	modd = 2;
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = mhm1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    mhpi = mh + i__;
	    mhmi = mh - i__;
	    w[i__] = y[mhmi + j * y_dim1] - y[mhpi + j * y_dim1];
	    w[mhpi] = y[mhmi + j * y_dim1] + y[mhpi + j * y_dim1];
/* L115: */
	}
	w[mh] = y[mh + j * y_dim1] * 2.;
	switch (modd) {
	    case 1:  goto L117;
	    case 2:  goto L116;
	}
L116:
	w[*m] = y[*m + j * y_dim1] * 2.;
L117:
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    y[i__ + j * y_dim1] = w[i__];
/* L118: */
	}
/* L119: */
    }
    k = iwbc + mhm1 - 1;
    i__ = iwba + mhm1;
    w[k] = 0.;
    w[i__] = 0.;
    w[k + 1] *= 2.;
    switch (modd) {
	case 1:  goto L120;
	case 2:  goto L121;
    }
L120:
    k = iwbb + mhm1 - 1;
    w[k] -= w[i__ - 1];
    w[iwbc - 1] += w[iwbb - 1];
    goto L122;
L121:
    w[iwbb - 1] = w[k + 1];
L122:
    goto L107;

/*     reverse columns when nperod = 4. */

L123:
    irev = 1;
    nby2 = *n / 2;
L124:
    i__1 = nby2;
    for (j = 1; j <= i__1; ++j) {
	mskip = *n + 1 - j;
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a1 = y[i__ + j * y_dim1];
	    y[i__ + j * y_dim1] = y[i__ + mskip * y_dim1];
	    y[i__ + mskip * y_dim1] = a1;
/* L125: */
	}
/* L126: */
    }
    switch (irev) {
	case 1:  goto L110;
	case 2:  goto L113;
    }
L127:
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = mhm1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    mhmi = mh - i__;
	    mhpi = mh + i__;
	    w[mhmi] = (y[mhpi + j * y_dim1] + y[i__ + j * y_dim1]) * .5;
	    w[mhpi] = (y[mhpi + j * y_dim1] - y[i__ + j * y_dim1]) * .5;
/* L128: */
	}
	w[mh] = y[mh + j * y_dim1] * .5;
	switch (modd) {
	    case 1:  goto L130;
	    case 2:  goto L129;
	}
L129:
	w[*m] = y[*m + j * y_dim1] * .5;
L130:
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    y[i__ + j * y_dim1] = w[i__];
/* L131: */
	}
/* L132: */
    }
L133:

/*     return storage requirements for w array. */

    w[1] = (doublereal) (ipstor + iwp - 1);
    return 0;
} /* genbun_ */

/* Subroutine */ int poisd2_(integer *mr, integer *nr, integer *istag, 
	doublereal *ba, doublereal *bb, doublereal *bc, doublereal *q, 
	integer *idimq, doublereal *b, doublereal *w, doublereal *d__, 
	doublereal *tcos, doublereal *p)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, l, m, n;
    static doublereal t, fi;
    static integer ip, kr, lr, jm1, jm2, jm3, jp1, jp2, jp3, ip1, jsh, jsp, 
	    nun, jst, ideg, jdeg, nodd, krpi;
    extern /* Subroutine */ int trix_(integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *), merge_(doublereal *, 
	    integer *, integer *, integer *, integer *, integer *);
    static integer irreg;
    extern /* Subroutine */ int cosgen_(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *);
    static integer noddpr, jstsav, ipstor;


/*     subroutine to solve poisson's equation for dirichlet boundary */
/*     conditions. */

/*     istag = 1 if the last diagonal block is the matrix a. */
/*     istag = 2 if the last diagonal block is the matrix a+i. */

    /* Parameter adjustments */
    --ba;
    --bb;
    --bc;
    q_dim1 = *idimq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --b;
    --w;
    --d__;
    --tcos;
    --p;

    /* Function Body */
    m = *mr;
    n = *nr;
    jsh = 0;
    fi = 1. / (doublereal) (*istag);
    ip = -m;
    ipstor = 0;
    switch (*istag) {
	case 1:  goto L101;
	case 2:  goto L102;
    }
L101:
    kr = 0;
    irreg = 1;
    if (n > 1) {
	goto L106;
    }
    tcos[1] = 0.;
    goto L103;
L102:
    kr = 1;
    jstsav = 1;
    irreg = 2;
    if (n > 1) {
	goto L106;
    }
    tcos[1] = -1.;
L103:
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	b[i__] = q[i__ + q_dim1];
/* L104: */
    }
    trix_(&c__1, &c__0, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[1], 
	    &w[1]);
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__ + q_dim1] = b[i__];
/* L105: */
    }
    goto L183;
L106:
    lr = 0;
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	p[i__] = 0.;
/* L107: */
    }
    nun = n;
    jst = 1;
    jsp = n;

/*     irreg = 1 when no irregularities have occurred, otherwise it is 2. */

L108:
    l = jst << 1;
    nodd = 2 - ((nun + 1) / 2 << 1) + nun;

/*     nodd = 1 when nun is odd, otherwise it is 2. */

    switch (nodd) {
	case 1:  goto L110;
	case 2:  goto L109;
    }
L109:
    jsp -= l;
    goto L111;
L110:
    jsp -= jst;
    if (irreg != 1) {
	jsp -= l;
    }
L111:

/*     regular reduction */

    cosgen_(&jst, &c__1, &c_b105, &c_b106, &tcos[1]);
    if (l > jsp) {
	goto L118;
    }
    i__1 = jsp;
    i__2 = l;
    for (j = l; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
	jm1 = j - jsh;
	jp1 = j + jsh;
	jm2 = j - jst;
	jp2 = j + jst;
	jm3 = jm2 - jsh;
	jp3 = jp2 + jsh;
	if (jst != 1) {
	    goto L113;
	}
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] * 2.;
	    q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + q[i__ + jp2 * 
		    q_dim1];
/* L112: */
	}
	goto L115;
L113:
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    t = q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] - q[i__ + jp1 * 
		    q_dim1] + q[i__ + jm2 * q_dim1] + q[i__ + jp2 * q_dim1];
	    b[i__] = t + q[i__ + j * q_dim1] - q[i__ + jm3 * q_dim1] - q[i__ 
		    + jp3 * q_dim1];
	    q[i__ + j * q_dim1] = t;
/* L114: */
	}
L115:
	trix_(&jst, &c__0, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[
		1], &w[1]);
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] += b[i__];
/* L116: */
	}
/* L117: */
    }

/*     reduction for last unknown */

L118:
    switch (nodd) {
	case 1:  goto L119;
	case 2:  goto L136;
    }
L119:
    switch (irreg) {
	case 1:  goto L152;
	case 2:  goto L120;
    }

/*     odd number of unknowns */

L120:
    jsp += l;
    j = jsp;
    jm1 = j - jsh;
    jp1 = j + jsh;
    jm2 = j - jst;
    jp2 = j + jst;
    jm3 = jm2 - jsh;
    switch (*istag) {
	case 1:  goto L123;
	case 2:  goto L121;
    }
L121:
    if (jst != 1) {
	goto L123;
    }
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1];
	q[i__ + j * q_dim1] = 0.;
/* L122: */
    }
    goto L130;
L123:
    switch (noddpr) {
	case 1:  goto L124;
	case 2:  goto L126;
    }
L124:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ip1 = ip + i__;
	b[i__] = (q[i__ + jm2 * q_dim1] - q[i__ + jm1 * q_dim1] - q[i__ + jm3 
		* q_dim1]) * .5 + p[ip1] + q[i__ + j * q_dim1];
/* L125: */
    }
    goto L128;
L126:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = (q[i__ + jm2 * q_dim1] - q[i__ + jm1 * q_dim1] - q[i__ + jm3 
		* q_dim1]) * .5 + q[i__ + jp2 * q_dim1] - q[i__ + jp1 * 
		q_dim1] + q[i__ + j * q_dim1];
/* L127: */
    }
L128:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] - 
		q[i__ + jp1 * q_dim1]) * .5;
/* L129: */
    }
L130:
    trix_(&jst, &c__0, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[1], &
	    w[1]);
    ip += m;
/* Computing MAX */
    i__2 = ipstor, i__1 = ip + m;
    ipstor = max(i__2,i__1);
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ip1 = ip + i__;
	p[ip1] = q[i__ + j * q_dim1] + b[i__];
	b[i__] = q[i__ + jp2 * q_dim1] + p[ip1];
/* L131: */
    }
    if (lr != 0) {
	goto L133;
    }
    i__2 = jst;
    for (i__ = 1; i__ <= i__2; ++i__) {
	krpi = kr + i__;
	tcos[krpi] = tcos[i__];
/* L132: */
    }
    goto L134;
L133:
    cosgen_(&lr, &jstsav, &c_b106, &fi, &tcos[jst + 1]);
    merge_(&tcos[1], &c__0, &jst, &jst, &lr, &kr);
L134:
    cosgen_(&kr, &jstsav, &c_b106, &fi, &tcos[1]);
    trix_(&kr, &kr, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[1], &w[
	    1]);
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ip1 = ip + i__;
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + b[i__] + p[ip1];
/* L135: */
    }
    lr = kr;
    kr += l;
    goto L152;

/*     even number of unknowns */

L136:
    jsp += l;
    j = jsp;
    jm1 = j - jsh;
    jp1 = j + jsh;
    jm2 = j - jst;
    jp2 = j + jst;
    jm3 = jm2 - jsh;
    switch (irreg) {
	case 1:  goto L137;
	case 2:  goto L138;
    }
L137:
    jstsav = jst;
    ideg = jst;
    kr = l;
    goto L139;
L138:
    cosgen_(&kr, &jstsav, &c_b106, &fi, &tcos[1]);
    cosgen_(&lr, &jstsav, &c_b106, &fi, &tcos[kr + 1]);
    ideg = kr;
    kr += jst;
L139:
    if (jst != 1) {
	goto L141;
    }
    irreg = 2;
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1];
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1];
/* L140: */
    }
    goto L150;
L141:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1] + (q[i__ + jm2 * q_dim1] - q[i__ + jm1 * 
		q_dim1] - q[i__ + jm3 * q_dim1]) * .5;
/* L142: */
    }
    switch (irreg) {
	case 1:  goto L143;
	case 2:  goto L145;
    }
L143:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + (q[i__ + j * q_dim1] - 
		q[i__ + jm1 * q_dim1] - q[i__ + jp1 * q_dim1]) * .5;
/* L144: */
    }
    irreg = 2;
    goto L150;
L145:
    switch (noddpr) {
	case 1:  goto L146;
	case 2:  goto L148;
    }
L146:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ip1 = ip + i__;
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + p[ip1];
/* L147: */
    }
    ip -= m;
    goto L150;
L148:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + q[i__ + j * q_dim1] - q[
		i__ + jm1 * q_dim1];
/* L149: */
    }
L150:
    trix_(&ideg, &lr, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[1], &
	    w[1]);
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] += b[i__];
/* L151: */
    }
L152:
    nun /= 2;
    noddpr = nodd;
    jsh = jst;
    jst <<= 1;
    if (nun >= 2) {
	goto L108;
    }

/*     start solution. */

    j = jsp;
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1];
/* L153: */
    }
    switch (irreg) {
	case 1:  goto L154;
	case 2:  goto L155;
    }
L154:
    cosgen_(&jst, &c__1, &c_b105, &c_b106, &tcos[1]);
    ideg = jst;
    goto L156;
L155:
    kr = lr + jst;
    cosgen_(&kr, &jstsav, &c_b106, &fi, &tcos[1]);
    cosgen_(&lr, &jstsav, &c_b106, &fi, &tcos[kr + 1]);
    ideg = kr;
L156:
    trix_(&ideg, &lr, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[1], &
	    w[1]);
    jm1 = j - jsh;
    jp1 = j + jsh;
    switch (irreg) {
	case 1:  goto L157;
	case 2:  goto L159;
    }
L157:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] - 
		q[i__ + jp1 * q_dim1]) * .5 + b[i__];
/* L158: */
    }
    goto L164;
L159:
    switch (noddpr) {
	case 1:  goto L160;
	case 2:  goto L162;
    }
L160:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ip1 = ip + i__;
	q[i__ + j * q_dim1] = p[ip1] + b[i__];
/* L161: */
    }
    ip -= m;
    goto L164;
L162:
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] + b[
		i__];
/* L163: */
    }
L164:

/*     start back substitution. */

    jst /= 2;
    jsh = jst / 2;
    nun <<= 1;
    if (nun > n) {
	goto L183;
    }
    i__2 = n;
    i__1 = l;
    for (j = jst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {
	jm1 = j - jsh;
	jp1 = j + jsh;
	jm2 = j - jst;
	jp2 = j + jst;
	if (j > jst) {
	    goto L166;
	}
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] + q[i__ + jp2 * q_dim1];
/* L165: */
	}
	goto L170;
L166:
	if (jp2 <= n) {
	    goto L168;
	}
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] + q[i__ + jm2 * q_dim1];
/* L167: */
	}
	if (jst < jstsav) {
	    irreg = 1;
	}
	switch (irreg) {
	    case 1:  goto L170;
	    case 2:  goto L171;
	}
L168:
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] + q[i__ + jm2 * q_dim1] + q[i__ + 
		    jp2 * q_dim1];
/* L169: */
	}
L170:
	cosgen_(&jst, &c__1, &c_b105, &c_b106, &tcos[1]);
	ideg = jst;
	jdeg = 0;
	goto L172;
L171:
	if (j + l > n) {
	    lr -= jst;
	}
	kr = jst + lr;
	cosgen_(&kr, &jstsav, &c_b106, &fi, &tcos[1]);
	cosgen_(&lr, &jstsav, &c_b106, &fi, &tcos[kr + 1]);
	ideg = kr;
	jdeg = lr;
L172:
	trix_(&ideg, &jdeg, &m, &ba[1], &bb[1], &bc[1], &b[1], &tcos[1], &d__[
		1], &w[1]);
	if (jst > 1) {
	    goto L174;
	}
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] = b[i__];
/* L173: */
	}
	goto L182;
L174:
	if (jp2 > n) {
	    goto L177;
	}
L175:
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] = (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1]
		     - q[i__ + jp1 * q_dim1]) * .5 + b[i__];
/* L176: */
	}
	goto L182;
L177:
	switch (irreg) {
	    case 1:  goto L175;
	    case 2:  goto L178;
	}
L178:
	if (j + jsh > n) {
	    goto L180;
	}
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    ip1 = ip + i__;
	    q[i__ + j * q_dim1] = b[i__] + p[ip1];
/* L179: */
	}
	ip -= m;
	goto L182;
L180:
	i__3 = m;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] = b[i__] + q[i__ + j * q_dim1] - q[i__ + jm1 *
		     q_dim1];
/* L181: */
	}
L182:
	;
    }
    l /= 2;
    goto L164;
L183:

/*     return storage requirements for p vectors. */

    w[1] = (doublereal) ipstor;
    return 0;
} /* poisd2_ */

/* Subroutine */ int poisn2_(integer *m, integer *n, integer *istag, integer *
	mixbnd, doublereal *a, doublereal *bb, doublereal *c__, doublereal *q,
	 integer *idimq, doublereal *b, doublereal *b2, doublereal *b3, 
	doublereal *w, doublereal *w2, doublereal *w3, doublereal *d__, 
	doublereal *tcos, doublereal *p)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2, i__3;
    static integer equiv_3[4];

    /* Local variables */
    static integer i__, j;
#define k (equiv_3)
    static doublereal t;
    static integer i1, i2;
#define k1 (equiv_3)
#define k2 (equiv_3 + 1)
#define k3 (equiv_3 + 2)
#define k4 (equiv_3 + 3)
    static doublereal fi;
    static integer ii, ip, jr, kr, lr, mr, nr, jm1, jm2, jm3, jp1, jp2, i2r, 
	    jp3, jr2;
    extern /* Subroutine */ int tri3_(integer *, doublereal *, doublereal *, 
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *);
    static doublereal fden;
    static integer nrod;
    static doublereal fnum;
    extern /* Subroutine */ int trix_(integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *), merge_(doublereal *, 
	    integer *, integer *, integer *, integer *, integer *);
    static integer i2rby2, nlast, jstep, jstop;
    static doublereal fistag;
    extern /* Subroutine */ int cosgen_(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *);
    static integer nlastp, nrodpr, jstart, ipstor;


/*     subroutine to solve poisson's equation with neumann boundary */
/*     conditions. */

/*     istag = 1 if the last diagonal block is a. */
/*     istag = 2 if the last diagonal block is a-i. */
/*     mixbnd = 1 if have neumann boundary conditions at both boundaries. */
/*     mixbnd = 2 if have neumann boundary conditions at bottom and */
/*     dirichlet condition at top.  (for this case, must have istag = 1.) */

    /* Parameter adjustments */
    --a;
    --bb;
    --c__;
    q_dim1 = *idimq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --b;
    --b2;
    --b3;
    --w;
    --w2;
    --w3;
    --d__;
    --tcos;
    --p;

    /* Function Body */
    fistag = (doublereal) (3 - *istag);
    fnum = 1. / (doublereal) (*istag);
    fden = (doublereal) (*istag - 1) * .5;
    mr = *m;
    ip = -mr;
    ipstor = 0;
    i2r = 1;
    jr = 2;
    nr = *n;
    nlast = *n;
    kr = 1;
    lr = 0;
    switch (*istag) {
	case 1:  goto L101;
	case 2:  goto L103;
    }
L101:
    i__1 = mr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__ + *n * q_dim1] *= .5;
/* L102: */
    }
    switch (*mixbnd) {
	case 1:  goto L103;
	case 2:  goto L104;
    }
L103:
    if (*n <= 3) {
	goto L155;
    }
L104:
    jr = i2r << 1;
    nrod = 1;
    if (nr / 2 << 1 == nr) {
	nrod = 0;
    }
    switch (*mixbnd) {
	case 1:  goto L105;
	case 2:  goto L106;
    }
L105:
    jstart = 1;
    goto L107;
L106:
    jstart = jr;
    nrod = 1 - nrod;
L107:
    jstop = nlast - jr;
    if (nrod == 0) {
	jstop -= i2r;
    }
    cosgen_(&i2r, &c__1, &c_b105, &c_b106, &tcos[1]);
    i2rby2 = i2r / 2;
    if (jstop >= jstart) {
	goto L108;
    }
    j = jr;
    goto L116;
L108:

/*     regular reduction. */

    i__1 = jstop;
    i__2 = jr;
    for (j = jstart; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
	jp1 = j + i2rby2;
	jp2 = j + i2r;
	jp3 = jp2 + i2rby2;
	jm1 = j - i2rby2;
	jm2 = j - i2r;
	jm3 = jm2 - i2rby2;
	if (j != 1) {
	    goto L109;
	}
	jm1 = jp1;
	jm2 = jp2;
	jm3 = jp3;
L109:
	if (i2r != 1) {
	    goto L111;
	}
	if (j == 1) {
	    jm2 = jp2;
	}
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] * 2.;
	    q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + q[i__ + jp2 * 
		    q_dim1];
/* L110: */
	}
	goto L113;
L111:
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    fi = q[i__ + j * q_dim1];
	    q[i__ + j * q_dim1] = q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] 
		    - q[i__ + jp1 * q_dim1] + q[i__ + jm2 * q_dim1] + q[i__ + 
		    jp2 * q_dim1];
	    b[i__] = fi + q[i__ + j * q_dim1] - q[i__ + jm3 * q_dim1] - q[i__ 
		    + jp3 * q_dim1];
/* L112: */
	}
L113:
	trix_(&i2r, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[
		1], &w[1]);
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] += b[i__];
/* L114: */
	}

/*     end of reduction for regular unknowns. */

/* L115: */
    }

/*     begin special reduction for last unknown. */

    j = jstop + jr;
L116:
    nlast = j;
    jm1 = j - i2rby2;
    jm2 = j - i2r;
    jm3 = jm2 - i2rby2;
    if (nrod == 0) {
	goto L128;
    }

/*     odd number of unknowns */

    if (i2r != 1) {
	goto L118;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = fistag * q[i__ + j * q_dim1];
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1];
/* L117: */
    }
    goto L126;
L118:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1] + (q[i__ + jm2 * q_dim1] - q[i__ + jm1 * 
		q_dim1] - q[i__ + jm3 * q_dim1]) * .5;
/* L119: */
    }
    if (nrodpr != 0) {
	goto L121;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + p[ii];
/* L120: */
    }
    ip -= mr;
    goto L123;
L121:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] + q[
		i__ + jm2 * q_dim1];
/* L122: */
    }
L123:
    if (lr == 0) {
	goto L124;
    }
    cosgen_(&lr, &c__1, &c_b105, &fden, &tcos[kr + 1]);
    goto L126;
L124:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = fistag * b[i__];
/* L125: */
    }
L126:
    cosgen_(&kr, &c__1, &c_b105, &fden, &tcos[1]);
    trix_(&kr, &lr, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &w[
	    1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] += b[i__];
/* L127: */
    }
    kr += i2r;
    goto L151;
L128:

/*     even number of unknowns */

    jp1 = j + i2rby2;
    jp2 = j + i2r;
    if (i2r != 1) {
	goto L135;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1];
/* L129: */
    }
    trix_(&c__1, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    ip = 0;
    ipstor = mr;
    switch (*istag) {
	case 1:  goto L133;
	case 2:  goto L130;
    }
L130:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	p[i__] = b[i__];
	b[i__] += q[i__ + *n * q_dim1];
/* L131: */
    }
    tcos[1] = 1.;
    tcos[2] = 0.;
    trix_(&c__1, &c__1, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + p[i__] + b[i__];
/* L132: */
    }
    goto L150;
L133:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	p[i__] = b[i__];
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + q[i__ + jp2 * q_dim1] * 
		2. + b[i__] * 3.;
/* L134: */
    }
    goto L150;
L135:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1] + (q[i__ + jm2 * q_dim1] - q[i__ + jm1 * 
		q_dim1] - q[i__ + jm3 * q_dim1]) * .5;
/* L136: */
    }
    if (nrodpr != 0) {
	goto L138;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	b[i__] += p[ii];
/* L137: */
    }
    goto L140;
L138:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = b[i__] + q[i__ + jp2 * q_dim1] - q[i__ + jp1 * q_dim1];
/* L139: */
    }
L140:
    trix_(&i2r, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], 
	    &w[1]);
    ip += mr;
/* Computing MAX */
    i__2 = ipstor, i__1 = ip + mr;
    ipstor = max(i__2,i__1);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	p[ii] = b[i__] + (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] - q[i__ 
		+ jp1 * q_dim1]) * .5;
	b[i__] = p[ii] + q[i__ + jp2 * q_dim1];
/* L141: */
    }
    if (lr == 0) {
	goto L142;
    }
    cosgen_(&lr, &c__1, &c_b105, &fden, &tcos[i2r + 1]);
    merge_(&tcos[1], &c__0, &i2r, &i2r, &lr, &kr);
    goto L144;
L142:
    i__2 = i2r;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = kr + i__;
	tcos[ii] = tcos[i__];
/* L143: */
    }
L144:
    cosgen_(&kr, &c__1, &c_b105, &fden, &tcos[1]);
    if (lr != 0) {
	goto L145;
    }
    switch (*istag) {
	case 1:  goto L146;
	case 2:  goto L145;
    }
L145:
    trix_(&kr, &kr, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &w[
	    1]);
    goto L148;
L146:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = fistag * b[i__];
/* L147: */
    }
L148:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	q[i__ + j * q_dim1] = q[i__ + jm2 * q_dim1] + p[ii] + b[i__];
/* L149: */
    }
L150:
    lr = kr;
    kr += jr;
L151:
    switch (*mixbnd) {
	case 1:  goto L152;
	case 2:  goto L153;
    }
L152:
    nr = (nlast - 1) / jr + 1;
    if (nr <= 3) {
	goto L155;
    }
    goto L154;
L153:
    nr = nlast / jr;
    if (nr <= 1) {
	goto L192;
    }
L154:
    i2r = jr;
    nrodpr = nrod;
    goto L104;
L155:

/*      begin solution */

    j = jr + 1;
    jm1 = j - i2r;
    jp1 = j + i2r;
    jm2 = nlast - i2r;
    if (nr == 2) {
	goto L184;
    }
    if (lr != 0) {
	goto L170;
    }
    if (*n != 3) {
	goto L161;
    }

/*     case n = 3. */

    switch (*istag) {
	case 1:  goto L156;
	case 2:  goto L168;
    }
L156:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + (q_dim1 << 1)];
/* L157: */
    }
    tcos[1] = 0.f;
    trix_(&c__1, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + (q_dim1 << 1)] = b[i__];
	b[i__] = b[i__] * 4. + q[i__ + q_dim1] + q[i__ + q_dim1 * 3] * 2.;
/* L158: */
    }
    tcos[1] = -2.;
    tcos[2] = 2.;
    i1 = 2;
    i2 = 0;
    trix_(&i1, &i2, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &w[
	    1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + (q_dim1 << 1)] += b[i__];
	b[i__] = q[i__ + q_dim1] + q[i__ + (q_dim1 << 1)] * 2.;
/* L159: */
    }
    tcos[1] = 0.;
    trix_(&c__1, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] = b[i__];
/* L160: */
    }
    jr = 1;
    i2r = 0;
    goto L194;

/*     case n = 2**p+1 */

L161:
    switch (*istag) {
	case 1:  goto L162;
	case 2:  goto L170;
    }
L162:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + j * q_dim1] + q[i__ + q_dim1] * .5 - q[i__ + jm1 * 
		q_dim1] + q[i__ + nlast * q_dim1] - q[i__ + jm2 * q_dim1];
/* L163: */
    }
    cosgen_(&jr, &c__1, &c_b105, &c_b106, &tcos[1]);
    trix_(&jr, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &
	    w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] = (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] - 
		q[i__ + jp1 * q_dim1]) * .5 + b[i__];
	b[i__] = q[i__ + q_dim1] + q[i__ + nlast * q_dim1] * 2. + q[i__ + j * 
		q_dim1] * 4.;
/* L164: */
    }
    jr2 = jr << 1;
    cosgen_(&jr, &c__1, &c_b106, &c_b106, &tcos[1]);
    i__2 = jr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	i1 = jr + i__;
	i2 = jr + 1 - i__;
	tcos[i1] = -tcos[i2];
/* L165: */
    }
    trix_(&jr2, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], 
	    &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] += b[i__];
	b[i__] = q[i__ + q_dim1] + q[i__ + j * q_dim1] * 2.;
/* L166: */
    }
    cosgen_(&jr, &c__1, &c_b105, &c_b106, &tcos[1]);
    trix_(&jr, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &
	    w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] = q[i__ + q_dim1] * .5 - q[i__ + jm1 * q_dim1] + b[
		i__];
/* L167: */
    }
    goto L194;

/*     case of general n with nr = 3 . */

L168:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + (q_dim1 << 1)];
	q[i__ + (q_dim1 << 1)] = 0.;
	b2[i__] = q[i__ + q_dim1 * 3];
	b3[i__] = q[i__ + q_dim1];
/* L169: */
    }
    jr = 1;
    i2r = 0;
    j = 2;
    goto L177;
L170:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + q_dim1] * .5 - q[i__ + jm1 * q_dim1] + q[i__ + j * 
		q_dim1];
/* L171: */
    }
    if (nrod != 0) {
	goto L173;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	b[i__] += p[ii];
/* L172: */
    }
    goto L175;
L173:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = b[i__] + q[i__ + nlast * q_dim1] - q[i__ + jm2 * q_dim1];
/* L174: */
    }
L175:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	t = (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1] - q[i__ + jp1 * 
		q_dim1]) * .5;
	q[i__ + j * q_dim1] = t;
	b2[i__] = q[i__ + nlast * q_dim1] + t;
	b3[i__] = q[i__ + q_dim1] + t * 2.f;
/* L176: */
    }
L177:
    *k1 = kr + (jr << 1) - 1;
    *k2 = kr + jr;
    tcos[*k1 + 1] = -2.f;
    *k4 = *k1 + 3 - *istag;
    i__2 = *k2 + *istag - 2;
    cosgen_(&i__2, &c__1, &c_b106, &fnum, &tcos[*k4]);
    *k4 = *k1 + *k2 + 1;
    i__2 = jr - 1;
    cosgen_(&i__2, &c__1, &c_b106, &c_b314, &tcos[*k4]);
    i__2 = *k1 + *k2;
    i__1 = jr - 1;
    merge_(&tcos[1], k1, k2, &i__2, &i__1, &c__0);
    *k3 = *k1 + *k2 + lr;
    cosgen_(&jr, &c__1, &c_b105, &c_b106, &tcos[*k3 + 1]);
    *k4 = *k3 + jr + 1;
    cosgen_(&kr, &c__1, &c_b105, &fden, &tcos[*k4]);
    i__2 = *k3 + jr;
    merge_(&tcos[1], k3, &jr, &i__2, &kr, k1);
    if (lr == 0) {
	goto L178;
    }
    cosgen_(&lr, &c__1, &c_b105, &fden, &tcos[*k4]);
    i__2 = *k3 + jr;
    i__1 = *k3 - lr;
    merge_(&tcos[1], k3, &jr, &i__2, &lr, &i__1);
    cosgen_(&kr, &c__1, &c_b105, &fden, &tcos[*k4]);
L178:
    *k3 = kr;
    *k4 = kr;
    tri3_(&mr, &a[1], &bb[1], &c__[1], k, &b[1], &b2[1], &b3[1], &tcos[1], &
	    d__[1], &w[1], &w2[1], &w3[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = b[i__] + b2[i__] + b3[i__];
/* L179: */
    }
    tcos[1] = 2.;
    trix_(&c__1, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + j * q_dim1] += b[i__];
	b[i__] = q[i__ + q_dim1] + q[i__ + j * q_dim1] * 2.f;
/* L180: */
    }
    cosgen_(&jr, &c__1, &c_b105, &c_b106, &tcos[1]);
    trix_(&jr, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &
	    w[1]);
    if (jr != 1) {
	goto L182;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] = b[i__];
/* L181: */
    }
    goto L194;
L182:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] = q[i__ + q_dim1] * .5 - q[i__ + jm1 * q_dim1] + b[
		i__];
/* L183: */
    }
    goto L194;
L184:
    if (*n != 2) {
	goto L188;
    }

/*     case  n = 2 */

    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + q_dim1];
/* L185: */
    }
    tcos[1] = 0.f;
    trix_(&c__1, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] = b[i__];
	b[i__] = (q[i__ + (q_dim1 << 1)] + b[i__]) * 2. * fistag;
/* L186: */
    }
    tcos[1] = -fistag;
    tcos[2] = 2.;
    trix_(&c__2, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] += b[i__];
/* L187: */
    }
    jr = 1;
    i2r = 0;
    goto L194;
L188:

/*     case of general n and nr = 2 . */

    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	b3[i__] = 0.f;
	b[i__] = q[i__ + q_dim1] + p[ii] * 2.f;
	q[i__ + q_dim1] = q[i__ + q_dim1] * .5 - q[i__ + jm1 * q_dim1];
	b2[i__] = (q[i__ + q_dim1] + q[i__ + nlast * q_dim1]) * 2.;
/* L189: */
    }
    *k1 = kr + jr - 1;
    tcos[*k1 + 1] = -2.;
    *k4 = *k1 + 3 - *istag;
    i__2 = kr + *istag - 2;
    cosgen_(&i__2, &c__1, &c_b106, &fnum, &tcos[*k4]);
    *k4 = *k1 + kr + 1;
    i__2 = jr - 1;
    cosgen_(&i__2, &c__1, &c_b106, &c_b314, &tcos[*k4]);
    i__2 = *k1 + kr;
    i__1 = jr - 1;
    merge_(&tcos[1], k1, &kr, &i__2, &i__1, &c__0);
    cosgen_(&kr, &c__1, &c_b105, &fden, &tcos[*k1 + 1]);
    *k2 = kr;
    *k4 = *k1 + *k2 + 1;
    cosgen_(&lr, &c__1, &c_b105, &fden, &tcos[*k4]);
    *k3 = lr;
    *k4 = 0;
    tri3_(&mr, &a[1], &bb[1], &c__[1], k, &b[1], &b2[1], &b3[1], &tcos[1], &
	    d__[1], &w[1], &w2[1], &w3[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] += b2[i__];
/* L190: */
    }
    tcos[1] = 2.f;
    trix_(&c__1, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1],
	     &w[1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + q_dim1] += b[i__];
/* L191: */
    }
    goto L194;
L192:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + nlast * q_dim1];
/* L193: */
    }
    goto L196;
L194:

/*     start back substitution. */

    j = nlast - jr;
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = q[i__ + nlast * q_dim1] + q[i__ + j * q_dim1];
/* L195: */
    }
L196:
    jm2 = nlast - i2r;
    if (jr != 1) {
	goto L198;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + nlast * q_dim1] = 0.f;
/* L197: */
    }
    goto L202;
L198:
    if (nrod != 0) {
	goto L200;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	ii = ip + i__;
	q[i__ + nlast * q_dim1] = p[ii];
/* L199: */
    }
    ip -= mr;
    goto L202;
L200:
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + nlast * q_dim1] -= q[i__ + jm2 * q_dim1];
/* L201: */
    }
L202:
    cosgen_(&kr, &c__1, &c_b105, &fden, &tcos[1]);
    cosgen_(&lr, &c__1, &c_b105, &fden, &tcos[kr + 1]);
    if (lr != 0) {
	goto L204;
    }
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	b[i__] = fistag * b[i__];
/* L203: */
    }
L204:
    trix_(&kr, &lr, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[1], &w[
	    1]);
    i__2 = mr;
    for (i__ = 1; i__ <= i__2; ++i__) {
	q[i__ + nlast * q_dim1] += b[i__];
/* L205: */
    }
    nlastp = nlast;
L206:
    jstep = jr;
    jr = i2r;
    i2r /= 2;
    if (jr == 0) {
	goto L222;
    }
    switch (*mixbnd) {
	case 1:  goto L207;
	case 2:  goto L208;
    }
L207:
    jstart = jr + 1;
    goto L209;
L208:
    jstart = jr;
L209:
    kr -= jr;
    if (nlast + jr > *n) {
	goto L210;
    }
    kr -= jr;
    nlast += jr;
    jstop = nlast - jstep;
    goto L211;
L210:
    jstop = nlast - jr;
L211:
    lr = kr - jr;
    cosgen_(&jr, &c__1, &c_b105, &c_b106, &tcos[1]);
    i__2 = jstop;
    i__1 = jstep;
    for (j = jstart; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {
	jm2 = j - jr;
	jp2 = j + jr;
	if (j != jr) {
	    goto L213;
	}
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] + q[i__ + jp2 * q_dim1];
/* L212: */
	}
	goto L215;
L213:
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    b[i__] = q[i__ + j * q_dim1] + q[i__ + jm2 * q_dim1] + q[i__ + 
		    jp2 * q_dim1];
/* L214: */
	}
L215:
	if (jr != 1) {
	    goto L217;
	}
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] = 0.f;
/* L216: */
	}
	goto L219;
L217:
	jm1 = j - i2r;
	jp1 = j + i2r;
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] = (q[i__ + j * q_dim1] - q[i__ + jm1 * q_dim1]
		     - q[i__ + jp1 * q_dim1]) * .5;
/* L218: */
	}
L219:
	trix_(&jr, &c__0, &mr, &a[1], &bb[1], &c__[1], &b[1], &tcos[1], &d__[
		1], &w[1]);
	i__3 = mr;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    q[i__ + j * q_dim1] += b[i__];
/* L220: */
	}
/* L221: */
    }
    nrod = 1;
    if (nlast + i2r <= *n) {
	nrod = 0;
    }
    if (nlastp != nlast) {
	goto L194;
    }
    goto L206;
L222:

/*     return storage requirements for p vectors. */

    w[1] = (doublereal) ipstor;
    return 0;
} /* poisn2_ */

#undef k4
#undef k3
#undef k2
#undef k1
#undef k


/* Subroutine */ int poisp2_(integer *m, integer *n, doublereal *a, 
	doublereal *bb, doublereal *c__, doublereal *q, integer *idimq, 
	doublereal *b, doublereal *b2, doublereal *b3, doublereal *w, 
	doublereal *w2, doublereal *w3, doublereal *d__, doublereal *tcos, 
	doublereal *p)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;
    static doublereal s, t;
    static integer lh, mr, nr, nrm1, nrmj, nrpj;
    extern /* Subroutine */ int poisd2_(integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *,
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), poisn2_(integer *, integer *, integer *, integer *,
	     doublereal *, doublereal *, doublereal *, doublereal *, integer *
	    , doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *);
    static integer ipstor;


/*     subroutine to solve poisson equation with periodic boundary */
/*     conditions. */

    /* Parameter adjustments */
    --a;
    --bb;
    --c__;
    q_dim1 = *idimq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --b;
    --b2;
    --b3;
    --w;
    --w2;
    --w3;
    --d__;
    --tcos;
    --p;

    /* Function Body */
    mr = *m;
    nr = (*n + 1) / 2;
    nrm1 = nr - 1;
    if (nr << 1 != *n) {
	goto L107;
    }

/*     even number of unknowns */

    i__1 = nrm1;
    for (j = 1; j <= i__1; ++j) {
	nrmj = nr - j;
	nrpj = nr + j;
	i__2 = mr;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = q[i__ + nrmj * q_dim1] - q[i__ + nrpj * q_dim1];
	    t = q[i__ + nrmj * q_dim1] + q[i__ + nrpj * q_dim1];
	    q[i__ + nrmj * q_dim1] = s;
	    q[i__ + nrpj * q_dim1] = t;
/* L101: */
	}
/* L102: */
    }
    i__1 = mr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__ + nr * q_dim1] *= 2.;
	q[i__ + *n * q_dim1] *= 2.;
/* L103: */
    }
    poisd2_(&mr, &nrm1, &c__1, &a[1], &bb[1], &c__[1], &q[q_offset], idimq, &
	    b[1], &w[1], &d__[1], &tcos[1], &p[1]);
    ipstor = (integer) w[1];
    i__1 = nr + 1;
    poisn2_(&mr, &i__1, &c__1, &c__1, &a[1], &bb[1], &c__[1], &q[nr * q_dim1 
	    + 1], idimq, &b[1], &b2[1], &b3[1], &w[1], &w2[1], &w3[1], &d__[1]
	    , &tcos[1], &p[1]);
/* Computing MAX */
    i__1 = ipstor, i__2 = (integer) w[1];
    ipstor = max(i__1,i__2);
    i__1 = nrm1;
    for (j = 1; j <= i__1; ++j) {
	nrmj = nr - j;
	nrpj = nr + j;
	i__2 = mr;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = (q[i__ + nrpj * q_dim1] + q[i__ + nrmj * q_dim1]) * .5;
	    t = (q[i__ + nrpj * q_dim1] - q[i__ + nrmj * q_dim1]) * .5;
	    q[i__ + nrmj * q_dim1] = s;
	    q[i__ + nrpj * q_dim1] = t;
/* L104: */
	}
/* L105: */
    }
    i__1 = mr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__ + nr * q_dim1] *= .5;
	q[i__ + *n * q_dim1] *= .5;
/* L106: */
    }
    goto L118;
L107:

/*     odd  number of unknowns */

    i__1 = nrm1;
    for (j = 1; j <= i__1; ++j) {
	nrpj = *n + 1 - j;
	i__2 = mr;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = q[i__ + j * q_dim1] - q[i__ + nrpj * q_dim1];
	    t = q[i__ + j * q_dim1] + q[i__ + nrpj * q_dim1];
	    q[i__ + j * q_dim1] = s;
	    q[i__ + nrpj * q_dim1] = t;
/* L108: */
	}
/* L109: */
    }
    i__1 = mr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__ + nr * q_dim1] *= 2.;
/* L110: */
    }
    lh = nrm1 / 2;
    i__1 = lh;
    for (j = 1; j <= i__1; ++j) {
	nrmj = nr - j;
	i__2 = mr;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = q[i__ + j * q_dim1];
	    q[i__ + j * q_dim1] = q[i__ + nrmj * q_dim1];
	    q[i__ + nrmj * q_dim1] = s;
/* L111: */
	}
/* L112: */
    }
    poisd2_(&mr, &nrm1, &c__2, &a[1], &bb[1], &c__[1], &q[q_offset], idimq, &
	    b[1], &w[1], &d__[1], &tcos[1], &p[1]);
    ipstor = (integer) w[1];
    poisn2_(&mr, &nr, &c__2, &c__1, &a[1], &bb[1], &c__[1], &q[nr * q_dim1 + 
	    1], idimq, &b[1], &b2[1], &b3[1], &w[1], &w2[1], &w3[1], &d__[1], 
	    &tcos[1], &p[1]);
/* Computing MAX */
    i__1 = ipstor, i__2 = (integer) w[1];
    ipstor = max(i__1,i__2);
    i__1 = nrm1;
    for (j = 1; j <= i__1; ++j) {
	nrpj = nr + j;
	i__2 = mr;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = (q[i__ + nrpj * q_dim1] + q[i__ + j * q_dim1]) * .5;
	    t = (q[i__ + nrpj * q_dim1] - q[i__ + j * q_dim1]) * .5;
	    q[i__ + nrpj * q_dim1] = t;
	    q[i__ + j * q_dim1] = s;
/* L113: */
	}
/* L114: */
    }
    i__1 = mr;
    for (i__ = 1; i__ <= i__1; ++i__) {
	q[i__ + nr * q_dim1] *= .5;
/* L115: */
    }
    i__1 = lh;
    for (j = 1; j <= i__1; ++j) {
	nrmj = nr - j;
	i__2 = mr;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = q[i__ + j * q_dim1];
	    q[i__ + j * q_dim1] = q[i__ + nrmj * q_dim1];
	    q[i__ + nrmj * q_dim1] = s;
/* L116: */
	}
/* L117: */
    }
L118:

/*     return storage requirements for p vectors. */

    w[1] = (doublereal) ipstor;
    return 0;
} /* poisp2_ */

/* Subroutine */ int tri3_(integer *m, doublereal *a, doublereal *b, 
	doublereal *c__, integer *k, doublereal *y1, doublereal *y2, 
	doublereal *y3, doublereal *tcos, doublereal *d__, doublereal *w1, 
	doublereal *w2, doublereal *w3)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, n;
    static doublereal x, z__, f1, f2, f3, f4;
    static integer k1, k2, k3, k4, l1, l2, l3, ip;
    static doublereal xx;
    static integer mm1, k2k3k4, kint1, lint1, lint2, lint3, kint2, kint3;


/*     subroutine to solve three linear systems whose common coefficient */
/*     matrix is a rational function in the matrix given by */

/*                  tridiagonal (...,a(i),b(i),c(i),...) */

    /* Parameter adjustments */
    --w3;
    --w2;
    --w1;
    --d__;
    --tcos;
    --y3;
    --y2;
    --y1;
    --k;
    --c__;
    --b;
    --a;

    /* Function Body */
    mm1 = *m - 1;
    k1 = k[1];
    k2 = k[2];
    k3 = k[3];
    k4 = k[4];
    f1 = (doublereal) (k1 + 1);
    f2 = (doublereal) (k2 + 1);
    f3 = (doublereal) (k3 + 1);
    f4 = (doublereal) (k4 + 1);
    k2k3k4 = k2 + k3 + k4;
    if (k2k3k4 == 0) {
	goto L101;
    }
    l1 = (integer) (f1 / f2);
    l2 = (integer) (f1 / f3);
    l3 = (integer) (f1 / f4);
    lint1 = 1;
    lint2 = 1;
    lint3 = 1;
    kint1 = k1;
    kint2 = kint1 + k2;
    kint3 = kint2 + k3;
L101:
    i__1 = k1;
    for (n = 1; n <= i__1; ++n) {
	x = tcos[n];
	if (k2k3k4 == 0) {
	    goto L107;
	}
	if (n != l1) {
	    goto L103;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w1[i__] = y1[i__];
/* L102: */
	}
L103:
	if (n != l2) {
	    goto L105;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w2[i__] = y2[i__];
/* L104: */
	}
L105:
	if (n != l3) {
	    goto L107;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w3[i__] = y3[i__];
/* L106: */
	}
L107:
	z__ = 1. / (b[1] - x);
	d__[1] = c__[1] * z__;
	y1[1] *= z__;
	y2[1] *= z__;
	y3[1] *= z__;
	i__2 = *m;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    z__ = 1. / (b[i__] - x - a[i__] * d__[i__ - 1]);
	    d__[i__] = c__[i__] * z__;
	    y1[i__] = (y1[i__] - a[i__] * y1[i__ - 1]) * z__;
	    y2[i__] = (y2[i__] - a[i__] * y2[i__ - 1]) * z__;
	    y3[i__] = (y3[i__] - a[i__] * y3[i__ - 1]) * z__;
/* L108: */
	}
	i__2 = mm1;
	for (ip = 1; ip <= i__2; ++ip) {
	    i__ = *m - ip;
	    y1[i__] -= d__[i__] * y1[i__ + 1];
	    y2[i__] -= d__[i__] * y2[i__ + 1];
	    y3[i__] -= d__[i__] * y3[i__ + 1];
/* L109: */
	}
	if (k2k3k4 == 0) {
	    goto L115;
	}
	if (n != l1) {
	    goto L111;
	}
	i__ = lint1 + kint1;
	xx = x - tcos[i__];
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    y1[i__] = xx * y1[i__] + w1[i__];
/* L110: */
	}
	++lint1;
	l1 = (integer) ((doublereal) lint1 * f1 / f2);
L111:
	if (n != l2) {
	    goto L113;
	}
	i__ = lint2 + kint2;
	xx = x - tcos[i__];
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    y2[i__] = xx * y2[i__] + w2[i__];
/* L112: */
	}
	++lint2;
	l2 = (integer) ((doublereal) lint2 * f1 / f3);
L113:
	if (n != l3) {
	    goto L115;
	}
	i__ = lint3 + kint3;
	xx = x - tcos[i__];
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    y3[i__] = xx * y3[i__] + w3[i__];
/* L114: */
	}
	++lint3;
	l3 = (integer) ((doublereal) lint3 * f1 / f4);
L115:
	;
    }
    return 0;
} /* tri3_ */

/* Subroutine */ int trix_(integer *idegbr, integer *idegcr, integer *m, 
	doublereal *a, doublereal *b, doublereal *c__, doublereal *y, 
	doublereal *tcos, doublereal *d__, doublereal *w)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, k, l;
    static doublereal x, z__, fb, fc;
    static integer ip;
    static doublereal xx;
    static integer mm1, lint;


/*     subroutine to solve a system of linear equations where the */
/*     coefficient matrix is a rational function in the matrix given by */
/*     tridiagonal  ( . . . , a(i), b(i), c(i), . . . ). */

    /* Parameter adjustments */
    --w;
    --d__;
    --tcos;
    --y;
    --c__;
    --b;
    --a;

    /* Function Body */
    mm1 = *m - 1;
    fb = (doublereal) (*idegbr + 1);
    fc = (doublereal) (*idegcr + 1);
    l = (integer) (fb / fc);
    lint = 1;
    i__1 = *idegbr;
    for (k = 1; k <= i__1; ++k) {
	x = tcos[k];
	if (k != l) {
	    goto L102;
	}
	i__ = *idegbr + lint;
	xx = x - tcos[i__];
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w[i__] = y[i__];
	    y[i__] = xx * y[i__];
/* L101: */
	}
L102:
	z__ = 1. / (b[1] - x);
	d__[1] = c__[1] * z__;
	y[1] *= z__;
	i__2 = mm1;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    z__ = 1. / (b[i__] - x - a[i__] * d__[i__ - 1]);
	    d__[i__] = c__[i__] * z__;
	    y[i__] = (y[i__] - a[i__] * y[i__ - 1]) * z__;
/* L103: */
	}
	z__ = b[*m] - x - a[*m] * d__[mm1];
	if (z__ != 0.) {
	    goto L104;
	}
	y[*m] = 0.;
	goto L105;
L104:
	y[*m] = (y[*m] - a[*m] * y[mm1]) / z__;
L105:
	i__2 = mm1;
	for (ip = 1; ip <= i__2; ++ip) {
	    i__ = *m - ip;
	    y[i__] -= d__[i__] * y[i__ + 1];
/* L106: */
	}
	if (k != l) {
	    goto L108;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    y[i__] += w[i__];
/* L107: */
	}
	++lint;
	l = (integer) ((doublereal) lint * fb / fc);
L108:
	;
    }
    return 0;
} /* trix_ */

/* Subroutine */ int cosgen_(integer *n, integer *ijump, doublereal *fnum, 
	doublereal *fden, doublereal *a)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double cos(doublereal);

    /* Local variables */
    static integer i__, k;
    static doublereal x, y;
    static integer k1, k2, k3, k4, k5;
    static doublereal pi;
    static integer np1;
    static doublereal dum, pibyn;
    extern doublereal pimach_(doublereal *);



/*     this subroutine computes required cosine values in ascending */
/*     order.  when ijump .gt. 1 the routine computes values */

/*        2*cos(j*pi/l) , j=1,2,...,l and j .ne. 0(mod n/ijump+1) */

/*     where l = ijump*(n/ijump+1). */


/*     when ijump = 1 it computes */

/*            2*cos((j-fnum)*pi/(n+fden)) ,  j=1, 2, ... ,n */

/*     where */
/*        fnum = 0.5, fden = 0.0,  for regular reduction values */
/*        fnum = 0.0, fden = 1.0, for b-r and c-r when istag = 1 */
/*        fnum = 0.0, fden = 0.5, for b-r and c-r when istag = 2 */
/*        fnum = 0.5, fden = 0.5, for b-r and c-r when istag = 2 */
/*                                in poisn2 only. */


    /* Parameter adjustments */
    --a;

    /* Function Body */
    pi = pimach_(&dum);
    if (*n == 0) {
	goto L105;
    }
    if (*ijump == 1) {
	goto L103;
    }
    k3 = *n / *ijump + 1;
    k4 = k3 - 1;
    pibyn = pi / (doublereal) (*n + *ijump);
    i__1 = *ijump;
    for (k = 1; k <= i__1; ++k) {
	k1 = (k - 1) * k3;
	k5 = (k - 1) * k4;
	i__2 = k4;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    x = (doublereal) (k1 + i__);
	    k2 = k5 + i__;
	    a[k2] = cos(x * pibyn) * -2.;
/* L101: */
	}
/* L102: */
    }
    goto L105;
L103:
    np1 = *n + 1;
    y = pi / ((doublereal) (*n) + *fden);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x = (doublereal) (np1 - i__) - *fnum;
	a[i__] = cos(x * y) * 2.;
/* L104: */
    }
L105:
    return 0;
} /* cosgen_ */

/* Subroutine */ int merge_(doublereal *tcos, integer *i1, integer *m1, 
	integer *i2, integer *m2, integer *i3)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer j, k, l, m;
    static doublereal x, y;
    static integer j1, j2;



/*     this subroutine merges two ascending strings of numbers in the */
/*     array tcos.  the first string is of length m1 and starts at */
/*     tcos(i1+1).  the second string is of length m2 and starts at */
/*     tcos(i2+1).  the merged string goes into tcos(i3+1). */


    /* Parameter adjustments */
    --tcos;

    /* Function Body */
    j1 = 1;
    j2 = 1;
    j = *i3;
    if (*m1 == 0) {
	goto L107;
    }
    if (*m2 == 0) {
	goto L104;
    }
L101:
    ++j;
    l = j1 + *i1;
    x = tcos[l];
    l = j2 + *i2;
    y = tcos[l];
    if (x - y <= 0.) {
	goto L102;
    } else {
	goto L103;
    }
L102:
    tcos[j] = x;
    ++j1;
    if (j1 > *m1) {
	goto L106;
    }
    goto L101;
L103:
    tcos[j] = y;
    ++j2;
    if (j2 <= *m2) {
	goto L101;
    }
    if (j1 > *m1) {
	goto L109;
    }
L104:
    k = j - j1 + 1;
    i__1 = *m1;
    for (j = j1; j <= i__1; ++j) {
	m = k + j;
	l = j + *i1;
	tcos[m] = tcos[l];
/* L105: */
    }
    goto L109;
L106:
    if (j2 > *m2) {
	goto L109;
    }
L107:
    k = j - j2 + 1;
    i__1 = *m2;
    for (j = j2; j <= i__1; ++j) {
	m = k + j;
	l = j + *i2;
	tcos[m] = tcos[l];
/* L108: */
    }
L109:
    return 0;
} /* merge_ */

doublereal pimach_(doublereal *dum)
{
    /* System generated locals */
    doublereal ret_val;


/*     this subprogram supplies the value of the constant pi correct to */
/*     machine precision where */

/*     pi=3.1415926535897932384626433832795028841971693993751058209749446 */

    ret_val = 3.14159265358979;
    return ret_val;
} /* pimach_ */

