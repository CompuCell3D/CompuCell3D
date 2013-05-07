/* wnorm.f -- translated by f2c (version 20090411).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c_nleq.h"

doublereal wnorm_(integer *n, doublereal *z__, doublereal *xw)
{
    /* System generated locals */
    integer i__1;
    doublereal ret_val, d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__;
    static doublereal s;

/*     ------------------------------------------------------------ */

/* *    Summary : */

/*     E N O R M : Return the norm to be used in exit (termination) */
/*                 criteria */

/* *    Input parameters */
/*     ================ */

/*     N         Int Number of equations/unknowns */
/*     Z(N)     Dble  The vector, of which the norm is to be computed */
/*     XW(N)    Dble  The scaling values of Z(N) */

/* *    Output */
/*     ====== */

/*     WNORM(N,Z,XW)  Dble  The mean square root norm of Z(N) subject */
/*                          to the scaling values in XW(N): */
/*                          = Sqrt( Sum(1,...N)((Z(I)/XW(I))**2) / N ) */

/*     ------------------------------------------------------------ */
/* *    End Prologue */
/* *    Begin */
    /* Parameter adjustments */
    --xw;
    --z__;

    /* Function Body */
    s = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = z__[i__] / xw[i__];
	s += d__1 * d__1;
/* L10: */
    }
    ret_val = sqrt(s / (doublereal) ((f2c_real) (*n)));
/*     End of function WNORM */
    return ret_val;
} /* wnorm_ */

