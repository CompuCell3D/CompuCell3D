/* nleq1e.f -- translated by f2c (version 20090411).
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

/* Table of constant values */

static integer c__50 = 50;
static integer c__1 = 1;
static integer c__100 = 100;
static integer c__3210 = 3210;

/* Subroutine */ int nleq1e_(integer *n, doublereal *x, doublereal *rtol, 
	integer *ierr)
{

    /* Format strings */
    static char fmt_1001[] = "(/,\002 Error: Bad input to parameter N suppli"
	    "ed\002,/,8x,\002choose 1 .LE. N .LE. \002,i3,\002 , your input i"
	    "s: N = \002,i5)";

    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

    /* Local variables */
    static integer i__;
    extern /* Subroutine */ int fcn_();
    static integer iwk[100], niw;
    static doublereal rwk[3210];
    static integer nrw, iopt[50];
    extern /* Subroutine */ int nleq1_(integer *, U_fp, f2c_real *, doublereal *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    integer *, integer *, doublereal *);
    static doublereal xscal[50];
    static f2c_real dummy;

    /* Fortran I/O blocks */
    static cilist io___5 = { 0, 6, 0, fmt_1001, 0 };


/* *    Begin Prologue NLEQ1E */
/*     ------------------------------------------------------------ */

/* *  Title */

/*     Numerical solution of nonlinear (NL) equations (EQ) */
/*     especially designed for numerically sensitive problems. */
/*     (E)asy-to-use driver routine for NLEQ1. */

/* *  Written by        U. Nowak, L. Weimann */
/* *  Purpose           Solution of systems of highly nonlinear equations */
/* *  Method            Damped affine invariant Newton method */
/*                     (see references below) */
/* *  Category          F2a. - Systems of nonlinear equations */
/* *  Keywords          Nonlinear equations, Newton methods */
/* *  Version           2.3 */
/* *  Revision          November 1991 */
/* *  Latest Change     November 1991 */
/* *  Library           CodeLib */
/* *  Code              Fortran 77, Double Precision */
/* *  Environment       Standard Fortran 77 environment on PC's, */
/*                     workstations and hosts. */
/* *  Copyright     (c) Konrad-Zuse-Zentrum fuer */
/*                     Informationstechnik Berlin (ZIB) */
/*                     Takustrasse 7, D-14195 Berlin-Dahlem */
/*                     phone : + 49/30/84185-0 */
/*                     fax   : + 49/30/84185-125 */
/* *  Contact           Lutz Weimann */
/*                     ZIB, Division Scientific Computing, */
/*                          Department Numerical Analysis and Modelling */
/*                     phone : + 49/30/84185-185 */
/*                     fax   : + 49/30/84185-107 */
/*                     e-mail: weimann@zib.de */

/* *    References: */

/*     /1/ P. Deuflhard: */
/*         Newton Methods for Nonlinear Problems. - */
/*         Affine Invariance and Adaptive Algorithms. */
/*         Series Computational Mathematics 35, Springer (2004) */

/*     /2/ U. Nowak, L. Weimann: */
/*         A Family of Newton Codes for Systems of Highly Nonlinear */
/*         Equations - Algorithm, Implementation, Application. */
/*         ZIB, Technical Report TR 90-10 (December 1990) */

/*  --------------------------------------------------------------- */

/* * Licence */
/*    You may use or modify this code for your own non commercial */
/*    purposes for an unlimited time. */
/*    In any case you should not deliver this code without a special */
/*    permission of ZIB. */
/*    In case you intend to use the code commercially, we oblige you */
/*    to sign an according licence agreement with ZIB. */

/* * Warranty */
/*    This code has been tested up to a certain level. Defects and */
/*    weaknesses, which may be included in the code, do not establish */
/*    any warranties by ZIB. ZIB does not take over any liabilities */
/*    which may follow from aquisition or application of this code. */

/* * Software status */
/*    This code is under care of ZIB and belongs to ZIB software class 1. */

/*     ------------------------------------------------------------ */

/* *    Summary: */
/*     ======== */
/*     Damped Newton-algorithm for systems of highly nonlinear */
/*     equations - damping strategy due to Ref. (1). */

/*     (The iteration is done by subroutine N1INT actually. NLEQ1E */
/*      calls the standard interface driver NLEQ1, which itself does */
/*      some house keeping and builds up workspace.) */

/*     Jacobian approximation by numerical differences. */

/*     The numerical solution of the arising linear equations is */
/*     done by means of the subroutines *GEFA and *GESL ( Gauss- */
/*     algorithm with column-pivoting and row-interchange; */
/*     replace '*' by 'S' or 'D' for single or double precision */
/*     version respectively). */

/*     ------------------------------------------------------------ */

/* *    External subroutine to be supplied by the user */
/*     ============================================== */

/*     FCN(N,X,F,IFAIL) Ext    Function subroutine - the problem function */
/*                             (The routine must be named exactly FCN !) */
/*       N              Int    Number of vector components (input) */
/*                             Must not be altered! */
/*       X(N)           Dble   Vector of unknowns (input) */
/*                             Must not be altered! */
/*       F(N)           Dble   Vector of function values (output) */
/*       IFAIL          Int    FCN evaluation-failure indicator. (output) */
/*                             On input:  Has always value 0 (zero). */
/*                             On output: Indicates failure of FCN eval- */
/*                                uation, if having a nonzero value. */
/*                             If <0: NLEQ1E will be terminated with */
/*                                    IFAIL returned via IERR. */
/*                             If =1: A new trial Newton iterate will be */
/*                                    computed, with the damping factor */
/*                                    reduced to it's half. */
/*                             If =2: A new trial Newton iterate will */
/*                                    computed, with the damping factor */
/*                                    reduced by a reduction factor, */
/*                                    which must be output through F(1) */
/*                                    by FCN, and it's value must be */
/*                                    >0 and < 1. */

/* *    Parameters list description */
/*     =========================== */

/*     Name   Type    In/Out Meaning */

/*     N      Int     In     Number of unknowns ( N .LE. 50 ) */
/*     X(N)   Dble    In     Initial estimate of the solution */
/*                    Out    Solution values ( or final values, */
/*                           respectively ) */
/*     RTOL   Dble    In     Required relative precision of */
/*                           solution components - */
/*                           RTOL.GE.EPMACH*TEN*N */
/*                    Out    Finally achieved accuracy */
/*     IERR   Int     Out    Return value parameter */
/*                           < 0 Termination forced by user function FCN */
/*                               due to IFAIL<0 on output, IERR is set */
/*                               to IFAIL */
/*                           = 0 successfull completion of the iteration, */
/*                               solution has been computed */
/*                           > 0 see list of error/warning messages below */

/* *   Error and warning messages: */
/*    =========================== */

/*      1    Termination, since Jacobian matrix became singular */
/*      2    Termination after 100 iterations */
/*      3    Termination, since damping factor became to small */
/*      4    Warning: Superlinear or quadratic convergence slowed down */
/*           near the solution. */
/*           Iteration has been stopped therefore with an approximation */
/*           of the solution not such accurate as requested by RTOL, */
/*           because possibly the RTOL requirement may be too stringent */
/*           (i.e. the nonlinear problem is ill-conditioned) */
/*      5    Warning: Iteration stopped with termination criterion */
/*           (using RTOL as requested precision) satisfied, but no */
/*           superlinear or quadratic convergence has been indicated yet. */
/*           Therefore, possibly the error estimate for the solution may */
/*           not match good enough the really achieved accuracy. */
/*     20    Bad input value to parameter N; 1 .LE. N .LE. 50 required */
/*     21    Nonpositive value for RTOL supplied */
/*     82    Termination, because user routine FCN returned with IFAIL>0 */

/*     Note   : in case of failure: */
/*        -    use better initial guess */
/*        -    or refine model */
/*        -    or use non-standard options and/or analytical Jacobian */
/*             via the standard interface NLEQ1 */

/* *    Machine dependent constants used: */
/*     ================================= */

/*     DOUBLE PRECISION EPMACH  in  N1PCHK, N1INT */
/*     DOUBLE PRECISION GREAT   in  N1PCHK */
/*     DOUBLE PRECISION SMALL   in  N1PCHK, N1INT, N1SCAL */

/* *    Subroutines called: NLEQ1 */

/*     ------------------------------------------------------------ */
/* *    End Prologue */

/*     Version: 2.3               Latest change: */
/*     ----------------------------------------- */

    /* Parameter adjustments */
    --x;

    /* Function Body */
/* *    Begin */
    niw = *n + 50;
    nrw = (*n + 13) * *n + 60;
/*     Checking dimensional parameter N */
    if (*n < 1 || *n > 50) {
	s_wsfe(&io___5);
	do_fio(&c__1, (char *)&c__50, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(integer));
	e_wsfe();
	*ierr = 20;
	return 0;
    }
    for (i__ = 1; i__ <= 50; ++i__) {
	iopt[i__ - 1] = 0;
/* L10: */
    }
    i__1 = niw;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iwk[i__ - 1] = 0;
/* L20: */
    }
    i__1 = nrw;
    for (i__ = 1; i__ <= i__1; ++i__) {
	rwk[i__ - 1] = 0.;
/* L30: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	xscal[i__ - 1] = 0.;
/* L40: */
    }
/*     Print errors, warnings, monitor and time monitor */
/*     to standard output */
    iopt[10] = 3;
    iopt[11] = 6;
    iopt[12] = 3;
    iopt[13] = 6;
    iopt[18] = 1;
    iopt[19] = 6;
/*     Maximum number of Newton iterations */
    iwk[30] = 100;

    nleq1_(n, (U_fp)fcn_, &dummy, &x[1], xscal, rtol, iopt, ierr, &c__100, 
	    iwk, &c__3210, rwk);
    if (*ierr == 82 && iwk[22] < 0) {
	*ierr = iwk[22];
    }
/*     End of subroutine NLEQ1E */
    return 0;
} /* nleq1e_ */

