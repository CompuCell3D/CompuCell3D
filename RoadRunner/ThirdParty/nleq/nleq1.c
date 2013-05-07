/* nleq1.f -- translated by f2c (version 20090411).
You must link the resulting object file with libf2c:
on Microsoft Windows system, link with libf2c.lib;
on Linux or Unix systems, link with .../path/to/libf2c.a -lm
or, if you install libf2c.a in a standard place, with -lf2c -lm
-- in that order, at the end of the command line, as in
cc *.o -lf2c -lm
Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

http://www.netlib.org/f2c/libf2c.zip
*/

#include "nleq1.h"
/* Table of constant values */

static integer c__1 = 1;
static integer c__0 = 0;
static integer c__2 = 2;
static integer c__3 = 3;
static integer c__4 = 4;
static integer c__5 = 5;
static integer c__9 = 9;

//#pragma comment(linker, "/EXPORT:NLEQ1=_NLEQ1@48")
/* Subroutine */
DLLEXPORT int STDCALL NLEQ1(integer *n, c_NLMFCN fcn, U_fp jac, doublereal *x,
	doublereal *xscal, doublereal *rtol, integer *iopt, integer *ierr,
	integer *liwk, integer *iwk, integer *lrwk, doublereal *rwk)
{

//DLLEXPORT int STDCALL NLEQ1(integer *n, U_fp fcn, U_fp jac, doublereal *x,
//	doublereal *xscal, doublereal *rtol, integer *iopt, integer *ierr,
//	integer *liwk, integer *iwk, integer *lrwk, doublereal *rwk)
//{	/* Initialized data */

	static char prodct[8] = "NLEQ1  ";

	/* Format strings */
	static char fmt_10000[] = "(\002   N L E Q 1  *****  V e r s i o n "
		" \002,\0022 . 3 ***\002,//,1x,\002Newton-Method \002,\002for the"
		" solution of nonlinear systems\002,//)";
	static char fmt_10050[] = "(\002 Real    Workspace declared as \002,i9"
		",\002 is used up to \002,i9,\002 (\002,f5.1,\002 percent)\002,//,"
		"\002 Integer Workspace declared as \002,i9,\002 is used up to"
		" \002,i9,\002 (\002,f5.1,\002 percent)\002,//)";
	static char fmt_10051[] = "(/,\002 N =\002,i4,//,\002 Prescribed relativ"
		"e \002,\002precision\002,d10.2,/)";
	static char fmt_10052[] = "(\002 The Jacobian is supplied by \002,a)";
	static char fmt_10055[] = "(\002 The Jacobian will be stored in \002,a"
		",\002 mode\002)";
	static char fmt_10056[] = "(\002 Lower bandwidth : \002,i3,\002   Upper "
		"bandwidth : \002,i3)";
	static char fmt_10057[] = "(\002 Automatic row scaling of the Jacobian i"
		"s \002,a,/)";
	static char fmt_10064[] = "(\002 Rank-1 updates are \002,a)";
	static char fmt_10065[] = "(\002 Problem is specified as being \002,a)";
	static char fmt_10066[] = "(\002 Bounded damping strategy is \002,a,:,/"
		",\002 Bounding factor is \002,d10.3)";
	static char fmt_10067[] = "(\002 Special mode: \002,a,\002 Newton iterat"
		"ion will be done\002)";
	static char fmt_10068[] = "(\002 Maximum permitted number of iteration s"
		"teps : \002,i6)";
	static char fmt_10069[] = "(//,\002 Internal parameters:\002,//,\002 Sta"
		"rting value for damping factor FCSTART = \002,d9.2,/,\002 Minimu"
		"m allowed damping factor FCMIN = \002,d9.2,/,\002 Rank-1 updates"
		" decision parameter SIGMA = \002,d9.2)";
	static char fmt_10080[] = "(/,\002   ******  Statistics * \002,a8,\002 *"
		"******\002,/,\002   ***  Newton iterations : \002,i7,\002  **"
		"*\002,/,\002   ***  Corrector steps   : \002,i7,\002  ***\002,/"
		",\002   ***  Rejected rk-1 st. : \002,i7,\002  ***\002,/,\002   "
		"***  Jacobian eval.    : \002,i7,\002  ***\002,/,\002   ***  Fun"
		"ction eval.    : \002,i7,\002  ***\002,/,\002   ***  ...  for Ja"
		"cobian : \002,i7,\002  ***\002,/,\002   ************************"
		"*************\002,/)";
	static char fmt_10090[] = "(//,\002 \002,20(\002*\002),\002Workspace Err"
		"or\002,20(\002*\002))";
	static char fmt_10091[] = "(/,\002 Real Workspace dimensioned as\002,1x,"
		"i9,1x,\002must be enlarged at least up to \002,i9,//)";
	static char fmt_10092[] = "(/,\002 Integer Workspace dimensioned as \002"
		",i9,\002 must be enlarged at least up \002,\002to \002,i9,//)";

	/* System generated locals */
	integer i__1;

	/* Builtin functions */
	integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

	/* Local variables */
	static integer m1, m2, l4, l5, l6, l7, l8, l9;
	static doublereal fc;
	static integer l11, l12, l13, l41, l51, l61, l62, l63, l71, l14, l20, ml,
		mu, l121, niw, nrw;
	extern /* Subroutine */ int n1int_(integer *, U_fp, U_fp, doublereal *,
		doublereal *, doublereal *, integer *, integer *, integer *,
		integer *, integer *, doublereal *, integer *, integer *, integer
		*, integer *, integer *, integer *, integer *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, doublereal *, integer *, integer *, integer *,
		integer *, integer *, integer *, integer *, integer *, integer *,
		integer *, integer *, integer *, integer *, integer *, integer *,
		logical *);
	static doublereal fcmin, perci, percr;
	static logical qvchk, qsucc;
	static integer luerr, lumon, lutim, nbroy, lusol, mstor;
	extern /* Subroutine */ int n1pchk_(integer *, doublereal *, doublereal *,
		doublereal *, integer *, integer *, integer *, integer *,
		integer *, doublereal *);
	static logical qrank1;
	static integer jacgen;
	static logical qbdamp;
	extern /* Subroutine */ int mondef_(integer *, char *, ftnlen);
	static integer nifrin;
	extern /* Subroutine */ int monini_(char *, integer *, ftnlen);
	static logical qinimo;
	static integer nonlin, nrfrin, nitmax, niwkfr;
	static logical qfcstr;
	extern /* Subroutine */ int monhlt_(void);
	static logical qsimpl;
	static integer mprerr, mprmon, nrwkfr, mprtim, mprsol;
	extern /* Subroutine */ int monprt_(void), monstr_(integer *);

	/* Fortran I/O blocks */
	static cilist io___14 = { 0, 0, 0, fmt_10000, 0 };
	static cilist io___50 = { 0, 0, 0, fmt_10050, 0 };
	static cilist io___51 = { 0, 0, 0, fmt_10051, 0 };
	static cilist io___52 = { 0, 0, 0, fmt_10052, 0 };
	static cilist io___53 = { 0, 0, 0, fmt_10052, 0 };
	static cilist io___54 = { 0, 0, 0, fmt_10052, 0 };
	static cilist io___55 = { 0, 0, 0, fmt_10055, 0 };
	static cilist io___56 = { 0, 0, 0, fmt_10055, 0 };
	static cilist io___57 = { 0, 0, 0, fmt_10056, 0 };
	static cilist io___58 = { 0, 0, 0, fmt_10057, 0 };
	static cilist io___59 = { 0, 0, 0, fmt_10057, 0 };
	static cilist io___62 = { 0, 0, 0, fmt_10064, 0 };
	static cilist io___63 = { 0, 0, 0, fmt_10064, 0 };
	static cilist io___64 = { 0, 0, 0, fmt_10065, 0 };
	static cilist io___65 = { 0, 0, 0, fmt_10065, 0 };
	static cilist io___66 = { 0, 0, 0, fmt_10065, 0 };
	static cilist io___67 = { 0, 0, 0, fmt_10065, 0 };
	static cilist io___68 = { 0, 0, 0, fmt_10066, 0 };
	static cilist io___69 = { 0, 0, 0, fmt_10066, 0 };
	static cilist io___70 = { 0, 0, 0, fmt_10067, 0 };
	static cilist io___71 = { 0, 0, 0, fmt_10067, 0 };
	static cilist io___73 = { 0, 0, 0, fmt_10068, 0 };
	static cilist io___77 = { 0, 0, 0, fmt_10069, 0 };
	static cilist io___78 = { 0, 0, 0, fmt_10080, 0 };
	static cilist io___79 = { 0, 0, 0, fmt_10090, 0 };
	static cilist io___80 = { 0, 0, 0, fmt_10091, 0 };
	static cilist io___81 = { 0, 0, 0, fmt_10092, 0 };


	/* *    Begin Prologue NLEQ1 */
	/*     ------------------------------------------------------------ */

	/* *  Title */

	/*     Numerical solution of nonlinear (NL) equations (EQ) */
	/*     especially designed for numerically sensitive problems. */

	/* *  Written by        U. Nowak, L. Weimann */
	/* *  Purpose           Solution of systems of highly nonlinear equations */
	/* *  Method            Damped affine invariant Newton method */
	/*                     (see references below) */
	/* *  Category          F2a. - Systems of nonlinear equations */
	/* *  Keywords          Nonlinear equations, Newton methods */
	/* *  Version           2.4 */
	/* *  Revision          May 2009 */
	/* *  Latest Change     May 2009 */
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
	/*    which may follow from acquisition or application of this code. */

	/* * Software status */
	/*    This code is under care of ZIB and belongs to ZIB software class 1. */

	/*     ------------------------------------------------------------ */

	/* *    Summary: */
	/*     ======== */
	/*     Damped Newton-algorithm for systems of highly nonlinear */
	/*     equations - damping strategy due to Ref. (1). */

	/*     (The iteration is done by subroutine N1INT currently. NLEQ1 */
	/*      itself does some house keeping and builds up workspace.) */

	/*     Jacobian approximation by numerical differences or user */
	/*     supplied subroutine JAC. */

	/*     The numerical solution of the arising linear equations is */
	/*     done by means of the subroutines *GETRF and *GETRS ( Gauss- */
	/*     algorithm with column-pivoting and row-interchange ) in the */
	/*     dense matrix case, or by the subroutines *GBTRF and *GBTRS in */
	/*     the band matrix case from LAPACK (replace '*' by 'S' or 'D' */
	/*     for single or double precision version respectively). */
	/*     For special purposes these routines may be substituted. */

	/*     This is a driver routine for the core solver N1INT. */

	/*     ------------------------------------------------------------ */

	/* *    Parameters list description (* marks inout parameters) */
	/*     ====================================================== */

	/* *    External subroutines (to be supplied by the user) */
	/*     ================================================= */

	/*     (Caution: Arguments declared as (input) must not */
	/*               be altered by the user subroutines ! ) */

	/*     FCN(N,X,F,IFAIL) Ext    Function subroutine */
	/*       N              Int    Number of vector components (input) */
	/*       X(N)           Dble   Vector of unknowns (input) */
	/*       F(N)           Dble   Vector of function values (output) */
	/*       IFAIL          Int    FCN evaluation-failure indicator. (output) */
	/*                             On input:  Has always value 0 (zero). */
	/*                             On output: Indicates failure of FCN eval- */
	/*                                uation, if having a value <= 2. */
	/*                             If <0: NLEQ1 will be terminated with */
	/*                                    error code = 82, and IFAIL stored */
	/*                                    to IWK(23). */
	/*                             If =1: A new trial Newton iterate will */
	/*                                    computed, with the damping factor */
	/*                                    reduced to it's half. */
	/*                             If =2: A new trial Newton iterate will */
	/*                                    computed, with the damping factor */
	/*                                    reduced by a reduct. factor, which */
	/*                                    must be output through F(1) by FCN, */
	/*                                    and it's value must be >0 and < 1. */
	/*                             Note, that if IFAIL = 1 or 2, additional */
	/*                             conditions concerning the damping factor, */
	/*                             e.g. the minimum damping factor or the */
	/*                             bounded damping strategy may also influ- */
	/*                             ence the value of the reduced damping */
	/*                             factor. */

	/*     JAC(N,LDJAC,X,DFDX,IFAIL) */
	/*                       Ext    Jacobian matrix subroutine */
	/*       N                 Int    Number of vector components (input) */
	/*       LDJAC             Int    Leading dimension of Jacobian array */
	/*                                (input) */
	/*       X(N)              Dble   Vector of unknowns (input) */
	/*       DFDX(LDJAC,N)     Dble   DFDX(i,k): partial derivative of */
	/*                                I-th component of FCN with respect */
	/*                                to X(k) (output) */
	/*       IFAIL             Int    JAC evaluation-failure indicator. */
	/*                                (output) */
	/*                                Has always value 0 (zero) on input. */
	/*                                Indicates failure of JAC evaluation */
	/*                                and causes termination of NLEQ1, */
	/*                                if set to a negative value on output */


	/* *    Input parameters of NLEQ1 */
	/*     ========================= */

	/*     N              Int    Number of unknowns */
	/*   * X(N)           Dble   Initial estimate of the solution */
	/*   * XSCAL(N)       Dble   User scaling (lower threshold) of the */
	/*                           iteration vector X(N) */
	/*   * RTOL           Dble   Required relative precision of */
	/*                           solution components - */
	/*                           RTOL.GE.EPMACH*TEN*N */
	/*   * IOPT(50)       Int    Array of run-time options. Set to zero */
	/*                           to get default values (details see below) */

	/* *    Output parameters of NLEQ1 */
	/*     ========================== */

	/*   * X(N)           Dble   Solution values ( or final values, */
	/*                           respectively ) */
	/*   * XSCAL(N)       Dble   After return with IERR.GE.0, it contains */
	/*                           the latest internal scaling vector used */
	/*                           After return with IERR.EQ.-1 in onestep- */
	/*                           mode it contains a possibly adapted */
	/*                           (as described below) user scaling vector: */
	/*                           If (XSCAL(I).LT. SMALL) XSCAL(I) = SMALL , */
	/*                           If (XSCAL(I).GT. GREAT) XSCAL(I) = GREAT . */
	/*                           For SMALL and GREAT, see section machine */
	/*                           constants below  and regard note 1. */
	/*   * RTOL           Dble   Finally achieved (relative) accuracy. */
	/*                           The estimated absolute error of component i */
	/*                           of x_out is approximately given by */
	/*                             abs_err(i) = RTOL * XSCAL_out(i) , */
	/*                           where (approximately) */
	/*                             XSCAL_out(i) = */
	/*                                max(abs(X_out(i)),XSCAL_in(i)). */
	/*     IERR           Int    Return value parameter */
	/*                           =-1 sucessfull completion of one iteration */
	/*                               step, subsequent iterations are needed */
	/*                               to get a solution. (stepwise mode only) */
	/*                           = 0 successfull completion of the iteration, */
	/*                               solution has been computed */
	/*                           > 0 see list of error messages below */

	/*     Note 1. */
	/*        The machine dependent values SMALL and EPMACH are */
	/*        gained from calls of the ZIBCONST. */

	/* *    Workspace parameters of NLEQ1 */
	/*     ============================= */

	/*     LIWK           Int    Declared dimension of integer workspace. */
	/*                           Required minimum (for standard linear system */
	/*                           solver) : N+50 */
	/*  *  IWK(LIWK)      Int    Integer Workspace */
	/*     LRWK           Int    Declared dimension of real workspace. */
	/*                           Required minimum (for standard linear system */
	/*                           solver and Jacobian computed by numerical */
	/*                           approximation - if the Jacobian is computed */
	/*                           by a user subroutine JAC, decrease the */
	/*                           expressions noted below by N): */
	/*                           for full case Jacobian: (N+NBROY+13)*N+61 */
	/*                           for a band-matrix Jacobian: */
	/*                              (2*ML+MU+NBROY+14)*N+61 with */
	/*                           ML = lower bandwidth , MU = upper bandwidth */
	/*                           NBROY = Maximum number of Broyden steps */
	/*                           (Default: if Broyden steps are enabled, e.g. */
	/*                                                IOPT(32)=1            - */
	/*                                       NBROY=N (full Jacobian), */
	/*                                            =ML+MU+1 (band Jacobian), */
	/*                                       but at least NBROY=10 */
	/*                                     else (if IOPT(32)=0) - */
	/*                                       NBROY=0 ; */
	/*                            see equally named IOPT and IWK-fields below) */
	/*   * RWK(LRWK)      Dble   Real Workspace */

	/*     Note 2a.  A test on sufficient workspace is made. If this */
	/*               test fails, IERR is set to 10 and an error-message */
	/*               is issued from which the minimum of required */
	/*               workspace size can be obtained. */

	/*     Note 2b.  The first 50 elements of IWK and RWK are partially */
	/*               used as input for internal algorithm parameters (for */
	/*               details, see below). In order to set the default values */
	/*               of these parameters, the fields must be set to zero. */
	/*               Therefore, it's recommanded always to initialize the */
	/*               first 50 elements of both workspaces to zero. */

	/* *   Options IOPT: */
	/*    ============= */

	/*     Pos. Name   Default  Meaning */

	/*       1  QSUCC  0        =0 (.FALSE.) initial call: */
	/*                             NLEQ1 is not yet initialized, i.e. this is */
	/*                             the first call for this nonlinear system. */
	/*                             At successfull return with MODE=1, */
	/*                             QSUCC is set to 1. */
	/*                          =1 (.TRUE.) successive call: */
	/*                             NLEQ1 is initialized already and is now */
	/*                             called to perform one or more following */
	/*                             Newton-iteration steps. */
	/*                             ATTENTION: */
	/*                                Don't destroy the contents of */
	/*                                IOPT(i) for 1 <= i <= 50 , */
	/*                                IWK(j)  for 1 <= j < NIWKFR and */
	/*                                RWK(k)  for 1 <= k < NRWKFR. */
	/*                                (Nevertheless, some of the options, e.g. */
	/*                                 FCMIN, SIGMA, MPR..., can be modified */
	/*                                 before successive calls.) */
	/*       2  MODE   0        =0 Standard mode initial call: */
	/*                             Return when the required accuracy for the */
	/*                             iteration vector is reached. User defined */
	/*                             parameters are evaluated and checked. */
	/*                             Standard mode successive call: */
	/*                             If NLEQ1 was called previously with MODE=1, */
	/*                             it performs all remaining iteration steps. */
	/*                          =1 Stepwise mode: */
	/*                             Return after one Newton iteration step. */
	/*       3  JACGEN 0        Method of Jacobian generation */
	/*                          =0 Standard method is JACGEN=2 */
	/*                          =1 User supplied subroutine JAC will be */
	/*                             called to generate Jacobian matrix */
	/*                          =2 Jacobian approximation by numerical */
	/*                             differentation (see subroutines N1JAC */
	/*                             and N1JACB) */
	/*                          =3 Jacobian approximation by numerical */
	/*                             differentation with feedback control */
	/*                             (see subroutines N1JCF and N1JCFB) */
	/*       4  MSTOR  0        =0 The Jacobian A is a dense matrix */
	/*                          =1 A is a band matrix */
	/*       5                  Reserved */
	/*       6  ML     0        Lower bandwidth of A (excluding the */
	/*                          diagonal); */
	/*                          IOPT(6) ignored, if IOPT(4).NE.1 */
	/*       7  MU     0        Upper bandwidth of A (excluding the */
	/*                          diagonal); */
	/*                          IOPT(7) ignored, if IOPT(4).NE.1 */
	/*       8                  Reserved */
	/*       9  ISCAL  0        Determines how to scale the iterate-vector: */
	/*                          =0 The user supplied scaling vector XSCAL is */
	/*                             used as a (componentwise) lower threshold */
	/*                             of the current scaling vector */
	/*                          =1 The vector XSCAL is always used as the */
	/*                             current scaling vector */
	/*      10                  Reserved */
	/*      11  MPRERR 0        Print error messages */
	/*                          =0 No output */
	/*                          =1 Error messages */
	/*                          =2 Warnings additionally */
	/*                          =3 Informal messages additionally */
	/*      12  LUERR  6        Logical unit number for error messages */
	/*      13  MPRMON 0        Print iteration Monitor */
	/*                          =0 No output */
	/*                          =1 Standard output */
	/*                          =2 Summary iteration monitor additionally */
	/*                          =3 Detailed iteration monitor additionally */
	/*                          =4,5,6 Outputs with increasing level addi- */
	/*                             tional increasing information for code */
	/*                             testing purposes. Level 6 produces */
	/*                             in general extremely large output! */
	/*      14  LUMON  6        Logical unit number for iteration monitor */
	/*      15  MPRSOL 0        Print solutions */
	/*                          =0 No output */
	/*                          =1 Initial values and solution values */
	/*                          =2 Intermediate iterates additionally */
	/*      16  LUSOL  6        Logical unit number for solutions */
	/*      17..18              Reserved */
	/*      19  MPRTIM 0        Output level for the time monitor */
	/*                          = 0 : no time measurement and no output */
	/*                          = 1 : time measurement will be done and */
	/*                                summary output will be written - */
	/*                                regard note 5a. */
	/*      20  LUTIM  6        Logical output unit for time monitor */
	/*      21..30              Reserved */
	/*      31  NONLIN 3        Problem type specification */
	/*                          =1 Linear problem */
	/*                             Warning: If specified, no check will be */
	/*                             done, if the problem is really linear, and */
	/*                             NLEQ1 terminates unconditionally after one */
	/*                             Newton-iteration step. */
	/*                          =2 Mildly nonlinear problem */
	/*                          =3 Highly nonlinear problem */
	/*                          =4 Extremely nonlinear problem */
	/*      32  QRANK1 0        =0 (.FALSE.) Rank-1 updates by Broyden- */
	/*                             approximation are inhibited. */
	/*                          =1 (.TRUE.) Rank-1 updates by Broyden- */
	/*                             approximation are allowed. */
	/*      33  QORDI  0        =0 (.FALSE.) Standard program mode */
	/*                          =1 (.TRUE.)  Special program mode: */
	/*                             Ordinary Newton iteration is done, e.g.: */
	/*                             No damping strategy and no monotonicity */
	/*                             test is applied */
	/*      34  QSIMPL 0        =0 (.FALSE.) Standard program mode */
	/*                          =1 (.TRUE.)  Special program mode: */
	/*                             Simplified Newton iteration is done, e.g.: */
	/*                             The Jacobian computed at the starting */
	/*                             point is fixed for all subsequent */
	/*                             iteration steps, and */
	/*                             no damping strategy and no monotonicity */
	/*                             test is applied. */
	/*      35  QNSCAL 0        Inhibit automatic row scaling: */
	/*                          =0 (.FALSE.) Automatic row scaling of */
	/*                             the linear system is activ: */
	/*                             Rows i=1,...,N will be divided by */
	/*                             max j=1,...,N (abs(a(i,j))) */
	/*                          =1 (.TRUE.) No row scaling of the linear */
	/*                             system. Recommended only for well row- */
	/*                             scaled nonlinear systems. */
	/*      36..37              Reserved */
	/*      38  IBDAMP          Bounded damping strategy switch: */
	/*                          =0 The default switch takes place, dependent */
	/*                             on the setting of NONLIN (=IOPT(31)): */
	/*                             NONLIN = 0,1,2,3 -> IBDAMP = off , */
	/*                             NONLIN = 4 -> IBDAMP = on */
	/*                          =1 means always IBDAMP = on */
	/*                          =2 means always IBDAMP = off */
	/*      39  IORMON          Convergence order monitor */
	/*                          =0 Standard option is IORMON=2 */
	/*                          =1 Convergence order is not checked, */
	/*                             the iteration will be always proceeded */
	/*                             until the solution has the required */
	/*                             precision RTOL (or some error condition */
	/*                             occured) */
	/*                          =2 Use additional 'weak stop' criterion: */
	/*                             Convergence order is monitored */
	/*                             and termination due to slowdown of the */
	/*                             convergence may occur. */
	/*                          =3 Use additional 'hard stop' criterion: */
	/*                             Convergence order is monitored */
	/*                             and termination due to superlinear */
	/*                             convergence slowdown may occur. */
	/*                          In case of termination due to convergence */
	/*                          slowdown, the warning code IERR=4 will be */
	/*                          set. */
	/*                          In cases, where the Newton iteration con- */
	/*                          verges but superlinear convergence order has */
	/*                          never been detected, the warning code IERR=5 */
	/*                          is returned. */
	/*      40..45              Reserved */
	/*      46..50              User options (see note 5b) */

	/*     Note 3: */
	/*         If NLEQ1 terminates with IERR=2 (maximum iterations) */
	/*         or  IERR=3 (small damping factor), you may try to continue */
	/*         the iteration by increasing NITMAX or decreasing FCMIN */
	/*         (see RWK) and setting QSUCC to 1. */

	/*     Note 4 : Storage of user supplied banded Jacobian */
	/*        In the band matrix case, the following lines may build */
	/*        up the analytic Jacobian A; */
	/*        Here AFL denotes the quadratic matrix A in dense form, */
	/*        and ABD the rectangular matrix A in banded form : */

	/*                   ML = IOPT(6) */
	/*                   MU = IOPT(7) */
	/*                   MH = MU+1 */
	/*                   DO 20 J = 1,N */
	/*                     I1 = MAX0(1,J-MU) */
	/*                     I2 = MIN0(N,J+ML) */
	/*                     DO 10 I = I1,I2 */
	/*                       K = I-J+MH */
	/*                       ABD(K,J) = AFL(I,J) */
	/*           10        CONTINUE */
	/*           20      CONTINUE */

	/*         The total number of rows needed in  ABD  is  ML+MU+1 . */
	/*         The  MU by MU  upper left triangle and the */
	/*         ML by ML  lower right triangle are not referenced. */

	/*     Note 5a: */
	/*        The integrated time monitor calls the machine dependent */
	/*        subroutine SECOND to get the current time stamp in form */
	/*        of a real number (Single precision). As delivered, this */
	/*        subroutine always return 0.0 as time stamp value. Refer */
	/*        to the compiler- or library manual of the FORTRAN compiler */
	/*        which you currently use to find out how to get the current */
	/*        time stamp on your machine. */

	/*     Note 5b: */
	/*         The user options may be interpreted by the user replacable */
	/*         routines N1SOUT, N1FACT, N1SOLV - the distributed version */
	/*         of N1SOUT currently uses IOPT(46) as follows: */
	/*         0 = standard plotdata output (may be postprocessed by a user- */
	/*             written graphical program) */
	/*         1 = plotdata output is suitable as input to the graphical */
	/*             package GRAZIL (based on GKS), which has been developed */
	/*             at ZIB. */


	/* *   Optional INTEGER input/output in IWK: */
	/*    ======================================= */

	/*     Pos. Name          Meaning */

	/*      1   NITER  IN/OUT Number of Newton-iterations */
	/*      2                 reserved */
	/*      3   NCORR  IN/OUT Number of corrector steps */
	/*      4   NFCN   IN/OUT Number of FCN-evaluations */
	/*      5   NJAC   IN/OUT Number of Jacobian generations or */
	/*                        JAC-calls */
	/*      6                 reserved */
	/*      7                 reserved */
	/*      8   NFCNJ  IN/OUT Number of FCN-evaluations for Jacobian */
	/*                        approximation */
	/*      9   NREJR1 IN/OUT Number of rejected Newton iteration steps */
	/*                        done with a rank-1 approximated Jacobian */
	/*     10..11             Reserved */
	/*     12   IDCODE IN/OUT Output: The 8 decimal digits program identi- */
	/*                        fication number ppppvvvv, consisting of the */
	/*                        program code pppp and the version code vvvv. */
	/*                        Input: If containing a negative number, */
	/*                        it will only be overwritten by the identi- */
	/*                        fication number, immediately followed by */
	/*                        a return to the calling program. */
	/*     13..15             Reserved */
	/*     16   NIWKFR OUT    First element of IWK which is free to be used */
	/*                        as workspace between Newton iteration steps */
	/*                        for standard linear solvers: 51 */
	/*     17   NRWKFR OUT    First element of RWK which is free to be used */
	/*                        as workspace between Newton iteration steps. */
	/*                        For standard linear solvers and numerically */
	/*                        approximated Jacobian computed by one of the */
	/*                        expressions: */
	/*                        (N+7+NBROY)*N+61        for a full Jacobian */
	/*                        (2*ML+MU+8+NBROY)*N+61  for a banded Jacobian */
	/*                        If the Jacobian is computed by a user routine */
	/*                        JAC, subtract N in both expressions. */
	/*     18   LIWKA  OUT    Length of IWK currently required */
	/*     19   LRWKA  OUT    Length of RWK currently required */
	/*     20..22             Reserved */
	/*     23   IFAIL  OUT    Set in case of failure of N1FACT (IERR=80), */
	/*                        N1SOLV (IERR=81), FCN (IERR=82) or JAC(IERR=83) */
	/*                        to the nonzero IFAIL value returned by the */
	/*                        routine indicating the failure . */
	/*     24   ICONV  OUT    Current status of of the convergence monitor */
	/*                        (only if convergence order monitor is on - */
	/*                         see IORMON(=IOPT(39))) */
	/*                        =0: No convergence indicated yet */
	/*                        =1: Damping factor is 1.0d0 */
	/*                        =2: Superlinear convergence in progress */
	/*                        =3: Quadratic convergence in progress */
	/*     25..30             Reserved */
	/*     31   NITMAX IN     Maximum number of permitted iteration */
	/*                        steps (default: 50) */
	/*     32                 Reserved */
	/*     33   NEW    IN/OUT Count of consecutive rank-1 updates */
	/*     34..35             Reserved */
	/*     36   NBROY  IN     Maximum number of possible consecutive */
	/*                        iterative Broyden steps. The total real */
	/*                        workspace needed (RWK) depends on this value */
	/*                        (see LRWK above). */
	/*                        Default is N (see parameter N), */
	/*                        if MSTOR=0 (=IOPT(4)), */
	/*                        and ML+MU+1 (=IOPT(6)+IOPT(7)+1), if MSTOR=1 */
	/*                        (but minimum is always 10) - */
	/*                        provided that Broyden is allowed. */
	/*                        If Broyden is inhibited, NBROY is always set to */
	/*                        zero. */
	/*     37..50             Reserved */

	/* *   Optional REAL input/output in RWK: */
	/*    ==================================== */

	/*     Pos. Name          Meaning */

	/*      1..16             Reserved */
	/*     17   CONV   OUT    The achieved relative accuracy after the */
	/*                        current step */
	/*     18   SUMX   OUT    Natural level (not Normx of printouts) */
	/*                        of the current iterate, i.e. Sum(DX(i)**2), */
	/*                        where DX = scaled Newton correction. */
	/*     19   DLEVF  OUT    Standard level (not Normf of printouts) */
	/*                        of the current iterate, i.e. Norm2(F(X)), */
	/*                        where F =  nonlinear problem function. */
	/*     20   FCBND  IN     Bounded damping strategy restriction factor */
	/*                        (Default is 10) */
	/*     21   FCSTRT IN     Damping factor for first Newton iteration - */
	/*                        overrides option NONLIN, if set (see note 6) */
	/*     22   FCMIN  IN     Minimal allowed damping factor (see note 6) */
	/*     23   SIGMA  IN     Broyden-approximation decision parameter */
	/*                        Required choice: SIGMA.GE.1. Increasing this */
	/*                        parameter make it less probable that the algo- */
	/*                        rith performs rank-1 updates. */
	/*                        Rank-1 updates are inhibited, if */
	/*                        SIGMA.GT.1/FCMIN is set. (see note 6) */
	/*     24   SIGMA2 IN     Decision parameter about increasing damping */
	/*                        factor to corrector if predictor is small. */
	/*                        Required choice: SIGMA2.GE.1. Increasing this */
	/*                        parameter make it less probable that the algo- */
	/*                        rith performs rank-1 updates. */
	/*     25                 Reserved */
	/*     26   AJDEL  IN     Jacobian approximation without feedback: */
	/*                        Relative pertubation for components */
	/*                        (Default: sqrt(epmach*10), epmach: relative */
	/*                         machine precision) */
	/*     27   AJMIN  IN     Jacobian approximation without feedback: */
	/*                        Threshold value (Default: 0.0d0) */
	/*                          The absolute pertubation for component k is */
	/*                          computed by */
	/*                          DELX := AJDEL*max(abs(Xk),AJMIN) */
	/*     28  ETADIF  IN     Jacobian approximation with feedback: */
	/*                        Target value for relative pertubation ETA of X */
	/*                        (Default: 1.0d-6) */
	/*     29  ETAINI  IN     Jacobian approximation with feedback: */
	/*                        Initial value for denominator differences */
	/*                        (Default: 1.0d-6) */
	/*     30..50             Reserved */

	/*     Note 6: */
	/*       The default values of the internal parameters may be obtained */
	/*       from the monitor output with at least IOPT field MPRMON set to 2 */
	/*       and by initializing the corresponding RWK-fields to zero. */

	/* *   Error and warning messages: */
	/*    =========================== */

	/*      1    Termination, since jacobian matrix became singular */
	/*      2    Termination after NITMAX iterations ( as indicated by */
	/*           input parameter NITMAX=IWK(31) ) */
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
	/*     10    Integer or real workspace too small */
	/*     20    Bad input to dimensional parameter N */
	/*     21    Nonpositive value for RTOL supplied */
	/*     22    Negative scaling value via vector XSCAL supplied */
	/*     30    One or more fields specified in IOPT are invalid */
	/*           (for more information, see error-printout) */
	/*     80    Error signalled by linear solver routine N1FACT, */
	/*           for more detailed information see IFAIL-value */
	/*           stored to IWK(23) */
	/*     81    Error signalled by linear solver routine N1SOLV, */
	/*           for more detailed information see IFAIL-value */
	/*           stored to IWK(23) */
	/*           (not used by standard routine N1SOLV) */
	/*     82    Error signalled by user routine FCN (Nonzero value */
	/*           returned via IFAIL-flag; stored to IWK(23) ) */
	/*     83    Error signalled by user routine JAC (Nonzero value */
	/*           returned via IFAIL-flag; stored to IWK(23) ) */

	/*     Note 7 : in case of failure: */
	/*        -    use non-standard options */
	/*        -    or turn to Newton-algorithm with rank strategy */
	/*        -    use another initial guess */
	/*        -    or reformulate model */
	/*        -    or apply continuation techniques */

	/* *    Machine dependent constants used: */
	/*     ================================= */

	/*     DOUBLE PRECISION EPMACH  in  N1PCHK, N1INT */
	/*     DOUBLE PRECISION GREAT   in  N1PCHK */
	/*     DOUBLE PRECISION SMALL   in  N1PCHK, N1INT, N1SCAL */

	/* *    Subroutines called: N1PCHK, N1INT */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */

	/* *    Summary of changes: */
	/*     =================== */

	/*     2.2.1  91, May 30     Time monitor included */
	/*     2.2.2  91, May 30     Bounded damping strategy implemented */
	/*     2.2.3  91, June 19    AJDEL, AJMIN as RWK-options for JACGEN.EQ.2, */
	/*                           ETADIF, ETAINI as RWK-opts. for JACGEN.EQ.3 */
	/*                           FCN-count changed for anal. Jacobian */
	/*     2.2.4  91, August  9  Convergence order monitor included */
	/*     2.2.5  91, August 13  Standard Broyden updates replaced by */
	/*                           iterative Broyden */
	/*     2.2.6  91, Sept.  16  Damping factor reduction by FCN-fail imple- */
	/*                           mented */
	/*     2.3    91, Dec.   20  New Release for CodeLib */
	/*            92, March  11  Level of accepted simplified correction */
	/*                           stored to RWK(IRWKI+4) */
	/*            00, July   12  RTOL output-value bug fixed */
	/*            06, Jan.   24  IERR=5 no longer returned if residuum of */
	/*                           final iterate is exactly zero */
	/*     2.4    09, May    29  Changed routines N1FACT and N1SOLV to use */
	/*                           LAPACK Routines DGETRF, DGETRS, DGBTRF and */
	/*                           DGBTRS instead of LINPACK routines to solve */
	/*                           linear systems. */
	/*            10, July   26  Subroutine N1INT: Initialization of unitialized */
	/*                           Variable FCMON fixed. */

	/*     ------------------------------------------------------------ */

	/*     PARAMETER (IRWKI=xx, LRWKI=yy) */
	/*     IRWKI: Start position of internally used RWK part */
	/*     LRWKI: Length of internally used RWK part */
	/*     (current values see parameter statement below) */

	/*     INTEGER L4,L5,L51,L6,L61,L62,L63,L7,L71,L8,L9,L10,L11,L12,L121, */
	/*             L13,L14,L20 */
	/*     Starting positions in RWK of formal array parameters of internal */
	/*     routine N1INT (dynamically determined in driver routine NLEQ1, */
	/*     dependent on N and options setting) */

	/*     Further RWK positions (only internally used) */

	/*     Position  Name     Meaning */

	/*     IRWKI     FCKEEP   Damping factor of previous successfull iter. */
	/*     IRWKI+1   FCA      Previous damping factor */
	/*     IRWKI+2   FCPRI    A priori estimate of damping factor */
	/*     IRWKI+3   DMYCOR   Number My of latest corrector damping factor */
	/*                        (kept for use in rank-1 decision criterium) */
	/*     IRWKI+4   SUMXS    natural level of accepted simplified correction */
	/*     IRWKI+(5..LRWKI-1) Free */

	/*     Internal arrays stored in RWK (see routine N1INT for descriptions) */

	/*     Position  Array         Type   Remarks */

	/*     L4        A(M1,N)       Perm   M1=N (full mode) or */
	/*                                    M1=2*IOPT(6)+IOPT(7)+1 (band mode) */
	/*     L41       DXSAVE(N,NBROY) */
	/*                             Perm   NBROY=IWK(36) (Default: N or 0) */
	/*     L5        DX(N)         Perm */
	/*     L51       DXQ(N)        Perm */
	/*     L6        XA(N)         Perm */
	/*     L61       F(N)          Perm */
	/*     L62       FW(N)         Perm */
	/*     L63       XWA(N)        Perm */
	/*     L7        FA(N)         Perm */
	/*     L71       ETA(N)        Perm   Only used for JACGEN=IOPT(3)=3 */
	/*     L9        XW(N)         Temp */
	/*     L11       DXQA(N)       Temp */
	/*     L12       T1(N)         Temp */
	/*     L121      T2(N)         Temp */
	/*     L13       T3(N)         Temp */
	/*     L14                     Temp   Start position of array workspace */
	/*                                    needed for linear solver */


	/*     Which version ? */

	/*     Version: 2.4.0.1           Latest change: */
	/*     ----------------------------------------- */

	/* Parameter adjustments */
	--xscal;
	--x;
	--iopt;
	--iwk;
	--rwk;

	/* Function Body */
	/* *    Begin */
	*ierr = 0;
	qvchk = iwk[12] < 0;
	iwk[12] = 21112401;
	if (qvchk) {
		return 0;
	}
	/*        Print error messages? */
	mprerr = iopt[11];
	luerr = iopt[12];
	if (luerr == 0) {
		luerr = 6;
		iopt[12] = luerr;
	}
	/*        Print iteration monitor? */
	mprmon = iopt[13];
	lumon = iopt[14];
	if (lumon <= 0 || lumon > 99) {
		lumon = 6;
		iopt[14] = lumon;
	}
	/*        Print intermediate solutions? */
	mprsol = iopt[15];
	lusol = iopt[16];
	if (lusol == 0) {
		lusol = 6;
		iopt[16] = lusol;
	}
	/*        Print time summary statistics? */
	mprtim = iopt[19];
	lutim = iopt[20];
	if (lutim == 0) {
		lutim = 6;
		iopt[20] = lutim;
	}
	qsucc = iopt[1] == 1;
	qinimo = mprmon >= 1 && ! qsucc;
	/*     Print NLEQ1 heading lines */
	if (qinimo) {
		/* L10000: */
		io___14.ciunit = lumon;
		s_wsfe(&io___14);
		e_wsfe();
	}
	/*     Check input parameters and options */
	n1pchk_(n, &x[1], &xscal[1], rtol, &iopt[1], ierr, liwk, &iwk[1], lrwk, &
		rwk[1]);
	/*     Exit, if any parameter error was detected till here */
	if (*ierr != 0) {
		return 0;
	}

	mstor = iopt[4];
	if (mstor == 0) {
		m1 = *n;
		m2 = *n;
	} else if (mstor == 1) {
		ml = iopt[6];
		mu = iopt[7];
		m1 = (ml << 1) + mu + 1;
		m2 = ml + mu + 1;
	}
	jacgen = iopt[3];
	if (jacgen == 0) {
		jacgen = 2;
	}
	iopt[3] = jacgen;
	qrank1 = iopt[32] == 1;
	qsimpl = iopt[34] == 1;
	if (qrank1) {
		nbroy = iwk[36];
		if (nbroy == 0) {
			nbroy = max(m2,10);
		}
		iwk[36] = nbroy;
	} else {
		nbroy = 0;
	}
	/*     WorkSpace: RWK */
	l4 = 61;
	l41 = l4 + m1 * *n;
	l5 = l41 + nbroy * *n;
	l51 = l5 + *n;
	l6 = l51 + *n;
	l61 = l6 + *n;
	l62 = l61 + *n;
	l63 = l62 + *n;
	l7 = l63 + *n;
	l71 = l7 + *n;
	if (jacgen != 3) {
		l8 = l71;
	} else {
		l8 = l71 + *n;
	}
	nrwkfr = l8;
	l9 = l8;
	l11 = l9 + *n;
	l12 = l11 + *n;
	l121 = l12 + *n;
	l13 = l121 + *n;
	l14 = l13 + *n;
	nrw = l14 - 1;
	/*     End WorkSpace at NRW */
	/*     WorkSpace: IWK */
	l20 = 51;
	niwkfr = l20;
	if (qrank1 || qsimpl) {
		niwkfr += *n;
	}
	niw = l20 - 1;
	/*     End WorkSpace at NIW */
	iwk[16] = niw + 1;
	iwk[17] = nrw + 1;
	nifrin = niw + 1;
	nrfrin = nrw + 1;

	if (nrw > *lrwk || niw > *liwk) {
		*ierr = 10;
	} else {
		if (qinimo) {
			percr = (doublereal) nrw / (doublereal) (*lrwk) * 100.;
			perci = (doublereal) niw / (doublereal) (*liwk) * 100.;
			/*         Print statistics concerning workspace usage */
			/* L10050: */
			io___50.ciunit = lumon;
			s_wsfe(&io___50);
			do_fio(&c__1, (char *)&(*lrwk), (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&nrw, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&percr, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*liwk), (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&niw, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&perci, (ftnlen)sizeof(doublereal));
			e_wsfe();
		}
		if (qinimo) {
			/* L10051: */
			io___51.ciunit = lumon;
			s_wsfe(&io___51);
			do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&(*rtol), (ftnlen)sizeof(doublereal));
			e_wsfe();
			/* L10052: */
			if (jacgen == 1) {
				io___52.ciunit = lumon;
				s_wsfe(&io___52);
				do_fio(&c__1, "a user subroutine", (ftnlen)17);
				e_wsfe();
			} else if (jacgen == 2) {
				io___53.ciunit = lumon;
				s_wsfe(&io___53);
				do_fio(&c__1, "numerical differentiation (without feedback s"
					"trategy)", (ftnlen)53);
				e_wsfe();
			} else if (jacgen == 3) {
				io___54.ciunit = lumon;
				s_wsfe(&io___54);
				do_fio(&c__1, "numerical differentiation (feedback strategy "
					"included)", (ftnlen)54);
				e_wsfe();
			}
			/* L10055: */
			if (mstor == 0) {
				io___55.ciunit = lumon;
				s_wsfe(&io___55);
				do_fio(&c__1, "full", (ftnlen)4);
				e_wsfe();
			} else if (mstor == 1) {
				io___56.ciunit = lumon;
				s_wsfe(&io___56);
				do_fio(&c__1, "banded", (ftnlen)6);
				e_wsfe();
				/* L10056: */
				io___57.ciunit = lumon;
				s_wsfe(&io___57);
				do_fio(&c__1, (char *)&ml, (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&mu, (ftnlen)sizeof(integer));
				e_wsfe();
			}
			/* L10057: */
			if (iopt[35] == 1) {
				io___58.ciunit = lumon;
				s_wsfe(&io___58);
				do_fio(&c__1, "inhibited", (ftnlen)9);
				e_wsfe();
			} else {
				io___59.ciunit = lumon;
				s_wsfe(&io___59);
				do_fio(&c__1, "allowed", (ftnlen)7);
				e_wsfe();
			}
		}
		nonlin = iopt[31];
		if (iopt[38] == 0) {
			qbdamp = nonlin == 4;
		}
		if (iopt[38] == 1) {
			qbdamp = TRUE_;
		}
		if (iopt[38] == 2) {
			qbdamp = FALSE_;
		}
		if (qbdamp) {
			if (rwk[20] < 1.) {
				rwk[20] = 10.;
			}
		}
		/* L10064: */
		if (qinimo) {
			if (qrank1) {
				io___62.ciunit = lumon;
				s_wsfe(&io___62);
				do_fio(&c__1, "allowed", (ftnlen)7);
				e_wsfe();
			} else {
				io___63.ciunit = lumon;
				s_wsfe(&io___63);
				do_fio(&c__1, "inhibited", (ftnlen)9);
				e_wsfe();
			}
			/* L10065: */
			if (nonlin == 1) {
				io___64.ciunit = lumon;
				s_wsfe(&io___64);
				do_fio(&c__1, "linear", (ftnlen)6);
				e_wsfe();
			} else if (nonlin == 2) {
				io___65.ciunit = lumon;
				s_wsfe(&io___65);
				do_fio(&c__1, "mildly nonlinear", (ftnlen)16);
				e_wsfe();
			} else if (nonlin == 3) {
				io___66.ciunit = lumon;
				s_wsfe(&io___66);
				do_fio(&c__1, "highly nonlinear", (ftnlen)16);
				e_wsfe();
			} else if (nonlin == 4) {
				io___67.ciunit = lumon;
				s_wsfe(&io___67);
				do_fio(&c__1, "extremely nonlinear", (ftnlen)19);
				e_wsfe();
			}
			/* L10066: */
			if (qbdamp) {
				io___68.ciunit = lumon;
				s_wsfe(&io___68);
				do_fio(&c__1, "active", (ftnlen)6);
				do_fio(&c__1, (char *)&rwk[20], (ftnlen)sizeof(doublereal));
				e_wsfe();
			} else {
				io___69.ciunit = lumon;
				s_wsfe(&io___69);
				do_fio(&c__1, "off", (ftnlen)3);
				e_wsfe();
			}
			/* L10067: */
			if (iopt[33] == 1) {
				io___70.ciunit = lumon;
				s_wsfe(&io___70);
				do_fio(&c__1, "Ordinary", (ftnlen)8);
				e_wsfe();
			}
			if (iopt[34] == 1) {
				io___71.ciunit = lumon;
				s_wsfe(&io___71);
				do_fio(&c__1, "Simplified", (ftnlen)10);
				e_wsfe();
			}
		}
		/*       Maximum permitted number of iteration steps */
		nitmax = iwk[31];
		if (nitmax <= 0) {
			nitmax = 50;
		}
		iwk[31] = nitmax;
		/* L10068: */
		if (qinimo) {
			io___73.ciunit = lumon;
			s_wsfe(&io___73);
			do_fio(&c__1, (char *)&nitmax, (ftnlen)sizeof(integer));
			e_wsfe();
		}
		/*       Initial damping factor for highly nonlinear problems */
		qfcstr = rwk[21] > 0.;
		if (! qfcstr) {
			rwk[21] = .01;
			if (nonlin == 4) {
				rwk[21] = 1e-4;
			}
		}
		/*       Minimal permitted damping factor */
		if (rwk[22] <= 0.) {
			rwk[22] = 1e-4;
			if (nonlin == 4) {
				rwk[22] = 1e-8;
			}
		}
		fcmin = rwk[22];
		/*       Rank1 decision parameter SIGMA */
		if (rwk[23] < 1.) {
			rwk[23] = 3.;
		}
		if (! qrank1) {
			rwk[23] = 10. / fcmin;
		}
		/*       Decision parameter about increasing too small predictor */
		/*       to greater corrector value */
		if (rwk[24] < 1.) {
			rwk[24] = 10. / fcmin;
		}
		/*       Starting value of damping factor (FCMIN.LE.FC.LE.1.0) */
		if (nonlin <= 2 && ! qfcstr) {
			/*         for linear or mildly nonlinear problems */
			fc = 1.;
		} else {
			/*         for highly or extremely nonlinear problems */
			fc = rwk[21];
		}
		/*       Simplied Newton iteration implies ordinary Newton it. mode */
		if (iopt[34] == 1) {
			iopt[33] = 1;
		}
		/*       If ordinary Newton iteration, factor is always one */
		if (iopt[33] == 1) {
			fc = 1.;
		}
		rwk[21] = fc;
		if (mprmon >= 2 && ! qsucc) {
			/* L10069: */
			io___77.ciunit = lumon;
			s_wsfe(&io___77);
			do_fio(&c__1, (char *)&rwk[21], (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&fcmin, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&rwk[23], (ftnlen)sizeof(doublereal));
			e_wsfe();
		}
		/*       Store lengths of currently required workspaces */
		iwk[18] = nifrin - 1;
		iwk[19] = nrfrin - 1;

		/*       Initialize and start time measurements monitor */

		if (iopt[1] == 0 && mprtim != 0) {
			monini_(" NLEQ1", &lutim, (ftnlen)6);
			mondef_(&c__0, "NLEQ1", (ftnlen)5);
			mondef_(&c__1, "FCN", (ftnlen)3);
			mondef_(&c__2, "Jacobi", (ftnlen)6);
			mondef_(&c__3, "Lin-Fact", (ftnlen)8);
			mondef_(&c__4, "Lin-Sol", (ftnlen)7);
			mondef_(&c__5, "Output", (ftnlen)6);
			monstr_(ierr);
		}



		*ierr = -1;
		/*       If IERR is unmodified on exit, successive steps are required */
		/*       to complete the Newton iteration */
		if (nbroy == 0) {
			nbroy = 1;
		}
		n1int_(n, (U_fp)fcn, (U_fp)jac, &x[1], &xscal[1], rtol, &nitmax, &
			nonlin, &iopt[1], ierr, lrwk, &rwk[1], &nrfrin, liwk, &iwk[1],
			&nifrin, &m1, &m2, &nbroy, &rwk[l4], &rwk[l41], &rwk[l5], &
			rwk[l51], &rwk[l6], &rwk[l63], &rwk[l61], &rwk[l7], &rwk[l71],
			&rwk[l9], &rwk[l62], &rwk[l11], &rwk[l12], &rwk[l121], &rwk[
				l13], &rwk[21], &rwk[22], &rwk[23], &rwk[24], &rwk[52], &rwk[
					51], &rwk[53], &rwk[54], &rwk[17], &rwk[18], &rwk[55], &rwk[
						19], &mstor, &mprerr, &mprmon, &mprsol, &luerr, &lumon, &
							lusol, &iwk[1], &iwk[3], &iwk[4], &iwk[5], &iwk[8], &iwk[9], &
							iwk[33], &iwk[24], &qbdamp);

						if (mprtim != 0 && *ierr != -1 && *ierr != 10) {
							monhlt_();
							monprt_();
						}

						/*       Free workspaces, so far not used between steps */
						iwk[16] = niwkfr;
						iwk[17] = nrwkfr;
	}
	/*     Print statistics */
	if (mprmon >= 1 && *ierr != -1 && *ierr != 10) {
		/* L10080: */
		io___78.ciunit = lumon;
		s_wsfe(&io___78);
		do_fio(&c__1, prodct, (ftnlen)8);
		do_fio(&c__1, (char *)&iwk[1], (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&iwk[3], (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&iwk[9], (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&iwk[5], (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&iwk[4], (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&iwk[8], (ftnlen)sizeof(integer));
		e_wsfe();
	}
	/*     Print workspace requirements, if insufficient */
	if (*ierr == 10) {
		/* L10090: */
		if (mprerr >= 1) {
			io___79.ciunit = luerr;
			s_wsfe(&io___79);
			e_wsfe();
		}
		if (nrw > *lrwk) {
			/* L10091: */
			if (mprerr >= 1) {
				io___80.ciunit = luerr;
				s_wsfe(&io___80);
				do_fio(&c__1, (char *)&(*lrwk), (ftnlen)sizeof(integer));
				i__1 = nrfrin - 1;
				do_fio(&c__1, (char *)&i__1, (ftnlen)sizeof(integer));
				e_wsfe();
			}
		}
		if (niw > *liwk) {
			/* L10092: */
			if (mprerr >= 1) {
				io___81.ciunit = luerr;
				s_wsfe(&io___81);
				do_fio(&c__1, (char *)&(*liwk), (ftnlen)sizeof(integer));
				i__1 = nifrin - 1;
				do_fio(&c__1, (char *)&i__1, (ftnlen)sizeof(integer));
				e_wsfe();
			}
		}
	}
	/*     End of subroutine NLEQ1 */
	return 0;
} /* nleq1_ */


/* Subroutine */ int n1pchk_(integer *n, doublereal *x, doublereal *xscal, 
	doublereal *rtol, integer *iopt, integer *ierr, integer *liwk, 
	integer *iwk, integer *lrwk, doublereal *rwk)
{
	/* Initialized data */

	static integer ioptl[50] = { 0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-9999999,-9999999,
		-9999999,-9999999,-9999999 };
	static integer ioptu[50] = { 1,1,3,1,0,9999999,9999999,0,1,0,3,99,6,99,3,
		99,0,0,1,99,0,0,0,0,0,0,0,0,0,0,4,1,1,1,1,0,0,2,3,0,0,0,0,0,0,
		9999999,9999999,9999999,9999999,9999999 };

	/* Format strings */
	static char fmt_10011[] = "(/,\002 Error: Bad input to dimensional param"
		"eter N supplied\002,/,8x,\002choose N positive, your input is: N"
		" = \002,i5)";
	static char fmt_10012[] = "(/,\002 Warning: User prescribed RTOL \002,a"
		",\002to \002,\002reasonable \002,a,\002 value RTOL = \002,d11.2)";
	static char fmt_10013[] = "(/,\002 Error: Negative value in XSCAL(\002,i"
		"5,\002) supplied\002)";
	static char fmt_10014[] = "(/,\002 Warning: XSCAL(\002,i5,\002) = \002,d"
		"9.2,\002 too small, \002,\002increased to\002,d9.2)";
	static char fmt_10015[] = "(/,\002 Warning: XSCAL(\002,i5,\002) = \002,d"
		"9.2,\002 too big, \002,\002decreased to\002,d9.2)";
	static char fmt_20001[] = "(\002 Invalid option specified: IOPT(\002,i2"
		",\002)=\002,i12,\002;\002,/,3x,\002range of permitted values is"
		" \002,i8,\002 to \002,i8)";

	/* System generated locals */
	integer i__1;

	/* Builtin functions */
	integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

	/* Local variables */
	extern /* Subroutine */ int zibconst_(doublereal *, doublereal *);
	static integer i__;
	static doublereal great, small;
	static integer luerr, mstor;
	static doublereal epmach, defscl;
	static integer nonlin;
	static doublereal tolmin, tolmax;
	static integer mprerr;

	/* Fortran I/O blocks */
	static cilist io___89 = { 0, 0, 0, fmt_10011, 0 };
	static cilist io___91 = { 0, 0, 0, "(/,A)", 0 };
	static cilist io___93 = { 0, 0, 0, fmt_10012, 0 };
	static cilist io___95 = { 0, 0, 0, fmt_10012, 0 };
	static cilist io___98 = { 0, 0, 0, fmt_10013, 0 };
	static cilist io___99 = { 0, 0, 0, fmt_10014, 0 };
	static cilist io___100 = { 0, 0, 0, fmt_10015, 0 };
	static cilist io___102 = { 0, 0, 0, fmt_20001, 0 };


	/* *    Begin Prologue N1PCHK */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     N 1 P C H K : Checking of input parameters and options */
	/*                   for NLEQ1. */

	/* *    Parameters: */
	/*     =========== */

	/*     See parameter description in driver routine. */

	/* *    Subroutines called: ZIBCONST */

	/* *    Machine dependent constants used: */
	/*     ================================= */

	/*     EPMACH = relative machine precision */
	/*     GREAT = squareroot of maxreal divided by 10 */
	/*     SMALL = squareroot of "smallest positive machine number */
	/*             divided by relative machine precision" */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */



	/* Parameter adjustments */
	--xscal;
	--x;
	--iopt;
	--iwk;
	--rwk;

	/* Function Body */

	zibconst_(&epmach, &small);
	great = 1. / small;
	*ierr = 0;
	/*        Print error messages? */
	mprerr = iopt[11];
	luerr = iopt[12];
	if (luerr <= 0 || luerr > 99) {
		luerr = 6;
		iopt[12] = luerr;
	}

	/*     Checking dimensional parameter N */
	if (*n <= 0) {
		if (mprerr >= 1) {
			io___89.ciunit = luerr;
			s_wsfe(&io___89);
			do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(integer));
			e_wsfe();
		}
		*ierr = 20;
	}

	/*     Problem type specification by user */
	nonlin = iopt[31];
	if (nonlin == 0) {
		nonlin = 3;
	}
	iopt[31] = nonlin;

	/*     Checking and conditional adaption of the user-prescribed RTOL */
	if (*rtol <= 0.) {
		if (mprerr >= 1) {
			io___91.ciunit = luerr;
			s_wsfe(&io___91);
			do_fio(&c__1, " Error: Nonpositive RTOL supplied", (ftnlen)33);
			e_wsfe();
		}
		*ierr = 21;
	} else {
		tolmin = epmach * 10. * (doublereal) (*n);
		if (*rtol < tolmin) {
			*rtol = tolmin;
			if (mprerr >= 2) {
				io___93.ciunit = luerr;
				s_wsfe(&io___93);
				do_fio(&c__1, "increased ", (ftnlen)10);
				do_fio(&c__1, "smallest", (ftnlen)8);
				do_fio(&c__1, (char *)&(*rtol), (ftnlen)sizeof(doublereal));
				e_wsfe();
			}
		}
		tolmax = .1;
		if (*rtol > tolmax) {
			*rtol = tolmax;
			if (mprerr >= 2) {
				io___95.ciunit = luerr;
				s_wsfe(&io___95);
				do_fio(&c__1, "decreased ", (ftnlen)10);
				do_fio(&c__1, "largest", (ftnlen)7);
				do_fio(&c__1, (char *)&(*rtol), (ftnlen)sizeof(doublereal));
				e_wsfe();
			}
		}
	}

	/*     Test user prescribed accuracy and scaling on proper values */
	if (*n <= 0) {
		return 0;
	}
	if (nonlin >= 3) {
		defscl = *rtol;
	} else {
		defscl = 1.;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		if (xscal[i__] < 0.) {
			if (mprerr >= 1) {
				io___98.ciunit = luerr;
				s_wsfe(&io___98);
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
				e_wsfe();
			}
			*ierr = 22;
		}
		if (xscal[i__] == 0.) {
			xscal[i__] = defscl;
		}
		if (xscal[i__] > 0. && xscal[i__] < small) {
			if (mprerr >= 2) {
				io___99.ciunit = luerr;
				s_wsfe(&io___99);
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&xscal[i__], (ftnlen)sizeof(doublereal))
					;
				do_fio(&c__1, (char *)&small, (ftnlen)sizeof(doublereal));
				e_wsfe();
			}
			xscal[i__] = small;
		}
		if (xscal[i__] > great) {
			if (mprerr >= 2) {
				io___100.ciunit = luerr;
				s_wsfe(&io___100);
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&xscal[i__], (ftnlen)sizeof(doublereal))
					;
				do_fio(&c__1, (char *)&great, (ftnlen)sizeof(doublereal));
				e_wsfe();
			}
			xscal[i__] = great;
		}
		/* L10: */
	}
	/*     Special dependence on Jacobian's storage mode for ML and MU */
	mstor = iopt[4];
	if (mstor == 0) {
		ioptu[5] = 0;
		ioptu[6] = 0;
	} else if (mstor == 1) {
		ioptu[5] = *n - 1;
		ioptu[6] = *n - 1;
	}
	/*     Checks options */
	for (i__ = 1; i__ <= 30; ++i__) {
		if (iopt[i__] < ioptl[i__ - 1] || iopt[i__] > ioptu[i__ - 1]) {
			*ierr = 30;
			if (mprerr >= 1) {
				io___102.ciunit = luerr;
				s_wsfe(&io___102);
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&iopt[i__], (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&ioptl[i__ - 1], (ftnlen)sizeof(integer)
					);
				do_fio(&c__1, (char *)&ioptu[i__ - 1], (ftnlen)sizeof(integer)
					);
				e_wsfe();
			}
		}
		/* L20: */
	}
	/*     End of subroutine N1PCHK */
	return 0;
} /* n1pchk_ */


/* Subroutine */ int n1int_(integer *n, S_fp fcn, S_fp jac, doublereal *x, 
	doublereal *xscal, doublereal *rtol, integer *nitmax, integer *nonlin,
	integer *iopt, integer *ierr, integer *lrwk, doublereal *rwk, 
	integer *nrwkfr, integer *liwk, integer *iwk, integer *niwkfr, 
	integer *m1, integer *m2, integer *nbroy, doublereal *a, doublereal *
	dxsave, doublereal *dx, doublereal *dxq, doublereal *xa, doublereal *
	xwa, doublereal *f, doublereal *fa, doublereal *eta, doublereal *xw, 
	doublereal *fw, doublereal *dxqa, doublereal *t1, doublereal *t2, 
	doublereal *t3, doublereal *fc, doublereal *fcmin, doublereal *sigma, 
	doublereal *sigma2, doublereal *fca, doublereal *fckeep, doublereal *
	fcpri, doublereal *dmycor, doublereal *conv, doublereal *sumx, 
	doublereal *sumxs, doublereal *dlevf, integer *mstor, integer *mprerr,
	integer *mprmon, integer *mprsol, integer *luerr, integer *lumon, 
	integer *lusol, integer *niter, integer *ncorr, integer *nfcn, 
	integer *njac, integer *nfcnj, integer *nrejr1, integer *new__, 
	integer *iconv, logical *qbdamp)
{
	/* Format strings */
	static char fmt_16003[] = "(///,2x,66(\002*\002))";
	static char fmt_16004[] = "(/,8x,\002It\002,7x,\002Normf \002,10x,\002No"
		"rmx \002,8x,\002Damp.Fct.\002,3x,\002New\002)";
	static char fmt_22201[] = "(/,\002 +++ aposteriori estimate +++\002,/"
		",\002 FCCOR  = \002,d18.10,\002  FC     = \002,d18.10,/\002 DMYC"
		"OR = \002,d18.10,\002  FCNUMP = \002,d18.10,/\002 FCDNM  = \002,"
		"d18.10,/,\002 ++++++++++++++++++++++++++++\002,/)";
	static char fmt_33201[] = "(/,\002 +++ apriori estimate +++\002,/,\002 F"
		"CPRI  = \002,d18.10,\002  FC     = \002,d18.10,/,\002 FCA    ="
		" \002,d18.10,\002  DMYPRI = \002,d18.10,/,\002 FCNUMP = \002,d18"
		".10,\002  FCDNM  = \002,d18.10,/,\002 +++++++++++++++++++++++"
		"+\002,/)";
	static char fmt_32001[] = "(\002 ** ICONV: \002,i1,\002  ALPHA: \002,d9."
		"2,\002  CONST-ALPHA: \002,d9.2,\002  CONST-LIN: \002,d9.2,\002 **"
		"\002,/,\002 **\002,11x,\002ALPHA-POST: \002,d9.2,\002 CHECK: "
		"\002,d9.2,25x,\002**\002)";
	static char fmt_36101[] = "(8x,i2,\002 FCN could not be evaluated  \002,"
		"8x,f7.5,4x,i2)";
	static char fmt_39001[] = "(/,\002 +++ corrector computation +++\002,/"
		",\002 FCCOR  = \002,d18.10,\002  FC     = \002,d18.10,/,\002 DMY"
		"COR = \002,d18.10,\002  FCNUMK = \002,d18.10,/,\002 FCDNM  = "
		"\002,d18.10,\002  FCA    = \002,d18.10,/,\002 ++++++++++++++++++"
		"+++++++++++\002,/)";
	static char fmt_39002[] = "(/,\002 +++ corrector setting 1 +++\002,/,"
		"\002 FC     = \002,d18.10,/,\002 +++++++++++++++++++++++++++\002"
		",/)";
	static char fmt_39003[] = "(/,\002 +++ corrector setting 2 +++\002,/,"
		"\002 FC     = \002,d18.10,/,\002 +++++++++++++++++++++++++++\002"
		",/)";
	static char fmt_31130[] = "(8x,i2,\002 Not accepted damping factor \002,"
		"8x,f7.5,4x,i2)";
	static char fmt_91001[] = "(///\002 Solution of nonlinear system \002"
		",\002of equations obtained within \002,i3,\002 iteration step"
		"s\002,//,\002 Achieved relative accuracy\002,d10.3)";
	static char fmt_91002[] = "(///\002 Solution of linear system \002,\002o"
		"f equations obtained by NLEQ1\002,//,\002 No estimate \002,\002a"
		"vailable for the achieved relative accuracy\002)";
	static char fmt_92101[] = "(/,\002 Iteration terminates due to \002,\002"
		"singular jacobian matrix\002,/)";
	static char fmt_92201[] = "(/,\002 Iteration terminates after NITMAX "
		"\002,\002=\002,i3,\002  Iteration steps\002)";
	static char fmt_92301[] = "(/,\002 Damping factor has become too \002"
		",\002small: lambda =\002,d10.3,2x,/)";
	static char fmt_92401[] = "(/,\002 Warning: Monotonicity test failed aft"
		"er \002,a,\002 convergence was already checked;\002,/,\002 RTOL "
		"requirement may be too stringent\002,/)";
	static char fmt_92402[] = "(/,\002 Warning: \002,a,\002 convergence slow"
		"ed down;\002,/,\002 RTOL requirement may be too stringent\002,/)";
	static char fmt_92410[] = "(/,\002 Warning: No quadratic or superlinear "
		"convergence \002,\002established yet\002,/,10x,\002your solution"
		" may perhaps may be less accurate \002,/,10x,\002as indicated by"
		" the standard error estimate\002)";
	static char fmt_92501[] = "(/,\002 Error \002,i5,\002 signalled by linea"
		"r solver N1FACT\002)";
	static char fmt_92601[] = "(/,\002 Error \002,i5,\002 signalled by linea"
		"r solver N1SOLV\002)";
	static char fmt_92701[] = "(/,\002 Error \002,i5,\002 signalled by user "
		"function FCN\002)";
	static char fmt_92801[] = "(/,\002 Error \002,i5,\002 signalled by user "
		"function JAC\002)";
	static char fmt_92810[] = "(\002 Try to find a better initial guess for "
		"the solution\002)";
	static char fmt_93100[] = "(/,\002    Achieved relative accuracy\002,d10"
		".3,2x)";

	/* System generated locals */
	integer a_dim1, a_offset, dxsave_dim1, dxsave_offset, i__1, i__2, i__3;
	doublereal d__1, d__2;
	logical L__1;

	/* Builtin functions */
	double sqrt(doublereal);
	integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen),
		s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
		e_wsle(void);
	double log(doublereal), pow_dd(doublereal *, doublereal *);

	/* Local variables */
	extern /* Subroutine */ int zibconst_(doublereal *, doublereal *);
	static integer i__, k, l1, l2, l3;
	static doublereal s1;
	static integer ml;
	static doublereal th;
	static integer mu;
	static doublereal fch;
	static logical qlu;
	static doublereal fck2, sum1, sum2, fcbh, alfa, beta;
	static integer mode, nred, liwl;
	static logical qrep;
	static integer lrwl;
	static doublereal alfa1, alfa2;
	extern /* Subroutine */ int n1jac_(S_fp, integer *, integer *, doublereal
		*, doublereal *, doublereal *, doublereal *, doublereal *, 
		doublereal *, integer *, doublereal *, integer *), n1jcf_(S_fp, 
		integer *, integer *, doublereal *, doublereal *, doublereal *, 
		doublereal *, doublereal *, doublereal *, doublereal *, 
		doublereal *, doublereal *, integer *, doublereal *, integer *);
	static doublereal clin0, clin1, fcbnd, ajdel;
	static integer ifail;
	static doublereal fcdnm, aprec;
	static integer iscal;
	static doublereal fccor, ajmin, fcmon;
	static logical qgenj;
	static doublereal conva, small;
	static integer niwla;
	static logical qsucc, qordi;
	static integer iloop, nrwla;
	extern /* Subroutine */ int monon_(integer *), n1prv1_(doublereal *, 
		doublereal *, doublereal *, integer *, integer *, integer *, 
		integer *, logical *), n1prv2_(doublereal *, doublereal *, 
		doublereal *, integer *, integer *, integer *, logical *, char *, 
		ftnlen);
	static doublereal dxnrm;
	extern /* Subroutine */ int n1jacb_(S_fp, integer *, integer *, integer *,
		doublereal *, doublereal *, doublereal *, doublereal *, 
		doublereal *, doublereal *, integer *, doublereal *, doublereal *,
		doublereal *, integer *);
	static doublereal sumxa;
	static logical qnext;
	extern doublereal wnorm_(integer *, doublereal *, doublereal *);
	extern /* Subroutine */ int n1jcfb_(S_fp, integer *, integer *, integer *,
		doublereal *, doublereal *, doublereal *, doublereal *, 
		doublereal *, doublereal *, doublereal *, doublereal *,
		doublereal *, integer *, doublereal *, doublereal *, doublereal *,
		integer *), n1fact_(integer *, integer *, integer *, integer *, 
		doublereal *, integer *, integer *, integer *, integer *, integer 
		*, integer *, doublereal *, integer *);
	static doublereal fcmin2;
	extern /* Subroutine */ int n1scal_(integer *, doublereal *, doublereal *,
		doublereal *, doublereal *, integer *, logical *, integer *, 
		integer *, doublereal *);
	static doublereal fcnmp2;
	extern /* Subroutine */ int n1scrb_(integer *, integer *, integer *, 
		integer *, doublereal *, doublereal *), n1scrf_(integer *, 
		integer *, doublereal *, doublereal *);
	static logical qrank1;
	static doublereal sumxa0, sumxa1;
	extern /* Subroutine */ int n1lvls_(integer *, doublereal *, doublereal *,
		doublereal *, doublereal *, doublereal *, doublereal *, 
		doublereal *, integer *, logical *);
	static doublereal sumxa2;
	extern /* Subroutine */ int n1solv_(integer *, integer *, integer *, 
		integer *, doublereal *, doublereal *, integer *, integer *, 
		integer *, integer *, integer *, integer *, doublereal *, integer 
		*);
	static doublereal alphaa;
	static integer jacgen;
	static doublereal calpha;
	extern /* Subroutine */ int n1sout_(integer *, doublereal *, integer *, 
		integer *, doublereal *, integer *, integer *, integer *, integer 
		*, integer *);
	static doublereal alphae, etadif, epmach, epdiff, alphak, calphk;
	static integer modefi;
	static doublereal fcredu, etaini;
	static logical qscale;
	static doublereal etamin, dlevfn, etamax, dlevxa;
	extern /* Subroutine */ int monoff_(integer *);
	static doublereal fcnumk;
	static logical qinisc, qjcrfr;
	static doublereal fcnump, dxanrm, rsmall;
	static integer niluse;
	static logical qlinit;
	static integer iormon;
	static doublereal dmypri;
	static logical qsimpl, qmixio;
	static integer nrluse, mprtim, idummy;
	static logical qmstop;
	static doublereal sumxte;

	/* Fortran I/O blocks */
	static cilist io___142 = { 0, 0, 0, fmt_16003, 0 };
	static cilist io___143 = { 0, 0, 0, fmt_16004, 0 };
	static cilist io___148 = { 0, 0, 0, fmt_22201, 0 };
	static cilist io___171 = { 0, 0, 0, fmt_33201, 0 };
	static cilist io___173 = { 0, 0, 0, 0, 0 };
	static cilist io___174 = { 0, 0, 0, 0, 0 };
	static cilist io___182 = { 0, 0, 0, fmt_32001, 0 };
	static cilist io___188 = { 0, 0, 0, fmt_36101, 0 };
	static cilist io___190 = { 0, 0, 0, 0, 0 };
	static cilist io___196 = { 0, 0, 0, fmt_39001, 0 };
	static cilist io___197 = { 0, 0, 0, 0, 0 };
	static cilist io___198 = { 0, 0, 0, fmt_39002, 0 };
	static cilist io___199 = { 0, 0, 0, fmt_39003, 0 };
	static cilist io___200 = { 0, 0, 0, fmt_31130, 0 };
	static cilist io___202 = { 0, 0, 0, fmt_91001, 0 };
	static cilist io___203 = { 0, 0, 0, fmt_91001, 0 };
	static cilist io___204 = { 0, 0, 0, fmt_91002, 0 };
	static cilist io___205 = { 0, 0, 0, fmt_92101, 0 };
	static cilist io___206 = { 0, 0, 0, fmt_92201, 0 };
	static cilist io___207 = { 0, 0, 0, fmt_92301, 0 };
	static cilist io___208 = { 0, 0, 0, fmt_92401, 0 };
	static cilist io___209 = { 0, 0, 0, fmt_92401, 0 };
	static cilist io___210 = { 0, 0, 0, fmt_92402, 0 };
	static cilist io___211 = { 0, 0, 0, fmt_92402, 0 };
	static cilist io___212 = { 0, 0, 0, fmt_92410, 0 };
	static cilist io___213 = { 0, 0, 0, fmt_92501, 0 };
	static cilist io___214 = { 0, 0, 0, fmt_92601, 0 };
	static cilist io___215 = { 0, 0, 0, fmt_92701, 0 };
	static cilist io___216 = { 0, 0, 0, fmt_92801, 0 };
	static cilist io___217 = { 0, 0, 0, fmt_92810, 0 };
	static cilist io___218 = { 0, 0, 0, fmt_93100, 0 };


	/* *    Begin Prologue N1INT */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     N 1 I N T : Core routine for NLEQ1 . */
	/*     Damped Newton-algorithm for systems of highly nonlinear */
	/*     equations especially designed for numerically sensitive */
	/*     problems. */

	/* *    Parameters: */
	/*     =========== */

	/*       N,FCN,JAC,X,XSCAL,RTOL */
	/*                         See parameter description in driver routine */

	/*       NITMAX      Int    Maximum number of allowed iterations */
	/*       NONLIN      Int    Problem type specification */
	/*                          (see IOPT-field NONLIN) */
	/*       IOPT        Int    See parameter description in driver routine */
	/*       IERR        Int    See parameter description in driver routine */
	/*       LRWK        Int    Length of real workspace */
	/*       RWK(LRWK)   Dble   Real workspace array */
	/*       NRWKFR      Int    First free position of RWK on exit */
	/*       LIWK        Int    Length of integer workspace */
	/*       IWK(LIWK)   Int    Integer workspace array */
	/*       NIWKFR      Int    First free position of IWK on exit */
	/*       M1          Int    Leading dimension of Jacobian array A */
	/*                          for full case Jacobian: N */
	/*                          for banded Jacobian: 2*ML+MU+1; */
	/*                          ML, MU see IOPT-description in driver routine */
	/*       M2          Int    for full case Jacobian: N */
	/*                          for banded Jacobian: ML+MU+1 */
	/*       NBROY       Int    Maximum number of possible consecutive */
	/*                          iterative Broyden steps. (See IWK(36)) */
	/*       A(M1,N)     Dble   Holds the Jacobian matrix (decomposed form */
	/*                          after call of linear decomposition */
	/*                          routine) */
	/*       DXSAVE(X,NBROY) */
	/*                   Dble   Used to save the quasi Newton corrections of */
	/*                          all previously done consecutive Broyden */
	/*                          steps. */
	/*       DX(N)       Dble   Current Newton correction */
	/*       DXQ(N)      Dble   Simplified Newton correction J(k-1)*X(k) */
	/*       XA(N)       Dble   Previous Newton iterate */
	/*       XWA(N)      Dble   Scaling factors used for latest decomposed */
	/*                          Jacobian for column scaling - may differ */
	/*                          from XW, if Broyden updates are performed */
	/*       F(N)        Dble   Function (FCN) value of current iterate */
	/*       FA(N)       Dble   Function (FCN) value of previous iterate */
	/*       ETA(N)      Dble   Jacobian approximation: updated scaled */
	/*                          denominators */
	/*       XW(N)       Dble   Scaling factors for iteration vector */
	/*       FW(N)       Dble   Scaling factors for rows of the system */
	/*       DXQA(N)     Dble   Previous Newton correction */
	/*       T1(N)       Dble   Workspace for linear solvers and internal */
	/*                          subroutines */
	/*       T2(N)       Dble   Workspace array for internal subroutines */
	/*       T3(N)       Dble   Workspace array for internal subroutines */
	/*       FC          Dble   Current Newton iteration damping factor. */
	/*       FCMIN       Dble   Minimum permitted damping factor. If */
	/*                          FC becomes smaller than this value, one */
	/*                          of the following may occur: */
	/*                          a.    Recomputation of the Jacobian */
	/*                                matrix by means of difference */
	/*                                approximation (instead of Rank1 */
	/*                                update), if Rank1 - update */
	/*                                previously was used */
	/*                          b.    Fail exit otherwise */
	/*       SIGMA       Dble   Decision parameter for rank1-updates */
	/*       SIGMA2      Dble   Decision parameter for damping factor */
	/*                          increasing to corrector value */
	/*       FCA         Dble   Previous Newton iteration damping factor. */
	/*       FCKEEP      Dble   Keeps the damping factor as it is at start */
	/*                          of iteration step. */
	/*       CONV        Dble   Scaled maximum norm of the Newton- */
	/*                          correction. Passed to RWK-field on output. */
	/*       SUMX        Dble   Square of the natural level (see equal- */
	/*                          named RWK-output field) */
	/*       SUMXS       Dble   Square of the "simplified" natural level */
	/*                          (see equal-named RWK-internal field) */
	/*       DLEVF       Dble   The standard level (see equal- */
	/*                          named RWK-output field) */
	/*       MSTOR       Int    see description of IOPT-field MSTOR */
	/*       MPRERR,MPRMON,MPRSOL,LUERR,LUMON,LUSOL, */
	/*       NITER,NCORR,NFCN,NJAC,NFCNJ,NREJR1,NEW : */
	/*                          See description of equal named IWK-fields */
	/*                          in the driver subroutine */
	/*       QBDAMP      Logic  Flag, that indicates, whether bounded damping */
	/*                          strategy is active: */
	/*                          .true.  = bounded damping strategy is active */
	/*                          .false. = normal damping strategy is active */

	/* *    Internal double variables */
	/*     ========================= */

	/*       AJDEL    See RWK(26) (num. diff. without feedback) */
	/*       AJMIN    See RWK(27) (num. diff. without feedback) */
	/*       CONVA    Holds the previous value of CONV . */
	/*       DMUE     Temporary value used during computation of damping */
	/*                factors predictor. */
	/*       EPDIFF   sqrt(10*epmach) (num. diff. with feedback) */
	/*       ETADIF   See description of RWK(28) (num. diff. with feedback) */
	/*       ETAINI   Initial value for all ETA-components (num. diff. fb.) */
	/*       ETAMAX   Maximum allowed pertubation (num. diff. with feedback) */
	/*       ETAMIN   Minimum allowed pertubation (num. diff. with feedback) */
	/*       FCDNM    Used to compute the denominator of the damping */
	/*                factor FC during computation of it's predictor, */
	/*                corrector and aposteriori estimate (in the case of */
	/*                performing a Rank1 update) . */
	/*       FCK2     Aposteriori estimate of FC. */
	/*       FCH      Temporary storage of new corrector FC. */
	/*       FCMIN2   FCMIN**2 . Used for FC-predictor computation. */
	/*       FCNUMP   Gets the numerator of the predictor formula for FC. */
	/*       FCNMP2   Temporary used for predictor numerator computation. */
	/*       FCNUMK   Gets the numerator of the corrector computation */
	/*                of FC . */
	/*       SUMXA    Natural level of the previous iterate. */
	/*       TH       Temporary variable used during corrector- and */
	/*                aposteriori computations of FC. */

	/* *    Internal integer variables */
	/*     ========================== */

	/*     IFAIL      Gets the return value from subroutines called from */
	/*                N1INT (N1FACT, N1SOLV, FCN, JAC) */
	/*     ISCAL      Holds the scaling option from the IOPT-field ISCAL */
	/*     MODE       Matrix storage mode (see IOPT-field MODE) */
	/*     NRED       Count of successive corrector steps */
	/*     NILUSE     Gets the amount of IWK used by the linear solver */
	/*     NRLUSE     Gets the amount of RWK used by the linear solver */
	/*     NIWLA      Index of first element of IWK provided to the */
	/*                linear solver */
	/*     NRWLA      Index of first element of RWK provided to the */
	/*                linear solver */
	/*     LIWL       Holds the maximum amount of integer workspace */
	/*                available to the linear solver */
	/*     LRWL       Holds the maximum amount of real workspace */
	/*                available to the linear solver */


	/* *    Internal logical variables */
	/*     ========================== */

	/*     QGENJ      Jacobian updating technique flag: */
	/*                =.TRUE.  : Call of analytical subroutine JAC or */
	/*                           numerical differentiation */
	/*                =.FALSE. : rank1- (Broyden-) update */
	/*     QINISC     Iterate initial-scaling flag: */
	/*                =.TRUE.  : at first call of N1SCAL */
	/*                =.FALSE. : at successive calls of N1SCAL */
	/*     QSUCC      See description of IOPT-field QSUCC. */
	/*     QJCRFR     Jacobian refresh flag: */
	/*                set to .TRUE. if damping factor gets too small */
	/*                and Jacobian was computed by rank1-update. */
	/*                Indicates, that the Jacobian needs to be recomputed */
	/*                by subroutine JAC or numerical differentation. */
	/*     QLINIT     Initialization state of linear solver workspace: */
	/*                =.FALSE. : Not yet initialized */
	/*                =.TRUE.  : Initialized - N1FACT has been called at */
	/*                           least one time. */
	/*     QSCALE     Holds the value of .NOT.QNSCAL. See description */
	/*                of IOPT-field QNSCAL. */

	/* *    Subroutines called: */
	/*     =================== */

	/*       N1FACT, N1SOLV, N1JAC,  N1JACB, N1JCF,  N1JCFB, N1LVLS, */
	/*       N1SCRF, N1SCRB, N1SOUT, N1PRV1, N1PRV2, N1SCAL, */
	/*       MONON,  MONOFF */

	/* *    Functions called: */
	/*     ================= */

	/*       ZIBCONST, WNORM */

	/* *    Machine constants used */
	/*     ====================== */


	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* WEI */

	/* Parameter adjustments */
	--t3;
	--t2;
	--t1;
	--dxqa;
	--fw;
	--xw;
	--eta;
	--fa;
	--f;
	--xwa;
	--xa;
	--dxq;
	--dx;
	--xscal;
	--x;
	--iopt;
	--rwk;
	--iwk;
	a_dim1 = *m1;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	dxsave_dim1 = *n;
	dxsave_offset = 1 + dxsave_dim1;
	dxsave -= dxsave_offset;

	/* Function Body */
	zibconst_(&epmach, &small);
	/* *    Begin */
	/*       ---------------------------------------------------------- */
	/*       1 Initialization */
	/*       ---------------------------------------------------------- */
	/*       1.1 Control-flags and -integers */
	qsucc = iopt[1] == 1;
	qscale = ! (iopt[35] == 1);
	qordi = iopt[33] == 1;
	qsimpl = iopt[34] == 1;
	qrank1 = iopt[32] == 1;
	iormon = iopt[39];
	if (iormon == 0) {
		iormon = 2;
	}
	iscal = iopt[9];
	mode = iopt[2];
	jacgen = iopt[3];
	qmixio = *lumon == *lusol && *mprmon != 0 && *mprsol != 0;
	qlu = ! qsimpl;
	mprtim = iopt[19];
	/*       ---------------------------------------------------------- */
	/*       1.2 Derivated dimensional parameters */
	if (*mstor == 0) {
		ml = 0;
	} else if (*mstor == 1) {
		ml = *m1 - *m2;
		mu = *m2 - 1 - ml;
	}
	/*       ---------------------------------------------------------- */
	/*       1.3 Derivated internal parameters */
	fcmin2 = *fcmin * *fcmin;
	rsmall = sqrt(*rtol * 10.);
	/*       ---------------------------------------------------------- */
	/*       1.4 Adaption of input parameters, if necessary */
	if (*fc < *fcmin) {
		*fc = *fcmin;
	}
	if (*fc > 1.) {
		*fc = 1.;
	}
	/*       ---------------------------------------------------------- */
	/*       1.5 Initial preparations */
	qjcrfr = FALSE_;
	qlinit = FALSE_;
	ifail = 0;
	fcbnd = 0.;
	if (*qbdamp) {
		fcbnd = rwk[20];
	}
	/*       ---------------------------------------------------------- */
	/*       1.5.1 Numerical differentiation related initializations */
	if (jacgen == 2) {
		ajdel = rwk[26];
		if (ajdel <= small) {
			ajdel = sqrt(epmach * 10.);
		}
		ajmin = rwk[27];
	} else if (jacgen == 3) {
		etadif = rwk[28];
		if (etadif <= small) {
			etadif = 1e-6;
		}
		etaini = rwk[29];
		if (etaini <= small) {
			etaini = 1e-6;
		}
		epdiff = sqrt(epmach * 10.);
		etamax = sqrt(epdiff);
		etamin = epdiff * etamax;
	}
	/*       ---------------------------------------------------------- */
	/*       1.5.2 Miscellaneous preparations of first iteration step */
	if (! qsucc) {
		*niter = 0;
		*ncorr = 0;
		*nrejr1 = 0;
		*nfcn = 0;
		*njac = 0;
		*nfcnj = 0;
		qgenj = TRUE_;
		qinisc = TRUE_;
		*fckeep = *fc;
		*fca = *fc;
		*fcpri = *fc;
		fck2 = *fc;
		fcmon = *fc;
		*conv = 0.;
		if (jacgen == 3) {
			i__1 = *n;
			for (l1 = 1; l1 <= i__1; ++l1) {
				eta[l1] = etaini;
				/* L1520: */
			}
		}
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			xa[l1] = x[l1];
			/* L1521: */
		}
		/* WEI */
		*iconv = 0;
		alphae = 0.;
		sumxa1 = 0.;
		sumxa0 = 0.;
		clin0 = 0.;
		qmstop = FALSE_;
		/*         ------------------------------------------------------ */
		/*         1.6 Print monitor header */
		if (*mprmon >= 2 && ! qmixio) {
			/* L16003: */
			io___142.ciunit = *lumon;
			s_wsfe(&io___142);
			e_wsfe();
			/* L16004: */
			io___143.ciunit = *lumon;
			s_wsfe(&io___143);
			e_wsfe();
		}
		/*         -------------------------------------------------------- */
		/*         1.7 Startup step */
		/*         -------------------------------------------------------- */
		/*         1.7.1 Computation of the residual vector */
		if (mprtim != 0) {
			monon_(&c__1);
		}
		(*fcn)(n, &x[1], &f[1], &ifail);
		if (mprtim != 0) {
			monoff_(&c__1);
		}
		++(*nfcn);
		/*     Exit, if ... */
		if (ifail != 0) {
			*ierr = 82;
			goto L4299;
		}
	} else {
		qinisc = FALSE_;
	}

	/*       Main iteration loop */
	/*       =================== */

	/*       Repeat */
L2:
	/*         -------------------------------------------------------- */
	/*         2 Startup of iteration step */
	if (! qjcrfr) {
		/*           ------------------------------------------------------ */
		/*           2.1 Scaling of variables X(N) */
		n1scal_(n, &x[1], &xa[1], &xscal[1], &xw[1], &iscal, &qinisc, &iopt[1]
		, lrwk, &rwk[1]);
		qinisc = FALSE_;
		if (*niter != 0) {
			/*             ---------------------------------------------------- */
			/*             2.2 Aposteriori estimate of damping factor */
			i__1 = *n;
			for (l1 = 1; l1 <= i__1; ++l1) {
				dxqa[l1] = dxq[l1];
				/* L2200: */
			}
			if (! qordi) {
				fcnump = 0.;
				i__1 = *n;
				for (l1 = 1; l1 <= i__1; ++l1) {
					/* Computing 2nd power */
					d__1 = dx[l1] / xw[l1];
					fcnump += d__1 * d__1;
					/* L2201: */
				}
				th = *fc - 1.;
				fcdnm = 0.;
				i__1 = *n;
				for (l1 = 1; l1 <= i__1; ++l1) {
					/* Computing 2nd power */
					d__1 = (dxqa[l1] + th * dx[l1]) / xw[l1];
					fcdnm += d__1 * d__1;
					/* L2202: */
				}
				/*               -------------------------------------------------- */
				/*               2.2.2 Decision criterion for Jacobian updating */
				/*                     technique: */
				/*                     QGENJ.EQ..TRUE. numerical differentation, */
				/*                     QGENJ.EQ..FALSE. rank1 updating */
				qgenj = TRUE_;
				if (*fc == *fcpri) {
					qgenj = *fc < 1. || *fca < 1. || *dmycor <= *fc * *sigma 
						|| ! qrank1 || *new__ + 2 > *nbroy;
					*fca = *fc;
				} else {
					*dmycor = *fca * *fca * .5 * sqrt(fcnump / fcdnm);
					if (*nonlin <= 3) {
						fccor = min(1.,*dmycor);
					} else {
						/* Computing MIN */
						d__1 = 1., d__2 = *dmycor * .5;
						fccor = min(d__1,d__2);
					}
					/* Computing MAX */
					d__1 = min(*fc,fccor);
					*fca = max(d__1,*fcmin);
					/* $Test-begin */
					if (*mprmon >= 5) {
						io___148.ciunit = *lumon;
						s_wsfe(&io___148);
						do_fio(&c__1, (char *)&fccor, (ftnlen)sizeof(
							doublereal));
						do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(
							doublereal));
						do_fio(&c__1, (char *)&(*dmycor), (ftnlen)sizeof(
							doublereal));
						do_fio(&c__1, (char *)&fcnump, (ftnlen)sizeof(
							doublereal));
						do_fio(&c__1, (char *)&fcdnm, (ftnlen)sizeof(
							doublereal));
						e_wsfe();
					}
					/* $Test-end */
				}
				fck2 = *fca;
				/*               ------------------------------------------------------ */
				/*               2.2.1 Computation of the numerator of damping */
				/*                     factor predictor */
				fcnmp2 = 0.;
				i__1 = *n;
				for (l1 = 1; l1 <= i__1; ++l1) {
					/* Computing 2nd power */
					d__1 = dxqa[l1] / xw[l1];
					fcnmp2 += d__1 * d__1;
					/* L221: */
				}
				fcnump *= fcnmp2;
			}
		}
	}
	qjcrfr = FALSE_;
	/*         -------------------------------------------------------- */
	/*         2.3 Jacobian matrix (stored to array A(M1,N)) */
	/*         -------------------------------------------------------- */
	/*         2.3.1 Jacobian generation by routine JAC or */
	/*               difference approximation (If QGENJ.EQ..TRUE.) */
	/*               - or - */
	/*               Rank-1 update of Jacobian (If QGENJ.EQ..FALSE.) */
	if (qgenj && (! qsimpl || *niter == 0)) {
		*new__ = 0;
		if (jacgen == 1) {
			if (mprtim != 0) {
				monon_(&c__2);
			}
			(*jac)(n, m1, &x[1], &a[a_offset], &ifail);
			if (mprtim != 0) {
				monoff_(&c__2);
			}
		} else {
			if (*mstor == 0) {
				if (mprtim != 0) {
					monon_(&c__2);
				}
				if (jacgen == 3) {
					n1jcf_((S_fp)fcn, n, n, &x[1], &f[1], &a[a_offset], &xw[1]
					, &eta[1], &etamin, &etamax, &etadif, conv, nfcnj,
						&t1[1], &ifail);
				}
				if (jacgen == 2) {
					n1jac_((S_fp)fcn, n, n, &x[1], &f[1], &a[a_offset], &xw[1]
					, &ajdel, &ajmin, nfcnj, &t1[1], &ifail);
				}
				if (mprtim != 0) {
					monoff_(&c__2);
				}
			} else if (*mstor == 1) {
				if (mprtim != 0) {
					monon_(&c__2);
				}
				if (jacgen == 3) {
					n1jcfb_((S_fp)fcn, n, m1, &ml, &x[1], &f[1], &a[a_offset],
						&xw[1], &eta[1], &etamin, &etamax, &etadif, conv,
						nfcnj, &t1[1], &t2[1], &t3[1], &ifail);
				}
				if (jacgen == 2) {
					n1jacb_((S_fp)fcn, n, m1, &ml, &x[1], &f[1], &a[a_offset],
						&xw[1], &ajdel, &ajmin, nfcnj, &t1[1], &t2[1], &
						t3[1], &ifail);
				}
				if (mprtim != 0) {
					monoff_(&c__2);
				}
			}
		}
		++(*njac);
		/*     Exit, If ... */
		if (jacgen == 1 && ifail < 0) {
			*ierr = 83;
			goto L4299;
		}
		if (jacgen != 1 && ifail != 0) {
			*ierr = 82;
			goto L4299;
		}
	} else if (! qsimpl) {
		++(*new__);
	}
	if (*new__ == 0 && (qlu || *niter == 0)) {
		/*             ------------------------------------------------------ */
		/*             2.3.2.1 Save scaling values */
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			xwa[l1] = xw[l1];
			/* L2321: */
		}
		/*             ------------------------------------------------------ */
		/*             2.3.2.2 Prepare Jac. for use by band-solver DGBFA/DGBSL */
		if (*mstor == 1) {
			i__1 = *n;
			for (l1 = 1; l1 <= i__1; ++l1) {
				for (l2 = *m2; l2 >= 1; --l2) {
					a[l2 + ml + l1 * a_dim1] = a[l2 + l1 * a_dim1];
					/* L2323: */
				}
				/* L2322: */
			}
		}
		/*             ------------------------------------------------------ */
		/*             2.4 Prepare solution of the linear system */
		/*             ------------------------------------------------------ */
		/*             2.4.1 internal column scaling of matrix A */
		if (*mstor == 0) {
			i__1 = *n;
			for (k = 1; k <= i__1; ++k) {
				s1 = -xw[k];
				i__2 = *n;
				for (l1 = 1; l1 <= i__2; ++l1) {
					a[l1 + k * a_dim1] *= s1;
					/* L2412: */
				}
				/* L2410: */
			}
		} else if (*mstor == 1) {
			i__1 = *n;
			for (k = 1; k <= i__1; ++k) {
				/* Computing MAX */
				i__2 = *m2 + 1 - k, i__3 = ml + 1;
				l2 = max(i__2,i__3);
				/* Computing MIN */
				i__2 = *n + *m2 - k;
				l3 = min(i__2,*m1);
				s1 = -xw[k];
				i__2 = l3;
				for (l1 = l2; l1 <= i__2; ++l1) {
					a[l1 + k * a_dim1] *= s1;
					/* L2415: */
				}
				/* L2413: */
			}
		}
		/*             ---------------------------------------------------- */
		/*             2.4.2 Row scaling of matrix A */
		if (qscale) {
			if (*mstor == 0) {
				n1scrf_(n, n, &a[a_offset], &fw[1]);
			} else if (*mstor == 1) {
				n1scrb_(n, m1, &ml, &mu, &a[a_offset], &fw[1]);
			}
		} else {
			i__1 = *n;
			for (k = 1; k <= i__1; ++k) {
				fw[k] = 1.;
				/* L242: */
			}
		}
	}
	/*         -------------------------------------------------------- */
	/*         2.4.3 Save and scale values of F(N) */
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		fa[l1] = f[l1];
		t1[l1] = f[l1] * fw[l1];
		/* L243: */
	}
	/*         -------------------------------------------------------- */
	/*         3 Central part of iteration step */
	/*         -------------------------------------------------------- */
	/*         3.1 Solution of the linear system */
	/*         -------------------------------------------------------- */
	/*         3.1.1 Decomposition of (N,N)-matrix A */
	if (! qlinit) {
		niwla = iwk[18] + 1;
		nrwla = iwk[19] + 1;
		liwl = *liwk - niwla + 1;
		lrwl = *lrwk - nrwla + 1;
	}
	if (*new__ == 0 && (qlu || *niter == 0)) {
		if (mprtim != 0) {
			monon_(&c__3);
		}
		n1fact_(n, m1, &ml, &mu, &a[a_offset], &iopt[1], &ifail, &liwl, &iwk[
			niwla], &niluse, &lrwl, &rwk[nrwla], &nrluse);
			if (mprtim != 0) {
				monoff_(&c__3);
			}
			if (! qlinit) {
				*niwkfr += niluse;
				*nrwkfr += nrluse;
				/*             Store lengths of currently required workspaces */
				iwk[18] = *niwkfr - 1;
				iwk[19] = *nrwkfr - 1;
			}
			/*       Exit Repeat If ... */
			if (ifail != 0) {
				if (ifail == 1) {
					*ierr = 1;
				} else {
					*ierr = 80;
				}
				goto L4299;
			}
	}
	qlinit = TRUE_;
	/*         -------------------------------------------------------- */
	/*         3.1.2 Solution of linear (N,N)-system */
	if (*new__ == 0) {
		if (mprtim != 0) {
			monon_(&c__4);
		}
		n1solv_(n, m1, &ml, &mu, &a[a_offset], &t1[1], &iopt[1], &ifail, &
			liwl, &iwk[niwla], &idummy, &lrwl, &rwk[nrwla], &idummy);
		if (mprtim != 0) {
			monoff_(&c__4);
		}
		/*     Exit Repeat If ... */
		if (ifail != 0) {
			*ierr = 81;
			goto L4299;
		}
	} else {
		alfa1 = 0.;
		alfa2 = 0.;
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
			/* Computing 2nd power */
			d__1 = xw[i__];
			alfa1 += dx[i__] * dxq[i__] / (d__1 * d__1);
			/* Computing 2nd power */
			d__1 = dx[i__];
			/* Computing 2nd power */
			d__2 = xw[i__];
			alfa2 += d__1 * d__1 / (d__2 * d__2);
			/* L3121: */
		}
		alfa = alfa1 / alfa2;
		beta = 1. - alfa;
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
			t1[i__] = (dxq[i__] + (*fca - 1.) * alfa * dx[i__]) / beta;
			/* L3122: */
		}
		if (*new__ == 1) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
				dxsave[i__ + dxsave_dim1] = dx[i__];
				/* L3123: */
			}
		}
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
			dxsave[i__ + (*new__ + 1) * dxsave_dim1] = t1[i__];
			dx[i__] = t1[i__];
			t1[i__] /= xw[i__];
			/* L3124: */
		}
	}
	/*         -------------------------------------------------------- */
	/*         3.2 Evaluation of scaled natural level function SUMX */
	/*             scaled maximum error norm CONV */
	/*             evaluation of (scaled) standard level function */
	/*             DLEVF ( DLEVF only, if MPRMON.GE.2 ) */
	/*             and computation of ordinary Newton corrections */
	/*             DX(N) */
	if (! qsimpl) {
		L__1 = *new__ == 0;
		n1lvls_(n, &t1[1], &xw[1], &f[1], &dx[1], conv, sumx, dlevf, mprmon, &
			L__1);
	} else {
		L__1 = *new__ == 0;
		n1lvls_(n, &t1[1], &xwa[1], &f[1], &dx[1], conv, sumx, dlevf, mprmon, 
			&L__1);
	}
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		xa[l1] = x[l1];
		/* L32: */
	}
	sumxa = *sumx;
	dlevxa = sqrt(sumxa / (doublereal) ((f2c_real) (*n)));
	conva = *conv;
	dxanrm = wnorm_(n, &dx[1], &xw[1]);
	/*         -------------------------------------------------------- */
	/*         3.3 A - priori estimate of damping factor FC */
	if (*niter != 0 && *nonlin != 1 && *new__ == 0 && ! qordi) {
		/*           ------------------------------------------------------ */
		/*           3.3.1 Computation of the denominator of a-priori */
		/*                 estimate */
		fcdnm = 0.;
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			/* Computing 2nd power */
			d__1 = (dx[l1] - dxqa[l1]) / xw[l1];
			fcdnm += d__1 * d__1;
			/* L331: */
		}
		fcdnm *= *sumx;
		/*           ------------------------------------------------------ */
		/*           3.3.2 New damping factor */
		/* Computing 2nd power */
		d__1 = *fca;
		if (fcdnm > fcnump * fcmin2 || *nonlin == 4 && d__1 * d__1 * fcnump < 
			fcdnm * 4.) {
				dmypri = *fca * sqrt(fcnump / fcdnm);
				*fcpri = min(dmypri,1.);
				if (*nonlin == 4) {
					/* Computing MIN */
					d__1 = dmypri * .5;
					*fcpri = min(d__1,1.);
				}
		} else {
			*fcpri = 1.;
			/* $Test-begin */
			dmypri = -1.;
			/* $Test-end */
		}
		/* $Test-begin */
		if (*mprmon >= 5) {
			io___171.ciunit = *lumon;
			s_wsfe(&io___171);
			do_fio(&c__1, (char *)&(*fcpri), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*fca), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&dmypri, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&fcnump, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&fcdnm, (ftnlen)sizeof(doublereal));
			e_wsfe();
		}
		/* $Test-end */
		*fc = max(*fcpri,*fcmin);
		if (*qbdamp) {
			fcbh = *fca * fcbnd;
			if (*fc > fcbh) {
				*fc = fcbh;
				if (*mprmon >= 4) {
					io___173.ciunit = *lumon;
					s_wsle(&io___173);
					do_lio(&c__9, &c__1, " *** incr. rest. act. (a prio) ***",
						(ftnlen)34);
					e_wsle();
				}
			}
			fcbh = *fca / fcbnd;
			if (*fc < fcbh) {
				*fc = fcbh;
				if (*mprmon >= 4) {
					io___174.ciunit = *lumon;
					s_wsle(&io___174);
					do_lio(&c__9, &c__1, " *** decr. rest. act. (a prio) ***",
						(ftnlen)34);
					e_wsle();
				}
			}
		}
	}
	/* WEI */
	if (iormon >= 2) {
		sumxa2 = sumxa1;
		sumxa1 = sumxa0;
		sumxa0 = dlevxa;
		if (sumxa0 == 0.) {
			sumxa0 = small;
		}
		/*           Check convergence rates (linear and superlinear) */
		/*           ICONV : Convergence indicator */
		/*                   =0: No convergence indicated yet */
		/*                   =1: Damping factor is 1.0d0 */
		/*                   =2: Superlinear convergence detected (alpha >=1.2) */
		/*                   =3: Quadratic convergence detected (alpha > 1.8) */
		fcmon = min(*fc,fcmon);
		if (fcmon < 1.) {
			*iconv = 0;
			alphae = 0.;
		}
		if (fcmon == 1. && *iconv == 0) {
			*iconv = 1;
		}
		if (*niter >= 1) {
			clin1 = clin0;
			clin0 = sumxa0 / sumxa1;
		}
		if (*iconv >= 1 && *niter >= 2) {
			alphak = alphae;
			alphae = 0.;
			if (clin1 <= .95) {
				alphae = log(clin0) / log(clin1);
			}
			if (alphak != 0.) {
				alphak = (alphae + alphak) * .5;
			}
			alphaa = min(alphak,alphae);
			calphk = calpha;
			calpha = 0.;
			if (alphae != 0.) {
				calpha = sumxa1 / pow_dd(&sumxa2, &alphae);
			}
			sumxte = sqrt(calpha * calphk) * pow_dd(&sumxa1, &alphak) - 
				sumxa0;
			if (alphaa >= 1.2 && *iconv == 1) {
				*iconv = 2;
			}
			if (alphaa > 1.8) {
				*iconv = 3;
			}
			if (*mprmon >= 4) {
				io___182.ciunit = *lumon;
				s_wsfe(&io___182);
				do_fio(&c__1, (char *)&(*iconv), (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&alphae, (ftnlen)sizeof(doublereal));
				do_fio(&c__1, (char *)&calpha, (ftnlen)sizeof(doublereal));
				do_fio(&c__1, (char *)&clin0, (ftnlen)sizeof(doublereal));
				do_fio(&c__1, (char *)&alphak, (ftnlen)sizeof(doublereal));
				do_fio(&c__1, (char *)&sumxte, (ftnlen)sizeof(doublereal));
				e_wsfe();
			}
			if (*iconv >= 2 && alphaa < .9) {
				if (iormon == 3) {
					*ierr = 4;
					goto L4299;
				} else {
					qmstop = TRUE_;
				}
			}
		}
	}
	fcmon = *fc;

	/*         -------------------------------------------------------- */
	/*         3.4 Save natural level for later computations of */
	/*             corrector and print iterate */
	fcnumk = *sumx;
	if (*mprmon >= 2) {
		if (mprtim != 0) {
			monon_(&c__5);
		}
		n1prv1_(dlevf, &dlevxa, fckeep, niter, new__, mprmon, lumon, &qmixio);
		if (mprtim != 0) {
			monoff_(&c__5);
		}
	}
	nred = 0;
	qnext = FALSE_;
	qrep = FALSE_;
	/*         QREP = ITER .GT. ITMAX   or  QREP = ITER .GT. 0 */

	/*         Damping-factor reduction loop */
	/*         ================================ */
	/*         DO (Until) */
L34:
	/*           ------------------------------------------------------ */
	/*           3.5 Preliminary new iterate */
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		x[l1] = xa[l1] + dx[l1] * *fc;
		/* L35: */
	}
	/*           ----------------------------------------------------- */
	/*           3.5.2 Exit, if problem is specified as being linear */
	/*     Exit Repeat If ... */
	if (*nonlin == 1) {
		*ierr = 0;
		goto L4299;
	}
	/*           ------------------------------------------------------ */
	/*           3.6.1 Computation of the residual vector */
	if (mprtim != 0) {
		monon_(&c__1);
	}
	(*fcn)(n, &x[1], &f[1], &ifail);
	if (mprtim != 0) {
		monoff_(&c__1);
	}
	++(*nfcn);
	/*     Exit, If ... */
	if (ifail < 0) {
		*ierr = 82;
		goto L4299;
	}
	if (ifail == 1 || ifail == 2) {
		if (qordi) {
			*ierr = 82;
			goto L4299;
		}
		if (ifail == 1) {
			fcredu = .5;
		} else {
			fcredu = f[1];
			/*     Exit, If ... */
			if (fcredu <= 0. || fcredu >= 1.) {
				*ierr = 82;
				goto L4299;
			}
		}
		if (*mprmon >= 2) {
			/* L36101: */
			io___188.ciunit = *lumon;
			s_wsfe(&io___188);
			do_fio(&c__1, (char *)&(*niter), (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*new__), (ftnlen)sizeof(integer));
			e_wsfe();
		}
		fch = *fc;
		*fc = fcredu * *fc;
		if (fch > *fcmin) {
			*fc = max(*fc,*fcmin);
		}
		if (*qbdamp) {
			fcbh = fch / fcbnd;
			if (*fc < fcbh) {
				*fc = fcbh;
				if (*mprmon >= 4) {
					io___190.ciunit = *lumon;
					s_wsle(&io___190);
					do_lio(&c__9, &c__1, " *** decr. rest. act. (FCN redu.) "
						"***", (ftnlen)37);
					e_wsle();
				}
			}
		}
		if (*fc < *fcmin) {
			*ierr = 3;
			goto L4299;
		}
		/*     Break DO (Until) ... */
		goto L3109;
	}
	if (qordi) {
		/*             ------------------------------------------------------- */
		/*             3.6.2 Convergence test for ordinary Newton iteration */
		/*     Exit Repeat If ... */
		if (dxanrm <= *rtol) {
			*ierr = 0;
			goto L4299;
		}
	} else {
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			t1[l1] = f[l1] * fw[l1];
			/* L361: */
		}
		/*             -------------------------------------------------- */
		/*             3.6.3 Solution of linear (N,N)-system */
		if (mprtim != 0) {
			monon_(&c__4);
		}
		n1solv_(n, m1, &ml, &mu, &a[a_offset], &t1[1], &iopt[1], &ifail, &
			liwl, &iwk[niwla], &idummy, &lrwl, &rwk[nrwla], &idummy);
		if (mprtim != 0) {
			monoff_(&c__4);
		}
		/*     Exit Repeat If ... */
		if (ifail != 0) {
			*ierr = 81;
			goto L4299;
		}
		if (*new__ > 0) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
				dxq[i__] = t1[i__] * xwa[i__];
				/* L3630: */
			}
			i__1 = *new__;
			for (iloop = 1; iloop <= i__1; ++iloop) {
				sum1 = 0.;
				sum2 = 0.;
				i__2 = *n;
				for (i__ = 1; i__ <= i__2; ++i__) {
					/* Computing 2nd power */
					d__1 = xw[i__];
					sum1 += dxq[i__] * dxsave[i__ + iloop * dxsave_dim1] / (
						d__1 * d__1);
					/* Computing 2nd power */
					d__1 = dxsave[i__ + iloop * dxsave_dim1] / xw[i__];
					sum2 += d__1 * d__1;
					/* L3631: */
				}
				beta = sum1 / sum2;
				i__2 = *n;
				for (i__ = 1; i__ <= i__2; ++i__) {
					dxq[i__] += beta * dxsave[i__ + (iloop + 1) * dxsave_dim1]
					;
					t1[i__] = dxq[i__] / xw[i__];
					/* L3632: */
				}
				/* L363: */
			}
		}
		/*             ---------------------------------------------------- */
		/*             3.6.4 Evaluation of scaled natural level function */
		/*                   SUMX */
		/*                   scaled maximum error norm CONV and evaluation */
		/*                   of (scaled) standard level function DLEVFN */
		if (! qsimpl) {
			L__1 = *new__ == 0;
			n1lvls_(n, &t1[1], &xw[1], &f[1], &dxq[1], conv, sumx, &dlevfn, 
				mprmon, &L__1);
		} else {
			L__1 = *new__ == 0;
			n1lvls_(n, &t1[1], &xwa[1], &f[1], &dxq[1], conv, sumx, &dlevfn, 
				mprmon, &L__1);
		}
		dxnrm = wnorm_(n, &dxq[1], &xw[1]);
		/*             ----------------------------------------------------- */
		/*             3.6.5 Convergence test */
		/*     Exit Repeat If ... */
		if (dxnrm <= *rtol && dxanrm <= rsmall && *fc == 1.) {
			*ierr = 0;
			goto L4299;
		}

		*fca = *fc;
		/*             ---------------------------------------------------- */
		/*             3.6.6 Evaluation of reduced damping factor */
		th = *fca - 1.;
		fcdnm = 0.;
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			/* Computing 2nd power */
			d__1 = (dxq[l1] + th * dx[l1]) / xw[l1];
			fcdnm += d__1 * d__1;
			/* L39: */
		}
		if (fcdnm != 0.) {
			*dmycor = *fca * *fca * .5 * sqrt(fcnumk / fcdnm);
		} else {
			*dmycor = 1e35;
		}
		if (*nonlin <= 3) {
			fccor = min(1.,*dmycor);
		} else {
			/* Computing MIN */
			d__1 = 1., d__2 = *dmycor * .5;
			fccor = min(d__1,d__2);
		}
		/* $Test-begin */
		if (*mprmon >= 5) {
			io___196.ciunit = *lumon;
			s_wsfe(&io___196);
			do_fio(&c__1, (char *)&fccor, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*dmycor), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&fcnumk, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&fcdnm, (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*fca), (ftnlen)sizeof(doublereal));
			e_wsfe();
		}
		/* $Test-end */
	}
	/*           ------------------------------------------------------ */
	/*           3.7 Natural monotonicity test */
	if (*sumx > sumxa && ! qordi) {
		/*             ---------------------------------------------------- */
		/*             3.8 Output of iterate */
		if (*mprmon >= 3) {
			if (mprtim != 0) {
				monon_(&c__5);
			}
			d__1 = sqrt(*sumx / (doublereal) ((f2c_real) (*n)));
			n1prv2_(&dlevfn, &d__1, fc, niter, mprmon, lumon, &qmixio, "*", (
				ftnlen)1);
			if (mprtim != 0) {
				monoff_(&c__5);
			}
		}
		if (qmstop) {
			*ierr = 4;
			goto L4299;
		}
		/* Computing MIN */
		d__1 = fccor, d__2 = *fc * .5;
		fch = min(d__1,d__2);
		if (*fc > *fcmin) {
			*fc = max(fch,*fcmin);
		} else {
			*fc = fch;
		}
		if (*qbdamp) {
			fcbh = *fca / fcbnd;
			if (*fc < fcbh) {
				*fc = fcbh;
				if (*mprmon >= 4) {
					io___197.ciunit = *lumon;
					s_wsle(&io___197);
					do_lio(&c__9, &c__1, " *** decr. rest. act. (a post) ***",
						(ftnlen)34);
					e_wsle();
				}
			}
		}
		/* WEI */
		fcmon = *fc;

		/* $Test-begin */
		if (*mprmon >= 5) {
			io___198.ciunit = *lumon;
			s_wsfe(&io___198);
			do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
			e_wsfe();
		}
		/* $Test-end */
		qrep = TRUE_;
		++(*ncorr);
		++nred;
		/*             ---------------------------------------------------- */
		/*             3.10 If damping factor is too small: */
		/*                  Refresh Jacobian,if current Jacobian was computed */
		/*                  by a Rank1-update, else fail exit */
		qjcrfr = *fc < *fcmin || *new__ > 0 && nred > 1;
		/*     Exit Repeat If ... */
		if (qjcrfr && *new__ == 0) {
			*ierr = 3;
			goto L4299;
		}
	} else {
		if (! qordi && ! qrep && fccor > *sigma2 * *fc) {
			if (*mprmon >= 3) {
				if (mprtim != 0) {
					monon_(&c__5);
				}
				d__1 = sqrt(*sumx / (doublereal) ((f2c_real) (*n)));
				n1prv2_(&dlevfn, &d__1, fc, niter, mprmon, lumon, &qmixio, 
					"+", (ftnlen)1);
				if (mprtim != 0) {
					monoff_(&c__5);
				}
			}
			*fc = fccor;
			/* $Test-begin */
			if (*mprmon >= 5) {
				io___199.ciunit = *lumon;
				s_wsfe(&io___199);
				do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
				e_wsfe();
			}
			/* $Test-end */
			qrep = TRUE_;
		} else {
			qnext = TRUE_;
		}
	}
L3109:
	if (! (qnext || qjcrfr)) {
		goto L34;
	}
	/*         UNTIL ( expression - negated above) */
	/*         End of damping-factor reduction loop */
	/*         ======================================= */
	if (qjcrfr) {
		/*           ------------------------------------------------------ */
		/*           3.11 Restore former values for repeting iteration */
		/*                step */
		++(*nrejr1);
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			x[l1] = xa[l1];
			/* L3111: */
		}
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			f[l1] = fa[l1];
			/* L3112: */
		}
		if (*mprmon >= 2) {
			/* L31130: */
			io___200.ciunit = *lumon;
			s_wsfe(&io___200);
			do_fio(&c__1, (char *)&(*niter), (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&(*new__), (ftnlen)sizeof(integer));
			e_wsfe();
		}
		*fc = *fckeep;
		*fca = fck2;
		if (*niter == 0) {
			*fc = *fcmin;
		}
		qgenj = TRUE_;
	} else {
		/*           ------------------------------------------------------ */
		/*           4 Preparations to start the following iteration step */
		/*           ------------------------------------------------------ */
		/*           4.1 Print values */
		if (*mprmon >= 3 && ! qordi) {
			if (mprtim != 0) {
				monon_(&c__5);
			}
			d__1 = sqrt(*sumx / (doublereal) ((f2c_real) (*n)));
			i__1 = *niter + 1;
			n1prv2_(&dlevfn, &d__1, fc, &i__1, mprmon, lumon, &qmixio, "*", (
				ftnlen)1);
			if (mprtim != 0) {
				monoff_(&c__5);
			}
		}
		/*           Print the natural level of the current iterate and return */
		/*           it in one-step mode */
		*sumxs = *sumx;
		*sumx = sumxa;
		if (*mprsol >= 2 && *niter != 0) {
			if (mprtim != 0) {
				monon_(&c__5);
			}
			n1sout_(n, &xa[1], &c__2, &iopt[1], &rwk[1], lrwk, &iwk[1], liwk, 
				mprsol, lusol);
			if (mprtim != 0) {
				monoff_(&c__5);
			}
		} else if (*mprsol >= 1 && *niter == 0) {
			if (mprtim != 0) {
				monon_(&c__5);
			}
			n1sout_(n, &xa[1], &c__1, &iopt[1], &rwk[1], lrwk, &iwk[1], liwk, 
				mprsol, lusol);
			if (mprtim != 0) {
				monoff_(&c__5);
			}
		}
		++(*niter);
		*dlevf = dlevfn;
		/*     Exit Repeat If ... */
		if (*niter >= *nitmax) {
			*ierr = 2;
			goto L4299;
		}
		*fckeep = *fc;
		/*           ------------------------------------------------------ */
		/*           4.2 Return, if in one-step mode */

		/* Exit Subroutine If ... */
		if (mode == 1) {
			iwk[18] = niwla - 1;
			iwk[19] = nrwla - 1;
			iopt[1] = 1;
			return 0;
		}
	}
	goto L2;
	/*       End Repeat */
L4299:
	/*       End of main iteration loop */
	/*       ========================== */
	/*       ---------------------------------------------------------- */
	/*       9 Exits */
	/*       ---------------------------------------------------------- */
	/*       9.1 Solution exit */
	aprec = -1.;

	if (*ierr == 0 || *ierr == 4) {
		if (*nonlin != 1) {
			if (! qordi) {
				if (*ierr == 0) {
					aprec = sqrt(*sumx / (doublereal) ((f2c_real) (*n)));
					i__1 = *n;
					for (l1 = 1; l1 <= i__1; ++l1) {
						x[l1] += dxq[l1];
						/* L91: */
					}
				} else {
					aprec = sqrt(sumxa / (doublereal) ((f2c_real) (*n)));
					if (alphaa > 0. && iormon == 3) {
						i__1 = *n;
						for (l1 = 1; l1 <= i__1; ++l1) {
							x[l1] += dx[l1];
							/* L92: */
						}
					}
				}
				/*             Print final monitor output */
				if (*mprmon >= 2) {
					if (*ierr == 0) {
						if (mprtim != 0) {
							monon_(&c__5);
						}
						d__1 = sqrt(*sumx / (doublereal) ((f2c_real) (*n)));
						i__1 = *niter + 1;
						n1prv2_(&dlevfn, &d__1, fc, &i__1, mprmon, lumon, &
							qmixio, "*", (ftnlen)1);
						if (mprtim != 0) {
							monoff_(&c__5);
						}
					} else if (iormon == 3) {
						if (mprtim != 0) {
							monon_(&c__5);
						}
						d__1 = sqrt(sumxa / (doublereal) ((f2c_real) (*n)));
						n1prv1_(&dlevfn, &d__1, fc, niter, new__, mprmon, 
							lumon, &qmixio);
						if (mprtim != 0) {
							monoff_(&c__5);
						}
					}
				}
				if (iormon >= 2) {
					if (*iconv <= 1 && alphae != 0. && alphak != 0.) {
						*ierr = 5;
					}
				}
			} else {
				/*           IF (QORDI) THEN */
				aprec = sqrt(sumxa / (doublereal) ((f2c_real) (*n)));
			}
			if (*mprmon >= 1) {
				/* L91001: */
				if (qordi || *ierr == 4) {
					io___202.ciunit = *lumon;
					s_wsfe(&io___202);
					do_fio(&c__1, (char *)&(*niter), (ftnlen)sizeof(integer));
					do_fio(&c__1, (char *)&aprec, (ftnlen)sizeof(doublereal));
					e_wsfe();
				} else {
					io___203.ciunit = *lumon;
					s_wsfe(&io___203);
					i__1 = *niter + 1;
					do_fio(&c__1, (char *)&i__1, (ftnlen)sizeof(integer));
					do_fio(&c__1, (char *)&aprec, (ftnlen)sizeof(doublereal));
					e_wsfe();
				}
			}
		} else {
			if (*mprmon >= 1) {
				/* L91002: */
				io___204.ciunit = *lumon;
				s_wsfe(&io___204);
				e_wsfe();
			}
		}
	}
	/*       ---------------------------------------------------------- */
	/*       9.2 Fail exit messages */
	/*       ---------------------------------------------------------- */
	/*       9.2.1 Termination, since jacobian matrix became singular */
	if (*ierr == 1 && *mprerr >= 1) {
		/* L92101: */
		io___205.ciunit = *luerr;
		s_wsfe(&io___205);
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.2 Termination after more than NITMAX iterations */
	if (*ierr == 2 && *mprerr >= 1) {
		/* L92201: */
		io___206.ciunit = *luerr;
		s_wsfe(&io___206);
		do_fio(&c__1, (char *)&(*nitmax), (ftnlen)sizeof(integer));
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.3 Damping factor FC became too small */
	if (*ierr == 3 && *mprerr >= 1) {
		/* L92301: */
		io___207.ciunit = *luerr;
		s_wsfe(&io___207);
		do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
		e_wsfe();
	}
	/* WEI */
	/*       ---------------------------------------------------------- */
	/*       9.2.4.1 Superlinear convergence slowed down */
	if (*ierr == 4 && *mprerr >= 1) {
		/* L92401: */
		/* L92402: */
		if (qmstop) {
			if (*iconv == 2) {
				io___208.ciunit = *luerr;
				s_wsfe(&io___208);
				do_fio(&c__1, "superlinear", (ftnlen)11);
				e_wsfe();
			}
			if (*iconv == 3) {
				io___209.ciunit = *luerr;
				s_wsfe(&io___209);
				do_fio(&c__1, "quadratic", (ftnlen)9);
				e_wsfe();
			}
		} else {
			if (*iconv == 2) {
				io___210.ciunit = *luerr;
				s_wsfe(&io___210);
				do_fio(&c__1, "superlinear", (ftnlen)11);
				e_wsfe();
			}
			if (*iconv == 3) {
				io___211.ciunit = *luerr;
				s_wsfe(&io___211);
				do_fio(&c__1, "quadratic", (ftnlen)9);
				e_wsfe();
			}
		}
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.4.2 Convergence criterion satisfied before superlinear */
	/*               convergence has been established */
	if (*ierr == 5 && dlevfn == 0.) {
		*ierr = 0;
	}
	if (*ierr == 5 && *mprerr >= 1) {
		/* L92410: */
		io___212.ciunit = *luerr;
		s_wsfe(&io___212);
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.5 Error exit due to linear solver routine N1FACT */
	if (*ierr == 80 && *mprerr >= 1) {
		/* L92501: */
		io___213.ciunit = *luerr;
		s_wsfe(&io___213);
		do_fio(&c__1, (char *)&ifail, (ftnlen)sizeof(integer));
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.6 Error exit due to linear solver routine N1SOLV */
	if (*ierr == 81 && *mprerr >= 1) {
		/* L92601: */
		io___214.ciunit = *luerr;
		s_wsfe(&io___214);
		do_fio(&c__1, (char *)&ifail, (ftnlen)sizeof(integer));
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.7 Error exit due to fail of user function FCN */
	if (*ierr == 82 && *mprerr >= 1) {
		/* L92701: */
		io___215.ciunit = *luerr;
		s_wsfe(&io___215);
		do_fio(&c__1, (char *)&ifail, (ftnlen)sizeof(integer));
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.2.8 Error exit due to fail of user function JAC */
	if (*ierr == 83 && *mprerr >= 1) {
		/* L92801: */
		io___216.ciunit = *luerr;
		s_wsfe(&io___216);
		do_fio(&c__1, (char *)&ifail, (ftnlen)sizeof(integer));
		e_wsfe();
	}
	if (*ierr >= 80 && *ierr <= 83) {
		iwk[23] = ifail;
	}
	if ((*ierr == 82 || *ierr == 83) && *niter <= 1 && *mprerr >= 1) {
		io___217.ciunit = *luerr;
		s_wsfe(&io___217);
		e_wsfe();
	}
	/*       ---------------------------------------------------------- */
	/*       9.3 Common exit */
	if (*mprerr >= 3 && *ierr != 0 && *ierr != 4 && *nonlin != 1) {
		/* L93100: */
		io___218.ciunit = *luerr;
		s_wsfe(&io___218);
		do_fio(&c__1, (char *)&conva, (ftnlen)sizeof(doublereal));
		e_wsfe();
		aprec = conva;
	}
	*rtol = aprec;
	*sumx = sumxa;
	if (*mprsol >= 2 && *niter != 0) {
		mode = 2;
		if (qordi) {
			mode = 3;
		}
		if (mprtim != 0) {
			monon_(&c__5);
		}
		n1sout_(n, &xa[1], &mode, &iopt[1], &rwk[1], lrwk, &iwk[1], liwk, 
			mprsol, lusol);
		if (mprtim != 0) {
			monoff_(&c__5);
		}
	} else if (*mprsol >= 1 && *niter == 0) {
		if (mprtim != 0) {
			monon_(&c__5);
		}
		n1sout_(n, &xa[1], &c__1, &iopt[1], &rwk[1], lrwk, &iwk[1], liwk, 
			mprsol, lusol);
		if (mprtim != 0) {
			monoff_(&c__5);
		}
	}
	if (! qordi) {
		if (*ierr != 4) {
			++(*niter);
		}
		*dlevf = dlevfn;
		if (*mprsol >= 1) {
			/*           Print Solution or final iteration vector */
			if (*ierr == 0) {
				modefi = 3;
			} else {
				modefi = 4;
			}
			if (mprtim != 0) {
				monon_(&c__5);
			}
			n1sout_(n, &x[1], &modefi, &iopt[1], &rwk[1], lrwk, &iwk[1], liwk,
				mprsol, lusol);
			if (mprtim != 0) {
				monoff_(&c__5);
			}
		}
	}
	/*       Return the latest internal scaling to XSCAL */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
		xscal[i__] = xw[i__];
		/* L93: */
	}
	/*       End of exits */
	/*       End of subroutine N1INT */
	return 0;
} /* n1int_ */


/* Subroutine */ int n1scal_(integer *n, doublereal *x, doublereal *xa, 
	doublereal *xscal, doublereal *xw, integer *iscal, logical *qinisc, 
	integer *iopt, integer *lrwk, doublereal *rwk)
{
	/* Format strings */
	static char fmt_10[] = "(\002  \002,d18.10,\002  \002,d18.10)";

	/* System generated locals */
	integer i__1;
	doublereal d__1, d__2, d__3, d__4;

	/* Builtin functions */
	integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
		e_wsle(void), s_wsfe(cilist *), do_fio(integer *, char *, ftnlen),
		e_wsfe(void);

	/* Local variables */
	extern /* Subroutine */ int zibconst_(doublereal *, doublereal *);
	static integer l1;
	static doublereal small;
	static integer lumon;
	static doublereal epmach;
	static integer mprmon;

	/* Fortran I/O blocks */
	static cilist io___225 = { 0, 0, 0, 0, 0 };
	static cilist io___226 = { 0, 0, 0, 0, 0 };
	static cilist io___227 = { 0, 0, 0, 0, 0 };
	static cilist io___228 = { 0, 0, 0, fmt_10, 0 };
	static cilist io___229 = { 0, 0, 0, 0, 0 };
	static cilist io___230 = { 0, 0, 0, 0, 0 };


	/* *    Begin Prologue SCALE */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     S C A L E : To be used in connection with NLEQ1 . */
	/*       Computation of the internal scaling vector XW used for the */
	/*       Jacobian matrix, the iterate vector and it's related */
	/*       vectors - especially for the solution of the linear system */
	/*       and the computations of norms to avoid numerical overflow. */

	/* *    Input parameters */
	/*     ================ */

	/*     N         Int     Number of unknowns */
	/*     X(N)      Dble    Current iterate */
	/*     XA(N)     Dble    Previous iterate */
	/*     XSCAL(N)  Dble    User scaling passed from parameter XSCAL */
	/*                       of interface routine NLEQ1 */
	/*     ISCAL     Int     Option ISCAL passed from IOPT-field */
	/*                       (for details see description of IOPT-fields) */
	/*     QINISC    Logical = .TRUE.  : Initial scaling */
	/*                       = .FALSE. : Subsequent scaling */
	/*     IOPT(50)  Int     Options array passed from NLEQ1 parameter list */
	/*     LRWK      Int     Length of f2c_real workspace */
	/*     RWK(LRWK) Dble    Real workspace (see description above) */

	/* *    Output parameters */
	/*     ================= */

	/*     XW(N)     Dble   Scaling vector computed by this routine */
	/*                      All components must be positive. The follow- */
	/*                      ing relationship between the original vector */
	/*                      X and the scaled vector XSCAL holds: */
	/*                      XSCAL(I) = X(I)/XW(I) for I=1,...N */

	/* *    Subroutines called: ZIBCONST */

	/* *    Machine constants used */
	/*     ====================== */


	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* Parameter adjustments */
	--xw;
	--xscal;
	--xa;
	--x;
	--iopt;
	--rwk;

	/* Function Body */
	zibconst_(&epmach, &small);
	/* *    Begin */
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		if (*iscal == 1) {
			xw[l1] = xscal[l1];
		} else {
			/* Computing MAX */
			d__3 = xscal[l1], d__4 = ((d__1 = x[l1], abs(d__1)) + (d__2 = xa[
				l1], abs(d__2))) * .5, d__3 = max(d__3,d__4);
				xw[l1] = max(d__3,small);
		}
		/* L1: */
	}
	/* $Test-Begin */
	mprmon = iopt[13];
	if (mprmon >= 6) {
		lumon = iopt[14];
		io___225.ciunit = lumon;
		s_wsle(&io___225);
		do_lio(&c__9, &c__1, " ", (ftnlen)1);
		e_wsle();
		io___226.ciunit = lumon;
		s_wsle(&io___226);
		do_lio(&c__9, &c__1, " ++++++++++++++++++++++++++++++++++++++++++", (
			ftnlen)43);
		e_wsle();
		io___227.ciunit = lumon;
		s_wsle(&io___227);
		do_lio(&c__9, &c__1, "      X-components   Scaling-components    ", (
			ftnlen)43);
		e_wsle();
		io___228.ciunit = lumon;
		s_wsfe(&io___228);
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			do_fio(&c__1, (char *)&x[l1], (ftnlen)sizeof(doublereal));
			do_fio(&c__1, (char *)&xw[l1], (ftnlen)sizeof(doublereal));
		}
		e_wsfe();
		io___229.ciunit = lumon;
		s_wsle(&io___229);
		do_lio(&c__9, &c__1, " ++++++++++++++++++++++++++++++++++++++++++", (
			ftnlen)43);
		e_wsle();
		io___230.ciunit = lumon;
		s_wsle(&io___230);
		do_lio(&c__9, &c__1, " ", (ftnlen)1);
		e_wsle();
	}
	/* $Test-End */
	/*     End of subroutine N1SCAL */
	return 0;
} /* n1scal_ */


/* Subroutine */ int n1scrf_(integer *m, integer *n, doublereal *a, 
	doublereal *fw)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2;
	doublereal d__1;

	/* Local variables */
	static integer j, k;
	static doublereal s1, s2;

	/* *    Begin Prologue SCROWF */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     S C R O W F : Row Scaling of a (M,N)-matrix in full storage */
	/*                   mode */

	/* *    Input parameters (* marks inout parameters) */
	/*     =========================================== */

	/*       M           Int    Number of rows of the matrix */
	/*       N           Int    Number of columns of the matrix */
	/*     * A(M,N)      Dble   Matrix to be scaled */

	/* *    Output parameters */
	/*     ================= */

	/*       FW(M)       Dble   Row scaling factors - FW(i) contains */
	/*                          the factor by which the i-th row of A */
	/*                          has been multiplied */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* *    Begin */
	/* Parameter adjustments */
	--fw;
	a_dim1 = *m;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	i__1 = *m;
	for (k = 1; k <= i__1; ++k) {
		s1 = 0.;
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
			s2 = (d__1 = a[k + j * a_dim1], abs(d__1));
			if (s2 > s1) {
				s1 = s2;
			}
			/* L2: */
		}
		if (s1 > 0.) {
			s1 = 1. / s1;
			fw[k] = s1;
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
				a[k + j * a_dim1] *= s1;
				/* L3: */
			}
		} else {
			fw[k] = 1.;
		}
		/* L1: */
	}
	/*     End of subroutine N1SCRF */
	return 0;
} /* n1scrf_ */


/* Subroutine */ int n1scrb_(integer *n, integer *lda, integer *ml, integer *
	mu, doublereal *a, doublereal *fw)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2, i__3;
	doublereal d__1;

	/* Local variables */
	static integer k, k1, l1, l2, l3, m2;
	static doublereal s1, s2;

	/* *    Begin Prologue SCROWB */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     S C R O W B : Row Scaling of a (N,N)-matrix in band storage */
	/*                   mode */

	/* *    Input parameters (* marks inout parameters) */
	/*     =========================================== */

	/*       N           Int    Number of rows and columns of the matrix */
	/*       LDA         Int    Leading dimension of the matrix array */
	/*       ML          Int    Lower bandwidth of the matrix (see IOPT) */
	/*       MU          Int    Upper bandwidth of the matrix (see IOPT) */
	/*     * A(LDA,N)    Dble   Matrix to be scaled (see Note in routine */
	/*                          DGBFA for details on storage technique) */

	/* *    Output parameters */
	/*     ================= */

	/*       FW(N)       Dble   Row scaling factors - FW(i) contains */
	/*                          the factor by which the i-th row of */
	/*                          the matrix has been multiplied */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* *    Begin */
	/* Parameter adjustments */
	--fw;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	m2 = *ml + *mu + 1;
	i__1 = *n;
	for (k = 1; k <= i__1; ++k) {
		s1 = 0.;
		/* Computing MAX */
		i__2 = 1, i__3 = k - *ml;
		l2 = max(i__2,i__3);
		/* Computing MIN */
		i__2 = *n, i__3 = k + *mu;
		l3 = min(i__2,i__3);
		k1 = m2 + k;
		i__2 = l3;
		for (l1 = l2; l1 <= i__2; ++l1) {
			s2 = (d__1 = a[k1 - l1 + l1 * a_dim1], abs(d__1));
			if (s2 > s1) {
				s1 = s2;
			}
			/* L2: */
		}
		if (s1 > 0.) {
			s1 = 1. / s1;
			fw[k] = s1;
			i__2 = l3;
			for (l1 = l2; l1 <= i__2; ++l1) {
				a[k1 - l1 + l1 * a_dim1] *= s1;
				/* L3: */
			}
		} else {
			fw[k] = 1.;
		}
		/* L1: */
	}
	/*     End of subroutine N1SCRB */
	return 0;
} /* n1scrb_ */


/* Subroutine */ int n1fact_(integer *n, integer *lda, integer *ml, integer *
	mu, doublereal *a, integer *iopt, integer *ifail, integer *liwk, 
	integer *iwk, integer *laiwk, integer *lrwk, doublereal *rwk, integer 
	*larwk)
{
	/* Format strings */
	static char fmt_10001[] = "(/,\002 Insuffient workspace for linear solve"
		"r,\002,\002 at least needed more needed : \002,/,\002 \002,a,"
		"\002 workspace : \002,i4)";

	/* System generated locals */
	integer a_dim1, a_offset, i__1;

	/* Builtin functions */
	integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

	/* Local variables */
	static integer luerr, mstor;
	extern /* Subroutine */ int dgbtrf_(integer *, integer *, integer *, 
		integer *, doublereal *, integer *, integer *, integer *), 
		dgetrf_(integer *, integer *, doublereal *, integer *, integer *, 
		integer *);
	static integer mprerr;

	/* Fortran I/O blocks */
	static cilist io___246 = { 0, 0, 0, fmt_10001, 0 };


	/* *    Begin Prologue FACT */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     F A C T : Call linear algebra subprogram for factorization of */
	/*               a (N,N)-matrix */

	/* *    Input parameters (* marks inout parameters) */
	/*     =========================================== */

	/*     N             Int    Order of the linear system */
	/*     LDA           Int    Leading dimension of the matrix array A */
	/*     ML            Int    Lower bandwidth of the matrix (only for */
	/*                          banded systems) */
	/*     MU            Int    Upper bandwidth of the matrix (only for */
	/*                          banded systems) */
	/*   * A(LDA,N)      Dble   Matrix storage. See main routine NLEQ1, */
	/*                          note 4 for how to store a banded Jacobian */
	/*     IOPT(50)      Int    Option vector passed from NLEQ1 */

	/* *    Output parameters */
	/*     ================= */

	/*     IFAIL         Int    Error indicator returned by this routine: */
	/*                          = 0 matrix decomposition successfull */
	/*                          = 1 decomposition failed - */
	/*                              matrix is numerically singular */
	/*                          =10 supplied (integer) workspace too small */

	/* *    Workspace parameters */
	/*     ==================== */

	/*     LIWK          Int    Length of integer workspace passed to this */
	/*                          routine (In) */
	/*     IWK(LIWK)     Int    Integer Workspace supplied for this routine */
	/*     LAIWK         Int    Length of integer Workspace used by this */
	/*                          routine (out) */
	/*     LRWK          Int    Length of f2c_real workspace passed to this */
	/*                          routine (In) */
	/*     RWK(LRWK)     Dble   Real Workspace supplied for this routine */
	/*     LARWK         Int    Length of f2c_real Workspace used by this */
	/*                          routine (out) */

	/* *    Subroutines called:  DGETRF, DGBTRF */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* *    Begin */
	/* Parameter adjustments */
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	--iopt;
	--iwk;
	--rwk;

	/* Function Body */
	*laiwk = *n;
	*larwk = 0;
	/*     ( Real workspace starts at RWK(NRWKFR) ) */
	/*     IF (LIWK.GE.LAIWK.AND.LRWK.GE.LARWK) THEN */
	if (*liwk >= *laiwk) {
		mstor = iopt[4];
		if (mstor == 0) {
			dgetrf_(n, n, &a[a_offset], lda, &iwk[1], ifail);
		} else if (mstor == 1) {
			dgbtrf_(n, n, ml, mu, &a[a_offset], lda, &iwk[1], ifail);
		}
		if (*ifail != 0) {
			*ifail = 1;
		}
	} else {
		*ifail = 10;
		mprerr = iopt[11];
		luerr = iopt[12];
		/* L10001: */
		if (*liwk < *laiwk && mprerr > 0) {
			io___246.ciunit = luerr;
			s_wsfe(&io___246);
			do_fio(&c__1, "Integer", (ftnlen)7);
			i__1 = *laiwk - *liwk;
			do_fio(&c__1, (char *)&i__1, (ftnlen)sizeof(integer));
			e_wsfe();
		}
	}
	return 0;
} /* n1fact_ */


/* Subroutine */ int n1solv_(integer *n, integer *lda, integer *ml, integer *
	mu, doublereal *a, doublereal *b, integer *iopt, integer *ifail, 
	integer *liwk, integer *iwk, integer *laiwk, integer *lrwk, 
	doublereal *rwk, integer *larwk)
{
	/* System generated locals */
	integer a_dim1, a_offset;

	/* Local variables */
	static integer mstor;
	extern /* Subroutine */ int dgbtrs_(char *, integer *, integer *, integer 
		*, integer *, doublereal *, integer *, integer *, doublereal *, 
		integer *, integer *, ftnlen), dgetrs_(char *, integer *, integer 
		*, doublereal *, integer *, integer *, doublereal *, integer *, 
		integer *, ftnlen);

	/* *    Begin Prologue SOLVE */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     S O L V E : Call linear algebra subprogram for solution of */
	/*                 the linear system A*Z = B */

	/* *    Parameters */
	/*     ========== */

	/*     N,LDA,ML,MU,A,IOPT,IFAIL,LIWK,IWK,LAIWK,LRWK,RWK,LARWK : */
	/*                        See description for subroutine N1FACT. */
	/*     B          Dble    In:  Right hand side of the linear system */
	/*                        Out: Solution of the linear system */

	/*     Subroutines called: DGETRS, DGBTRS */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* *    Begin */
	/* Parameter adjustments */
	--b;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;
	--iopt;
	--iwk;
	--rwk;

	/* Function Body */
	mstor = iopt[4];
	if (mstor == 0) {
		dgetrs_("N", n, &c__1, &a[a_offset], lda, &iwk[1], &b[1], n, ifail, (
			ftnlen)1);
	} else if (mstor == 1) {
		dgbtrs_("N", n, ml, mu, &c__1, &a[a_offset], lda, &iwk[1], &b[1], n, 
			ifail, (ftnlen)1);
	}
	return 0;
} /* n1solv_ */


/* Subroutine */ int n1lvls_(integer *n, doublereal *dx1, doublereal *xw, 
	doublereal *f, doublereal *dxq, doublereal *conv, doublereal *sumx, 
	doublereal *dlevf, integer *mprmon, logical *qdscal)
{
	/* System generated locals */
	integer i__1;
	doublereal d__1;

	/* Builtin functions */
	double sqrt(doublereal);

	/* Local variables */
	static integer l1;
	static doublereal s1;

	/* *    Begin Prologue LEVELS */

	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     L E V E L S : To be used in connection with NLEQ1 . */
	/*     provides descaled solution, error norm and level functions */

	/* *    Input parameters (* marks inout parameters) */
	/*     =========================================== */

	/*       N              Int    Number of parameters to be estimated */
	/*       DX1(N)         Dble   array containing the scaled Newton */
	/*                             correction */
	/*       XW(N)          Dble   Array containing the scaling values */
	/*       F(N)           Dble   Array containing the residuum */

	/* *    Output parameters */
	/*     ================= */

	/*       DXQ(N)         Dble   Array containing the descaled Newton */
	/*                             correction */
	/*       CONV           Dble   Scaled maximum norm of the Newton */
	/*                             correction */
	/*       SUMX           Dble   Scaled natural level function value */
	/*       DLEVF          Dble   Standard level function value (only */
	/*                             if needed for print) */
	/*       MPRMON         Int    Print information parameter (see */
	/*                             driver routine NLEQ1 ) */
	/*       QDSCAL         Logic  .TRUE., if descaling of DX1 required, */
	/*                             else .FALSE. */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* *    Begin */
	/* Parameter adjustments */
	--dxq;
	--f;
	--xw;
	--dx1;

	/* Function Body */
	if (*qdscal) {
		/*       ------------------------------------------------------------ */
		/*       1.2 Descaling of solution DX1 ( stored to DXQ ) */
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			dxq[l1] = dx1[l1] * xw[l1];
			/* L12: */
		}
	}
	/*     ------------------------------------------------------------ */
	/*     2 Evaluation of scaled natural level function SUMX and */
	/*       scaled maximum error norm CONV */
	*conv = 0.;
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		s1 = (d__1 = dx1[l1], abs(d__1));
		if (s1 > *conv) {
			*conv = s1;
		}
		/* L20: */
	}
	*sumx = 0.;
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		/* Computing 2nd power */
		d__1 = dx1[l1];
		*sumx += d__1 * d__1;
		/* L21: */
	}
	/*     ------------------------------------------------------------ */
	/*     3 Evaluation of (scaled) standard level function DLEVF */
	*dlevf = 0.;
	i__1 = *n;
	for (l1 = 1; l1 <= i__1; ++l1) {
		/* Computing 2nd power */
		d__1 = f[l1];
		*dlevf += d__1 * d__1;
		/* L3: */
	}
	*dlevf = sqrt(*dlevf / (doublereal) ((f2c_real) (*n)));
	/*     End of subroutine N1LVLS */
	return 0;
} /* n1lvls_ */


/* Subroutine */ int n1jac_(S_fp fcn, integer *n, integer *lda, doublereal *x,
	doublereal *fx, doublereal *a, doublereal *yscal, doublereal *ajdel, 
	doublereal *ajmin, integer *nfcn, doublereal *fu, integer *ifail)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2;
	doublereal d__1, d__2, d__3, d__4;

	/* Builtin functions */
	double d_sign(doublereal *, doublereal *);

	/* Local variables */
	static integer i__, k;
	static doublereal u, w;

	/* * Begin Prologue N1JAC */

	/*  --------------------------------------------------------------------- */

	/* * Title */

	/*  Evaluation of a dense Jacobian matrix using finite difference */
	/*  approximation adapted for use in nonlinear systems solver NLEQ1 */

	/* * Environment       Fortran 77 */
	/*                    Double Precision */
	/*                    Sun 3/60, Sun OS */
	/* * Latest Revision   May 1990 */


	/* * Parameter list description */
	/*  -------------------------- */

	/* * External subroutines (to be supplied by the user) */
	/*  ------------------------------------------------- */

	/*  FCN        Ext     FCN (N, X, FX, IFAIL) */
	/*                     Subroutine in order to provide right-hand */
	/*                     side of first-order differential equations */
	/*    N        Int     Number of rows and columns of the Jacobian */
	/*    X(N)     Dble    The current scaled iterates */
	/*    FX(N)    Dble    Array containing FCN(X) */
	/*    IFAIL    Int     Return code */
	/*                     Whenever a negative value is returned by FCN */
	/*                     routine N1JAC is terminated immediately. */


	/* * Input parameters (* marks inout parameters) */
	/*  ---------------- */

	/*  N          Int     Number of rows and columns of the Jacobian */
	/*  LDA        Int     Leading Dimension of array A */
	/*  X(N)       Dble    Array containing the current scaled */
	/*                     iterate */
	/*  FX(N)      Dble    Array containing FCN(X) */
	/*  YSCAL(N)   Dble    Array containing the scaling factors */
	/*  AJDEL      Dble    Perturbation of component k: abs(Y(k))*AJDEL */
	/*  AJMIN      Dble    Minimum perturbation is AJMIN*AJDEL */
	/*  NFCN       Int  *  FCN - evaluation count */

	/* * Output parameters (* marks inout parameters) */
	/*  ----------------- */

	/*  A(LDA,N)   Dble    Array to contain the approximated */
	/*                     Jacobian matrix ( dF(i)/dx(j)in A(i,j)) */
	/*  NFCN       Int  *  FCN - evaluation count adjusted */
	/*  IFAIL      Int     Return code non-zero if Jacobian could not */
	/*                     be computed. */

	/* * Workspace parameters */
	/*  -------------------- */

	/*  FU(N)      Dble    Array to contain FCN(x+dx) for evaluation of */
	/*                     the numerator differences */

	/* * Called */
	/*  ------ */

	/*  --------------------------------------------------------------------- */

	/* * End Prologue */

	/* * Local variables */
	/*  --------------- */


	/* * Begin */

	/* Parameter adjustments */
	--fu;
	--yscal;
	--fx;
	--x;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	*ifail = 0;
	i__1 = *n;
	for (k = 1; k <= i__1; ++k) {
		w = x[k];
		/* Computing MAX */
		d__3 = (d__1 = x[k], abs(d__1)), d__3 = max(d__3,*ajmin), d__4 = 
			yscal[k];
		d__2 = max(d__3,d__4) * *ajdel;
		u = d_sign(&d__2, &x[k]);
		x[k] = w + u;

		(*fcn)(n, &x[1], &fu[1], ifail);
		++(*nfcn);
		if (*ifail != 0) {
			goto L99;
		}

		x[k] = w;
		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + k * a_dim1] = (fu[i__] - fx[i__]) / u;
			/* L11: */
		}
		/* L1: */
	}

L99:
	return 0;


	/* * End of N1JAC */

} /* n1jac_ */

/* Subroutine */ int n1jacb_(S_fp fcn, integer *n, integer *lda, integer *ml, 
	doublereal *x, doublereal *fx, doublereal *a, doublereal *yscal, 
	doublereal *ajdel, doublereal *ajmin, integer *nfcn, doublereal *fu, 
	doublereal *u, doublereal *w, integer *ifail)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
	doublereal d__1, d__2, d__3, d__4;

	/* Builtin functions */
	double d_sign(doublereal *, doublereal *);

	/* Local variables */
	static integer i__, k, i1, i2, jj, mh, mu, ldab;

	/* * Begin Prologue N1JACB */

	/*  --------------------------------------------------------------------- */

	/* * Title */

	/*  Evaluation of a banded Jacobian matrix using finite difference */
	/*  approximation adapted for use in nonlinear systems solver NLEQ1 */

	/* * Environment       Fortran 77 */
	/*                    Double Precision */
	/*                    Sun 3/60, Sun OS */
	/* * Latest Revision   May 1990 */


	/* * Parameter list description */
	/*  -------------------------- */

	/* * External subroutines (to be supplied by the user) */
	/*  ------------------------------------------------- */

	/*  FCN        Ext     FCN (N, X, FX, IFAIL) */
	/*                     Subroutine in order to provide right-hand */
	/*                     side of first-order differential equations */
	/*    N        Int     Number of rows and columns of the Jacobian */
	/*    X(N)     Dble    The current scaled iterates */
	/*    FX(N)    Dble    Array containing FCN(X) */
	/*    IFAIL    Int     Return code */
	/*                     Whenever a negative value is returned by FCN */
	/*                     routine N1JACB is terminated immediately. */


	/* * Input parameters (* marks inout parameters) */
	/*  ---------------- */

	/*  N          Int     Number of rows and columns of the Jacobian */
	/*  LDA        Int     Leading Dimension of array A */
	/*  ML         Int     Lower bandwidth of the Jacobian matrix */
	/*  X(N)       Dble    Array containing the current scaled */
	/*                     iterate */
	/*  FX(N)      Dble    Array containing FCN(X) */
	/*  YSCAL(N)   Dble    Array containing the scaling factors */
	/*  AJDEL      Dble    Perturbation of component k: abs(Y(k))*AJDEL */
	/*  AJMIN      Dble    Minimum perturbation is AJMIN*AJDEL */
	/*  NFCN       Int  *  FCN - evaluation count */

	/* * Output parameters (* marks inout parameters) */
	/*  ----------------- */

	/*  A(LDA,N)   Dble    Array to contain the approximated */
	/*                     Jacobian matrix ( dF(i)/dx(j)in A(i,j)) */
	/*  NFCN       Int  *  FCN - evaluation count adjusted */
	/*  IFAIL      Int     Return code non-zero if Jacobian could not */
	/*                     be computed. */

	/* * Workspace parameters */
	/*  -------------------- */

	/*  FU(N)      Dble    Array to contain FCN(x+dx) for evaluation of */
	/*                     the numerator differences */
	/*  U(N)       Dble    Array to contain dx(i) */
	/*  W(N)       Dble    Array to save original components of X */

	/* * Called */
	/*  ------ */


	/* * Constants */
	/*  --------- */


	/*  --------------------------------------------------------------------- */

	/* * End Prologue */

	/* * Local variables */
	/*  --------------- */


	/* * Begin */

	/* Parameter adjustments */
	--w;
	--u;
	--fu;
	--yscal;
	--fx;
	--x;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	*ifail = 0;
	mu = *lda - (*ml << 1) - 1;
	ldab = *ml + mu + 1;
	i__1 = ldab;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
			a[i__ + k * a_dim1] = 0.;
			/* L11: */
		}
		/* L1: */
	}

	i__1 = ldab;
	for (jj = 1; jj <= i__1; ++jj) {
		i__2 = *n;
		i__3 = ldab;
		for (k = jj; i__3 < 0 ? k >= i__2 : k <= i__2; k += i__3) {
			w[k] = x[k];
			/* Computing MAX */
			d__3 = (d__1 = x[k], abs(d__1)), d__3 = max(d__3,*ajmin), d__4 = 
				yscal[k];
			d__2 = max(d__3,d__4) * *ajdel;
			u[k] = d_sign(&d__2, &x[k]);
			x[k] = w[k] + u[k];
			/* L21: */
		}

		(*fcn)(n, &x[1], &fu[1], ifail);
		++(*nfcn);
		if (*ifail != 0) {
			goto L99;
		}

		i__3 = *n;
		i__2 = ldab;
		for (k = jj; i__2 < 0 ? k >= i__3 : k <= i__3; k += i__2) {
			x[k] = w[k];
			/* Computing MAX */
			i__4 = 1, i__5 = k - mu;
			i1 = max(i__4,i__5);
			/* Computing MIN */
			i__4 = *n, i__5 = k + *ml;
			i2 = min(i__4,i__5);
			mh = mu + 1 - k;
			i__4 = i2;
			for (i__ = i1; i__ <= i__4; ++i__) {
				a[mh + i__ + k * a_dim1] = (fu[i__] - fx[i__]) / u[k];
				/* L221: */
			}
			/* L22: */
		}

		/* L2: */
	}

L99:
	return 0;


	/* * End of N1JACB */

} /* n1jacb_ */

/* Subroutine */ int n1jcf_(S_fp fcn, integer *n, integer *lda, doublereal *x,
	doublereal *fx, doublereal *a, doublereal *yscal, doublereal *eta, 
	doublereal *etamin, doublereal *etamax, doublereal *etadif, 
	doublereal *conv, integer *nfcn, doublereal *fu, integer *ifail)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2;
	doublereal d__1, d__2, d__3, d__4;

	/* Builtin functions */
	double d_sign(doublereal *, doublereal *), sqrt(doublereal);

	/* Local variables */
	static integer i__, k;
	static doublereal u, w, hg;
	static integer is;
	static doublereal fhi, sumd;
	static logical qfine;

	/* * Begin Prologue N1JCF */

	/*  --------------------------------------------------------------------- */

	/* * Title */

	/*  Approximation of dense Jacobian matrix for nonlinear systems solver */
	/*  NLEQ1 with feed-back control of discretization and rounding errors */

	/* * Environment       Fortran 77 */
	/*                    Double Precision */
	/*                    Sun 3/60, Sun OS */
	/* * Latest Revision   May 1990 */


	/* * Parameter list description */
	/*  -------------------------- */

	/* * External subroutines (to be supplied by the user) */
	/*  ------------------------------------------------- */

	/*  FCN        Ext     FCN (N, X, FX, IFAIL) */
	/*                     Subroutine in order to provide right-hand */
	/*                     side of first-order differential equations */
	/*    N        Int     Number of rows and columns of the Jacobian */
	/*    X(N)     Dble    The current scaled iterates */
	/*    FX(N)    Dble    Array containing FCN(X) */
	/*    IFAIL    Int     Return code */
	/*                     Whenever a negative value is returned by FCN */
	/*                     routine N1JCF is terminated immediately. */


	/* * Input parameters (* marks inout parameters) */
	/*  ---------------- */

	/*  N          Int     Number of rows and columns of the Jacobian */
	/*  LDA        Int     Leading dimension of A (LDA .GE. N) */
	/*  X(N)       Dble    Array containing the current scaled */
	/*                     iterate */
	/*  FX(N)      Dble    Array containing FCN(X) */
	/*  YSCAL(N)   Dble    Array containing the scaling factors */
	/*  ETA(N)     Dble *  Array containing the scaled denominator */
	/*                     differences */
	/*  ETAMIN     Dble    Minimum allowed scaled denominator */
	/*  ETAMAX     Dble    Maximum allowed scaled denominator */
	/*  ETADIF     Dble    DSQRT (1.1*EPMACH) */
	/*                     EPMACH = machine precision */
	/*  CONV       Dble    Maximum norm of last (unrelaxed) Newton correction */
	/*  NFCN       Int  *  FCN - evaluation count */

	/* * Output parameters (* marks inout parameters) */
	/*  ----------------- */

	/*  A(LDA,N)   Dble    Array to contain the approximated */
	/*                     Jacobian matrix ( dF(i)/dx(j)in A(i,j)) */
	/*  ETA(N)     Dble *  Scaled denominator differences adjusted */
	/*  NFCN       Int  *  FCN - evaluation count adjusted */
	/*  IFAIL      Int     Return code non-zero if Jacobian could not */
	/*                     be computed. */

	/* * Workspace parameters */
	/*  -------------------- */

	/*  FU(N)      Dble    Array to contain FCN(x+dx) for evaluation of */
	/*                     the numerator differences */

	/* * Called */
	/*  ------ */


	/* * Constants */
	/*  --------- */


	/*  --------------------------------------------------------------------- */

	/* * End Prologue */

	/* * Local variables */
	/*  --------------- */


	/* * Begin */

	/* Parameter adjustments */
	--fu;
	--eta;
	--yscal;
	--fx;
	--x;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	i__1 = *n;
	for (k = 1; k <= i__1; ++k) {
		is = 0;
		/*        DO (Until) */
L11:
		w = x[k];
		d__1 = eta[k] * yscal[k];
		u = d_sign(&d__1, &x[k]);
		x[k] = w + u;
		(*fcn)(n, &x[1], &fu[1], ifail);
		++(*nfcn);
		/*           Exit, If ... */
		if (*ifail != 0) {
			goto L99;
		}
		x[k] = w;
		sumd = 0.;
		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
			/* Computing MAX */
			d__3 = (d__1 = fx[i__], abs(d__1)), d__4 = (d__2 = fu[i__], abs(
				d__2));
			hg = max(d__3,d__4);
			fhi = fu[i__] - fx[i__];
			if (hg != 0.) {
				/* Computing 2nd power */
				d__1 = fhi / hg;
				sumd += d__1 * d__1;
			}
			a[i__ + k * a_dim1] = fhi / u;
			/* L111: */
		}
		sumd = sqrt(sumd / (doublereal) (*n));
		qfine = TRUE_;
		if (sumd != 0. && is == 0) {
			/* Computing MIN */
			/* Computing MAX */
			d__3 = *etamin, d__4 = sqrt(*etadif / sumd) * eta[k];
			d__1 = *etamax, d__2 = max(d__3,d__4);
			eta[k] = min(d__1,d__2);
			is = 1;
			qfine = *conv < .1 || sumd >= *etamin;
		}
		if (! qfine) {
			goto L11;
		}
		/*        UNTIL ( expression - negated above) */
		/* L1: */
	}

	/*     Exit from DO-loop */
L99:

	return 0;

	/* * End of subroutine N1JCF */

} /* n1jcf_ */

/* Subroutine */ int n1jcfb_(S_fp fcn, integer *n, integer *lda, integer *ml, 
	doublereal *x, doublereal *fx, doublereal *a, doublereal *yscal, 
	doublereal *eta, doublereal *etamin, doublereal *etamax, doublereal *
	etadif, doublereal *conv, integer *nfcn, doublereal *fu, doublereal *
	u, doublereal *w, integer *ifail)
{
	/* System generated locals */
	integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
	doublereal d__1, d__2, d__3, d__4;

	/* Builtin functions */
	double d_sign(doublereal *, doublereal *), sqrt(doublereal);

	/* Local variables */
	static integer i__, k, i1, i2;
	static doublereal hg;
	static integer jj, mh, is, mu;
	static doublereal fhi;
	static integer ldab;
	static doublereal sumd;
	static logical qfine;

	/* * Begin Prologue N1JCFB */

	/*     --------------------------------------------------------------------- */

	/* * Title */

	/*  Approximation of banded Jacobian matrix for nonlinear systems solver */
	/*  NLEQ1 with feed-back control of discretization and rounding errors */

	/* * Environment       Fortran 77 */
	/*                    Double Precision */
	/*                    Sun 3/60, Sun OS */
	/* * Latest Revision   May 1990 */


	/* * Parameter list description */
	/*  -------------------------- */

	/* * External subroutines (to be supplied by the user) */
	/*  ------------------------------------------------- */

	/*  FCN        Ext     FCN (N, X, FX, IFAIL) */
	/*                     Subroutine in order to provide right-hand */
	/*                     side of first-order differential equations */
	/*    N        Int     Number of rows and columns of the Jacobian */
	/*    X(N)     Dble    The current scaled iterates */
	/*    FX(N)    Dble    Array containing FCN(X) */
	/*    IFAIL    Int     Return code */
	/*                     Whenever a negative value is returned by FCN */
	/*                     routine N1JCFB is terminated immediately. */


	/* * Input parameters (* marks inout parameters) */
	/*  ---------------- */

	/*  N          Int     Number of rows and columns of the Jacobian */
	/*  LDA        Int     Leading dimension of A (LDA .GE. ML+MU+1) */
	/*  ML         Int     Lower bandwidth of Jacobian matrix */
	/*  X(N)       Dble    Array containing the current scaled */
	/*                     iterate */
	/*  FX(N)      Dble    Array containing FCN(X) */
	/*  YSCAL(N)   Dble    Array containing the scaling factors */
	/*  ETA(N)     Dble *  Array containing the scaled denominator */
	/*                     differences */
	/*  ETAMIN     Dble    Minimum allowed scaled denominator */
	/*  ETAMAX     Dble    Maximum allowed scaled denominator */
	/*  ETADIF     Dble    DSQRT (1.1*EPMACH) */
	/*                     EPMACH = machine precision */
	/*  CONV       Dble    Maximum norm of last (unrelaxed) Newton correction */
	/*  NFCN       Int  *  FCN - evaluation count */

	/* * Output parameters (* marks inout parameters) */
	/*  ----------------- */

	/*  A(LDA,N)   Dble    Array to contain the approximated */
	/*                     Jacobian matrix ( dF(i)/dx(j)in A(i,j)) */
	/*  ETA(N)     Dble *  Scaled denominator differences adjusted */
	/*  NFCN       Int  *  FCN - evaluation count adjusted */
	/*  IFAIL      Int     Return code non-zero if Jacobian could not */
	/*                     be computed. */

	/* * Workspace parameters */
	/*  -------------------- */

	/*  FU(N)      Dble    Array to contain FCN(x+dx) for evaluation of */
	/*                     the numerator differences */
	/*  U(N)       Dble    Array to contain dx(i) */
	/*  W(N)       Dble    Array to save original components of X */

	/* * Called */
	/*  ------ */


	/* * Constants */
	/*  --------- */


	/*  --------------------------------------------------------------------- */

	/* * End Prologue */

	/* * Local variables */
	/*  --------------- */


	/* * Begin */

	/* Parameter adjustments */
	--w;
	--u;
	--fu;
	--eta;
	--yscal;
	--fx;
	--x;
	a_dim1 = *lda;
	a_offset = 1 + a_dim1;
	a -= a_offset;

	/* Function Body */
	mu = *lda - (*ml << 1) - 1;
	ldab = *ml + mu + 1;
	i__1 = ldab;
	for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
			a[i__ + k * a_dim1] = 0.;
			/* L11: */
		}
		/* L1: */
	}

	i__1 = ldab;
	for (jj = 1; jj <= i__1; ++jj) {
		is = 0;
		/*        DO (Until) */
L21:
		i__2 = *n;
		i__3 = ldab;
		for (k = jj; i__3 < 0 ? k >= i__2 : k <= i__2; k += i__3) {
			w[k] = x[k];
			d__1 = eta[k] * yscal[k];
			u[k] = d_sign(&d__1, &x[k]);
			x[k] = w[k] + u[k];
			/* L211: */
		}

		(*fcn)(n, &x[1], &fu[1], ifail);
		++(*nfcn);
		/*           Exit, If ... */
		if (*ifail != 0) {
			goto L99;
		}

		i__3 = *n;
		i__2 = ldab;
		for (k = jj; i__2 < 0 ? k >= i__3 : k <= i__3; k += i__2) {
			x[k] = w[k];
			sumd = 0.;
			/* Computing MAX */
			i__4 = 1, i__5 = k - mu;
			i1 = max(i__4,i__5);
			/* Computing MIN */
			i__4 = *n, i__5 = k + *ml;
			i2 = min(i__4,i__5);
			mh = mu + 1 - k;
			i__4 = i2;
			for (i__ = i1; i__ <= i__4; ++i__) {
				/* Computing MAX */
				d__3 = (d__1 = fx[i__], abs(d__1)), d__4 = (d__2 = fu[i__], 
					abs(d__2));
				hg = max(d__3,d__4);
				fhi = fu[i__] - fx[i__];
				if (hg != 0.) {
					/* Computing 2nd power */
					d__1 = fhi / hg;
					sumd += d__1 * d__1;
				}
				a[mh + i__ + k * a_dim1] = fhi / u[k];
				/* L213: */
			}
			sumd = sqrt(sumd / (doublereal) (*n));
			qfine = TRUE_;
			if (sumd != 0. && is == 0) {
				/* Computing MIN */
				/* Computing MAX */
				d__3 = *etamin, d__4 = sqrt(*etadif / sumd) * eta[k];
				d__1 = *etamax, d__2 = max(d__3,d__4);
				eta[k] = min(d__1,d__2);
				is = 1;
				qfine = *conv < .1 || sumd >= *etamin;
			}
			/* L212: */
		}
		if (! qfine) {
			goto L21;
		}
		/*        UNTIL ( expression - negated above) */
		/* L2: */
	}

	/*     Exit from DO-loop */
L99:

	return 0;

	/* * End of subroutine N1JCFB */

} /* n1jcfb_ */


/* Subroutine */ int n1prv1_(doublereal *dlevf, doublereal *dlevx, doublereal 
	*fc, integer *niter, integer *new__, integer *mprmon, integer *lumon, 
	logical *qmixio)
{
	/* Format strings */
	static char fmt_1[] = "(2x,66(\002*\002))";
	static char fmt_2[] = "(8x,\002It\002,7x,\002Normf \002,10x,\002Normx"
		" \002,20x,\002New\002)";
	static char fmt_3[] = "(8x,\002It\002,7x,\002Normf \002,10x,\002Normx"
		" \002,8x,\002Damp.Fct.\002,3x,\002New\002)";
	static char fmt_4[] = "(6x,i4,5x,d10.3,2x,4x,d10.3,17x,i2)";
	static char fmt_5[] = "(6x,i4,5x,d10.3,6x,d10.3,6x,f7.5,4x,i2)";
	static char fmt_6[] = "(2x,66(\002*\002))";

	/* Builtin functions */
	integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

	/* Fortran I/O blocks */
	static cilist io___284 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___285 = { 0, 0, 0, fmt_2, 0 };
	static cilist io___286 = { 0, 0, 0, fmt_3, 0 };
	static cilist io___287 = { 0, 0, 0, fmt_4, 0 };
	static cilist io___288 = { 0, 0, 0, fmt_5, 0 };
	static cilist io___289 = { 0, 0, 0, fmt_6, 0 };


	/* *    Begin Prologue N1PRV1 */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     N 1 P R V 1 : Printing of intermediate values (Type 1 routine) */

	/* *    Parameters */
	/*     ========== */

	/*     DLEVF, DLEVX   See descr. of internal double variables of N1INT */
	/*     FC,NITER,NEW,MPRMON,LUMON */
	/*                  See parameter descr. of subroutine N1INT */
	/*     QMIXIO Logical  = .TRUE.  , if LUMON.EQ.LUSOL */
	/*                     = .FALSE. , if LUMON.NE.LUSOL */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/*     Print Standard - and natural level */
	if (*qmixio) {
		/* L1: */
		io___284.ciunit = *lumon;
		s_wsfe(&io___284);
		e_wsfe();
		/* L2: */
		if (*mprmon >= 3) {
			io___285.ciunit = *lumon;
			s_wsfe(&io___285);
			e_wsfe();
		}
		/* L3: */
		if (*mprmon == 2) {
			io___286.ciunit = *lumon;
			s_wsfe(&io___286);
			e_wsfe();
		}
	}
	/* L4: */
	if (*mprmon >= 3 || *niter == 0) {
		io___287.ciunit = *lumon;
		s_wsfe(&io___287);
		do_fio(&c__1, (char *)&(*niter), (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&(*dlevf), (ftnlen)sizeof(doublereal));
		do_fio(&c__1, (char *)&(*dlevx), (ftnlen)sizeof(doublereal));
		do_fio(&c__1, (char *)&(*new__), (ftnlen)sizeof(integer));
		e_wsfe();
	}
	/* L5: */
	if (*mprmon == 2 && *niter != 0) {
		io___288.ciunit = *lumon;
		s_wsfe(&io___288);
		do_fio(&c__1, (char *)&(*niter), (ftnlen)sizeof(integer));
		do_fio(&c__1, (char *)&(*dlevf), (ftnlen)sizeof(doublereal));
		do_fio(&c__1, (char *)&(*dlevx), (ftnlen)sizeof(doublereal));
		do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
		do_fio(&c__1, (char *)&(*new__), (ftnlen)sizeof(integer));
		e_wsfe();
	}
	if (*qmixio) {
		/* L6: */
		io___289.ciunit = *lumon;
		s_wsfe(&io___289);
		e_wsfe();
	}
	/*     End of subroutine N1PRV1 */
	return 0;
} /* n1prv1_ */


/* Subroutine */ int n1prv2_(doublereal *dlevf, doublereal *dlevx, doublereal 
	*fc, integer *niter, integer *mprmon, integer *lumon, logical *qmixio,
	char *cmark, ftnlen cmark_len)
{
	/* Format strings */
	static char fmt_1[] = "(2x,66(\002*\002))";
	static char fmt_2[] = "(8x,\002It\002,7x,\002Normf \002,10x,\002Normx"
		" \002,8x,\002Damp.Fct.\002)";
	static char fmt_3[] = "(6x,i4,5x,d10.3,4x,a1,1x,d10.3,2x,4x,f7.5)";
	static char fmt_4[] = "(2x,66(\002*\002))";

	/* Builtin functions */
	integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

	/* Fortran I/O blocks */
	static cilist io___290 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___291 = { 0, 0, 0, fmt_2, 0 };
	static cilist io___292 = { 0, 0, 0, fmt_3, 0 };
	static cilist io___293 = { 0, 0, 0, fmt_4, 0 };


	/* *    Begin Prologue N1PRV2 */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     N 1 P R V 2 : Printing of intermediate values (Type 2 routine) */

	/* *    Parameters */
	/*     ========== */

	/*     DLEVF, DLEVX   See descr. of internal double variables of N2INT */
	/*     FC,NITER,MPRMON,LUMON */
	/*                  See parameter descr. of subroutine N2INT */
	/*     QMIXIO Logical  = .TRUE.  , if LUMON.EQ.LUSOL */
	/*                     = .FALSE. , if LUMON.NE.LUSOL */
	/*     CMARK Char*1    Marker character to be printed before DLEVX */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/*     Print Standard - and natural level, and damping */
	/*     factor */
	if (*qmixio) {
		/* L1: */
		io___290.ciunit = *lumon;
		s_wsfe(&io___290);
		e_wsfe();
		/* L2: */
		io___291.ciunit = *lumon;
		s_wsfe(&io___291);
		e_wsfe();
	}
	/* L3: */
	io___292.ciunit = *lumon;
	s_wsfe(&io___292);
	do_fio(&c__1, (char *)&(*niter), (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&(*dlevf), (ftnlen)sizeof(doublereal));
	do_fio(&c__1, cmark, (ftnlen)1);
	do_fio(&c__1, (char *)&(*dlevx), (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&(*fc), (ftnlen)sizeof(doublereal));
	e_wsfe();
	if (*qmixio) {
		/* L4: */
		io___293.ciunit = *lumon;
		s_wsfe(&io___293);
		e_wsfe();
	}
	/*     End of subroutine N1PRV2 */
	return 0;
} /* n1prv2_ */


/* Subroutine */ int n1sout_(integer *n, doublereal *x, integer *mode, 
	integer *iopt, doublereal *rwk, integer *nrw, integer *iwk, integer *
	niw, integer *mprint, integer *luout)
{
	/* Format strings */
	static char fmt_1[] = "(\002  \002,a,\002 data:\002,/)";
	static char fmt_101[] = "(\002  Start data:\002,/,\002  N =\002,i5,//"
		",\002  Format: iteration-number, (x(i),i=1,...N), \002,\002Normf"
		" , Normx \002,/)";
	static char fmt_2[] = "(\002 \002,i5)";
	static char fmt_3[] = "((12x,3(d18.10,1x)))";
	static char fmt_10[] = "(\002&name com\002,i3.3,:,255(7(\002, com\002,i3"
		".3,:),/))";
	static char fmt_15[] = "(\002&def  com\002,i3.3,:,255(7(\002, com\002,i3"
		".3,:),/))";
	static char fmt_16[] = "(6x,\002: X=1, Y=\002,i3)";
	static char fmt_20[] = "(\002&data \002,i5)";
	static char fmt_21[] = "((6x,4(d18.10)))";
	static char fmt_30[] = "(\002&wktype 3111\002,/,\002&atext x 'iter'\002)";
	static char fmt_35[] = "(\002&vars = com\002,i3.3,/,\002&atext y 'x\002,"
		"i3,\002'\002,/,\002&run\002)";
	static char fmt_36[] = "(\002&vars = com\002,i3.3,/,\002&atext y '\002"
		",a,\002'\002,/,\002&run\002)";

	/* System generated locals */
	integer i__1, i__2;
	doublereal d__1;

	/* Builtin functions */
	integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);
	double sqrt(doublereal);

	/* Local variables */
	static integer i__, l1;
	static logical qgraz, qnorm;

	/* Fortran I/O blocks */
	static cilist io___296 = { 0, 0, 0, fmt_101, 0 };
	static cilist io___297 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___298 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___299 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___300 = { 0, 0, 0, fmt_2, 0 };
	static cilist io___301 = { 0, 0, 0, fmt_3, 0 };
	static cilist io___303 = { 0, 0, 0, fmt_3, 0 };
	static cilist io___304 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___305 = { 0, 0, 0, fmt_1, 0 };
	static cilist io___306 = { 0, 0, 0, fmt_10, 0 };
	static cilist io___308 = { 0, 0, 0, fmt_15, 0 };
	static cilist io___309 = { 0, 0, 0, fmt_16, 0 };
	static cilist io___310 = { 0, 0, 0, fmt_20, 0 };
	static cilist io___311 = { 0, 0, 0, fmt_21, 0 };
	static cilist io___312 = { 0, 0, 0, fmt_21, 0 };
	static cilist io___313 = { 0, 0, 0, fmt_30, 0 };
	static cilist io___314 = { 0, 0, 0, fmt_35, 0 };
	static cilist io___315 = { 0, 0, 0, fmt_36, 0 };


	/* *    Begin Prologue SOLOUT */
	/*     ------------------------------------------------------------ */

	/* *    Summary : */

	/*     S O L O U T : Printing of iterate (user customizable routine) */

	/* *    Input parameters */
	/*     ================ */

	/*     N         Int Number of equations/unknowns */
	/*     X(N)   Double iterate vector */
	/*     MODE          =1 This routine is called before the first */
	/*                      Newton iteration step */
	/*                   =2 This routine is called with an intermedi- */
	/*                      ate iterate X(N) */
	/*                   =3 This is the last call with the solution */
	/*                      vector X(N) */
	/*                   =4 This is the last call with the final, but */
	/*                      not solution vector X(N) */
	/*     IOPT(50)  Int The option array as passed to the driver */
	/*                   routine(elements 46 to 50 may be used */
	/*                   for user options) */
	/*     MPRINT    Int Solution print level */
	/*                   (see description of IOPT-field MPRINT) */
	/*     LUOUT     Int the solution print unit */
	/*                   (see description of see IOPT-field LUSOL) */


	/* *    Workspace parameters */
	/*     ==================== */

	/*     NRW, RWK, NIW, IWK    see description in driver routine */

	/* *    Use of IOPT by this routine */
	/*     =========================== */

	/*     Field 46:       =0 Standard output */
	/*                     =1 GRAZIL suitable output */

	/*     ------------------------------------------------------------ */
	/* *    End Prologue */
	/* *    Begin */
	/* Parameter adjustments */
	--x;
	--iopt;
	--rwk;
	--iwk;

	/* Function Body */
	qnorm = iopt[46] == 0;
	qgraz = iopt[46] == 1;
	if (qnorm) {
		/* L1: */
		if (*mode == 1) {
			/* L101: */
			io___296.ciunit = *luout;
			s_wsfe(&io___296);
			do_fio(&c__1, (char *)&(*n), (ftnlen)sizeof(integer));
			e_wsfe();
			io___297.ciunit = *luout;
			s_wsfe(&io___297);
			do_fio(&c__1, "Initial", (ftnlen)7);
			e_wsfe();
		} else if (*mode == 3) {
			io___298.ciunit = *luout;
			s_wsfe(&io___298);
			do_fio(&c__1, "Solution", (ftnlen)8);
			e_wsfe();
		} else if (*mode == 4) {
			io___299.ciunit = *luout;
			s_wsfe(&io___299);
			do_fio(&c__1, "Final", (ftnlen)5);
			e_wsfe();
		}
		/* L2: */
		/*        WRITE          NITER */
		io___300.ciunit = *luout;
		s_wsfe(&io___300);
		do_fio(&c__1, (char *)&iwk[1], (ftnlen)sizeof(integer));
		e_wsfe();
		/* L3: */
		io___301.ciunit = *luout;
		s_wsfe(&io___301);
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			do_fio(&c__1, (char *)&x[l1], (ftnlen)sizeof(doublereal));
		}
		e_wsfe();
		/*        WRITE          DLEVF,   DLEVX */
		io___303.ciunit = *luout;
		s_wsfe(&io___303);
		do_fio(&c__1, (char *)&rwk[19], (ftnlen)sizeof(doublereal));
		d__1 = sqrt(rwk[18] / (doublereal) ((f2c_real) (*n)));
		do_fio(&c__1, (char *)&d__1, (ftnlen)sizeof(doublereal));
		e_wsfe();
		if (*mode == 1 && *mprint >= 2) {
			io___304.ciunit = *luout;
			s_wsfe(&io___304);
			do_fio(&c__1, "Intermediate", (ftnlen)12);
			e_wsfe();
		} else if (*mode >= 3) {
			io___305.ciunit = *luout;
			s_wsfe(&io___305);
			do_fio(&c__1, "End", (ftnlen)3);
			e_wsfe();
		}
	}
	if (qgraz) {
		if (*mode == 1) {
			/* L10: */
			io___306.ciunit = *luout;
			s_wsfe(&io___306);
			i__1 = *n + 2;
			for (i__ = 1; i__ <= i__1; ++i__) {
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
			}
			e_wsfe();
			/* L15: */
			io___308.ciunit = *luout;
			s_wsfe(&io___308);
			i__1 = *n + 2;
			for (i__ = 1; i__ <= i__1; ++i__) {
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
			}
			e_wsfe();
			/* L16: */
			io___309.ciunit = *luout;
			s_wsfe(&io___309);
			i__1 = *n + 2;
			do_fio(&c__1, (char *)&i__1, (ftnlen)sizeof(integer));
			e_wsfe();
		}
		/* L20: */
		/*        WRITE          NITER */
		io___310.ciunit = *luout;
		s_wsfe(&io___310);
		do_fio(&c__1, (char *)&iwk[1], (ftnlen)sizeof(integer));
		e_wsfe();
		/* L21: */
		io___311.ciunit = *luout;
		s_wsfe(&io___311);
		i__1 = *n;
		for (l1 = 1; l1 <= i__1; ++l1) {
			do_fio(&c__1, (char *)&x[l1], (ftnlen)sizeof(doublereal));
		}
		e_wsfe();
		/*        WRITE          DLEVF,   DLEVX */
		io___312.ciunit = *luout;
		s_wsfe(&io___312);
		do_fio(&c__1, (char *)&rwk[19], (ftnlen)sizeof(doublereal));
		d__1 = sqrt(rwk[18] / (doublereal) ((f2c_real) (*n)));
		do_fio(&c__1, (char *)&d__1, (ftnlen)sizeof(doublereal));
		e_wsfe();
		if (*mode >= 3) {
			/* L30: */
			io___313.ciunit = *luout;
			s_wsfe(&io___313);
			e_wsfe();
			/* L35: */
			io___314.ciunit = *luout;
			s_wsfe(&io___314);
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
				do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
			}
			e_wsfe();
			/* L36: */
			io___315.ciunit = *luout;
			s_wsfe(&io___315);
			i__1 = *n + 1;
			do_fio(&c__1, (char *)&i__1, (ftnlen)sizeof(integer));
			do_fio(&c__1, "Normf ", (ftnlen)6);
			i__2 = *n + 2;
			do_fio(&c__1, (char *)&i__2, (ftnlen)sizeof(integer));
			do_fio(&c__1, "Normx ", (ftnlen)6);
			e_wsfe();
			/* 39       FORMAT('&stop') */
			/*         WRITE(LUOUT,39) */
		}
	}
	/*     End of subroutine N1SOUT */
	return 0;
} /* n1sout_ */

