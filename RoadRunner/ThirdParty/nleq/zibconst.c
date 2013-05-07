/* zibconst.f -- translated by f2c (version 20090411).
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

/* Subroutine */ int zibconst_(doublereal *epmach, doublereal *small)
{

/* ********************************************************************* */
/*  Set Approximations to machine constants. */
/*  This routine is machine dependent. */
/* ********************************************************************* */

/*  Output parameters */
/*    EPMACH    DOUBLE     Relative machine precision */
/*    SMALL     DOUBLE     SQRT of smallest positive machine number */

/*  The proposed values should work on Intel compatible cpus, */
/*  PowerPCs and Sun sparcs */

/* ********************************************************************* */

    *epmach = 1e-17;
    *small = 1e-150;
    return 0;
} /* zibconst_ */

