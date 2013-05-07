/* zibsec.f -- translated by f2c (version 20090411).
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

/* Subroutine */ int zibsec_(f2c_real *cptim, integer *ifail)
{

/* ********************************************************************* */
/*  Set CPTIM to cpu time in seconds. */
/*  This routine is machine dependent. */
/* ********************************************************************* */

/*  Output parameters */
/*    CPTIM    REAL        Cpu time in seconds */
/*    IFAIL    INTEGER     Errorcode */

/*  CSUN code works on sun and MacOSX (g77). */
/*  CXLF code works on MacOSX (xlf) */
/*  CVIS code works with Visual Fortran (Windows) */

/* ********************************************************************* */


/* SUN      REAL RTIME(2) */
/* XLF      REAL(4) etime_ */
/* XLF      TYPE TB_TYPE */
/* XLF      SEQUENCE */
/* XLF      REAL(4) USRTIME */
/* XLF      REAL(4) SYSTIME */
/* XLF      END TYPE */
/* XLF      TYPE (TB_TYPE) ETIME_STRUCT */

    *cptim = 0.f;
    *ifail = 0;

/* SUN      CPTIM = ETIME(RTIME) */
/* SUN      CPTIM = RTIME(1) */

/* XLF      CPTIM = etime_(ETIME_STRUCT) */

/* VIS      CALL CPU_TIME(CPTIM) */

    return 0;
} /* zibsec_ */

