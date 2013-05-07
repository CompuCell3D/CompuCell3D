/* zibmon.f -- translated by f2c (version 20090411).
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

static integer c__1 = 1;
static integer c__9 = 9;
static integer c__4 = 4;

/* Subroutine */ int mon_0_(int n__, char *texth, integer *iounit, integer *
	indx, char *nameh, integer *iret, doublereal *aver, ftnlen texth_len, 
	ftnlen nameh_len)
{
    /* Initialized data */

    static f2c_real cptim = 0.f;
    static integer moni = 6;
    static integer info = 1;
    static integer ioncnt = -1;
    static logical qdisab = FALSE_;
    static logical qstart = FALSE_;

    /* Format strings */
    static char fmt_9000[] = "(///)";
    static char fmt_9010[] = "(1x,77(\002#\002))";
    static char fmt_9020[] = "(\002 #   \002,a,t75,\002   #\002)";
    static char fmt_9030[] = "(\002 #   \002,a40,1x,a31,\002#\002)";
    static char fmt_9040[] = "(\002 #   \002,a11,f11.3,5x,a13,f11.3,21x"
	    ",\002#\002)";
    static char fmt_9050[] = "(\002 #   \002,2x,\002Name\002,14x,\002Call"
	    "s\002,7x,\002Time\002,4x,\002Av-time\002,4x,\002% Total\002,6x"
	    ",\002% Sum   #\002)";
    static char fmt_9060[] = "(\002 #   \002,a17,i8,f11.3,f11.4,f11.2,14x"
	    ",\002#\002)";
    static char fmt_9070[] = "(\002 #   \002,a17,i8,f11.3,f11.4,f11.2,f11.2,"
	    "3x,\002#\002)";
    static char fmt_9080[] = "(/,\002 ++ Error in subroutine \002,a,\002 +"
	    "+\002,/,4x,a)";
    static char fmt_9090[] = "(/,\002 ++ Error in subroutine \002,a,\002 +"
	    "+\002,/,4x,a,\002 (\002,i6,\002).\002)";
    static char fmt_9100[] = "(/,\002 ++ Error in subroutine \002,a,\002 +"
	    "+\002,4x,a,/,4x,a,(10i4))";
    static char fmt_9110[] = "(/,\002 ++ Error in subroutine \002,a,\002 +"
	    "+\002,4x,a,i3,1x,a,/,4x,a,(10i4))";

    /* System generated locals */
    integer i__1;
    icilist ici__1;

    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer s_wsfi(icilist *), do_fio(integer *, char *, ftnlen), e_wsfi(void)
	    , s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void), s_wsfe(cilist *), e_wsfe(void);

    /* Local variables */
    static integer i__;
    static f2c_real pc1[26], pc2[26], sec[26];
    static logical qon[26];
    static f2c_real sum, sum0, asec[26];
    static char name__[17*26], text[30];
    static integer ifail, ncall[26], indact[20];
    extern /* Subroutine */ int zibsec_(f2c_real *, integer *);
    static integer maxind;

    /* Fortran I/O blocks */
    static cilist io___17 = { 0, 0, 0, 0, 0 };
    static cilist io___18 = { 0, 0, 0, 0, 0 };
    static cilist io___23 = { 0, 0, 0, fmt_9000, 0 };
    static cilist io___24 = { 0, 0, 0, fmt_9010, 0 };
    static cilist io___25 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___26 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___27 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___28 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___29 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___30 = { 0, 0, 0, fmt_9030, 0 };
    static cilist io___31 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___32 = { 0, 0, 0, fmt_9040, 0 };
    static cilist io___33 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___34 = { 0, 0, 0, fmt_9050, 0 };
    static cilist io___35 = { 0, 0, 0, fmt_9060, 0 };
    static cilist io___36 = { 0, 0, 0, fmt_9070, 0 };
    static cilist io___37 = { 0, 0, 0, fmt_9020, 0 };
    static cilist io___38 = { 0, 0, 0, fmt_9010, 0 };
    static cilist io___39 = { 0, 0, 0, fmt_9000, 0 };
    static cilist io___40 = { 0, 0, 0, fmt_9080, 0 };
    static cilist io___41 = { 0, 0, 0, fmt_9090, 0 };
    static cilist io___42 = { 0, 0, 0, fmt_9080, 0 };
    static cilist io___43 = { 0, 0, 0, fmt_9080, 0 };
    static cilist io___44 = { 0, 0, 0, fmt_9080, 0 };
    static cilist io___45 = { 0, 0, 0, fmt_9080, 0 };
    static cilist io___46 = { 0, 0, 0, fmt_9090, 0 };
    static cilist io___47 = { 0, 0, 0, fmt_9090, 0 };
    static cilist io___48 = { 0, 0, 0, fmt_9100, 0 };
    static cilist io___49 = { 0, 0, 0, fmt_9090, 0 };
    static cilist io___50 = { 0, 0, 0, fmt_9110, 0 };
    static cilist io___51 = { 0, 0, 0, fmt_9080, 0 };
    static cilist io___52 = { 0, 0, 0, fmt_9080, 0 };



/* * Begin Prologue MON */

/*  --------------------------------------------------------------------- */

/* * Title */

/*  Time monitor to discover cpu time consumption of interesting parts */
/*  of a code. */

/* * Written by        U. Nowak, U. Poehle, L. Weimann */
/* * Purpose           Cpu time measuring */
/* * Category          ... */
/* * File              mon.f */
/* * Version           2.1 */
/* * Latest Change     96/03/04 (1.3) */
/* * Library           CodeLib */
/* * Code              Fortran 77 */
/*                    Single precision */

/*  --------------------------------------------------------------------- */

/* * Summary */
/*  ------- */

/*  MON provides different entries */
/*   - to initialize the time monitor, */
/*   - to start and to stop the measuring, */
/*   - to start and to stop a number of stop-watches which may be nested, */
/*   - to print a summary table, */
/*   - to define text labels for the summary table, and */
/*   - to store the vector of average cpu times */

/*  Errors detected during the measuring will disable the time monitor */
/*  but will not affect the calling program. */

/*  --------------------------------------------------------------------- */

/* * Parameter list description */
/*  -------------------------- */

/* * Input parameters */
/*  ---------------- */

/*  TEXTH      Char    Text to identify the measuring (up to 31 */
/*                     characters will be printed in the headline of the */
/*                     summary table) */
/*                     Used by MONINI */

/*  IOUNIT     Int     Output unit for error messages and summary table */
/*                     Used by MONINI */

/*  INDX       Int     Number of a range of source code lines (a distinct */
/*                     part of the algorithm normally) for which the */
/*                     cpu time will be measured by a kind of stop-watch */
/*                     Used by MONDEF, MONON, and MONOFF */

/*  NAMEH      Char    Text to label the summary lines (up to 17 */
/*                     characters will be printed) */
/*                     Used by MONDEF */

/* * Output parameters */
/*  ----------------- */

/*  IRET       Int     Return code of ZIBSEC */
/*                     Used by MONSTR */

/*  AVER       Dble    Vector of average cpu time for all measurings */
/*                     Used by MONGET */


/* * End Prologue */
/*  ------------ */


/* * Constants */
/*  --------- */

/*  MAXTAB     Int     Maximum number of stop-watches */
/*  MNEST      Int     Maximum number of nested measurings */


/* * Local Variables */
/*  --------------- */

/*  ASEC       Real    Array of averages per call */
/*  INDACT     Int     Array for indices of nested measurings */
/*  NCALL      Int     Array of call counts */
/*  PC1        Real    Array of per cent values with respect to all */
/*                     measurings */
/*  PC2        Real    Array of per cent values with respect to the sum */
/*                     of all stop-watches */
/*  QDISAB     Log     Time monitor disabled */
/*  QON        Log     Array reflecting if an index is active */
/*  QSTART     Log     Time monitor started */
/*  SEC        Real    Array of cpu time measurings */








    switch(n__) {
	case 1: goto L_monini;
	case 2: goto L_mondef;
	case 3: goto L_monstr;
	case 4: goto L_monon;
	case 5: goto L_monoff;
	case 6: goto L_monhlt;
	case 7: goto L_monprt;
	case 8: goto L_monget;
	}


    return 0;



L_monini:
/*     Initialize monitor. */
/*     Has to be called first. May be called again after MONHLT. */

    s_copy(text, texth, (ftnlen)30, texth_len);
    moni = *iounit;

    if (ioncnt > 0 && ! qdisab) {
	goto L1070;
    }

    maxind = 0;
    ioncnt = 0;
    qdisab = FALSE_;

    for (i__ = 0; i__ <= 25; ++i__) {
	sec[i__] = 0.f;
	asec[i__] = 0.f;
	ncall[i__] = 0;
	qon[i__] = FALSE_;
	ici__1.icierr = 0;
	ici__1.icirnum = 1;
	ici__1.icirlen = 17;
	ici__1.iciunit = name__ + i__ * 17;
	ici__1.icifmt = "(A, I2)";
	s_wsfi(&ici__1);
	do_fio(&c__1, "Part ", (ftnlen)5);
	do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	e_wsfi();
/* L1000: */
    }
    s_copy(name__, "General", (ftnlen)17, (ftnlen)7);

    for (i__ = 1; i__ <= 20; ++i__) {
	indact[i__ - 1] = 0;
/* L1010: */
    }

    return 0;



L_mondef:
/*     Define one monitor entry. */
/*     May be called at any time before MONPRT. */

    if (qdisab) {
	return 0;
    }
    if (*indx < 0 || *indx > 25) {
	goto L1080;
    }

    s_copy(name__ + *indx * 17, nameh, (ftnlen)17, nameh_len);

    return 0;



L_monstr:
/*     Start monitor measurings. */
/*     Has to be called once after initialization. */

    if (qdisab) {
	return 0;
    }
    if (ioncnt < 0) {
	goto L1090;
    }
    if (ioncnt > 0) {
	goto L1100;
    }
    if (qon[0]) {
	goto L1110;
    }

    ifail = 0;
    zibsec_(&cptim, &ifail);

    if (ifail == 0) {
/*        Switch on general stop-watch */
	sec[0] = -cptim;
	qon[0] = TRUE_;
	ioncnt = 1;
	qstart = TRUE_;
    }
    *iret = ifail;

    return 0;



L_monon:
/*     Start one measuring. */
/*     A running stop-watch will be deactivated until the new stop-watch */
/*     stops. */

    if (! qstart) {
	return 0;
    }
    if (qdisab) {
	return 0;
    }
    if (ioncnt < 1) {
	goto L1120;
    }
    if (*indx > 25 || *indx <= 0) {
	goto L1130;
    }
    if (qon[*indx]) {
	goto L1140;
    }

    maxind = max(maxind,*indx);
    zibsec_(&cptim, &ifail);

/*     Hold actual stop-watch */
    sec[indact[ioncnt - 1]] += cptim;

/*     Switch on new stop-watch */
    if (info > 1) {
	io___17.ciunit = moni;
	s_wsle(&io___17);
	do_lio(&c__9, &c__1, " Enter ", (ftnlen)7);
	do_lio(&c__9, &c__1, name__ + *indx * 17, (ftnlen)17);
	do_lio(&c__4, &c__1, (char *)&sec[*indx], (ftnlen)sizeof(f2c_real));
	e_wsle();
    }

    ++ioncnt;
    if (ioncnt > 20) {
	goto L1150;
    }

    indact[ioncnt - 1] = *indx;
    sec[*indx] -= cptim;
    qon[*indx] = TRUE_;

    return 0;



L_monoff:
/*     Stop one measuring. */
/*     May be called for the stop-watch started most recently. */
/*     The stop-watch deactivated most recently will be activated. */

    if (! qstart) {
	return 0;
    }
    if (qdisab) {
	return 0;
    }
    if (*indx > 25 || *indx <= 0) {
	goto L1160;
    }
    if (indact[ioncnt - 1] != *indx) {
	goto L1170;
    }

    zibsec_(&cptim, &ifail);

/*     Switch off actual stop-watch */
    qon[*indx] = FALSE_;
    sec[*indx] += cptim;
    ++ncall[*indx];
    --ioncnt;
    if (info > 1) {
	io___18.ciunit = moni;
	s_wsle(&io___18);
	do_lio(&c__9, &c__1, " Exit ", (ftnlen)6);
	do_lio(&c__9, &c__1, name__ + *indx * 17, (ftnlen)17);
	do_lio(&c__4, &c__1, (char *)&sec[*indx], (ftnlen)sizeof(f2c_real));
	e_wsle();
    }

/*     Continue previous stop-watch */
    sec[indact[ioncnt - 1]] -= cptim;

    return 0;



L_monhlt:
/*     Terminate monitor. */
/*     Stops all active stop-watches. */

    if (! qstart) {
	return 0;
    }
    if (qdisab) {
	return 0;
    }

    zibsec_(&cptim, &ifail);

    for (i__ = ioncnt; i__ >= 1; --i__) {
	qon[indact[i__ - 1]] = FALSE_;
	sec[indact[i__ - 1]] += cptim;
	++ncall[indact[i__ - 1]];
/* L1020: */
    }

    ioncnt = 0;
/*     This means that the time monitor has to be started by MONSTR */
/*     before another measuring. */

    return 0;




L_monprt:
/*     Print statistics. */
/*     May be called after MONHLT only. */

    if (ioncnt > 0) {
	goto L1180;
    }
    if (! qstart) {
	goto L1190;
    }

    sum = 1e-10f;
    i__1 = maxind;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sum += sec[i__];
	if (ncall[i__] > 0) {
	    asec[i__] = sec[i__] / (f2c_real) ncall[i__];
	}
/* L1030: */
    }
    sum0 = sum + sec[0];
    if (ncall[0] > 0) {
	asec[0] = sec[0] / (f2c_real) ncall[0];
    }

    i__1 = maxind;
    for (i__ = 1; i__ <= i__1; ++i__) {
	pc1[i__] = sec[i__] * 100.f / sum0;
	pc2[i__] = sec[i__] * 100.f / sum;
/* L1040: */
    }
    pc1[0] = sec[0] * 100.f / sum0;

    io___23.ciunit = moni;
    s_wsfe(&io___23);
    e_wsfe();
    io___24.ciunit = moni;
    s_wsfe(&io___24);
    e_wsfe();
    io___25.ciunit = moni;
    s_wsfe(&io___25);
    do_fio(&c__1, " ", (ftnlen)1);
    e_wsfe();

    if (qdisab) {
	io___26.ciunit = moni;
	s_wsfe(&io___26);
	do_fio(&c__1, " ", (ftnlen)1);
	e_wsfe();
	io___27.ciunit = moni;
	s_wsfe(&io___27);
	do_fio(&c__1, "Warning  The following results may be misleading", (
		ftnlen)48);
	e_wsfe();
	io___28.ciunit = moni;
	s_wsfe(&io___28);
	do_fio(&c__1, "because an error occured and disabled the time monitor"
		, (ftnlen)54);
	e_wsfe();
    }

    io___29.ciunit = moni;
    s_wsfe(&io___29);
    do_fio(&c__1, " ", (ftnlen)1);
    e_wsfe();
    io___30.ciunit = moni;
    s_wsfe(&io___30);
    do_fio(&c__1, "Results from time monitor program for:", (ftnlen)38);
    do_fio(&c__1, text, (ftnlen)30);
    e_wsfe();

    io___31.ciunit = moni;
    s_wsfe(&io___31);
    do_fio(&c__1, " ", (ftnlen)1);
    e_wsfe();
    io___32.ciunit = moni;
    s_wsfe(&io___32);
    do_fio(&c__1, "Total time:", (ftnlen)11);
    do_fio(&c__1, (char *)&sum0, (ftnlen)sizeof(f2c_real));
    do_fio(&c__1, "Sum of parts:", (ftnlen)13);
    do_fio(&c__1, (char *)&sum, (ftnlen)sizeof(f2c_real));
    e_wsfe();

    io___33.ciunit = moni;
    s_wsfe(&io___33);
    do_fio(&c__1, " ", (ftnlen)1);
    e_wsfe();
    io___34.ciunit = moni;
    s_wsfe(&io___34);
    e_wsfe();

    io___35.ciunit = moni;
    s_wsfe(&io___35);
    do_fio(&c__1, name__, (ftnlen)17);
    do_fio(&c__1, (char *)&ncall[0], (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&sec[0], (ftnlen)sizeof(f2c_real));
    do_fio(&c__1, (char *)&asec[0], (ftnlen)sizeof(f2c_real));
    do_fio(&c__1, (char *)&pc1[0], (ftnlen)sizeof(f2c_real));
    e_wsfe();

    i__1 = maxind;
    for (i__ = 1; i__ <= i__1; ++i__) {
	io___36.ciunit = moni;
	s_wsfe(&io___36);
	do_fio(&c__1, name__ + i__ * 17, (ftnlen)17);
	do_fio(&c__1, (char *)&ncall[i__], (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&sec[i__], (ftnlen)sizeof(f2c_real));
	do_fio(&c__1, (char *)&asec[i__], (ftnlen)sizeof(f2c_real));
	do_fio(&c__1, (char *)&pc1[i__], (ftnlen)sizeof(f2c_real));
	do_fio(&c__1, (char *)&pc2[i__], (ftnlen)sizeof(f2c_real));
	e_wsfe();
/* L1050: */
    }



    io___37.ciunit = moni;
    s_wsfe(&io___37);
    do_fio(&c__1, " ", (ftnlen)1);
    e_wsfe();
    io___38.ciunit = moni;
    s_wsfe(&io___38);
    e_wsfe();
    io___39.ciunit = moni;
    s_wsfe(&io___39);
    e_wsfe();

    return 0;



L_monget:
/*     Store average cpu times per call. */
/*     May be called at any time after MONPRT. */

    if (! qstart) {
	return 0;
    }
    if (qdisab) {
	return 0;
    }

    i__1 = maxind;
    for (i__ = 0; i__ <= i__1; ++i__) {
	aver[i__] = (doublereal) asec[i__];
/* L1060: */
    }

    return 0;

/*  Error exits */

L1070:
    io___40.ciunit = moni;
    s_wsfe(&io___40);
    do_fio(&c__1, "MONINI", (ftnlen)6);
    do_fio(&c__1, "Time monitor is running already.", (ftnlen)32);
    e_wsfe();
    goto L1200;

L1080:
    io___41.ciunit = moni;
    s_wsfe(&io___41);
    do_fio(&c__1, "MONDEF", (ftnlen)6);
    do_fio(&c__1, "Index out of range", (ftnlen)18);
    do_fio(&c__1, (char *)&(*indx), (ftnlen)sizeof(integer));
    e_wsfe();
    goto L1200;

L1090:
    io___42.ciunit = moni;
    s_wsfe(&io___42);
    do_fio(&c__1, "MONSTR", (ftnlen)6);
    do_fio(&c__1, "Time monitor has to be initialized by MONINI first.", (
	    ftnlen)51);
    e_wsfe();
    goto L1200;

L1100:
    io___43.ciunit = moni;
    s_wsfe(&io___43);
    do_fio(&c__1, "MONSTR", (ftnlen)6);
    do_fio(&c__1, "Time monitor is running already.", (ftnlen)32);
    e_wsfe();
    goto L1200;

L1110:
    io___44.ciunit = moni;
    s_wsfe(&io___44);
    do_fio(&c__1, "MONSTR", (ftnlen)6);
    do_fio(&c__1, "Time monitor has been started already.", (ftnlen)38);
    e_wsfe();
    goto L1200;

L1120:
    io___45.ciunit = moni;
    s_wsfe(&io___45);
    do_fio(&c__1, "MONON", (ftnlen)5);
    do_fio(&c__1, "Time monitor is not yet started.", (ftnlen)32);
    e_wsfe();
    goto L1200;

L1130:
    io___46.ciunit = moni;
    s_wsfe(&io___46);
    do_fio(&c__1, "MONON", (ftnlen)5);
    do_fio(&c__1, "Index out of range", (ftnlen)18);
    do_fio(&c__1, (char *)&(*indx), (ftnlen)sizeof(integer));
    e_wsfe();
    goto L1200;

L1140:
    io___47.ciunit = moni;
    s_wsfe(&io___47);
    do_fio(&c__1, "MONON", (ftnlen)5);
    do_fio(&c__1, "Measuring is running already for this INDX", (ftnlen)42);
    do_fio(&c__1, (char *)&(*indx), (ftnlen)sizeof(integer));
    e_wsfe();
    goto L1200;

L1150:
    io___48.ciunit = moni;
    s_wsfe(&io___48);
    do_fio(&c__1, "MONON", (ftnlen)5);
    do_fio(&c__1, "Nesting is too deep.", (ftnlen)20);
    do_fio(&c__1, "The following indices are active", (ftnlen)32);
    i__1 = ioncnt;
    for (i__ = 0; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&indact[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_wsfe();
    goto L1200;

L1160:
    io___49.ciunit = moni;
    s_wsfe(&io___49);
    do_fio(&c__1, "MONOFF", (ftnlen)6);
    do_fio(&c__1, "Index out of range", (ftnlen)18);
    do_fio(&c__1, (char *)&(*indx), (ftnlen)sizeof(integer));
    e_wsfe();
    goto L1200;

L1170:
    io___50.ciunit = moni;
    s_wsfe(&io___50);
    do_fio(&c__1, "MONOFF", (ftnlen)6);
    do_fio(&c__1, "Measuring ", (ftnlen)10);
    do_fio(&c__1, (char *)&(*indx), (ftnlen)sizeof(integer));
    do_fio(&c__1, "cannot be stopped.", (ftnlen)18);
    do_fio(&c__1, "The following indices are active", (ftnlen)32);
    i__1 = ioncnt;
    for (i__ = 0; i__ <= i__1; ++i__) {
	do_fio(&c__1, (char *)&indact[i__ - 1], (ftnlen)sizeof(integer));
    }
    e_wsfe();
    goto L1200;

L1180:
    io___51.ciunit = moni;
    s_wsfe(&io___51);
    do_fio(&c__1, "MONPRT", (ftnlen)6);
    do_fio(&c__1, "Time monitor is still running.", (ftnlen)30);
    e_wsfe();
    goto L1200;

L1190:
    io___52.ciunit = moni;
    s_wsfe(&io___52);
    do_fio(&c__1, "MONPRT", (ftnlen)6);
    do_fio(&c__1, "Time monitor was not started.", (ftnlen)29);
    e_wsfe();
    goto L1200;

L1200:
    qdisab = TRUE_;
    return 0;





/*  End subroutine monitor */

} /* mon_ */

/* Subroutine */ int mon_(void)
{
    return mon_0_(0, (char *)0, (integer *)0, (integer *)0, (char *)0, (
	    integer *)0, (doublereal *)0, (ftnint)0, (ftnint)0);
    }

/* Subroutine */ int monini_(char *texth, integer *iounit, ftnlen texth_len)
{
    return mon_0_(1, texth, iounit, (integer *)0, (char *)0, (integer *)0, (
	    doublereal *)0, texth_len, (ftnint)0);
    }

/* Subroutine */ int mondef_(integer *indx, char *nameh, ftnlen nameh_len)
{
    return mon_0_(2, (char *)0, (integer *)0, indx, nameh, (integer *)0, (
	    doublereal *)0, (ftnint)0, nameh_len);
    }

/* Subroutine */ int monstr_(integer *iret)
{
    return mon_0_(3, (char *)0, (integer *)0, (integer *)0, (char *)0, iret, (
	    doublereal *)0, (ftnint)0, (ftnint)0);
    }

/* Subroutine */ int monon_(integer *indx)
{
    return mon_0_(4, (char *)0, (integer *)0, indx, (char *)0, (integer *)0, (
	    doublereal *)0, (ftnint)0, (ftnint)0);
    }

/* Subroutine */ int monoff_(integer *indx)
{
    return mon_0_(5, (char *)0, (integer *)0, indx, (char *)0, (integer *)0, (
	    doublereal *)0, (ftnint)0, (ftnint)0);
    }

/* Subroutine */ int monhlt_(void)
{
    return mon_0_(6, (char *)0, (integer *)0, (integer *)0, (char *)0, (
	    integer *)0, (doublereal *)0, (ftnint)0, (ftnint)0);
    }

/* Subroutine */ int monprt_(void)
{
    return mon_0_(7, (char *)0, (integer *)0, (integer *)0, (char *)0, (
	    integer *)0, (doublereal *)0, (ftnint)0, (ftnint)0);
    }

/* Subroutine */ int monget_(doublereal *aver)
{
    return mon_0_(8, (char *)0, (integer *)0, (integer *)0, (char *)0, (
	    integer *)0, aver, (ftnint)0, (ftnint)0);
    }

