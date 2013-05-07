/*
 * -----------------------------------------------------------------
 * $Revision: 601 $
 * $Date: 2012-07-10 12:56:42 -0700 (Tue, 10 Jul 2012) $
 * -----------------------------------------------------------------
 * Programmer(s): Allan Taylor, Alan Hindmarsh, Radu Serban, and
 *                Aaron Collier @ LLNL
 *  -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * KINBBDPRE module header file (private version)
 * -----------------------------------------------------------------
 */

#ifndef _KINBBDPRE_IMPL_H
#define _KINBBDPRE_IMPL_H

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

#include <kinsol/kinsol_bbdpre.h>
#include <sundials/sundials_band.h>
#include "kinsol_impl.h"

/*
 * -----------------------------------------------------------------
 * Definition of KBBDData
 * -----------------------------------------------------------------
 */

typedef struct KBBDPrecDataRec {

  /* passed by user to KINBBDPrecAlloc, used by pset/psolve functions */

  long int mudq, mldq, mukeep, mlkeep;
  KINLocalFn gloc;
  KINCommFn gcomm;

  /* relative error for the Jacobian DQ routine */

  realtype rel_uu;

  /* allocated for use by KINBBDPrecSetup */

  N_Vector vtemp3;

  /* set by KINBBDPrecSetup and used by KINBBDPrecSolve */

  DlsMat PP;
  long int *lpivots;

  /* set by KINBBDPrecAlloc and used by KINBBDPrecSetup */

  long int n_local;

  /* available for optional output */

  long int rpwsize;
  long int ipwsize;
  long int nge;

  /* pointer to KINSol memory */

  void *kin_mem;

} *KBBDPrecData;

/*
 *-----------------------------------------------------------------
 * KINBBDPRE error messages
 *-----------------------------------------------------------------
 */

#define MSGBBD_MEM_NULL    "KINSOL Memory is NULL."
#define MSGBBD_LMEM_NULL   "Linear solver memory is NULL. One of the SPILS linear solvers must be attached."
#define MSGBBD_MEM_FAIL    "A memory request failed."
#define MSGBBD_BAD_NVECTOR "A required vector operation is not implemented."
#define MSGBBD_FUNC_FAILED "The gloc or cfn routine failed in an unrecoverable manner."
#define MSGBBD_PMEM_NULL   "BBD peconditioner memory is NULL. IDABBDPrecInit must be called."

#ifdef __cplusplus
}
#endif

#endif
