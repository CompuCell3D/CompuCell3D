/*
 * -----------------------------------------------------------------
 * $Revision: 601 $
 * $Date: 2012-07-10 12:56:42 -0700 (Tue, 10 Jul 2012) $
 * ----------------------------------------------------------------- 
 * Programmer(s): Michael Wittman, Alan C. Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * Implementation header file for the CVBBDPRE module.
 * -----------------------------------------------------------------
 */

#ifndef _CVBBDPRE_IMPL_H
#define _CVBBDPRE_IMPL_H

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

#include <cvode/cvode_bbdpre.h>
#include <sundials/sundials_band.h>

/*
 * -----------------------------------------------------------------
 * Type: CVBBDPrecData
 * -----------------------------------------------------------------
 */

typedef struct CVBBDPrecDataRec {

  /* passed by user to CVBBDPrecAlloc and used by PrecSetup/PrecSolve */

  long int mudq, mldq, mukeep, mlkeep;
  realtype dqrely;
  CVLocalFn gloc;
  CVCommFn cfn;

  /* set by CVBBDPrecSetup and used by CVBBDPrecSolve */

  DlsMat savedJ;
  DlsMat savedP;
  long int *lpivots;

  /* set by CVBBDPrecAlloc and used by CVBBDPrecSetup */

  long int n_local;

  /* available for optional output */

  long int rpwsize;
  long int ipwsize;
  long int nge;

  /* pointer to cvode_mem */

  void *cvode_mem;

} *CVBBDPrecData;

/*
 * -----------------------------------------------------------------
 * CVBBDPRE error messages
 * -----------------------------------------------------------------
 */

#define MSGBBD_MEM_NULL    "Integrator memory is NULL."
#define MSGBBD_LMEM_NULL   "Linear solver memory is NULL. One of the SPILS linear solvers must be attached."
#define MSGBBD_MEM_FAIL    "A memory request failed."
#define MSGBBD_BAD_NVECTOR "A required vector operation is not implemented."
#define MSGBBD_PMEM_NULL   "BBD peconditioner memory is NULL. CVBBDPrecInit must be called."
#define MSGBBD_FUNC_FAILED "The gloc or cfn routine failed in an unrecoverable manner."

#ifdef __cplusplus
}
#endif

#endif
