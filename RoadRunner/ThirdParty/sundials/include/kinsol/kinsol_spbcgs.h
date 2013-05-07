/*
 * -----------------------------------------------------------------
 * $Revision: 601 $
 * $Date: 2012-07-10 12:56:42 -0700 (Tue, 10 Jul 2012) $
 * -----------------------------------------------------------------
 * Programmer(s): Aaron Collier @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2004, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * This is the public header file for the KINSOL scaled preconditioned
 * Bi-CGSTAB linear solver module, KINSPBCG.
 * -----------------------------------------------------------------
 */

#ifndef _KINSPBCG_H
#define _KINSPBCG_H

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

#include <kinsol/kinsol_spils.h>
#include <sundials/sundials_spbcgs.h>

/*
 * -----------------------------------------------------------------
 * Function : KINSpbcg
 * -----------------------------------------------------------------
 * KINSpbcg links the main KINSOL solver module with the SPBCG
 * linear solver module. The routine establishes the inter-module
 * interface by setting the generic KINSOL pointers linit, lsetup,
 * lsolve, and lfree to KINSpbcgInit, KINSpbcgSetup, KINSpbcgSolve,
 * and KINSpbcgFree, respectively.
 *
 *  kinmem  pointer to an internal memory block allocated during a
 *          prior call to KINCreate
 *
 *  maxl  maximum allowable dimension of Krylov subspace (passing
 *        a value of 0 (zero) will cause the default value
 *        KINSPILS_MAXL (predefined constant) to be used)
 *
 * If successful, KINSpbcg returns KINSPILS_SUCCESS. If an error
 * occurs, then KINSpbcg returns an error code (negative integer
 * value).
 *
 * -----------------------------------------------------------------
 * KINSpbcg Return Values
 * -----------------------------------------------------------------
 * The possible return values for the KINSpbcg subroutine are the
 * following:
 *
 * KINSPILS_SUCCESS : means the KINSPBCG linear solver module
 *                    (implementation of the Bi-CGSTAB method) was
 *                    successfully initialized - allocated system
 *                    memory and set shared variables to default
 *                    values [0]
 *
 * KINSPILS_MEM_NULL : means a NULL KINSOL memory block pointer
 *                     was given (must call the KINCreate and
 *                     KINMalloc memory allocation subroutines
 *                     prior to calling KINSpbcg) [-1]
 *
 * KINSPILS_MEM_FAIL : means either insufficient system resources
 *                     were available to allocate memory for the
 *                     main KINSPBCG data structure (type
 *                     KINSpbcgMemRec), or the SpbcgMalloc subroutine
 *                     failed (unable to allocate enough system
 *                     memory for vector storate and/or the main
 *                     SPBCG data structure (type SpbcgMemRec)) [-4]
 *
 * KINSPILS_ILL_INPUT : means either a supplied parameter was invalid,
 *                      or the NVECTOR implementation is NOT
 *                      compatible [-3]
 *
 * The above constants are defined in kinsol_spils.h
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT int KINSpbcg(void *kinmem, int maxl);


#ifdef __cplusplus
}
#endif

#endif
