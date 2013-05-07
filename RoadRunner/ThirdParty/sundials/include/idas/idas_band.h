/*
 * -----------------------------------------------------------------
 * $Revision: 601 $
 * $Date: 2012-07-10 12:56:42 -0700 (Tue, 10 Jul 2012) $
 * ----------------------------------------------------------------- 
 * Programmer(s): Alan C. Hindmarsh, and Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * This is the header file for the IDAS band linear solver
 * module, IDABAND. It interfaces between the band module and the
 * integrator when a banded linear solver is appropriate.
 *
 * Part I contains type definitions and function prototypes for using
 * IDABAND on forward problems (DAE integration and/or FSA)
 *
 * Part II contains type definitions and function prototypes for using
 * IDABAND on adjoint (backward) problems
 * -----------------------------------------------------------------
 */

#ifndef _IDASBAND_H
#define _IDASBAND_H

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

#include <idas/idas_direct.h>
#include <sundials/sundials_band.h>

/*
 * -----------------------------------------------------------------
 * Function : IDABand
 * -----------------------------------------------------------------
 * A call to the IDABand function links the main integrator       
 * with the IDABAND linear solver module.                         
 *                                                                
 * ida_mem is the pointer to the integrator memory returned by    
 *         IDACreate.                                                   
 *                                                                
 * mupper is the upper bandwidth of the banded Jacobian matrix.   
 *                                                                
 * mlower is the lower bandwidth of the banded Jacobian matrix.   
 *                                                                
 * The return values of IDABand are:                              
 *     IDADLS_SUCCESS   = 0  if successful                            
 *     IDADLS_LMEM_FAIL = -1 if there was a memory allocation failure 
 *     IDADLS_ILL_INPUT = -2 if the input was illegal or NVECTOR bad. 
 *                                                                
 * NOTE: The band linear solver assumes a serial implementation   
 *       of the NVECTOR package. Therefore, IDABand will first
 *       test for a compatible N_Vector internal representation
 *       by checking that the N_VGetArrayPointer function exists.
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT int IDABand(void *ida_mem,
                            long int Neq, long int mupper, long int mlower);

/*
 * -----------------------------------------------------------------
 * Function: IDABandB
 * -----------------------------------------------------------------
 * IDABandB links the main IDAS integrator with the IDABAND
 * linear solver for the backward integration.
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT int IDABandB(void *idaadj_mem, int which, 
                             long int NeqB, long int mupperB, long int mlowerB);

#ifdef __cplusplus
}
#endif

#endif
