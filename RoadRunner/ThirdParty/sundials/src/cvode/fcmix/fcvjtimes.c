/*
 * -----------------------------------------------------------------
 * $Revision: 1.3 $
 * $Date: 2007/04/30 19:28:59 $
 * ----------------------------------------------------------------- 
 * Programmer(s): Alan C. Hindmarsh, Radu Serban and
 *                Aaron Collier @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * -----------------------------------------------------------------
 * The C function FCVJtimes is to interface between the
 * CVSP* module and the user-supplied Jacobian-vector
 * product routine FCVJTIMES. Note the use of the generic name
 * FCV_JTIMES below.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>

#include "fcvode.h"     /* actual fn. names, prototypes and global vars.*/
#include "cvode_impl.h" /* definition of CVodeMem type                  */

#include <cvode/cvode_spils.h>

/***************************************************************************/

/* Prototype of the Fortran routine */

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

  extern void FCV_JTIMES(realtype*, realtype*,            /* V, JV      */
                         realtype*, realtype*, realtype*, /* T, Y, FY   */
                         realtype*,                       /* H          */
                         long int*, realtype*,            /* IPAR, RPAR */
                         realtype*,                       /* WRK        */
                         int*);                           /* IER        */

#ifdef __cplusplus
}
#endif

/***************************************************************************/

void FCV_SPILSSETJAC(int *flag, int *ier)
{
  CVodeMem cv_mem;

  if (*flag == 0) {
    *ier = CVSpilsSetJacTimesVecFn(CV_cvodemem, NULL);
  } else {
    cv_mem = (CVodeMem) CV_cvodemem;
    *ier = CVSpilsSetJacTimesVecFn(CV_cvodemem, FCVJtimes);
  }
}

/***************************************************************************/

/* C function  FCVJtimes to interface between CVODE and  user-supplied
   Fortran routine FCVJTIMES for Jacobian * vector product.
   Addresses of v, Jv, t, y, fy, h, and work are passed to FCVJTIMES,
   using the routine N_VGetArrayPointer from NVECTOR.
   A return flag ier from FCVJTIMES is returned by FCVJtimes.
   Auxiliary data is assumed to be communicated by common blocks. */

int FCVJtimes(N_Vector v, N_Vector Jv, realtype t, 
              N_Vector y, N_Vector fy,
              void *user_data, N_Vector work)
{
  realtype *vdata, *Jvdata, *ydata, *fydata, *wkdata;
  realtype h;
  FCVUserData CV_userdata;

  int ier = 0;
  
  CVodeGetLastStep(CV_cvodemem, &h);

  vdata   = N_VGetArrayPointer(v);
  Jvdata  = N_VGetArrayPointer(Jv);
  ydata   = N_VGetArrayPointer(y);
  fydata  = N_VGetArrayPointer(fy);
  wkdata  = N_VGetArrayPointer(work);

  CV_userdata = (FCVUserData) user_data;
 
  FCV_JTIMES (vdata, Jvdata, &t, ydata, fydata, &h, 
              CV_userdata->ipar, CV_userdata->rpar, wkdata, &ier);

  return(ier);
}
