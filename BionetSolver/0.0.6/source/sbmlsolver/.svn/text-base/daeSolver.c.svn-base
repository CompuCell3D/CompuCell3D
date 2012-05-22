/*
  Last changed Time-stamp: <2007-09-19 14:59:33 raim>
  $Id: daeSolver.c,v 1.14 2007/09/20 01:16:12 raimc Exp $
*/
/* 
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * authors have no obligations to provide maintenance, support,
 * updates, enhancements or modifications.  In no event shall the
 * authors be liable to any party for direct, indirect, special,
 * incidental or consequential damages, including lost profits, arising
 * out of the use of this software and its documentation, even if the
 * authors have been advised of the possibility of such damage.  See
 * the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Rainer Machne
 *
 * Contributor(s):
 *
 */
/*! \defgroup ida IDA DAE Solver: x(t)
    \ingroup integrator

    \brief NOT COMPILED CODE: This module contains the functions that
    will call SUNDIALS IDA solver routines for DAE systems, once
    implemented.

    This code is not working yet. It is not compiled with the current
    package. It is included in the documentation merely to motivate
    people to help us implement this functionality. The main problem
    is that ODE construction currently can't decide which variables
    are to be defined via algebraic constraints. Contact us, if you
    want to help!
*/
/*@{*/

#include <stdio.h>
#include <stdlib.h>

/* Header Files for CVODE */
#include "ida/ida.h"    
#include "ida/ida_dense.h"
#include "cvodes/cvodes_dense.h"
#include "nvector/nvector_serial.h"  

#include "sbmlsolver/cvodeData.h"
#include "sbmlsolver/processAST.h"
#include "sbmlsolver/odeModel.h"
#include "sbmlsolver/variableIndex.h"
#include "sbmlsolver/solverError.h"
#include "sbmlsolver/integratorInstance.h"
#include "sbmlsolver/cvodeSolver.h"
#include "sbmlsolver/daeSolver.h"

/* Prototypes of functions called by IDA */

static int fRes(realtype tres, N_Vector yy, N_Vector yp,
		N_Vector resval, void *rdata);

static int JacRes(long int Neq, realtype tt, N_Vector yy, N_Vector yp,
		  N_Vector resvec, realtype cj, void *jdata, DenseMat JJ,
		  N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);


static void
IntegratorInstance_freeIDASpecSolverStructures(integratorInstance_t *);

/* The Hot Stuff! */
/** 
*/

SBML_ODESOLVER_API int IntegratorInstance_idaOneStep(integratorInstance_t *engine)
{
    int i, flag;
    realtype *ydata = NULL;
    
    cvodeSolver_t *solver = engine->solver;
    cvodeData_t *data = engine->data;
/*     cvodeSettings_t *opt = engine->opt; */
/*     cvodeResults_t *results = engine->results; */
    odeModel_t *om = engine->om;
    
    /* !!!! calling CVODE !!!! */
    flag = -1; /* IDASolver(solver->cvode_mem, solver->tout, &(solver->t),
		  solver->y, solver->dy, IDA_NORMAL); */

    if ( flag != IDA_SUCCESS )
      {
	char *message[] =
	  {
	    /*  0 IDA_SUCCESS */
	    "Success",
	    /**/
	    /*  1 IDA_ROOT_RETURN */
	    /*   "CVode succeeded, and found one or more roots" */
	    /*  2 IDA_TSTOP_RETURN */
	    /*   "CVode succeeded and returned at tstop" */
	    /**/
	    /* -1 IDA_MEM_NULL -1 (old CVODE_NO_MEM) */
	    "The cvode_mem argument was NULL",
	    /* -2 IDA_ILL_INPUT */
	    "One of the inputs to CVode is illegal. This "
	    "includes the situation when a component of the "
	    "error weight vectors becomes < 0 during "
	    "internal time-stepping. The ILL_INPUT flag "
	    "will also be returned if the linear solver "
	    "routine CV--- (called by the user after "
	    "calling CVodeMalloc) failed to set one of the "
	    "linear solver-related fields in cvode_mem or "
	    "if the linear solver's init routine failed. In "
	    "any case, the user should see the printed "
	    "error message for more details.",
	    /* -3 IDA_NO_MALLOC */
	    "cvode_mem was not allocated",
	    /* -4 IDA_TOO_MUCH_WORK */
	    "The solver took %g internal steps but could not "
	    "compute variable values for time %g",
	    /* -5 IDA_TOO_MUCH_ACC */
	    "The solver could not satisfy the accuracy " 
	    "requested for some internal step.",
	    /* -6 IDA_ERR_FAILURE */
	    "Error test failures occurred too many times "
	    "during one internal time step or "
	    "occurred with |h| = hmin.",
	    /* -7 IDA_CONV_FAILURE */
	    "Convergence test failures occurred too many "
	    "times during one internal time step or occurred "
	    "with |h| = hmin.",
	    /* -8 IDA_LINIT_FAIL */
	    "CVode -- Initial Setup: "
	    "The linear solver's init routine failed.",
	    /* -9 IDA_LSETUP_FAIL */
	    "The linear solver's setup routine failed in an "
	    "unrecoverable manner.",
	    /* -10 IDA_LSOLVE_FAIL */
	    "The linear solver's solve routine failed in an "
	    "unrecoverable manner.",
	    /* -11 IDA_MEM_FAIL */
	    "A memory allocation failed. "
	    "(including an attempt to increase maxord)",
	    /* -12 IDA_RTFUNC_NULL */
	    "nrtfn > 0 but g = NULL.",
	    /* -13 IDA_NO_SLDET */
	    "CVodeGetNumStabLimOrderReds -- Illegal attempt "
	    "to call without enabling SLDET.",
	    /* -14 IDA_BAD_K */
	    "CVodeGetDky -- Illegal value for k.",
	    /* -15 IDA_BAD_T */
	    "CVodeGetDky -- Illegal value for t.",
	    /* -16 IDA_BAD_DKY */
	    "CVodeGetDky -- dky = NULL illegal.",
	    /* -17 IDA_PDATA_NULL */
	    "???",
	  };
	    
	SolverError_error(
			  ERROR_ERROR_TYPE,
			  flag,
			  message[flag * -1],
			  solver->tout);
	SolverError_error(
			  WARNING_ERROR_TYPE,
			  SOLVER_ERROR_INTEGRATION_NOT_SUCCESSFUL,
			  "Integration not successful. Results are not complete.");

	return 0 ; /* Error - stop integration*/
      }
    
    ydata = NV_DATA_S(solver->y);

    
    /* update cvodeData time dependent variables */    
    for ( i=0; i<om->neq; i++ )
      data->value[i] = ydata[i];

    /* update rest of data with internal default function */
    return IntegratorInstance_updateData(engine);

}


/************* CVODES integrator setup functions ************/


/* creates CVODES structures and fills cvodeSolver 
   return 1 => success
   return 0 => failure
*/
int
IntegratorInstance_createIdaSolverStructures(integratorInstance_t *engine)
{
  int i, flag, neq, nalg;
  realtype *ydata, *abstoldata, *dydata;
  
  odeModel_t *om = engine->om;
  cvodeData_t *data = engine->data;
  cvodeSolver_t *solver = engine->solver;
  cvodeSettings_t *opt = engine->opt;
  
  neq = engine->om->neq;   /* number of ODEs */
  nalg = engine->om->nalg; /* number of algebraic constraints */
  
  /* construct jacobian, if wanted and not yet existing */
  if ( opt->UseJacobian && om->jacob == NULL ) 
    /* reset UseJacobian option, depending on success */
    engine->UseJacobian = ODEModel_constructJacobian(om);
  else if ( !opt->UseJacobian )
  {
    /* free jacobian from former runs (not necessary, frees also
       unsuccessful jacobians from former runs ) */
    ODEModel_freeJacobian(om);
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_MODEL_NOT_SIMPLIFIED,
		      "Jacobian matrix construction skipped.");
    engine->UseJacobian = om->jacobian;
  }
  /* construct algebraic `Jacobian' (or do that in constructJacobian */
  
  /* CVODESolverStructures from former runs must be freed */
  if ( engine->run > 1 )
    IntegratorInstance_freeIDASolverStructures(engine);
  
  
    /*
     * Allocate y, abstol vectors
     */
  solver->y = N_VNew_Serial(neq + nalg);
  CVODE_HANDLE_ERROR((void *)solver->y, "N_VNew_Serial for vector y", 0);
  
  solver->dy = N_VNew_Serial(neq + nalg);
  CVODE_HANDLE_ERROR((void *)solver->dy, "N_VNew_Serial for vector dy", 0);
    
  solver->abstol = N_VNew_Serial(neq + nalg);
  CVODE_HANDLE_ERROR((void *)solver->abstol,
		     "N_VNew_Serial for vector abstol", 0);
  
  /*
   * Initialize y, abstol vectors
   */
  ydata      = NV_DATA_S(solver->y);
  abstoldata = NV_DATA_S(solver->abstol);
  dydata     = NV_DATA_S(solver->dy);
  
  for ( i=0; i<neq; i++ )
  {
    /* Set initial value vector components of y and y' */
    ydata[i] = data->value[i];
    /* Set absolute tolerance vector components,
       currently the same absolute error is used for all y */ 
    abstoldata[i] = opt->Error;
    dydata[i] = evaluateAST(om->ode[i], data);
  }
  /* set initial value vector components for algebraic rule variables  */
    
  /* scalar relative tolerance: the same for all y */
  solver->reltol = opt->RError;

  /*
   * Call IDACreate to create the solver memory:
   *
   */
  solver->cvode_mem = IDACreate();
  CVODE_HANDLE_ERROR((void *)(solver->cvode_mem), "IDACreate", 0);

  /*
   * Call IDAMalloc to initialize the integrator memory:
   *
   * cvode_mem  pointer to the CVode memory block returned by CVodeCreate
   * fRes         user's right hand side function
   * t0         initial value of time
   * y          the dependent variable vector
   * dy         the ODE value vector
   * IDA_SV     specifies scalar relative and vector absolute tolerances
   * reltol     the scalar relative tolerance
   * abstol     pointer to the absolute tolerance vector
   */
  flag = IDAMalloc(solver->cvode_mem, fRes, solver->t0, solver->y,
		   solver->dy, IDA_SV, solver->reltol, solver->abstol);
  CVODE_HANDLE_ERROR(&flag, "IDAMalloc", 1);

  /* 
   * Link the main integrator with data for right-hand side function
   */ 
  flag = IDASetRdata(solver->cvode_mem, engine->data);
  CVODE_HANDLE_ERROR(&flag, "IDASetRdata", 1);
    
  /*
   * Link the main integrator with the IDADENSE linear solver
   */
  flag = IDADense(solver->cvode_mem, neq);
  CVODE_HANDLE_ERROR(&flag, "IDADense", 1);


  /*
   * Set the routine used by the IDADense linear solver
   * to approximate the Jacobian matrix to ...
   */
  if ( opt->UseJacobian == 1 ) 
    /* ... user-supplied routine JacRes : put JacRes instead of NULL
       when working */
    flag = IDADenseSetJacFn(solver->cvode_mem, NULL, data);
  else 
    /* ... the internal default difference quotient routine IDADenseDQJac */
    flag = IDADenseSetJacFn(solver->cvode_mem, NULL, NULL);    
    
  CVODE_HANDLE_ERROR(&flag, "IDADenseSetJacFn", 1);
     
  return 1; /* OK */
}

/* frees N_V vector structures, and the cvode_mem solver */
void IntegratorInstance_freeIDASolverStructures(integratorInstance_t *engine)
{
  /* Free CVODE structures: the same for both */ 
  IntegratorInstance_freeCVODESolverStructures(engine);
}

/* frees only sensitivity structure, not used at the moment  */
static void
IntegratorInstance_freeIDASpecSolverStructures(integratorInstance_t *engine)
{
  /* Free sensitivity vector yS */
  N_VDestroy_Serial(engine->solver->dy);
  engine->solver->dy = NULL;
}

/** \brief Prints some final statistics of the calls to CVODE routines
 */

SBML_ODESOLVER_API void IntegratorInstance_printIDAStatistics(integratorInstance_t *engine, FILE *f)
{
  /* print IDA statistics */
  IntegratorInstance_printCVODEStatistics(engine, f);
  /* print additional IDA statistics ...*/
}

/***************** Functions Called by the CVODE Solver ******************/

/**
   fRes routine. Compute the system residual function.
   This function is called by IDA's integration routines every time
   needed. 
*/

static int
fRes(realtype t, N_Vector y, N_Vector dy, N_Vector r, void *f_data)
{
  
  int i;
  realtype *ydata, *dydata, *resdata;
  cvodeData_t *data;
  data   = (cvodeData_t *) f_data;
  ydata  = NV_DATA_S(y);
  dydata = NV_DATA_S(dy);
  resdata  = NV_DATA_S(r);
  
  /* update ODE variables from CVODE */
  for ( i=0; i<data->model->neq; i++ ) 
    data->value[i] = ydata[i];

  /* update algebraic constraint defined variables */

  
  /* update assignment rules */
  for ( i=0; i<data->model->nass; i++ ) 
    data->value[data->model->neq+i] =
      evaluateAST(data->model->assignment[i],data);

  /* update time  */
  data->currenttime = t;

  /* evaluate residual functions:
     for available ODEs: 0 = dY/dt - dY/dt
     for algebraicRules: 0 = algebraic rule */
  for ( i=0; i<data->model->neq; i++ ) 
    resdata[i] = evaluateAST(data->model->ode[i],data) - dydata[i];

  for ( i=0 ; i<data->model->nalg; i++ ) 
    resdata[i] = evaluateAST(data->model->algebraic[i],data);

  return 0;
}


/*
  Jacobian residual routine. Compute J(t,y).
  This function is (optionally) called by IDA's integration routines
  every time needed.
  Very similar to the f routine, it evaluates the Jacobian matrix
  equations with IDA's current values and writes the results
  back to IDA's internal vector DENSE_ELEM(J,i,j).
*/

static int
JacRes(long int N, realtype t, N_Vector y, N_Vector dy,
       N_Vector resvec, realtype cj, void *jac_data, DenseMat J,
       N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
{
  
  int i, j;
  realtype *ydata;
  cvodeData_t *data;
  data  = (cvodeData_t *) jac_data;
  ydata = NV_DATA_S(y);

  /* update ODE variables from CVODE */
  for ( i=0; i<data->model->neq; i++ ) {
    data->value[i] = ydata[i];
  }
  /* update algebraic constraint defined variables */

  /* update assignment rules */
  for ( i=0; i<data->model->nass; i++ ) {
    data->value[data->model->neq+i] =
      evaluateAST(data->model->assignment[i],data);
  }
  /* update time */
  data->currenttime = t;

  /* evaluate Jacobian*/
  for ( i=0; i<data->model->neq; i++ ) {
    for ( j=0; j<data->model->neq; j++ ) {
      DENSE_ELEM(J,i,j) = evaluateAST(data->model->jacob[i][j], data);
      if ( i == j )
	DENSE_ELEM(J, i, j) -= cj;
    }
  }
  
  for ( i=0; i<data->model->nalg; i++ ) 
    for ( j=0; j<data->model->nalg; j++ ) 
      DENSE_ELEM(J,i,j) = 1.; /* algebraic jacobian here!! */

  return (0);
}


/** @} */
/* End of file */
