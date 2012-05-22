/*
  Last changed Time-stamp: <2009-02-12 15:24:24 raim>
  $Id: sensSolver.c,v 1.80 2009/02/12 06:24:04 raimc Exp $
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
 *     Rainer Machne, James Lu and Stefan Müller
 *
 * Contributor(s):
 *
 */

/*! \defgroup sensi CVODES Forward Sensitivity:  dx(t)/dp
  \ingroup cvode
  \brief This module contains the functions that set up and
  call SUNDIALS CVODES forward sensitivity analysis routines.
    

*/
/*@{*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Header Files for CVODE */
#include "cvodes/cvodes.h"    
#include "cvodes/cvodes_dense.h"
#include "nvector/nvector_serial.h"  

#include "sbmlsolver/cvodeData.h"
#include "sbmlsolver/processAST.h"
#include "sbmlsolver/odeModel.h"
#include "sbmlsolver/variableIndex.h"
#include "sbmlsolver/solverError.h"
#include "sbmlsolver/integratorInstance.h"
#include "sbmlsolver/cvodeSolver.h"
#include "sbmlsolver/sensSolver.h"
#include "sbmlsolver/odeSolver.h"
#include "sbmlsolver/interpol.h"
#include "sbmlsolver/variableIndex.h"

#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>
#include "sbmlsolver/ASTIndexNameNode.h"

/* 
 * fS routine. Compute sensitivity r.h.s. for param[iS]
 */
static int fS(int Ns, realtype t, N_Vector y, N_Vector ydot,
	      int iS, N_Vector yS, N_Vector ySdot,
	      void *fS_data, N_Vector tmp1, N_Vector tmp2);


static int fA(realtype t, N_Vector y, N_Vector yA, N_Vector yAdot,
	      void *fA_data);


static int JacA(long int NB, DenseMat JA, realtype t,
		N_Vector y, N_Vector yA, N_Vector fyA, void *jac_dataA,
		N_Vector tmp1A, N_Vector tmp2A, N_Vector tmp3A);

static int fQA(realtype t, N_Vector y, N_Vector yA, N_Vector qAdot,
	       void *fQ_dataA);

static int fQS(realtype t, N_Vector y, N_Vector qdot, void *fQ_data);

static int fQFIM(realtype t, N_Vector y, N_Vector qdot, void *fQ_data);

static int ODEModel_construct_vector_v_FromObjectiveFunction(odeModel_t *);

static ASTNode_t *copyRevertDataAST(const ASTNode_t *);


/* The Hot Stuff! */
/** \brief Calls CVODES to provide forward sensitivities after a call to
    cvodeOneStep.

    produces appropriate error messages on failures and returns 1 if
    the integration can continue, 0 otherwise.  
*/

int IntegratorInstance_getForwardSens(integratorInstance_t *engine)
{
  int i, j, flag;
  realtype *ySdata = NULL;
   
  cvodeSolver_t *solver;
  cvodeData_t *data;
  cvodeSettings_t *opt;
  cvodeResults_t *results;
    
  solver = engine->solver;
  data = engine->data;
  opt = engine->opt;
  results = engine->results;

  /* getting sensitivities */
  flag = CVodeGetSens(solver->cvode_mem, solver->t, solver->yS);
    
  if ( flag != CV_SUCCESS )
    return flag;
  else
  {
    for ( j=0; j<data->nsens; j++ )
    {
      ySdata = NV_DATA_S(solver->yS[j]);
      for ( i=0; i<data->neq; i++ )
      {
	data->sensitivity[i][j] = ySdata[i];
/* 	/\* store results *\/ */
/* 	if ( opt->StoreResults ) */
/* 	  results->sensitivity[i][j][solver->iout-1] = ySdata[i]; */
      }
    }
  }

  return flag;
}

/* The Hot Stuff! */
/** \brief Calls CVODES to provide adjoint sensitivities after a call to
    cvodeOneStep.

    NOTE: does not do interpolation based on CVodeGetSens. 

    produces appropriate error messages on failures and returns 1 if
    the integration can continue, 0 otherwise.  
*/

int IntegratorInstance_getAdjSens(integratorInstance_t *engine)
{
  int i;
  realtype *yAdata = NULL;
   
  cvodeSolver_t *solver;
  cvodeData_t *data;
  cvodeSettings_t *opt;
  cvodeResults_t *results;
    
  solver = engine->solver;
  data = engine->data;
  opt = engine->opt;
  results = engine->results;
  
  yAdata = NV_DATA_S(solver->yA);
  
  for ( i=0; i<data->neq; i++ )
  {
    data->adjvalue[i] = yAdata[i];

    /* store results */
    if ( opt->AdjStoreResults )
      results->adjvalue[i][solver->iout-1] = yAdata[i];
  }
    
  return 1;
}




/************* CVODES integrator setup functions ************/


/* creates CVODES forward sensitivity solver structures
   return 1 => success
   return 0 => failure
*/
int
IntegratorInstance_createCVODESSolverStructures(integratorInstance_t *engine)
{
  int i, j, reinit, flag, sensMethod;
 /*  realtype *abstoldata, *ySdata; */
  int found;

  odeModel_t *om = engine->om;
  odeSense_t *os = engine->os;
  cvodeData_t *data = engine->data;
  cvodeSolver_t *solver = engine->solver;
  cvodeSettings_t *opt = engine->opt;
  CVSensRhs1Fn sensRhsFunction = NULL;
  CVDenseJacFnB adjointJACFunction = NULL;
  CVQuadRhsFnB adjointQuadFunction = NULL;
  CVRhsFnB adjointRHSFunction = NULL;
  
  /* adjoint specific*/
  int method, iteration;

  if( !engine->AdjointPhase )
  {
    /*****  adding sensitivity specific structures ******/

    /* set rhs function for sensitivity */
    if ( om->jacobian && os->sensitivity )
    {
      if ( opt->compileFunctions )
      {
	/* this is currently one of two ways to compilation
	   of odeSense_t RHS functions, the other is below for
	   adjoint functions ! */
	sensRhsFunction = ODESense_getCompiledCVODESenseFunction(os);
	if ( !sensRhsFunction ) return 0;  /*!!! use CVODE_HANDLE_ERROR */
      }
      else
	sensRhsFunction = fS ;
    }
    
    /* if the sens. problem dimension has changed since
        a former run, free all sensitivity structures */
    if ( engine->solver->nsens != os->nsens )
    {
      /* free forward sens structures, including CVodeQuad */
      IntegratorInstance_freeForwardSensitivity(engine);
      engine->solver->nsens = os->nsens;
      /* remember this for quadrature initialization */
    }

    /*
     * construct sensitivity yS and  absolute tolerance senstol
     * structures if they are not available from a previous run
     * with the same sens. problem dimension nsens
     */
    
    if ( solver->senstol == NULL )
    {
      solver->senstol = N_VNew_Serial(os->nsens);
      CVODE_HANDLE_ERROR((void *)solver->senstol,
			 "N_VNewSerial for senstol", 0);
    }

    /* remember: if yS had to be reconstructed, then also
       the sense solver structure CVodeSens needs reconstruction
       rather then mere re-initiation */
    reinit = 1;
    if ( solver->yS == NULL )
    {
      N_Vector y;

      y = N_VNew_Serial(data->neq);
      solver->yS = N_VCloneVectorArray_Serial(os->nsens, y);
      CVODE_HANDLE_ERROR((void *)solver->yS, "N_VCloneVectorArray_Serial", 0);
      reinit = 0;
      N_VDestroy_Serial(y);
    }

    /* initialize sensitivity and sensitiity tolerance
       arrays, yS and senstol, resp.:
       yS are 0.0 or 1.0 (variables) in a new run or
       old values in case of events!! */    
    for ( j=0; j<os->nsens; j++ )
    {
      NV_Ith_S(solver->senstol,j) =  NV_Ith_S(solver->abstol,0);
      for ( i=0; i<data->neq; i++ )
       NV_Ith_S(solver->yS[j], i) = data->sensitivity[i][j]; 
    }  

    /*
     * set forward sensitivity method
     */
    sensMethod = 0;
    if ( opt->SensMethod == 0 ) sensMethod = CV_SIMULTANEOUS;
    else if ( opt->SensMethod == 1 ) sensMethod = CV_STAGGERED;
    else if ( opt->SensMethod == 2 ) sensMethod = CV_STAGGERED1;

    if ( reinit == 0 )
    {
      flag = CVodeSensMalloc(solver->cvode_mem, os->nsens,
			     sensMethod, solver->yS);
      CVODE_HANDLE_ERROR(&flag, "CVodeSensMalloc", 1);    
    }
    else
    {
      flag = CVodeSensReInit(solver->cvode_mem, sensMethod, solver->yS);
      CVODE_HANDLE_ERROR(&flag, "CVodeSensReInit", 1);	  
    }

    /* *** set parameter values or R.H.S function fS *****/
    /* NOTES: */
    /* !!! plist could later be used to specify requested parameters
       for sens.analysis !!! */
    
    /* was construction of jacobian and parametric matrix successfull ? */
    /* if only init.cond. sensitivities, only the jacobian
      matrix required, and os->sensitivity will also be 1 even though
      there is no matrix */
    if ( os->sensitivity && om->jacobian )
    {
      flag = CVodeSetSensRhs1Fn(solver->cvode_mem, sensRhsFunction, data);
      CVODE_HANDLE_ERROR(&flag, "CVodeSetSensRhs1Fn Matrix", 1);
      data->use_p = 0; /* don't use data->p */
    }
    else
    {
      flag = CVodeSetSensRhs1Fn(solver->cvode_mem, NULL, NULL);
      CVODE_HANDLE_ERROR(&flag, "CVodeSetSensRhs1Fn NULL", 1);
      data->use_p = 1; /* use data->p, as fS is not available */

      /* difference quotient method: CV_FORWARD or CV_CENTERED
       see, cvs_guide.pdf CVODES doc */
      CVodeSetSensDQMethod(solver->cvode_mem, CV_CENTERED, 0.0);
      CVODE_HANDLE_ERROR(&flag, "CVodeSetSensDQMethod", 1);
    }

    /* initializing and setting data->p, used if the jacobian or
     the parametric matrix are not available */
    for ( i=0; i<os->nsens; i++ )
      data->p[i] = data->p_orig[i] = data->value[os->index_sens[i]];

    flag = CVodeSetSensParams(solver->cvode_mem, data->p, NULL, NULL);
    CVODE_HANDLE_ERROR(&flag, "CVodeSetSensParams", 1);
    
    /* set tolerances for sensitivity equations */
   /*  CVodeSetSensTolerances(solver->cvode_mem, CV_SS, */
/*     			   NV_Ith_S(solver->senstol,0), &(opt->AdjError)); */
    
    /* difference FALSE/TRUE ? */
    flag = CVodeSetSensErrCon(solver->cvode_mem, FALSE);
    CVODE_HANDLE_ERROR(&flag, "CVodeSetSensErrCon", 1);
    

    /*  If linear functional exists, initialize quadrature computation  */
    if ( !opt->doFIM )
    {
      if ( (om->ObjectiveFunction == NULL) && (om->vector_v != NULL) )
      {
	if ( solver->qS == NULL )
	{
	  /* initialize new qS quadrature */
	  solver->qS = N_VNew_Serial(os->nsens);
	  CVODE_HANDLE_ERROR((void *) solver->qS,
			     "N_VNew_Serial for vector qS", 0);
	  for ( i=0; i<os->nsens; i++ )
	    NV_Ith_S(solver->qS, i) = 0.0;
	  
	  /* if q exists, nsens has size 1 and quad can be reused */
	  if ( solver->q )
	  {
	    N_VDestroy_Serial(engine->solver->q);
	    engine->solver->q = NULL;
	    
	    flag = CVodeQuadReInit(solver->cvode_mem, fQS, solver->qS);
	    CVODE_HANDLE_ERROR(&flag, "CVodeQuadReInit fQS", 1);
	  }
	  else
	  { /* NO QUAD EXIST, CALL MALLOC*/
	    flag = CVodeQuadMalloc(solver->cvode_mem, fQS, solver->qS);
	    CVODE_HANDLE_ERROR(&flag, "CVodeQuadMalloc for qS", 1);
	  }
	}
	/* if qS still exists then the dimension hasn't changed */
	else
	{
	  /* just use existing quadrature */
	  for ( i=0; i<os->nsens; i++ )
	    NV_Ith_S(solver->qS, i) = 0.0;
	  flag = CVodeQuadReInit(solver->cvode_mem, fQS, solver->qS);
	  CVODE_HANDLE_ERROR(&flag, "CVodeQuadReInit fQS", 1);
	}
	
	
	flag = CVodeSetQuadFdata(solver->cvode_mem, engine);
	CVODE_HANDLE_ERROR(&flag, "CVodeSetQuadFdata", 1);
	
	/* set quadrature tolerance for objective function 
	   to be the same as the forward solution tolerances */
	/*!!! error init causes strange problems */
	/*       flag = CVodeSetQuadErrCon(solver->cvode_mem, TRUE, */
	/* 				CV_SS, solver->reltol, &(opt->Error)); */
	/*       CVODE_HANDLE_ERROR(&flag, "CVodeSetQuadErrCon", 1); */
      }
    }
    else /* do FIM */
    {
      if ( solver->qFIM == NULL )
      {
	/* initialize new qFIM quadrature */
	solver->qFIM = N_VNew_Serial(os->nsens*os->nsens);
	CVODE_HANDLE_ERROR((void *) solver->qFIM,
			   "N_VNew_Serial for vector qFIM", 0);
	for ( i=0; i<os->nsens*os->nsens; i++ )
	  NV_Ith_S(solver->qFIM, i) = 0.0;
	
	/* if q exists, nsens has size 1 and quad can be reused */
	if ( solver->q )
	{
	  N_VDestroy_Serial(engine->solver->q);
	  engine->solver->q = NULL;
	  
	  flag = CVodeQuadReInit(solver->cvode_mem, fQFIM, solver->qFIM);
	  CVODE_HANDLE_ERROR(&flag, "CVodeQuadReInit fQFIM", 1);
	}
	else
	{ /* NO QUAD EXIST, CALL MALLOC*/
	  flag = CVodeQuadMalloc(solver->cvode_mem, fQFIM, solver->qFIM);
	  CVODE_HANDLE_ERROR(&flag, "CVodeQuadMalloc for qFIM", 1);
	}
      }
      /* if qFIM still exists then the dimension hasn't changed */
      else
      {
	/* just use existing quadrature */
	for ( i=0; i<os->nsens*os->nsens; i++ )
	  NV_Ith_S(solver->qFIM, i) = 0.0;
	flag = CVodeQuadReInit(solver->cvode_mem, fQFIM, solver->qFIM);
	CVODE_HANDLE_ERROR(&flag, "CVodeQuadReInit fQFIM", 1);
      }
      
      
      flag = CVodeSetQuadFdata(solver->cvode_mem, engine);
      CVODE_HANDLE_ERROR(&flag, "CVodeSetQuadFdata for FIM", 1);
      
      /* set quadrature tolerance for objective function 
	 to be the same as the forward solution tolerances */
      /*!!! error init causes strange problems */
      /*       flag = CVodeSetQuadErrCon(solver->cvode_mem, TRUE, */
      /* 				CV_SS, solver->reltol, &(opt->Error)); */
      /*       CVODE_HANDLE_ERROR(&flag, "CVodeSetQuadErrCon", 1); */
      
    } /* end FIM */
    
  } 
  else /* Adjoint Phase */    
  {
    /* set rhs function for sensitivity */
    if ( opt->compileFunctions )
    {
      adjointJACFunction =
	ODEModel_getCompiledCVODEAdjointJacobianFunction(om);
      if ( !adjointJACFunction ) return 0;/*!!! use CVODE_HANDLE_ERROR  */
      
      /* set adjoint quadrature function for sensitivity */
      if ( os->sensitivity )
      {
	/* this is currently one of two ways to compilation
	   of odeSense_t RHS functions, the other is above for
	   forward sensitivity functions ! */
	adjointQuadFunction = ODESense_getCompiledCVODEAdjointQuadFunction(os);
	if ( !adjointQuadFunction ) return 0;/*!!! use CVODE_HANDLE_ERROR */
      }

      adjointRHSFunction = ODEModel_getCompiledCVODEAdjointRHSFunction(om);
      if ( !adjointRHSFunction ) return 0; /*!!! use CVODE_HANDLE_ERROR */
    }    
    else
    {
      adjointJACFunction = JacA;
      adjointQuadFunction = fQA ;
      adjointRHSFunction = fA;
    }
    
    /* remember which of the adj RHS function is being used */
    om->current_AdjRHS  = adjointRHSFunction;
    om->current_AdjJAC  = adjointJACFunction;
    os->current_AdjQAD = adjointQuadFunction;
 
      /*  If ObjectiveFunction exists, compute vector_v from it */ 
    if ( om->ObjectiveFunction != NULL ) 
    {
      flag = ODEModel_construct_vector_v_FromObjectiveFunction(om);
      /*!!! TODO : add this to solverError messages */
      if (flag != 1){
	fprintf(stderr, "error in constructing vector_v\n");
	return flag;
      }
    }

    /* update initial adjoint state if discrete experimental data is
       observed */
    if ( opt->observation_data_type == 1 )
    {
      /* set current time and state values for evaluating vector_v  */
      data->currenttime = solver->t;
    
      found = 0;
      if ( fabs(engine->results->time[opt->PrintStep] - solver->t) < 1e-5)
      {
	found++;
	for ( j=0; j<om->neq; j++ )
	  data->value[j] = engine->results->value[j][opt->PrintStep];
	
      }

      if ( found != 1 )
      {
	fprintf(stderr, "ERROR in updating the initial adjoint data.\n");
	SolverError_error(FATAL_ERROR_TYPE,
			  SOLVER_ERROR_INITIALIZE_ADJDATA,
			  "Failed to get state value at time %g.",solver->t);
        return 0;
      }

      /* in discrete data case, set the initial adjoint solution to
	 the evaluated value of vector_v */
      om->compute_vector_v=1;
      data->TimeSeriesIndex = data->model->time_series->n_time-1 ;
      for ( i=0; i<om->neq; i++ )
	data->adjvalue[i] = -evaluateAST( data->model->vector_v[i], data);
      om->compute_vector_v=0;

    } 


    /*  Allocate yA, abstolA vectors */
    if ( solver->yA == NULL )
    {
      solver->yA = N_VNew_Serial(om->neq);
      CVODE_HANDLE_ERROR((void *)solver->yA, "N_VNew_Serial for vector yA", 0);
    }

    if ( solver->abstolA == NULL )
    {
      solver->abstolA = N_VNew_Serial(om->neq);
      CVODE_HANDLE_ERROR((void *)solver->abstolA,
			 "N_VNew_Serial for vector abstolA", 0);
    }

    /**
     * Initialize y, abstol vectors
     */
    for ( i=0; i<om->neq; i++ )
    {
      /* Set initial value vector components of yA */
          NV_Ith_S(solver->yA, i) = data->adjvalue[i]; 

      /* Set absolute tolerance vector components,
	 currently the same absolute error is used for all y */ 
         NV_Ith_S(solver->abstolA, i) = opt->AdjError; 
    }
    
    /* scalar relative tolerance: the same for all y */
    solver->reltolA = opt->AdjRError;

    /* Adjoint specific allocations   */
    /**
     * Call CVodeCreateB to create the non-linear solver memory:\n
     *
     Nonlinear Solver:\n
     * CV_BDF         Backward Differentiation Formula method\n
     * CV_ADAMS       Adams-Moulton method\n
     Iteration Method:\n
     * CV_NEWTON      Newton iteration method\n
     * CV_FUNCTIONAL  functional iteration method\n
     */
    if ( opt->CvodeMethod == 1 ) method = CV_ADAMS;
    else method = CV_BDF;
    
    if ( opt->IterMethod == 1 ) iteration = CV_FUNCTIONAL;
    else iteration = CV_NEWTON;

    /* Error if neither ObjectiveFunction nor vector_v has been set  */
    if( (om->ObjectiveFunction == NULL) && (om->vector_v == NULL) )
      return 0;
   
     if ( engine->adjrun == 1 )
    {
      flag = CVodeCreateB(solver->cvadj_mem, method, iteration);
      CVODE_HANDLE_ERROR(&flag, "CVodeCreateB", 1);

      flag = CVodeMallocB(solver->cvadj_mem, adjointRHSFunction, solver->t0,
			  solver->yA, CV_SV, solver->reltolA,
			  solver->abstolA);
      CVODE_HANDLE_ERROR(&flag, "CVodeMallocB", 1);
     
      flag = CVDenseB(solver->cvadj_mem, om->neq);
      CVODE_HANDLE_ERROR(&flag, "CVDenseB", 1);

    }
    else
    {
      flag = CVodeReInitB(solver->cvadj_mem, adjointRHSFunction, solver->t0,
			  solver->yA, CV_SV, solver->reltolA,
			  solver->abstolA);
      CVODE_HANDLE_ERROR(&flag, "CVodeReInitB", 1);
    }
     
    flag = CVodeSetFdataB(solver->cvadj_mem, engine->data);
    CVODE_HANDLE_ERROR(&flag, "CVodeSetFdataB", 1);

    /*!!! could NULL be passed here if jacobian is not available ??*/
    flag = CVDenseSetJacFnB(solver->cvadj_mem, adjointJACFunction,
			    engine->data);
    CVODE_HANDLE_ERROR(&flag, "CVDenseSetJacFnB", 1);

    /* set adjoint max steps to be same as that for forward */
    flag = CVodeSetMaxNumStepsB(solver->cvadj_mem, opt->Mxstep);
    CVODE_HANDLE_ERROR(&flag, "CVodeSetMaxNumStepsB", 1);

   
    if ( solver->qA == NULL )
    {
      solver->qA = N_VNew_Serial(os->nsens);
      CVODE_HANDLE_ERROR((void *) solver->qA,
			 "N_VNew_Serial for vector qA failed", 0);

      /* Init solver->qA = 0.0;*/
      for( i=0; i<os->nsens; i++ )
	NV_Ith_S(solver->qA, i) = 0.0;
  
      flag = CVodeQuadMallocB(solver->cvadj_mem, adjointQuadFunction,
			      solver->qA);
      CVODE_HANDLE_ERROR(&flag, "CVodeQuadMallocB", 1);
    }
    else
    {
      /* Init solver->qA = 0.0;*/
      for( i=0; i<os->nsens; i++ )
	NV_Ith_S(solver->qA, i) = 0.0;
  
      flag = CVodeQuadReInitB(solver->cvadj_mem, adjointQuadFunction,
			      solver->qA);
      CVODE_HANDLE_ERROR(&flag, "CVodeQuadReInitB", 1);
    }
 
    /*  Allocate abstolQA vector */
    if ( solver->abstolQA == NULL )
    {
/*       solver->abstolQA = N_VNew_Serial(engine->om->neq); */
      solver->abstolQA = N_VNew_Serial(engine->os->nsens);
      CVODE_HANDLE_ERROR((void *)solver->abstolQA,
			 "N_VNew_Serial for vector quad abstol failed", 0);
    }
      
   /*  for ( i=0; i<engine->om->neq; i++ ) */
    for ( i=0; i<engine->os->nsens; i++ )
    {
      /* Set absolute tolerance vector components,
	 currently the same absolute error is used for all y */ 
      NV_Ith_S(solver->abstolQA, i) = opt->AdjError;
    } 

    solver->reltolQA = solver->reltolA;
 
    flag = CVodeSetQuadFdataB(solver->cvadj_mem, data);
    CVODE_HANDLE_ERROR(&flag, "CVodeSetQuadFdataB", 1);

    flag = CVodeSetQuadErrConB(solver->cvadj_mem, TRUE,
			       CV_SS, solver->reltolA, &(opt->AdjError) );
    CVODE_HANDLE_ERROR(&flag, "CVodeSetQuadErrConB", 1);


    /* END adjoint phase */
  } 

  return 1; /* OK */

}

/** \brief Prints some final statistics of the calls to CVODES
    forward and adjoint sensitivity analysis routines
*/

SBML_ODESOLVER_API int IntegratorInstance_printCVODESStatistics(integratorInstance_t *engine, FILE *f)
{
  int flag;
  long int nfSe, nfeS, nsetupsS, nniS, ncfnS, netfS;
  long int nstA, nfeA, nsetupsA, njeA, nniA, ncfnA, netfA;
  cvodeSolver_t *solver = engine->solver;
  void *cvode_memB;

  if ( engine->opt->Sensitivity )
  { 
     /* print additional CVODES forward sensitivity statistics */
    fprintf(f, "##\n## CVodes Forward Sensitivity Statistics:\n");

    flag = CVodeGetNumSensRhsEvals(solver->cvode_mem, &nfSe);
    CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSensRhsEvals", 1);
    flag = CVodeGetNumRhsEvalsSens(solver->cvode_mem, &nfeS);
    CVODE_HANDLE_ERROR(&flag, "CVodeGetNumRhsEvalsSens", 1);
    flag = CVodeGetNumSensLinSolvSetups(solver->cvode_mem, &nsetupsS);
    CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSensLinSolvSetups", 1);
    flag = CVodeGetNumSensErrTestFails(solver->cvode_mem, &netfS);
    CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSensErrTestFails", 1);
    flag = CVodeGetNumSensNonlinSolvIters(solver->cvode_mem, &nniS);
    CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSensNonlinSolvIters", 1);
    flag = CVodeGetNumSensNonlinSolvConvFails(solver->cvode_mem, &ncfnS);
    CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSensNonlinSolvConvFails", 1);

    fprintf(f, "## nfSe    = %5ld    nfeS     = %5ld\n", nfSe, nfeS);
    fprintf(f, "## netfs   = %5ld    nsetupsS = %5ld\n", netfS, nsetupsS);
    fprintf(f, "## nniS    = %5ld    ncfnS    = %5ld\n", nniS, ncfnS);
  }
  
  if ( (engine->opt->DoAdjoint) && (solver->cvadj_mem != NULL) )
  { 
      /* print additional CVODES adjoint sensitivity statistics */
     fprintf(f, "##\n## CVode Adjoint Sensitivity Statistics:\n");
     cvode_memB = CVadjGetCVodeBmem(solver->cvadj_mem);
     flag = CVodeGetNumSteps(cvode_memB, &nstA);
     CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSteps", 1);
     flag = CVodeGetNumRhsEvals(cvode_memB, &nfeA);
     CVODE_HANDLE_ERROR(&flag, "CVodeGetNumSensRhsEvals", 1);
     flag = CVodeGetNumLinSolvSetups(cvode_memB, &nsetupsA);
     CVODE_HANDLE_ERROR(&flag, "CVodeGetNumLinSolvSetups", 1);
     flag = CVDenseGetNumJacEvals(cvode_memB, &njeA);
     CVODE_HANDLE_ERROR(&flag, "CVDenseGetNumJacEvals", 1);
     flag = CVodeGetNonlinSolvStats(cvode_memB, &nniA, &ncfnA);
     CVODE_HANDLE_ERROR(&flag, "CVodeGetNonlinSolvStats", 1);
     flag = CVodeGetNumErrTestFails(cvode_memB, &netfA);
     CVODE_HANDLE_ERROR(&flag, "CVodeGetNumErrTestFails", 1);
     fprintf(f, "## nstA = %-6ld nfeA  = %-6ld nsetupsA = %-6ld njeA = %ld\n",
	     nstA, nfeA, nsetupsA, njeA); 
     fprintf(f, "## nniA = %-6ld ncfnA = %-6ld netfA = %ld\n",
	     nniA, ncfnA, netfA);
     fprintf(f, "## ncheck = %-6d\n", engine->opt->ncheck);
  }

  return(1);

}

/** \brief Sets a linear objective function (for sensitivity solvers) via
    a text input file 
*/
SBML_ODESOLVER_API int IntegratorInstance_setLinearObjectiveFunction(integratorInstance_t *engine, char *v_file)
{
  FILE *fp;
  char *line, *token;
  int i;
  ASTNode_t **vector_v, *tempAST;
  odeModel_t *om = engine->om;
 
  if ( om->vector_v != NULL  )
  {     
    for ( i=0; i<om->neq; i++ )
      ASTNode_free(om->vector_v[i]);
    free(om->vector_v);
  }

  ASSIGN_NEW_MEMORY_BLOCK(vector_v, om->neq, ASTNode_t *, 0);

  if ( (fp = fopen(v_file, "r")) == NULL )
  {
    SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_VECTOR_V_FAILED,
		      "File %s not found in reading vector_v", v_file);
    return 0;
  }


  /* loop over lines */
  for ( i=0; (line = get_line(fp)) != NULL; i++ )
  {
    /* read column 0 */
    token = strtok(line, " ");
    /* skip empty lines and comment lines */
    if ( token == NULL || *token == '#' )
    {
      free(line);
      i--;
      continue;
    }
    /* check variable order */
    if ( i == om->neq )
       SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_VECTOR_V_FAILED,
			 "Inconsistent number of variables (>) "
			 "in setting vector_v from file %s", v_file);
    
    if ( strcmp(token, om->names[i]) != 0 )
      SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_VECTOR_V_FAILED,
			"Inconsistent variable order "
			"in setting vector_v from file %s", v_file); 
    
    /* read column 1 */
    token = strtok(NULL, "");
    tempAST = SBML_parseFormula(token);

    vector_v[i] = indexAST(tempAST, om->neq, om->names);
    ASTNode_free(tempAST);
    free(line);
  }

  fclose(fp);

  if ( i < om->neq )
  {
    SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_VECTOR_V_FAILED,
		      "read_v_file(): inconsistent number of variables "
		      "required NEQ: %d, provided from file: %d "
		      "in file %s", om->neq, i, v_file); 
  }
  om->vector_v = vector_v;
  
  return 1;
}


/** \brief Sets a general objective function (for ODE solution) via a
    text input file */
SBML_ODESOLVER_API int IntegratorInstance_setObjectiveFunction(integratorInstance_t *engine, char *ObjFunc_file)
{
  int i;
  FILE *fp;
  char *line = NULL, *line_formula = NULL, *token;
  ASTNode_t *ObjectiveFunction, *tempAST;
  odeModel_t *om = engine->om;

  /* If objective function exists, free it */
  if ( om->ObjectiveFunction != NULL  )
  {
    ASTNode_free(om->ObjectiveFunction);
    om->ObjectiveFunction = NULL;
  }

  if ( (fp = fopen(ObjFunc_file, "r")) == NULL )
  {
    SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_OBJECTIVE_FUNCTION_FAILED,
		      "File %s not found in reading objective function",
		      ObjFunc_file);
    return 0;
  }

  /* a very obfuscated way to skip comment lines and read exactly one
     line of objective function */
  for ( i=0; (line = get_line(fp)) != NULL; i++ )
  {   
    token = strtok(line, "");
    if (token == NULL || *token == '#')
    {
      free(line);
      i--;
    }
    else
    {
      if ( line_formula != NULL  )
        free(line_formula);
      ASSIGN_NEW_MEMORY_BLOCK(line_formula, strlen(line)+1, char, 0); 
      strcpy(line_formula, line); 
      if ( line != NULL  )
        free(line); 
    }
  }
  
  fclose(fp);


  if( i > 1)
  {
   SolverError_error(FATAL_ERROR_TYPE,
		     SOLVER_ERROR_OBJECTIVE_FUNCTION_FAILED,
		     "Error in processing objective function from file %s, "
		     "%d lines", ObjFunc_file, i); 
   return 0;
  }

  tempAST = SBML_parseFormula(line_formula);

  ObjectiveFunction = indexAST(tempAST, om->neq, om->names);
  ASTNode_free(tempAST);
 
  if ( line != NULL )
    free(line);
  if ( line_formula != NULL )
    free(line_formula);  

  om->ObjectiveFunction = ObjectiveFunction;
  
  return 1;
}



/** \brief Sets a general objective function (for ODE solution) via a string
*/
SBML_ODESOLVER_API int IntegratorInstance_setObjectiveFunctionFromString(integratorInstance_t *engine, char *str)
{
  ASTNode_t *ast, *temp_ast;
  odeModel_t *om = engine->om;
  
  if ( om->ObjectiveFunction != NULL  )
  {
    ASTNode_free(om->ObjectiveFunction);
    om->ObjectiveFunction = NULL;
  }

 
  temp_ast = SBML_parseFormula(str);
  ast = indexAST(temp_ast, om->neq, om->names);
  om->ObjectiveFunction = ast;
  
  ASTNode_free(temp_ast);
  
  return 1;
    
}

static int ODEModel_construct_vector_v_FromObjectiveFunction(odeModel_t *om)
{  
  int i, j, failed;
  ASTNode_t *fprime, *ObjFun;
  List_t *names;

  if ( om == NULL ) 
    return 0;
  
  if ( om->ObjectiveFunction == NULL ) 
    return 0;  
   
  /* if vector_v exists, free it */
  if ( om->vector_v != NULL )
  {
    for (  i=0; i<om->neq; i++  )
      if ( om->vector_v[i] != NULL )
	ASTNode_free(om->vector_v[i]);      
    free(om->vector_v);   
  }

  /******************** Calculate dJ/dx ************************/
  failed = 0; 
  ASSIGN_NEW_MEMORY_BLOCK(om->vector_v, om->neq, ASTNode_t *, 0);

  ObjFun = copyAST(om->ObjectiveFunction);

  for ( i=0; i<om->neq; i++ )
  {   
    fprime = differentiateAST(ObjFun, om->names[i]);
    om->vector_v[i] = fprime;

    /* check if the AST contains a failure notice */
    names = ASTNode_getListOfNodes(fprime,
				   (ASTNodePredicate) ASTNode_isName);

    for ( j=0; j<List_size(names); j++ ) 
      if ( strcmp(ASTNode_getName(List_get(names,j)),
		  "differentiation_failed") == 0 ) 
	failed++;

    List_free(names); 
  }

  ASTNode_free(ObjFun);

  if ( failed != 0 )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_OBJECTIVE_FUNCTION_V_VECTOR_COULD_NOT_BE_CONSTRUCTED,
		      "%d entries of vector_v could not be "
		      "constructed, due to failure of differentiation. " ,
		      failed);   
  }
 
  return 1;
}



/** \brief Sets a general objective function (for ODE solution) via a
    text input file
*/
SBML_ODESOLVER_API int IntegratorInstance_readTimeSeriesData(integratorInstance_t *engine, char *TimeSeriesData_file)
{
  int i;
  char *name;

  int n_data;      /* number of relevant data columns */
  int *col;        /* positions of relevant columns in data file */
  int *index;      /* corresponding indices in variable list */
  int n_time;      /* number of data rows */
 
  
  odeModel_t *om = engine->om;
  int n_var = om->neq;       /* number ofvariable names */
  char **var = om->names;      /* variable names */
  time_series_t *ts;  /*   = om->time_series; */


  if ( om->time_series != NULL )
    free_data( om->time_series );
  
  /* alloc mem */
  /* ts = space(sizeof(time_series_t)); */
  ASSIGN_NEW_MEMORY_BLOCK(ts, 1, time_series_t, 0);

  /* alloc mem for index lists */
  ts->n_var = n_var;
  ASSIGN_NEW_MEMORY_BLOCK(ts->var,   n_var, char   *, 0);
  ASSIGN_NEW_MEMORY_BLOCK(ts->data,  n_var, double *, 0); 
  ASSIGN_NEW_MEMORY_BLOCK(ts->data2, n_var, double *, 0);
    
  /* initialize index lists */
  for ( i=0; i<n_var; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(name,  strlen(var[i])+1, char , 0);
    strcpy(name, var[i]);
    ts->var[i]   = name;
    ts->data[i]  = NULL;
    ts->data2[i] = NULL;
  }

  /* alloc temp mem for column info */
  ASSIGN_NEW_MEMORY_BLOCK(col,   n_var, int, 0);
  ASSIGN_NEW_MEMORY_BLOCK(index, n_var, int, 0);

  /* read header line */
  n_data = read_header_line(TimeSeriesData_file, n_var, var, col, index);
  ts->n_data = n_data;
	
  /* count number of lines */
  n_time = read_columns(TimeSeriesData_file, 0, NULL, NULL, NULL);
  ts->n_time = n_time;

  /* alloc mem for data */
  for ( i=0; i<n_data; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(ts->data[index[i]], n_time, double, 0);
    ASSIGN_NEW_MEMORY_BLOCK(ts->data2[index[i]], n_time, double, 0);
  }
  /* ts->time = space(n_time * sizeof(double)); */
  ASSIGN_NEW_MEMORY_BLOCK(ts->time,  n_time, double, 0);

  /* read data */
  read_columns(TimeSeriesData_file, n_data, col, index, ts);

  /* free temp mem */
  free(col);
  free(index);

  /* initialize interpolation type */
  ts->type = 3;

  /* calculate second derivatives, if data is treated as being continuous */
  for ( i=0; i<n_var; i++ )
    if ( ts->data[i] != NULL )
      spline(ts->n_time, ts->time, ts->data[i], ts->data2[i]);

  ts->last = 0;
    
  /* alloc mem for warnings */
  ASSIGN_NEW_MEMORY_BLOCK(ts->mess,  2, char *, 0);
  ASSIGN_NEW_MEMORY_BLOCK(ts->warn,  2, int, 0); 

  /* initialize warnings */
  ts->mess[0] = "argument out of range (left) ";
  ts->mess[1] = "argument out of range (right)";
  for ( i=0; i<2; i++ )
    ts->warn[i] = 0;   

  om->time_series = ts;

 return 1;
}


/** \brief Perform necessary quadratures for ODE/forward/adjoint
          sensitivity In forward phase, if nonlinear objective is
          present (by calling II_setObjectiveFunction) it is computed;
          alternatively, if linear objective is present (prior call to
          II_setLinearObj), it is computed.

          In adjoint phase, the backward quadrature for linear
          objective is performed.
 */
SBML_ODESOLVER_API int IntegratorInstance_CVODEQuad(integratorInstance_t *engine)
{
  int flag, i, j;
  cvodeSolver_t *solver = engine->solver;
  cvodeSettings_t *opt = engine->opt;
  int iS;
  odeModel_t *om = engine->om;
  odeSense_t *os = engine->os;
  cvodeData_t *data = engine->data;
  
  if( engine->AdjointPhase )
  {
    /* if continuous data is observed, compute quadrature;
       if discrete data is observed, quadrature has been computed already */
    if ( opt->observation_data_type == 0  )
    {  
      flag = CVodeGetQuadB(solver->cvadj_mem, solver->qA);
      CVODE_HANDLE_ERROR(&flag, "CVodeGetQuadB", 1);
    }

    /* For adj sensitivity components corresponding to IC as parameter */
    for( iS=0; iS<os->nsens; iS++  )
      if ( os->index_sensP[iS] == -1 )
      {
        NV_Ith_S(solver->qA, iS) = - data->adjvalue[os->index_sens[iS]];
      }
  }
  else
  {
     
    if ( opt->observation_data_type == 0  )
    {  /* if continuous data is observed, compute quadrature */
      /* If an objective function exists */
      if( om->ObjectiveFunction != NULL )
      {
	flag = CVodeGetQuad(solver->cvode_mem, solver->tout, solver->q);
	CVODE_HANDLE_ERROR(&flag, "CVodeGetQuad ObjectiveFunction", 1);
      }

      /* If doing forward sensitivity analysis and vector_v exists,
	 compute sensitivity quadrature */
      if (!opt->doFIM)
      {
	if( opt->Sensitivity && om->ObjectiveFunction == NULL &&
	    om->vector_v != NULL  )
	{
	  flag = CVodeGetQuad(solver->cvode_mem, solver->tout, solver->qS);
	  CVODE_HANDLE_ERROR(&flag, "CVodeGetQuad V_Vector", 1);
	}
      }
      else /* doFIM */
      {
	flag = CVodeGetQuad(solver->cvode_mem, solver->tout, solver->qFIM);
	CVODE_HANDLE_ERROR(&flag, "CVodeGetQuad FIM", 1);
	
	/* copy results to matrix FIM */
	for (i=0; i<os->nsens; i++)
	  for (j=0; j<os->nsens; j++)
	    data->FIM[i][j] = NV_Ith_S(solver->qFIM, i*os->nsens + j);
      }
    }
    else
    {
      /* else solver->q or solver->qS already contain the objective
	 and its sensitivity respectively */  
    }
  }
  return(1);
}


/** \brief Prints computed quadratures for ODE/forward/adjoint
           sensitivity. In forward phase, if nonlinear objective is
           present (by calling II_setObjectiveFunction) it is printed;
           alternatively, if linear objective is present (prior call
           to II_setLinearObj), it is printed.

	   In adjoint phase, the backward quadrature for linear
	   objective is printed.
*/

SBML_ODESOLVER_API int IntegratorInstance_printQuad(integratorInstance_t *engine, FILE *f)
{
  
  int j;
  double value;
  char *formula;
  odeModel_t *om = engine->om; 
  odeSense_t *os = engine->os; 
  ASTNode_t *tempAST; 

  if ( engine->AdjointPhase )
  {
  /*  fprintf(f, "\nExpression for integrand of linear objective J: \n"); */
    for(j=0;j<om->neq;j++)
    {       
      /*      Append "_data" to observation data in the vector_v AST */
      /*      tempAST = copyRevertDataAST(om->vector_v[j]); */
      /*      fprintf(f, "%d-th component: %s \n" , j, SBML_formulaToString(tempAST) ); */
      /*      ASTNode_free(tempAST); */
    }
    
    for ( j=0; j<os->nsens; j++ )
      {
	value = NV_Ith_S(engine->solver->qA, j);
	fprintf(f, "dJ/dp_%d=%0.15g ", j, value);
      }
    fprintf(f, "\n");
  }
  else
  {
    if ( om->ObjectiveFunction != NULL  ) 
    {  
      /*  Append "_data" to observation data in the vector_v AST */
      /*   tempAST = copyRevertDataAST(om->ObjectiveFunction); */
      /*        fprintf(f, "\nExpression for integrand of objective J: %s \n\n" , */
      /* 	       SBML_formulaToString(tempAST) ); */
      value = NV_Ith_S(engine->solver->q, 0);
      fprintf(f, "Computed J=%0.15g \n", value);
      /*  ASTNode_free(tempAST); */
    }
    else if ( engine->om->vector_v != NULL  )
    {
      fprintf(f, "\nExpression for integrand of linear objective J: \n");
      for(j=0;j<om->neq;j++)
      {    
        /* Append "_data" to observation data in the vector_v AST  */
	tempAST = copyRevertDataAST(om->vector_v[j]);
	formula = SBML_formulaToString(tempAST);
	
	fprintf(f, "%d-th component: %s \n" , j, formula);
	free(formula);
        ASTNode_free(tempAST);
      }      

      /*!!! TODO : clarify why valgrind reports
	"==16330== Conditional jump or move depends on
	           uninitialised value(s)" for the following print command!
		   see file sundials-2.3.0/include/nvector/nvector_serial.h */
      for ( j=0; j<os->nsens; j++ )
      {
	value = NV_Ith_S(engine->solver->qS, j);
	fprintf(f, "dJ/dp_%d=%0.15g ", j, value);
      }
      fprintf(f, "\n");
    }
    else fprintf(f, "\nNo quadrature was performed \n");
  }

  return(1);
}


/** \brief Writes computed quadratures for the computed objective, or 
           the adjoint adjoint sensitivity if the solver is in adjoint phase.
*/

SBML_ODESOLVER_API int IntegratorInstance_writeQuad(integratorInstance_t *engine, realtype *data)
{
  
  int j;
  odeModel_t *om = engine->om; 
  odeSense_t *os = engine->os; 

  data = (realtype *) data;

  if ( engine->AdjointPhase )
  { 
   for ( j=0; j<os->nsens; j++ )
     data[j] = NV_Ith_S(engine->solver->qA, j);
  }
  else
  {
      if ( om->ObjectiveFunction != NULL  )
	  data[0] = NV_Ith_S(engine->solver->q, 0); 
      else if  ( engine->opt->Sensitivity )
	  for ( j=0; j<os->nsens; j++ )
	      data[j] = NV_Ith_S(engine->solver->qS, j);
  }

  return(1);
}

/* Extension of copyAST, for adding to the AST having ASTNode_isSetData
   by attaching the string extension "_data" to variable names */
static ASTNode_t *copyRevertDataAST(const ASTNode_t *f)
{
  int i;
  ASTNode_t *copy;
  char *tempstr = NULL;
  char *tempstr2 = NULL;

  copy = ASTNode_create();

  /* DISTINCTION OF CASES */
  /* integers, reals */
  if ( ASTNode_isInteger(f) ) 
    ASTNode_setInteger(copy, ASTNode_getInteger(f));
  else if ( ASTNode_isReal(f) ) 
    ASTNode_setReal(copy, ASTNode_getReal(f));
  /* variables */
  else if ( ASTNode_isName(f) )
  {
    if ( ASTNode_isSetIndex((ASTNode_t *)f) )
    {
      ASTNode_free(copy);
      copy = ASTNode_createIndexName();
      ASTNode_setIndex(copy, ASTNode_getIndex((ASTNode_t *)f));
    }

    if ( !ASTNode_isSetData((ASTNode_t *)f) )  
         ASTNode_setName(copy, ASTNode_getName(f));
    else
    {   
        /*  ASTNode_setData(copy); */
      tempstr  = (char *)ASTNode_getName(f);
      ASSIGN_NEW_MEMORY_BLOCK(tempstr2, strlen(tempstr)+5, char, NULL);
      strncpy(tempstr2, tempstr, strlen(tempstr) );
      strncat(tempstr2, "_data", 5);
      ASTNode_setName(copy, tempstr2 );
      free(tempstr);
    }
    
  }
  /* constants, functions, operators */
  else
  {
    ASTNode_setType(copy, ASTNode_getType(f));
    /* user-defined functions: name must be set */
    if ( ASTNode_getType(f) == AST_FUNCTION ) 
      ASTNode_setName(copy, ASTNode_getName(f));
    for ( i=0; i<ASTNode_getNumChildren(f); i++ ) 
      ASTNode_addChild(copy, copyRevertDataAST(ASTNode_getChild(f,i)));
  }

  return copy;
}



/************* Additional Function for Sensitivity Analysis **************/

/**
 * fS routine: Called by CVODES to compute the sensitivity RHS for one
 * parameter.
 *
 CVODES sensitivity analysis calls this function any time required,
 with current values for variables x, time t and sensitivities
 s. The function evaluates df/dx * s + df/dp for one p and writes the
 results back to CVODE's N_Vector(ySdot) vector. The function is
 not `static' only for including it in the documentation!
*/

static int fS(int Ns, realtype t, N_Vector y, N_Vector ydot, 
	      int iS, N_Vector yS, N_Vector ySdot, 
	      void *fS_data, N_Vector tmp1, N_Vector tmp2)
{
  int i;
  realtype *ydata, *ySdata, *dySdata;
  cvodeData_t *data;
  data  = (cvodeData_t *) fS_data;

  ydata = NV_DATA_S(y);
  ySdata = NV_DATA_S(yS);
  dySdata = NV_DATA_S(ySdot);

  /** update ODE variables from CVODE */
  for ( i=0; i<data->model->neq; i++ ) data->value[i] = ydata[i];
  
  /** update time */
  data->currenttime = t;

  /** evaluate sensitivity RHS: df/dx * s + df/dp for one p */
  for ( i=0; i<data->model->neq; i++ ) 
  {
    dySdata[i] = 0;
    /* add parameter sensitivity */
    /*!!! TODO: evaluation of nonzero-elements in the parameter matrix dY/dP */
    if ( data->os->index_sensP[iS] != -1 &&
	 data->os->sensLogic[i][data->os->index_sensP[iS]] )
    {
#ifdef ARITHMETIC_TEST
      dySdata[i] +=
	data->os->senscode[i][data->os->index_sensP[iS]]->evaluate(data);
#else
      dySdata[i] +=
	evaluateAST(data->os->sens[i][data->os->index_sensP[iS]], data);
#endif
    }
  }

  /* add variable sensitivities */
  for ( i=0; i<data->model->sparsesize; i++ )
  {
    nonzeroElem_t *nonzero = data->model->jacobSparse[i];
    
#ifdef ARITHMETIC_TEST
    dySdata[nonzero->i] += nonzero->ijcode->evaluate(data) * ySdata[nonzero->j];
#else
    dySdata[nonzero->i] += evaluateAST(nonzero->ij, data)  * ySdata[nonzero->j];
#endif
    
  }
  return (0);
}

/********* Additional Function for Adjoint Sensitivity Analysis **********/

/**
 * fA routine: Called by CVODES to compute the adjoint sensitivity RHS for one
 * parameter.
 *
 CVODES adjoint sensitivity analysis calls this function any time required,
 with current values for variables y, yA, and time t.
 The function evaluates -[df/dx]^T * yA + v
 and writes the results back to CVODE's N_Vector yAdot.
*/

static int fA(realtype t, N_Vector y, N_Vector yA, N_Vector yAdot,
	      void *fA_data)
{
  int i;
  realtype *ydata, *yAdata, *dyAdata;
  cvodeData_t *data;
  data  = (cvodeData_t *) fA_data;
  
  ydata = NV_DATA_S(y);
  yAdata = NV_DATA_S(yA);
  dyAdata = NV_DATA_S(yAdot);

  /* update ODE variables from CVODE  */
  for ( i=0; i<data->model->neq; i++ ) data->value[i] = ydata[i];
 
  /* update time */
  data->currenttime = t;

  /* evaluate adjoint sensitivity RHS: -[df/dx]^T * yA + v */
  for(i=0; i<data->model->neq; i++)
  {
    dyAdata[i] = 0;  
    /*  Vector v contribution, if continuous data is used */
    if(data->model->discrete_observation_data==0)
      dyAdata[i] += evaluateAST(data->model->vector_v[i], data);
  }
     
  for ( i=0; i<data->model->sparsesize; i++ )
  {
    nonzeroElem_t *nonzero = data->model->jacobSparse[i];

#ifdef ARITHMETIC_TEST
    dyAdata[nonzero->j] -= nonzero->ijcode->evaluate(data) * yAdata[nonzero->i];
#else
    dyAdata[nonzero->j] -= evaluateAST(nonzero->ij, data)  * yAdata[nonzero->i];
#endif    
  }  
  return (0);
}

/**
   Adjoint Jacobian routine: Compute JB(t,x) = -[df/dx]^T
   
   This function is (optionally) called by CVODES integration routines
   every time as required.

   Very similar to the fA routine, it evaluates the Jacobian matrix
   equations with CVODE's current values and writes the results
   back to CVODE's internal vector DENSE_ELEM(J,i,j).
*/

static int JacA(long int NB, DenseMat JB, realtype t,
		N_Vector y, N_Vector yB, N_Vector fyB, void *jac_dataB,
		N_Vector tmp1B, N_Vector tmp2B, N_Vector tmp3B)
{

  int i;
  realtype *ydata;
  cvodeData_t *data;
  data  = (cvodeData_t *) jac_dataB;
  ydata = NV_DATA_S(y);
  
  /** update ODE variables from CVODE */
  for ( i=0; i<data->model->neq; i++ ) data->value[i] = ydata[i];

  /** update time */
  data->currenttime = t;

  /** evaluate Jacobian JB = -[df/dx]^T */
  for ( i=0; i<data->model->sparsesize; i++ )
  {
    nonzeroElem_t *nonzero = data->model->jacobSparse[i];    
#ifdef ARITHMETIC_TEST
    DENSE_ELEM(JB, nonzero->j,nonzero->i) = - nonzero->ijcode->evaluate(data);
#else
    DENSE_ELEM(JB, nonzero->j,nonzero->i) = - evaluateAST(nonzero->ij, data);
#endif 
  }
  return (0);
}


static int fQA(realtype t, N_Vector y, N_Vector yA, 
	       N_Vector qAdot, void *fA_data)
{ 
  int i, j;
  realtype *ydata, *yAdata, *dqAdata;
  cvodeData_t *data;
  data  = (cvodeData_t *) fA_data;

  ydata = NV_DATA_S(y);
  yAdata = NV_DATA_S(yA);
  dqAdata = NV_DATA_S(qAdot);

 
  /* update ODE variables from CVODE  */  
  for ( i=0; i<data->model->neq; i++ ) data->value[i] = ydata[i];
 
  /* update time */
  data->currenttime = t;

  /* evaluate quadrature integrand: yA^T * df/dp */
  for ( j=0; j<data->os->nsens; j++ )
    dqAdata[j] = 0.0;
  
  for ( i=0; i<data->os->sparsesize; i++ )
  {
    nonzeroElem_t *nonzero = data->os->sensSparse[i];        
#ifdef ARITHMETIC_TEST
    dqAdata[nonzero->j] += yAdata[nonzero->i] * nonzero->ijcode->evaluate(data);
#else
    dqAdata[nonzero->j] += yAdata[nonzero->i] * evaluateAST(nonzero->ij, data);
#endif 
  }
  return (0);
}


static int fQS(realtype t, N_Vector y, N_Vector qdot, void *fQ_data)
{
  int i, j, flag;
  realtype *ydata, *dqdata;
  cvodeData_t *data;
  cvodeSolver_t *solver; 
  integratorInstance_t *engine;
  N_Vector *yS;
  N_Vector yy;
  
  engine = (integratorInstance_t *) fQ_data;
  solver = engine->solver;
  data  =  engine->data;

  ydata = NV_DATA_S(y);
  dqdata = NV_DATA_S(qdot);

  /* update ODE variables from CVODE  */  
  for ( i=0; i<data->model->neq; i++ ) data->value[i] = ydata[i];
 
  /* update time */
  data->currenttime = t;

  /* update sensitivities */
  yy = N_VNew_Serial(data->model->neq);
  yS = N_VCloneVectorArray_Serial(data->os->nsens, yy);
  N_VDestroy_Serial(yy);

  /*  At t=0, yS is initialized to 0. In this case, CvodeGetSens
      shouldn't be used as it gives nan's */
  if ( t != 0 )
  {
    flag = CVodeGetSens(solver->cvode_mem, t, yS);
    if ( flag < 0 )
    {
      SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_CVODE_MALLOC_FAILED,
			"SUNDIALS_ERROR: CVodeGetSens failed "
			"with flag %d", flag);
      exit(EXIT_FAILURE);
    }
  }  


  /* evaluate quadrature integrand: (y-ydata) * yS_i for each i */
  for ( i=0; i<data->os->nsens; i++ )
  {
    dqdata[i] = 0.0;
    for ( j=0; j<data->model->neq; j++ )
      dqdata[i] += evaluateAST(engine->om->vector_v[j], data) *
	NV_Ith_S(yS[i], j);
  }

  N_VDestroyVectorArray_Serial(yS, data->os->nsens);

  return (0);
}

static int fQFIM(realtype t, N_Vector y, N_Vector qdot, void *fQ_data)
{
  int i, j, k, flag;
  realtype *dqdata;
  cvodeData_t *data;
  cvodeSolver_t *solver; 
  integratorInstance_t *engine;
  N_Vector *yS;
  N_Vector yy;
  
  engine = (integratorInstance_t *) fQ_data;
  solver = engine->solver;
  data  =  engine->data;

  dqdata = NV_DATA_S(qdot);

  /* update time */
  data->currenttime = t;

  /* update sensitivities */
  yy = N_VNew_Serial(data->model->neq);
  yS = N_VCloneVectorArray_Serial(data->os->nsens, yy);
  N_VDestroy_Serial(yy);

  /*  At t=0, yS is initialized to 0. In this case, CvodeGetSens
      shouldn't be used as it gives nan's */
  if ( t != 0 )
  {
    flag = CVodeGetSens(solver->cvode_mem, t, yS);
    if ( flag < 0 )
    {
      SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_CVODE_MALLOC_FAILED,
			"SUNDIALS_ERROR: CVodeGetSens failed "
			"with flag %d", flag);
      exit(EXIT_FAILURE);
    }
  }  


  /* evaluate quadrature integrand: < yS_i , yS_j > for each i,j */
  for ( i=0; i<data->os->nsens; i++ )
  {
    for ( j=0; j<data->os->nsens; j++ )
    {
    dqdata[i*data->os->nsens + j] = 0.0;
    for ( k=0; k<data->model->neq; k++ )
      dqdata[i*data->os->nsens + j] += data->weights[k] *
	NV_Ith_S(yS[i], k) * NV_Ith_S(yS[j], k) ;
    }
  }

  N_VDestroyVectorArray_Serial(yS, data->os->nsens);

  return (0);
}


/** @} */
/* End of file */

