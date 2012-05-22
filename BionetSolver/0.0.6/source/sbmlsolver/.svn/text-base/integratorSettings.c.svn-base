/*   Last changed Time-stamp: <2009-02-12 18:14:02 raim> */
/*   $Id: integratorSettings.c,v 1.52 2009/02/12 09:25:05 raimc Exp $ */
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
/*! \defgroup setttings Integrator Settings
    \ingroup integration 
    \brief This module contains all functions to set integration options
    in integratorSettings
    
    With these functions an application can choose integration time,
    methods and options like error tolerances.
*/
/*@{*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sbmlsolver/integratorSettings.h"
#include "sbmlsolver/solverError.h"
#include "sbmlsolver/interpol.h"
#include "sbmlsolver/util.h"


/** Creates a settings structure with default values
*/

SBML_ODESOLVER_API cvodeSettings_t *CvodeSettings_create()
{
  return CvodeSettings_createWithTime(1., 10);
}


/** Creates a settings structure with default Values
    for Errors, MxStep and Swicthes
*/

SBML_ODESOLVER_API cvodeSettings_t *CvodeSettings_createWithTime(double Time, int PrintStep)
{
  /* !!! CHANGE DEFAULT VALUES HERE!!! */
  return CvodeSettings_createWith(Time, PrintStep,
				  1e-18, 1e-10, 10000, 0, 0,
				  1, 0, 0, 0, 1, 0, 0);
}


/** Print all cvodeSettings
*/

SBML_ODESOLVER_API void CvodeSettings_dump(cvodeSettings_t *set)
{
  printf("\n");
  printf("SOSlib INTEGRATION SETTINGS\n");
  printf("1) CVODE SPECIFIC SETTINGS:\n");
  printf("absolute error tolerance for each output time:   %g\n",
	 set->Error);
  printf("relative error tolerance for each output time:   %g\n",
	 set->RError);
  printf("max. nr. of steps to reach next output time:     %d\n",
	 set->Mxstep);
  printf("Nonlinear solver method:                         %d: %s\n"
	 "          Maximum Order:                         %d\n",
	 set->CvodeMethod, CvodeSettings_getMethod(set), set->MaxOrder);
  printf("Iteration method:                                %d: %s\n",
	 set->IterMethod, CvodeSettings_getIterMethod(set));
  printf("Sensitivity:                                     %s\n",
	 set->Sensitivity ? "1: yes " : "0: no");
  printf("     method:                                     %d: %s\n",
	 set->SensMethod, CvodeSettings_getSensMethod(set));
  printf("2) SOSlib SPECIFIC SETTINGS:\n");
  printf("Jacobian matrix: %s\n", set->UseJacobian ?
	 "1: generate Jacobian" : "0: CVODE's internal approximation");
  printf("Indefinitely:    %s\n", set->Indefinitely ?
	 "1: infinite integration" :
	 "0: finite integration");
  printf("Event Handling:  %s\n", set->HaltOnEvent ?
	 "1: stop integration" :
	 "0: keep integrating");
  printf("Steady States:   %s\n", set->SteadyState ?
	 "1: stop integrating" :
	 "0: keep integrating");
  printf("Steady state threshold: %g\n", set->ssThreshold);
  printf("Store Results:   %s\n", set->StoreResults ?
	 "1: store results (only for finite integration)" :
	 "0: don't store results");  
  printf("3) TIME SETTINGS:\n");
  if ( set->Indefinitely )
    printf("Infinite integration with time step %g", set->Time);
  else {
    printf("endtime: %g\n", set->TimePoints[set->PrintStep]);
    printf("steps:   %d", set->PrintStep);
  }
  printf("\n");
  printf("\n");
}


/** Creates a settings structure from input values - WARNING:
    this function's type signature will change with time,
    as new settings will be required for other solvers!
*/

SBML_ODESOLVER_API cvodeSettings_t *CvodeSettings_createWith(double Time, int PrintStep, double Error, double RError, int Mxstep, int Method, int IterMethod, int UseJacobian, int Indefinitely, int HaltOnEvent, int HaltOnSteadyState, int StoreResults, int Sensitivity, int SensMethod)
{

  cvodeSettings_t *set;
  ASSIGN_NEW_MEMORY(set, struct cvodeSettings, NULL);

  /* 1. Setting SBML ODE Solver integration parameters */
  CvodeSettings_setErrors(set, Error, RError, Mxstep);
  /* set non-linear solver defaults (BDF, Newton, max.order 5*/
  set->CvodeMethod = Method;
  set->IterMethod = IterMethod;
  if ( Method == 0 )
    set->MaxOrder = 5;
  else
    set->MaxOrder = 12;
  set->compileFunctions = 0;
  set->ResetCvodeOnEvent = 1;
  CvodeSettings_setSwitches(set, UseJacobian, Indefinitely,
			    HaltOnEvent, HaltOnSteadyState, StoreResults,
			    Sensitivity, SensMethod);

  /* 2. Setting Requested Time Series */
  /* Unless indefinite integration, generate a TimePoints array  */
  if  ( !Indefinitely ) 
    /* ... generate default TimePoint array */
    CvodeSettings_setTime(set, Time, PrintStep);

  /* 3. Setting no selected parameters/ICs */
  set->sensIDs = NULL;
  set->nsens = 0; 

  /* Default: not doing adjoint solution  */
  set->DoAdjoint = 0;
  /* set->AdjointPhase = 0; */
 
  /* default: use continuous observation */
  set->observation_data_type = 0;

  /* deactivate TSTOP mode of CVODE */
  set->SetTStop = 0;
   
  /* do not trigger numerical refinement upon detection of negative state values */
  set->DetectNegState = 0;

  return set;
}


/** Creates a settings structure and copies all values from input
*/
/*!!! TODO : check whether clone copies all values, find better solution */
SBML_ODESOLVER_API cvodeSettings_t *CvodeSettings_clone(cvodeSettings_t *set)
{
  int i;
  cvodeSettings_t *clone;
  ASSIGN_NEW_MEMORY(clone, struct cvodeSettings, NULL);
    
  /* Setting SBML ODE Solver integration parameters */
  CvodeSettings_setErrors(clone, set->Error, set->RError, set->Mxstep);
  CvodeSettings_setSwitches(clone, set->UseJacobian, set->Indefinitely,
			    set->HaltOnEvent, set->SteadyState,
			    set->StoreResults,
			    set->Sensitivity, set->SensMethod);

  CvodeSettings_setMethod(clone, set->CvodeMethod, set->MaxOrder);
  CvodeSettings_setIterMethod(clone, set->IterMethod);

  clone->compileFunctions = set->compileFunctions;
  clone->ResetCvodeOnEvent = set->ResetCvodeOnEvent;
  
  /* Unless indefinite integration is chosen, generate a TimePoints array  */
  if  ( !clone->Indefinitely ) {    
    ASSIGN_NEW_MEMORY_BLOCK(clone->TimePoints,clone->PrintStep+1,double,NULL);
    /* copy TimePoint array */
    for ( i=0; i<=clone->PrintStep; i++ ) 
      clone->TimePoints[i] = set->TimePoints[i];    
    
    /* if adjoint TimePoint array exists, also copy it */
    if ( set->AdjTimePoints != NULL  ){
      ASSIGN_NEW_MEMORY_BLOCK(clone->AdjTimePoints,clone->PrintStep+1,double,NULL);
      /* copy TimePoint array */
      for ( i=0; i<=clone->PrintStep; i++ ) 
	clone->AdjTimePoints[i] = set->AdjTimePoints[i];     
    }
  }
  return clone;
}


/** Sets absolute and relative error tolerances and maximum number of
    internal steps during CVODE integration 
*/

SBML_ODESOLVER_API void CvodeSettings_setErrors(cvodeSettings_t *set, double Error, double RError, int Mxstep) 
{
  CvodeSettings_setError(set, Error);
  CvodeSettings_setRError(set, RError);
  CvodeSettings_setMxstep(set, Mxstep);    
}


/** Sets absolute error tolerance 
*/

SBML_ODESOLVER_API void CvodeSettings_setError(cvodeSettings_t *set, double Error)
{
  set->Error = Error;
}


/** Sets relative error tolerance 
*/

SBML_ODESOLVER_API void CvodeSettings_setRError(cvodeSettings_t *set, double RError)
{
  set->RError = RError;
}

/** Sets maximum number of internal steps during CVODE integration */

SBML_ODESOLVER_API void CvodeSettings_setMxstep(cvodeSettings_t *set, int Mxstep)
{
  set->Mxstep = Mxstep;  
}

/** Sets flag that tells solver that the adjoint solution is desired */
SBML_ODESOLVER_API void CvodeSettings_setDoAdj(cvodeSettings_t *set) 
{
  set->DoAdjoint = 1;
}

/** Sets flag that tells solver that the adjoint solution is desired   */
SBML_ODESOLVER_API void CvodeSettings_unsetDoAdj(cvodeSettings_t *set) 
{
  set->DoAdjoint = 0;
}


/** Sets absolute and relative error tolerances and maximum number of
    internal steps during CVODE adjoint integration 
*/

SBML_ODESOLVER_API void CvodeSettings_setAdjErrors(cvodeSettings_t *set, double Error, double RError) 
{
  CvodeSettings_setAdjError(set, Error);
  CvodeSettings_setAdjRError(set, RError);
  /* CvodeSettings_setAdjMxstep(set, Mxstep);*/    
}

/** Sets absolute error tolerance for adjoint integration  
*/

SBML_ODESOLVER_API void CvodeSettings_setAdjError(cvodeSettings_t *set, double Error)
{
  set->AdjError = Error;
}

/** Sets relative error tolerance for adjoint integration  
*/

SBML_ODESOLVER_API void CvodeSettings_setAdjRError(cvodeSettings_t *set, double RError)
{
  set->AdjRError = RError;
}


/** Sets the number of forward steps saved, prior to doing the adjoint integration  
*/

SBML_ODESOLVER_API void CvodeSettings_setnSaveSteps(cvodeSettings_t *set, int nSaveSteps)
{
  set->nSaveSteps = nSaveSteps;
}

/** Set method non-linear solver methods, and its maximum order (currently
    the latter cannot really be set, but default to 5 for BDF or 12 for
    Adams-Moulton!!
    
    CvodeMethod: 0: BDF (default); 1: Adams-Moulton,\n
    MaxOrder: maximum order (default: 5 for BDF, 12 for Adams-Moulton.

*/

SBML_ODESOLVER_API void CvodeSettings_setMethod(cvodeSettings_t *set, int CvodeMethod, int MaxOrder)
{
  /* CvodeMethod == 0: default BDF method
     Method == 1: Adams-Moulton method */
  if ( 0 <= CvodeMethod &&  CvodeMethod < 2 )
  {
    set->CvodeMethod = CvodeMethod;
    set->MaxOrder = MaxOrder;
  }
  /* else error message !! ? */
}


/** Set method for CVODE integration

    0: NEWTON (default)\n
    1: FUNCTIONAL
*/

SBML_ODESOLVER_API void CvodeSettings_setIterMethod(cvodeSettings_t *set, int i)
{
  /* i == 0: default NEWTON iteration
     i == 1: FUNCTIONAL iteraction */
  if ( 0 <= i && i < 2 ) set->IterMethod = i;
  else set->IterMethod = 0;
}


/** NOT USED!

    Sets maximum order of BDF or Adams-Moulton method, respectively,
    currently this settings is NOT USED; defaults to 5 for BDF and 12
    for Adams-Moulton
*/

SBML_ODESOLVER_API void CvodeSettings_setMaxOrder(cvodeSettings_t *set, int MaxOrder)
{
  set->MaxOrder = MaxOrder;  
}

/** Activates the detection of negative state values. CVodes would then perform
    numerical refinement if negative state inputs are encountered. */
SBML_ODESOLVER_API void CvodeSettings_setDetectNegState(cvodeSettings_t *set, int i)
{
  set->DetectNegState = i;
}

/** Sets whether the simulator uses compiled functions
    for computing ODEs, the Jacabian or events
*/
/*!!! either implement compilation or catch during integratio */
SBML_ODESOLVER_API void CvodeSettings_setCompileFunctions(cvodeSettings_t *set, int compileFunctions)
{
  set->compileFunctions = compileFunctions;

}


/** Activates the TSTOP mode of CVODES. This is highly recommended when
    IntegratorInstance_setVariableValue affects ODE right hand side
    equations (e.g. rate laws), PLEASE CLICK AND READ MORE BELOW

    Setting TStop is not required for all model types, but mainly necessary
    for very "unstiff" (easy-to-solve) models, for which CVODE internally can
    integrate far beyond the next time step. It will then not evaluate the
    right hand side of the ODE system for one or more output time steps and
    thus not realize the change.   */
SBML_ODESOLVER_API void CvodeSettings_setTStop(cvodeSettings_t *set, int i)
{
  set->SetTStop = i;
}

/** Sets integration switches in cvodeSettings. WARNING: this
    function's type signature will change with time, as new settings
    will be required for other solvers!
*/

SBML_ODESOLVER_API void CvodeSettings_setSwitches(cvodeSettings_t *set, int UseJacobian, int Indefinitely, int HaltOnEvent, int HaltOnSteadyState, int StoreResults, int Sensitivity, int SensMethod)
{  
  set->UseJacobian = UseJacobian;
  set->Indefinitely = Indefinitely;
  set->HaltOnEvent = HaltOnEvent;
  set->StoreResults = StoreResults;
  CvodeSettings_setHaltOnSteadyState(set, HaltOnSteadyState);
  CvodeSettings_setSensitivity(set, Sensitivity);
  CvodeSettings_setSensMethod(set, SensMethod);
}


/** Calculates a time point series from Endtime and Printstep and sets
    the time series in cvodeSettings. Returns 1, if sucessful and 0, if
    not.
*/

SBML_ODESOLVER_API int CvodeSettings_setTime(cvodeSettings_t *set, double EndTime, int PrintStep)
{
  int i, j;
  double *timeseries;
  ASSIGN_NEW_MEMORY_BLOCK(timeseries, PrintStep, double, 0);
  
  for ( i=1; i<=PrintStep; i++ ) timeseries[i-1] = i * EndTime/PrintStep;
  
  j = CvodeSettings_setTimeSeries(set, timeseries, PrintStep);
  free(timeseries);
  return j;
}

/** Copies a predefined timeseries into cvodeSettings. Assigns memory for
    an array of requested time points with size PrintStep + 1 (including
    initial time 0). Returns 1, if sucessful and 0, if not. */

SBML_ODESOLVER_API int CvodeSettings_setTimeSeries(cvodeSettings_t *set,
						   double *timeseries,
						   int PrintStep)
{
  int i;

  if ( set->TimePoints != NULL )
    free(set->TimePoints);
  
  ASSIGN_NEW_MEMORY_BLOCK(set->TimePoints, PrintStep+1, double, 0);    
  set->Time = timeseries[PrintStep-1];
  set->PrintStep = PrintStep;
  set->TimePoints[0] = 0.0;
  for ( i=1; i<=PrintStep; i++ ) 
    set->TimePoints[i] = timeseries[i-1];

  return 1;
}

/** Calculates adjoint time point series from Endtime and Printstep and sets
    the time series in cvodeSettings. Returns 1, if sucessful and 0, if
    not.
*/

SBML_ODESOLVER_API int CvodeSettings_setAdjTime(cvodeSettings_t *set, double EndTime, int PrintStep)
{
  int i, j;
  double *timeseries;
  ASSIGN_NEW_MEMORY_BLOCK(timeseries, PrintStep, double, 0);  

  /* Adjoint time series goes backwards, from EndTime to 0 */
  for ( i=1; i<=PrintStep; i++ )
    timeseries[i-1] = (PrintStep - i) * EndTime/PrintStep;
 
  j = CvodeSettings_setAdjTimeSeries(set, timeseries, PrintStep, EndTime);

  free(timeseries);
  return j;
}


/** Copies a predefined adjoint time series into cvodeSettings.
    Assigns memory for an array of requested time points with size
    AdjPrintStep + 1 (including initial time EndTime).
    Returns 1, if sucessful and 0, if  not. */

SBML_ODESOLVER_API  int CvodeSettings_setAdjTimeSeries(cvodeSettings_t *set,
						       double *timeseries,
						       int AdjPrintStep,
						       double EndTime)
{
  int i;

  if ( set->AdjTimePoints != NULL )
   free(set->AdjTimePoints);

  ASSIGN_NEW_MEMORY_BLOCK(set->AdjTimePoints, AdjPrintStep+1, double, 0);    

  set->AdjTime = timeseries[AdjPrintStep-1];
  set->AdjPrintStep = AdjPrintStep;

  /* Adjoint is integrated backwards from EndTime to time=0.0
     (initial time for forward)  */
  set->AdjTimePoints[0] = EndTime;

  for ( i=1; i<= AdjPrintStep; i++ ) 
    set->AdjTimePoints[i] = timeseries[i-1];
  
  return 1;
}

/** Gets the type of observation data,
    values are:
    0 ... continuous data
    1 ... discrete data  */
SBML_ODESOLVER_API int CvodeSettings_getObservationDataType(cvodeSettings_t *set)
{
  return set->observation_data_type;
}


/** Sets flag that tells solver that the (experimental) data should be treated as discrete entities  */
SBML_ODESOLVER_API void CvodeSettings_setDiscreteObservation(cvodeSettings_t *set) 
{
  set->observation_data_type = 1;
}

/** unsets flag that tells solver that the (experimental) data should be treated as discrete entities  */
SBML_ODESOLVER_API void CvodeSettings_unsetDiscreteObservation(cvodeSettings_t *set) 
{
  set->observation_data_type = 0;
}

/** Sets flag that tells solver to compute FIM (instead of gradient to objective) */
SBML_ODESOLVER_API void CvodeSettings_setFIM(cvodeSettings_t *set) 
{
  set->doFIM = 1;
}

/** Sets flag that tells solver not to compute FIM (but the gradient as "normal")  */
SBML_ODESOLVER_API void CvodeSettings_unsetFIM(cvodeSettings_t *set) 
{
  set->doFIM = 0;
}


/* Reads time point column of data, for use in
   CvodeSettings_setForwAdjTimeSeriesFromData   */
static int read_time(char *file, double *timepoints)
{    
  FILE *fp;
  char *line, *token;
  int i;

  /* open file */
  if ( (fp = fopen(file, "r")) == NULL )
    fatal(stderr, "read_time(): file not found");

  /* find data lines */
  for ( i=0; (line = get_line(fp)) != NULL; i++ )
  {  
    /* column 0 */
    token = strtok(line, " ");
    /* skip empty lines and comment lines (including header line) */
    if ( token == NULL || *token == '#' )
    {
      free(line);
      i--;
      continue;
    }
    sscanf(token, "%lf",  &(timepoints[i]));  
    free(line);
  }

  /* free */
  if ( fp != 0 ){
	fclose(fp);
  }

  return i;
}


/** Reads experimental data time points and use them in cvodeSettings.
    TimeSteps are given by the time values of data, with InterStep number of time points in
    between each 'data' time.
    Assigns memory for (forward and adjoint) arrays of requested time points with size PrintStep + 1 
    (including initial time 0). Returns 1, if sucessful and 0, if not. */

SBML_ODESOLVER_API int CvodeSettings_setForwAdjTimeSeriesFromData(cvodeSettings_t *set, char *TimeSeriesData_file, int InterStep)
{
  int i, n_time, OffSet, TotalNumStep;
  double *DataTimePoints;  
  double ZeroTol = 1e-5, NextDataTime, TimeStep;
  div_t d;

 if ( set->TimePoints != NULL )
  free(set->TimePoints);

  /* count number of lines */
  n_time = read_columns(TimeSeriesData_file, 0, NULL, NULL, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(DataTimePoints, n_time, double, 0);
  
  /* read time data */
  read_time(TimeSeriesData_file, DataTimePoints);

  OffSet = 0;
  if ( fabs(DataTimePoints[0] - 0.0) > ZeroTol  )
    OffSet = 1;
 
  TotalNumStep = (n_time-1) * (1+InterStep) + 1 + OffSet;
  ASSIGN_NEW_MEMORY_BLOCK(set->TimePoints, TotalNumStep, double, 0);

  for (i=0; i< TotalNumStep-OffSet; i++)
  {  
    set->TimePoints[0] = 0.0;
    d = div(i, 1+InterStep);

    if( d.rem == 0){
      set->TimePoints[OffSet+i] = DataTimePoints[d.quot];
    }
    else{

      if (d.quot == n_time-1 ){
	NextDataTime =  DataTimePoints[d.quot];
      }
      else{
        NextDataTime =  DataTimePoints[d.quot+1];
      }


      TimeStep = NextDataTime - DataTimePoints[d.quot];
      set->TimePoints[OffSet+i] = DataTimePoints[d.quot] +
	((double) d.rem/(1+InterStep) * TimeStep);
    }

  }
  free(DataTimePoints);
  set->PrintStep = TotalNumStep-1;
  set->Time = ((double) set->TimePoints[set->PrintStep]);
  set->OffSet = OffSet;
  set->InterStep = InterStep; 


 if (set->AdjTimePoints != NULL)
  free(set->AdjTimePoints);

 ASSIGN_NEW_MEMORY_BLOCK(set->AdjTimePoints, TotalNumStep, double, 0);

 for ( i=0; i<TotalNumStep; i++ )  
     set->AdjTimePoints[i] = ((double) set->TimePoints[TotalNumStep-i-1]);

 set->AdjTime= 0.0;
 set->AdjPrintStep =  set->PrintStep; 

  return 1;
}


SBML_ODESOLVER_API int CvodeSettings_setTimePointsFromExpm(cvodeSettings_t *set, time_series_t *expm, int InterStep)
{
  int i, n_time, OffSet, TotalNumStep;
  double *DataTimePoints;  
  double ZeroTol = 1e-5, NextDataTime, TimeStep;
  div_t d;

  if (set->TimePoints != NULL)
    free(set->TimePoints);
  
  /* count number of lines */
  n_time = expm->n_time;
  ASSIGN_NEW_MEMORY_BLOCK(DataTimePoints, n_time, double, 0);
  
  /* read time data */
  for ( i=0; i<n_time; i++ )
      DataTimePoints[i] = expm->time[i];

  OffSet = 0;
  if ( fabs(DataTimePoints[0] - 0.0) > ZeroTol  )
    OffSet = 1;
 
  TotalNumStep = (n_time-1) * (1+InterStep) + 1 + OffSet;
  ASSIGN_NEW_MEMORY_BLOCK(set->TimePoints, TotalNumStep, double, 0);

  for ( i=0; i< TotalNumStep-OffSet; i++ )
  {  

    set->TimePoints[0] = 0.0;
    d = div(i, 1+InterStep);

    if( d.rem == 0 )
    {
      set->TimePoints[OffSet+i] = DataTimePoints[d.quot];
    }
    else
    {
      if ( d.quot == n_time-1 )
      {
	NextDataTime =  DataTimePoints[d.quot];
      }
      else
      {
        NextDataTime =  DataTimePoints[d.quot+1];
      }


      TimeStep = NextDataTime - DataTimePoints[d.quot];
      set->TimePoints[OffSet+i] = DataTimePoints[d.quot] +
	((double) d.rem/(1+InterStep) * TimeStep);
    }

  }
  
  free(DataTimePoints);
  
  set->PrintStep = TotalNumStep-1;
  set->Time = ((double) set->TimePoints[set->PrintStep]);
  set->OffSet = OffSet;
  set->InterStep = InterStep; 


 if ( set->AdjTimePoints != NULL )
  free(set->AdjTimePoints);

 ASSIGN_NEW_MEMORY_BLOCK(set->AdjTimePoints, TotalNumStep, double, 0);

 for ( i=0; i<TotalNumStep; i++ )  
     set->AdjTimePoints[i] = ((double) set->TimePoints[TotalNumStep-i-1]);
 
 set->AdjTime= 0.0;
 set->AdjPrintStep =  set->PrintStep; 
 
 return 1;
}


/** Sets the ith time step for the integration, where
    0 < i <= PrintStep.
    
    Returns 1, if sucessful and 0, if not.\n
    The first time is always 0 and can not be set at the moment, as
    an SBML input file is supposed to start with 0.
*/

SBML_ODESOLVER_API int CvodeSettings_setTimeStep(cvodeSettings_t *set, int i, double time)
{
  if ( 0 < i && i <= set->PrintStep )
  {
    set->TimePoints[i] = time;
    return 1;
  }
  else
    return 0;
}


/** Sets use of generated Jacobian matrix (i=1) or
    of CVODE's internal approximation (i=0).

    If construction of the Jacobian matrix fails, the internal
    approximation will be used in any case!
*/

SBML_ODESOLVER_API void CvodeSettings_setJacobian(cvodeSettings_t *set, int i)
{
  set->UseJacobian = i;
}


/** Sets indefinite integration (i=1).

    For indefinite integration
    Time will be used as integration step and PrintStep will
    be ignored.
*/

SBML_ODESOLVER_API void CvodeSettings_setIndefinitely(cvodeSettings_t *set, int i)
{ /* set->Time will be used and PrintStep */
  set->Indefinitely = i;
}


/** Sets event handling, stop (1, default) or don't stop (0) the
    integration upon triggering of an event.

    If i==1, the integration will stop upon
    detection of an event and evaluation of event assignments.
    
    If i==0 the integration continues after evaluation of event
    assignments. CAUTION: the accuracy of event evaluations depends
    on the chosen printstep values!
*/

SBML_ODESOLVER_API void CvodeSettings_setHaltOnEvent(cvodeSettings_t *set, int i)
{
  set->HaltOnEvent = i;
}

/** Sets discontinuity handling internals, if set to 1 (default!)
    the CVODES solver will be re-initialized whenever the r.h.s.
    of the ODE system is changed (by an event assignment or the user).

    If the user is certain that the discontinuous changes applied
    won't affect the results, this option can be set to 0 and the
    solver will only be initialized for changes of the
    l.h.s. (i.e. y(t)) of the model, where this is absolutely necessary.    
*/

SBML_ODESOLVER_API void CvodeSettings_setResetCvodeOnEvent(cvodeSettings_t *set, int i)
{
  set->ResetCvodeOnEvent = i;
}

/** Sets steady state handling (replacing
    CvodeSettings_setSteadyState): if set to 1, the integration will
    stop upon an approximate detection of a steady state, which is
    here defined as some threshold value of the mean value and
    standard deviation of current ODE values. A warnig message will be
    issued via SOSlib error management. This functions also sets the
    default threshold for steady state detection to 1e-11. An
    alternative threshold can be set via
    CvodeSettings_setSteadyStateThreshold.
*/


SBML_ODESOLVER_API void CvodeSettings_setHaltOnSteadyState(cvodeSettings_t *set, int i)
{
  set->SteadyState = i;
  CvodeSettings_setSteadyStateThreshold(set, 1e-11);
}

/** Sets steady state threshold: The mean value + standard deviation
    of the ODE values must be lower then this threshold. If steady
    state detection is switched on, the passed value will be used as a
    threshold for aborting simulation. If steady state detection is
    switched off, the function IntegratorInstance_checkSteadyState can
    be used by the calling application and will return 1, if a steady
    state is detected.    
*/

SBML_ODESOLVER_API void CvodeSettings_setSteadyStateThreshold(cvodeSettings_t *set, double ssThreshold)
{
  set->ssThreshold = ssThreshold;
}

/** Results will only be stored, if i==1 and if a finite integration
    has been chosen (default value or
    CvodeSettings_setIndefinitely(settings, 0)). The results can be
    retrieved after integration has been finished. If i==0 or infinite
    integration has been chosen, results can only be retrieved during
    integration via variableIndex interface or dump functions for the
    integratorInstance.
*/

SBML_ODESOLVER_API void CvodeSettings_setStoreResults(cvodeSettings_t *set, int i)
{
  set->StoreResults = i;
}



/** Activate sensitivity analysis with 1; also sets to default
    sensitivity method `simultaneous' (setSensMethod(set, 0));
*/

SBML_ODESOLVER_API void CvodeSettings_setSensitivity(cvodeSettings_t *set, int i)
{
  set->Sensitivity = i;
  CvodeSettings_setSensMethod(set, 0);
}



/** Set a list of SBML IDs of model constants or ODE variables
    for sensitivity analysis; if NULL is passed instead of a character
    array a former setting is freed and default sensitivty analysis
    (for all model constants, but not for initial conditions) will be
    performed.
*/

SBML_ODESOLVER_API int CvodeSettings_setSensParams(cvodeSettings_t *set, char **sensIDs, int nsens)
{ 
  int i;

  CvodeSettings_unsetSensParams(set);
  
  if ( sensIDs != NULL )
  {
    ASSIGN_NEW_MEMORY_BLOCK(set->sensIDs, nsens, char *, 0);
    for ( i=0; i<nsens; i++ )
    {
      ASSIGN_NEW_MEMORY_BLOCK(set->sensIDs[i],
			      strlen(sensIDs[i])+1, char, 0);
      strcpy(set->sensIDs[i], sensIDs[i]);
    }
    set->nsens = nsens;
  }
  return 1;
}

/** De-activate defined sensitivities, and activated default
    sensitivity for all parameters of the model */
SBML_ODESOLVER_API void CvodeSettings_unsetSensParams(cvodeSettings_t *set)
{
  int i;
  if ( set->sensIDs != NULL )
    for ( i=0; i<set->nsens; i++ )
      free(set->sensIDs[i]);
  free(set->sensIDs);
  set->sensIDs = NULL;
  set->nsens = 0;  
}

/** Set method for sensitivity analysis:
    0: simultaneous 1: staggered, 2: staggered1.    
*/


SBML_ODESOLVER_API void CvodeSettings_setSensMethod(cvodeSettings_t *set, int i)
{
  if ( 0 <= i && i < 3 ) set->SensMethod = i;
  else set->SensMethod = 0;
}


/**** cvodeSettings get methods ****/

/** Returns the last time point of integration or -1, if
    Indefinitely is set to TRUE (1);   
*/

SBML_ODESOLVER_API double CvodeSettings_getEndTime(cvodeSettings_t *set)
{
  if ( !set->Indefinitely ) return set->Time;
  else return -1.;
}


/** Returns the time step of integration; if infinite integration has
    been chosen or a non-uniform time series has been set, this is
    only the first time step.
*/

SBML_ODESOLVER_API double CvodeSettings_getTimeStep(cvodeSettings_t *set)
{
  if ( !set->Indefinitely ) return set->TimePoints[1];
  else return set->Time;
}


/**  Returns the number of integration steps or -1, if
     infinite integration has been chosen
*/

SBML_ODESOLVER_API int CvodeSettings_getPrintsteps(cvodeSettings_t *set)
{
  if ( !set->Indefinitely ) return set->PrintStep;
  else return -1;
}


/** Returns the time of the ith time step, where
    0 <= i < PrintStep, unless
    infinite integration has been chosen
*/

SBML_ODESOLVER_API double CvodeSettings_getTime(cvodeSettings_t *set, int i)
{
  if ( !set->Indefinitely ) return set->TimePoints[i];
  else return i * set->Time;
}


/**  Returns the absolute error tolerance
*/

SBML_ODESOLVER_API double CvodeSettings_getError(cvodeSettings_t *set)
{
  return set->Error;
}


/** Returns the relative error tolerance
*/

SBML_ODESOLVER_API double CvodeSettings_getRError(cvodeSettings_t *set)
{
  return set->RError;
}


/** Returns the maximum number of internal time steps taken
    by CVODE to reach the next output time (printstep)
*/

SBML_ODESOLVER_API int CvodeSettings_getMxstep(cvodeSettings_t *set)
{
  return set->Mxstep;
}

/** returns whether the simulator will use compiled functions to compute ODEs, the Jacbian or events
*/
SBML_ODESOLVER_API int CvodeSettings_getCompileFunctions(cvodeSettings_t *set)
{
  return set->compileFunctions;
}

/** returns whether the CVODE integrator will be freed and restarted eveytime a event occurs
*/
SBML_ODESOLVER_API int CvodeSettings_getResetCvodeOnEvent(cvodeSettings_t *set)
{
  return set->ResetCvodeOnEvent;
}

/** Get non-linear solver method (BDF or ADAMS-MOULTON)
*/

SBML_ODESOLVER_API char *CvodeSettings_getMethod(cvodeSettings_t *set)
{
  char *meth[2];
  meth[0] = "BDF";
  meth[1] = "ADAMS-MOULTON";
  return meth[set->CvodeMethod];
}

/** Get maximum order of non-linear solver method
*/

SBML_ODESOLVER_API int CvodeSettings_getMaxOrder(cvodeSettings_t *set)
{
  return set->MaxOrder;
}

/** Get non-linear solver iteration type (NEWTON or FUNCTIONAL)
*/

SBML_ODESOLVER_API char *CvodeSettings_getIterMethod(cvodeSettings_t *set)
{
  char *meth[2];
  meth[0] = "NEWTON";
  meth[1] = "FUNCTIONAL";
  return meth[set->IterMethod];
}

/** Returns 1, if the automatically generated
    or 0 if CVODE's internal approximation
    of the jacobian matrix will be used by CVODE 
*/

SBML_ODESOLVER_API int CvodeSettings_getJacobian(cvodeSettings_t *set)
{
  return set->UseJacobian;
}


/** Returns 1, if infinite integration has been chosen,
    and 0 otherwise
*/

SBML_ODESOLVER_API int CvodeSettings_getIndefinitely(cvodeSettings_t *set)
{
  return set->Indefinitely;
}


/** Returns 1, if integration should stop upon an event trigger
    and 0 if integration should continue after evaluation of
    event assignments
*/

SBML_ODESOLVER_API int CvodeSettings_getHaltOnEvent(cvodeSettings_t *set)
{
  return set->HaltOnEvent;
}


/** Returns 1, if integration should stop upon detection of a
    steady state, and 0 if integration should continue 
*/

SBML_ODESOLVER_API int CvodeSettings_getHaltOnSteadyState(cvodeSettings_t *set)
{
  return set->SteadyState;
}

/** Returns to threshold used in steady state detection via
    IntegratorInstance_checkSteadyState */

SBML_ODESOLVER_API double CvodeSettings_getSteadyStateThreshold(cvodeSettings_t *set)
{
  return set->ssThreshold;
}

  
/** Returns 1, if integration results should be stored internally,
    and 0 if not.

    If set to 0 current values can be retrieved during an integration
    loop, and the values at the end time of integration afterwards.
*/

SBML_ODESOLVER_API int CvodeSettings_getStoreResults(cvodeSettings_t *set)
{
  return set->StoreResults;
}


/** Returns 1, if sensitivity analysis is requested and CVODES
    will be used.
*/

SBML_ODESOLVER_API int CvodeSettings_getSensitivity(cvodeSettings_t *set)
{
  return set->Sensitivity;
}


/** Get sensitivity method `simultaneous', `staggered'  or `staggered1'
*/

SBML_ODESOLVER_API char *CvodeSettings_getSensMethod(cvodeSettings_t *set)
{
  char *meth[3];
  meth[0] = "simultaneous";
  meth[1] = "staggered";
  meth[2] = "staggered1";
  return meth[set->SensMethod];
}


/** Frees cvodeSettings.
*/

SBML_ODESOLVER_API void CvodeSettings_free(cvodeSettings_t *set)
{
  int i;
  
  if ( set->TimePoints != NULL ) free(set->TimePoints);
  if ( set->AdjTimePoints != NULL ) free(set->AdjTimePoints);
  if ( set->sensIDs != NULL )    
    for ( i=0; i<set->nsens; i++ )
      free(set->sensIDs[i]);
  free(set->sensIDs);
  free(set);
}

/** @} */

/** Creates a settings structure from a timeSettings structure
    and fills rest with default values
*/

cvodeSettings_t *CvodeSettings_createFromTimeSettings(timeSettings_t *time)
{
  return CvodeSettings_createWithTime(time->tend, time->nout);
}



/* for when timeSettings might become a separate structure,
 not used at the moment */
timeSettings_t *TimeSettings_create(double t0, double tend, int nout) {

  timeSettings_t *time;
  time = 0;
  
  time = (timeSettings_t *)calloc(1, sizeof(timeSettings_t));

  if ( time != 0 ) {
	time->t0 = t0;
	  time->tend = tend;
	  time->nout = nout;
	time->tmult = (t0-tend) / nout;
  }
  return time;				  

}

void
TimeSettings_free(timeSettings_t *time)
{
  free(time);
}




/* End of file */
