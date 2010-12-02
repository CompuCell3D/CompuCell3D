/*
  Last changed Time-stamp: <2009-02-12 17:38:22 raim>
  $Id: cvodeData.h,v 1.16 2009/03/27 15:55:03 fbergmann Exp $
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
 *     Andrew Finney
 *     
 */

#ifndef _CVODEDATA_H_
#define _CVODEDATA_H_

typedef struct cvodeData cvodeData_t ;
typedef struct cvodeResults cvodeResults_t ;

/* Header Files for CVODE: required only for realtype *p  */
#include "nvector/nvector_serial.h"

#include <stdio.h>
#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

#include "sbmlsolver/integratorSettings.h"
#include "sbmlsolver/exportdefs.h"
#include "sbmlsolver/odeModel.h"
#include "sbmlsolver/variableIndex.h"

/** Contains the data needed for AST formula evaluation and odeModel
    integration and usually corresponds to an odeModel
      
    cvodeData is used for storing current variable, sensitivity and time
    values. It also holds the internal version of ODEs for optimized for
    a specific integration run and current (on or off) states of event
    triggers and of steady state approximation and the number of integration
    runs with this instance of cvodeData.
      
    API Interface functions to retrieve values directly from cvodeData,
    instead of via IntegratorInstance_getVariableValue()) will be available
    in one of the next releases!
*/
struct cvodeData {

  odeModel_t *model;
  /** the sensitivity structures, matrices etc. as constructed
      from odeModel_t */
  odeSense_t *os;
  /** objective function and experimental data for adjoint solver */
  objFunc_t *of;
  

  /** number of ODEs */
  int neq;

  /* current values: variables x(t), assigned and constant parameters */
  /** total number of values (variables x(t) + parameters p) */
  int nvalues; 
  /** value array is used to write and read the current values of
      all variables x(t) and parameters p of the system (of which
      there are `nvalues') */  
  double *value; 
  int allRulesUpdated; /* flag if all rules are up-to-date with current
			  variables and event assignments */

  /** the current time of the integration */
  float currenttime;

  /* current sensitivities: dx(t)/dp */
  /** number of requested sensitivities dx(t)/dp */
  int nsens;
  /** current values of sensitivities dx(t)/dp */
  double **sensitivity; 
  /** current and original values of parameters in sensitivity
      analysis, required only if no r.h.s function fS is available
      (missing Jacobi or missing parametrix matrix) */
  unsigned int use_p;
  realtype *p;
  realtype *p_orig;
  
  /** cvode settings: start- and end times, timesteps, number of timesteps,
      error tolerances, maximum number of steps, etc... */
  cvodeSettings_t *opt;
  
  /** trigger flags: check if triggers were active or not
      at the previous time step */
  int nevents;
  int *trigger;

  /** steady state flag: check if steady state was found */
  int steadystate; 

  /** Results: time series of integration are stored in this
      structure (see above) */
  cvodeResults_t *results;
  

  /* Adjoint specific  */
  /** value array is used to write and read the current values of
      all adjoint variables \psi(t) (of which there are `neq') */  
  double *adjvalue;  
 
  /* for computing vector_v using discrete observation data */
  int TimeSeriesIndex;

  /* Fisher Information Matrix */

  double** FIM;
  double* weights; /* for the inner product defining the entries of the FIM */

} ;

/** Stores CVODE specific integration results, data correspond
    to the respective values in cvodeData */
struct cvodeResults {
   /** counter for calculated time steps (without initial conditions),
       this number can be lower then the same variable in cvodeData,
       in case the integration is prematurely stopped. */
  int nout;
  /** contains the specific time steps for an integration */
  double *time;

  /** number of variables for which results exist */
  int nvalues;     
  /** the following arrays represent the time series of all variables
      and parameters of the model */
  double **value;

  /* number of variables x(t) for which sensitivities are calculated */
  int neq;
  /** number of parameters p for sens. analysis */
  int nsens;
  int *index_sens;
  /** time course of sensitivities dx(t)/dp */
  double ***sensitivity;
  /** time course of directional sensitivities \sum_i dx(t)/dp_i \delta p_i */
  double **directional;  
 
  /** Adjoint specific stuff */

  /** dimension of the adjoint solution  */
 /*  int nadjeq; */
  /**  the time series of all adjoint variables */
  double **adjvalue;

} ;

#ifdef __cplusplus
extern "C" {
#endif
  
  /* create data for formula evaluation */
  SBML_ODESOLVER_API cvodeData_t *CvodeData_create(odeModel_t *);
  SBML_ODESOLVER_API void CvodeData_free(cvodeData_t *);
  /*!!! TODO : should the initialization of data really be public???*/
  SBML_ODESOLVER_API void CvodeData_initializeValues(cvodeData_t *);
  SBML_ODESOLVER_API int CvodeData_initialize(cvodeData_t *, cvodeSettings_t *, odeModel_t *);
  
  /* get values from cvodeResults */
  SBML_ODESOLVER_API double CvodeResults_getTime(cvodeResults_t *, int);
  SBML_ODESOLVER_API double CvodeResults_getValue(cvodeResults_t *, variableIndex_t *, int);
  SBML_ODESOLVER_API int CvodeResults_getNout(cvodeResults_t *);
  SBML_ODESOLVER_API double CvodeResults_getSensitivityByNum(cvodeResults_t *,  int value, int parameter, int timestep);
  SBML_ODESOLVER_API double CvodeResults_getSensitivity(cvodeResults_t *,  variableIndex_t *y,  variableIndex_t *p, int timestep);
  SBML_ODESOLVER_API double** CvodeResults_getFisherInformationMatrix(cvodeResults_t *results); /* FIM */
  SBML_ODESOLVER_API void CvodeResults_computeDirectional(cvodeResults_t *results, double *dp);
  SBML_ODESOLVER_API void CvodeResults_free(cvodeResults_t *);

#ifdef __cplusplus
};
#endif


/* internal functions used by integratorInstance.c */
cvodeResults_t *CvodeResults_create(cvodeData_t *, int);
int CvodeResults_allocateSens(cvodeResults_t *, int neq, int nsens, int nout);
int CvodeData_initializeSensitivities(cvodeData_t *,cvodeSettings_t *,
				      odeModel_t *, odeSense_t *);
/* Adjoint specific internal functions used by integratorInstance.c */
int CvodeResults_allocateAdjSens(cvodeResults_t *, int neq, int nadjsens, int nout);

#endif

/* End of file */
