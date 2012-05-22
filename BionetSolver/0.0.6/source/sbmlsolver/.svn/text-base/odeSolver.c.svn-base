/*
  Last changed Time-stamp: <2008-10-16 18:48:01 raim>
  $Id: odeSolver.c,v 1.48 2008/10/16 17:27:50 raimc Exp $
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
 *     Stefanie Widder
 *     Christoph Flamm
 *     Akira Funahashi
 *     Stefan Müller
 *     Andrew Finney
 *     Norihiro Kikuchi
 */
/*! \defgroup odeSolver High Level Interfaces
    \brief This module contains high level interfaces to SOSlib, which
    take an SBML model and integratorSettings as input and return results
    mapped back to SBML structures.

    Please see sbmlResults for interfaces to the returned result
    structures. */
/*@{*/

#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include <string.h>
#include <time.h>

#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

#include "sbmlsolver/odeSolver.h"
#include "sbmlsolver/solverError.h"

static int globalizeParameter(Model_t *, char *id, char *rid);
static int localizeParameter(Model_t *, char *id, char *rid);
static int SBMLResults_createSens(SBMLResults_t *, cvodeData_t *);

/** Solves the timeCourses for a SBML model, passed via a libSBML
    SBMLDocument structure and according to passed integration
    settings and returns the SBMLResults structure.
*/

SBML_ODESOLVER_API SBMLResults_t *SBML_odeSolver(SBMLDocument_t *d, cvodeSettings_t *set)
{

  SBMLDocument_t *d2 = NULL;
  Model_t *m = NULL;
  SBMLResults_t *results;
  
  /** Convert SBML Document level 1 to level 2, and
      get the contained model   */
  if ( SBMLDocument_getLevel(d) != 2 )
  {
    d2 = convertModel(d);
    if ( d2 == NULL ) return NULL;
    m = SBMLDocument_getModel(d2);    
  }
  else m = SBMLDocument_getModel(d);

  if ( m == NULL )
  {
   if ( d2 != NULL ) SBMLDocument_free(d2);
   return NULL;
  }

  /** Call Model_odeSolver */
  results = Model_odeSolver(m, set);
  
  /** free temporary level 2 version of the document */
  if ( d2 != NULL ) SBMLDocument_free(d2);
  
  return(results);
  
}


/** Solves the timeCourses for a SBML model, passed via a libSBML
    SBMLDocument structure and according to passed 
    integration and parameter variation settings and returns
    the SBMLResultsMatrix containing SBMLResults for each
    varied parameter (columns) and each of its values (rows).
*/

SBML_ODESOLVER_API SBMLResultsArray_t *SBML_odeSolverBatch(SBMLDocument_t *d, cvodeSettings_t *set, varySettings_t *vs) 
{

  SBMLDocument_t *d2 = NULL;
  Model_t *m = NULL;
  SBMLResultsArray_t *resA;
  
  /** Convert SBML Document level 1 to level 2, and
      get the contained model
  */
  if ( SBMLDocument_getLevel(d) != 2 )
  {
    d2 = convertModel(d);
    if ( d2 == NULL ) return NULL;
    m = SBMLDocument_getModel(d2);
  }
  else m = SBMLDocument_getModel(d);

  if ( m == NULL )
  {
   if ( d2 != NULL ) SBMLDocument_free(d2);
   return NULL;
  }

  /** Call Model_odeSolverBatch */  
  resA = Model_odeSolverBatch(m, set, vs);
  /** free temporary level 2 version of the document */
  if ( d2 != NULL ) SBMLDocument_free(d2);
  
  return resA;
    
}


/** Solves the timeCourses for a SBML model, passed via a libSBML
    Model_t  (must be level 2 SBML!!) and according to passed
    integration settings and returns the SBMLResults structure.

*/

SBML_ODESOLVER_API SBMLResults_t *Model_odeSolver(Model_t *m, cvodeSettings_t *set)
{
  odeModel_t *om;
  integratorInstance_t *ii; 
  SBMLResults_t *results;
  int errorCode = 0;
  
  /** At first, ODEModel_create, attempts to construct a simplified
     SBML model with reactions replaced by ODEs. SBML RateRules,
     AssignmentRules,AlgebraicRules and Events are copied to the
     simplified model. AlgebraicRules or missing mathematical
     expressions produce fatal errors and appropriate messages. All
     function definitions are replaced by their values or expressions
     respectively in all remaining formulas (ie. rules and events). If
     that conversion was successful, an internal model structure
     (odeModel) is created, that contains indexed versions of all
     formulae (AFM's AST_INDEX) where the index of a former AST_NAME
     corresponds to its position in a value array (double *), that is
     used to store current values and to evaluate AST formulae during
     integration.
  */

  om = ODEModel_create(m);      
  if ( om == NULL ) return NULL;
  
  /**
     Second, an integratorInstance is created from the odeModel
     and the passed cvodeSettings. If that worked out ...
  */
  
  ii = IntegratorInstance_create(om, set);
  if ( ii == NULL )
  {
    ODEModel_free(om);
    return NULL;
  }

  /** .... the integrator loop can be started,
      that invoking CVODE to move one time step and store.
      The function will also handle events and
      check for steady states.
  */
  while ( !IntegratorInstance_timeCourseCompleted(ii) && !errorCode )
    if ( !IntegratorInstance_integrateOneStep(ii) )
      break;
  
  /* !!! on fatals: above created structures should be freed before return
     !!! */
  /* RETURN_ON_FATALS_WITH(NULL);  */

  /** finally, map cvode results back to SBML compartments, species
      and parameters  */
  results = SBMLResults_fromIntegrator(m, ii);

  /* free integration data */
  IntegratorInstance_free(ii);
  /* free odeModel */
  ODEModel_free(om);
  
  /* ... well done. */
  return(results);
}


/** Solves the timeCourses for a SBML model, passed via a libSBML
    Model_t (must be level 2 SBML!!) structure and according to passed 
    integration and parameter variation settings and returns
    the SBMLResultsArray containing SBMLResults for each
    parameter combination
*/


SBML_ODESOLVER_API SBMLResultsArray_t *Model_odeSolverBatch(Model_t *m, cvodeSettings_t *set, varySettings_t *vs)
{
  int i, j; 
  odeModel_t *om;
  integratorInstance_t *ii;
  variableIndex_t **vi = NULL;
  SBMLResultsArray_t *resA;

  char *local_param;
  
  int errorCode = 0;


  resA = SBMLResultsArray_allocate(vs->nrdesignpoints);
  if ( resA == NULL ) return NULL;
 
  /** At first, globalize all local (kineticLaw) parameters to be varied */
  for ( i=0; i<vs->nrparams; i++ ) 
    /* ** modified after suggestion by Norihiro Kikuchi ** */
    if ( vs->rid[i] != NULL && strlen(vs->rid[i]) > 0 ) 
      globalizeParameter(m, vs->id[i], vs->rid[i]);     
    
 
  /** Then create internal odeModel: attempts to construct a simplified
     SBML model with reactions replaced by ODEs.
     See comments in Model_odeSolver for details.  */
  om = ODEModel_create(m);      
  if ( om == NULL )
  {
    
    /** localize parameters again, unfortunately the new globalized
	parameter cannot be freed currently  */
    for ( i=0; i<vs->nrparams; i++ )
      /*  ** modified after suggestion by Norihiro Kikuchi ** */
      if ( vs->rid[i] != NULL  && strlen(vs->rid[i]) > 0 ) 
	localizeParameter(m, vs->id[i], vs->rid[i]);     
    SBMLResultsArray_free(resA);
    return NULL;
  }
 
  /** an integratorInstance is created from the odeModel and the passed
     cvodeSettings. If that worked out ...  */  
  ii = IntegratorInstance_create(om, set);
  if ( ii == NULL )
  {
    
    /** localize parameters again, unfortunately the new globalized
	parameter cannot be freed currently  */
    for ( i=0; i<vs->nrparams; i++ )
      /*  ** modified after suggestion by Norihiro Kikuchi ** */
      if ( vs->rid[i] != NULL  && strlen(vs->rid[i]) > 0 ) 
	localizeParameter(m, vs->id[i], vs->rid[i]);     
    SBMLResultsArray_free(resA);
    ODEModel_free(om);
    return NULL;
  }

  ASSIGN_NEW_MEMORY_BLOCK(vi, vs->nrparams, struct variableIndex *, NULL);
			  
  for ( j=0; j<vs->nrparams; j++ )
  {
    /* get the index for parameter i
    ** modified after suggestion by Norihiro Kikuchi ** */ 
    if ( vs->rid[j] != NULL  && strlen(vs->rid[j]) > 0 )
    {
      ASSIGN_NEW_MEMORY_BLOCK(local_param,
			      strlen(vs->id[j]) + strlen(vs->rid[j]) + 4,
			      char , 0);
      sprintf(local_param, "r_%s_%s", vs->rid[j], vs->id[j]);
      
      vi[j] = ODEModel_getVariableIndex(om, local_param);
      free(local_param);
    }
    else
      vi[j] = ODEModel_getVariableIndex(om, vs->id[j]);

    /* if ( vi[j] == NULL ) return NULL; */ /*!!! TODO : handle NULL */
  }
      
  /** now, work through the passed designpoints in varySettings */
  for ( i=0; i<vs->nrdesignpoints; i++ )
  {
    for ( j=0; j<vs->nrparams; j++ )
      IntegratorInstance_setVariableValue(ii, vi[j], vs->params[i][j]);
    
    while ( !IntegratorInstance_timeCourseCompleted(ii) && !errorCode )
      if ( !IntegratorInstance_integrateOneStep(ii) )
	break;
    /*!!! TODO : on fatals: above created structures should be freed
       !!before return ! */
    /* RETURN_ON_FATALS_WITH(NULL); */
        
    /** map cvode results back to SBML compartments, species and
	parameters  */
    resA->results[i] = SBMLResults_fromIntegrator(m, ii);
    IntegratorInstance_reset(ii);

  }
  
  /* free variableIndex, used for setting values */
  for ( j=0; j<vs->nrparams; j++ )
    VariableIndex_free(vi[j]);
  free(vi);

  /** localize parameters again, unfortunately the new globalized
     parameter cannot be freed currently  */
  for ( i=0; i<vs->nrparams; i++ )
    /*  ** modified after suggestion by Norihiro Kikuchi ** */
    if ( vs->rid[i] != NULL  && strlen(vs->rid[i]) > 0 ) 
      localizeParameter(m, vs->id[i], vs->rid[i]);     

 
  /* free integration data */
  IntegratorInstance_free(ii);
  /* free odeModel */
  ODEModel_free(om);
  /* ... well done. */
  return(resA);

}

static int globalizeParameter(Model_t *m, char *id, char *rid)
{
  int i, found;
  Reaction_t *r;
  KineticLaw_t *kl;
  Parameter_t *p, *p_global;
  ASTNode_t *math;
  char *newname;
 
  r = Model_getReactionById (m, (const char *) rid);
  
  if ( r == NULL ) return(0);
  
  kl = Reaction_getKineticLaw(r);
  math = (ASTNode_t *)KineticLaw_getMath(kl);

  ASSIGN_NEW_MEMORY_BLOCK(newname, strlen(id) + strlen(rid) + 4, char , 0);
  sprintf(newname, "r_%s_%s", rid, id);
  AST_replaceNameByName(math, (const char *) id,  (const char *) newname);

  found = 0;
  
  for ( i=0; i<KineticLaw_getNumParameters(kl); i++ )
  {
    p = KineticLaw_getParameter(kl, i);
    if ( strcmp(Parameter_getId(p), id) == 0 )
    {
      p_global = Parameter_clone(p);
      Parameter_setId(p_global, newname);
      Model_addParameter(m, p_global);
      Parameter_free(p_global);
      found = 1;
    }
  }
  free(newname);  
  return (found);
}

static int localizeParameter(Model_t *m, char *id, char *rid)
{
  int found;
  Reaction_t *r;
  KineticLaw_t *kl;
  ListOf_t     *pl;
  Parameter_t *p;
  ASTNode_t *math;
  char *newname;
  
  r = Model_getReactionById (m, (const char *) rid);
  
  if ( r == NULL ) return 0;
  
  kl = Reaction_getKineticLaw(r);
  math = (ASTNode_t *)KineticLaw_getMath(kl);  
  ASSIGN_NEW_MEMORY_BLOCK(newname, strlen(id) + strlen(rid) + 4, char , 0);
  sprintf(newname, "r_%s_%s", rid, id);
  AST_replaceNameByName(math, (const char *) newname, (const char *) id);

  /* just freeing the last parameter, one for each `rid',
     only if the globalized parameter is present */
  found = 0;
  if ( Model_getParameterById(m, newname) != NULL )
  {
    found = 1;
    pl = Model_getListOfParameters(m);
    p = (Parameter_t *) ListOf_remove(pl, ListOf_size(pl) - 1);
    Parameter_free(p);
  }  

  free(newname);
  return found;

}

/** @} */


/*! \addtogroup sbmlResults */
/*@{*/

/** Maps the integration results from internal data structures
    back to SBML structures (compartments, species, parameters
    and reaction fluxes)
*/


SBML_ODESOLVER_API SBMLResults_t *SBMLResults_fromIntegrator(Model_t *m, integratorInstance_t *ii)
{
  int i, j, k, flag;
  Reaction_t *r;
  KineticLaw_t *kl;
  ASTNode_t **kls;
  timeCourseArray_t *tcA;
  timeCourse_t *tc;
  SBMLResults_t *sbml_results;

  odeModel_t *om = ii->om;
  cvodeData_t *data = ii->data;
  cvodeResults_t *cv_results = ii->results;

  /* check if data is available */
  if ( data == NULL ) return(NULL);
  else if ( cv_results == NULL ) return(NULL);

  sbml_results = SBMLResults_create(m, cv_results->nout+1);

  /* Allocating temporary kinetic law ASTs, for evaluation of fluxes */

  ASSIGN_NEW_MEMORY_BLOCK(kls, Model_getNumReactions(m), ASTNode_t *, NULL);

  for ( i=0; i<Model_getNumReactions(m); i++ )
  {
    r = Model_getReaction(m, i);
    kl = Reaction_getKineticLaw(r);
    kls[i] = copyAST(KineticLaw_getMath(kl));
    AST_replaceNameByParameters(kls[i], KineticLaw_getListOfParameters(kl));
    AST_replaceConstants(m, kls[i]);
  }
  
  
  /*  filling results for each calculated timepoint.  */
  for ( i=0; i<sbml_results->time->timepoints; i++ )
  {    
    /* writing time steps */
    sbml_results->time->values[i] = cv_results->time[i];
    
    /* updating time and values in cvodeData_t *for calculations */
    data->currenttime = cv_results->time[i]; 
    for ( j=0; j<data->nvalues; j++ ) data->value[j] = cv_results->value[j][i];

    /* filling time courses for SBML species  */
    tcA = sbml_results->species;  
    for ( j=0; j<tcA->num_val; j++ )
    {
      tc = tcA->tc[j];
      /* search in cvodeData_t for values */
      for ( k=0; k<data->nvalues; k++ )
	if ( (strcmp(tc->name, om->names[k]) == 0) )
	  tc->values[i] = cv_results->value[k][i];
    }
    
    /* filling variable compartment time courses */
    tcA = sbml_results->compartments;  
    for ( j=0; j<tcA->num_val; j++ )
    {
      tc = tcA->tc[j];
      /* search in cvodeData_t for values */
      for ( k=0; k<data->nvalues; k++ )
	if ( (strcmp(tc->name, om->names[k]) == 0) )
	  tc->values[i] = cv_results->value[k][i];
    }         

    /* filling variable parameter time courses */
    tcA = sbml_results->parameters;  
    for ( j=0; j<tcA->num_val; j++ )
    {
      tc = tcA->tc[j];
      /* search in cvodeData_t for values */
      for ( k=0; k<data->nvalues; k++ ) 
	if ( (strcmp(tc->name, om->names[k]) == 0) ) 
	  tc->values[i] = cv_results->value[k][i];
    }

    /* filling reaction flux time courses */
    tcA = sbml_results->fluxes;
    for ( j=0; j<tcA->num_val; j++ )
    {
      tc = tcA->tc[j];
      tc->values[i] = evaluateAST(kls[j], data);
    }

  }

  /* freeing temporary kinetic law ASTs */
  for ( i=0; i<Model_getNumReactions(m); i++ )
    ASTNode_free(kls[i]);
  free(kls);

  /* filling sensitivities */
  flag = 0;
  if ( cv_results->nsens > 0 )
    flag = SBMLResults_createSens(sbml_results, data);
  if ( flag == 0 )
    sbml_results->nsens = 0;
   
  return(sbml_results);
}

static int SBMLResults_createSens(SBMLResults_t *Sres, cvodeData_t *data)
{
  int i, j, k;
  odeModel_t *om = data->model;
  odeSense_t *os = data->os;
  cvodeResults_t *res = data->results;
  timeCourse_t *tc;

  Sres->nsens = res->nsens;
  
  ASSIGN_NEW_MEMORY_BLOCK(Sres->param, res->nsens, char *, 0);  
  for ( i=0; i<res->nsens; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(Sres->param[i],
			    strlen(om->names[os->index_sens[i]]+1), char, 0);
    sprintf(Sres->param[i], "%s", om->names[os->index_sens[i]]);
  }
  for ( i=0; i<res->neq; i++ )
  {
    tc = SBMLResults_getTimeCourse(Sres, om->names[i]);
    ASSIGN_NEW_MEMORY_BLOCK(tc->sensitivity, res->nsens, double *, 0);
    for ( j=0; j<res->nsens; j++ )
    {
      ASSIGN_NEW_MEMORY_BLOCK(tc->sensitivity[j], res->nout, double, 0);
      for ( k=0; k<res->nout; k++ )
	tc->sensitivity[j][k] = res->sensitivity[i][j][k];
    } 
  }
  return(1);
}

/** @} */

/*! \defgroup varySettings Parameter Variation Settings
    \ingroup odeSolver
    
    \brief Create the varySettings structure with a series of
    parameter values used for the batch functions
    
*/
/*@{*/


/** Allocate varySettings structure for settings for parameter
    variation batch runs: nrparams is the number of parameters to be
    varied, and nrdesignpoints is the number of values to be tested
    for each parameter.
*/
/* NEW: rows:designpoints, columns:parameters */
SBML_ODESOLVER_API varySettings_t *VarySettings_allocate(int nrparams, int nrdesignpoints)
{
  int i;
  varySettings_t *vs;
  ASSIGN_NEW_MEMORY(vs, struct varySettings, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(vs->id, nrparams, char *, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(vs->rid, nrparams, char *, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(vs->params, nrdesignpoints, double *, NULL);
  for ( i=0; i<nrdesignpoints; i++ ) 
    ASSIGN_NEW_MEMORY_BLOCK(vs->params[i], nrparams, double, NULL);

  vs->nrdesignpoints = nrdesignpoints;
  vs->nrparams = nrparams;
  /* set conuter to 0, will be used as counter in addDesignPoints */
  vs->cnt_params = 0;
  vs->cnt_points = 0;
  return(vs);
}
/** Add a parameter to the varySettings;
    returns number of already set parameters for success,
    or 0 for failure, please check SolverError for errors.

    For local (reaction/kineticLaw)
    parameters, the reaction id must be passed as `rid'. For global
    parameters `rid' must be passed as NULL.

*/

SBML_ODESOLVER_API int VarySettings_addParameter(varySettings_t *vs, char *id, char *rid)
{
  if ( vs->cnt_params >= vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		SOLVER_ERROR_VARY_SETTINGS,
		"VarySettings_addParameter:\t"
		"Allocated parameter array already full, #%d parameters",
		vs->cnt_params);
    return 0;
  }

  VarySettings_setName(vs, vs->cnt_params, id, rid);
  /* count and return already filled parametervalues */
  return(vs->cnt_params++);  
}

/** Adds values for all parameters to be varied
    for the next design point;
    returns number of already set design points for success,
    or 0 for failure, please check SolverError for errors.
*/
SBML_ODESOLVER_API int VarySettings_addDesignPoint(varySettings_t *vs,
						   double *params)
{
  int i;
  if ( vs->cnt_points >= vs->nrdesignpoints )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		SOLVER_ERROR_VARY_SETTINGS,
		"VarySettings_addDesignPoints:\t"
		"Allocated design point array already full, #%d design points",
		vs->cnt_points);
    return 0;
  }

  for ( i=0; i<vs->nrparams; i++ )
    vs->params[vs->cnt_points][i] = params[i];
  
  return (vs->cnt_points++);
}



/** Get the name (SBML ID) of the ith parameterr;
    returns NULL for failure, please see SolverError messages.

    Passed values:      \n
    0 <= i < nrparams   \n
    as used for varySettings_allocate
*/

SBML_ODESOLVER_API const char *VarySettings_getName(varySettings_t *vs, int i)
{   
  if ( i >= vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		SOLVER_ERROR_VARY_SETTINGS,
		"VarySettings_getReactionName:\t"
		"Requested Value %d not found in varySettings"
		" # parameters: %d",
		i, vs->nrparams);
    return NULL;
  }
  return((const char *) vs->id[i]);
}



/** Get the name (SBML ID) of the reaction of the ith parameter;
    returns NULL for failure, please see SolverError messages.

    Passed values:      \n
    0 <= i < nrparams   \n
    as used for varySettings_allocate.
    Returns NULL, if the parameter is global.
*/

SBML_ODESOLVER_API const char *VarySettings_getReactionName(varySettings_t *vs, int i)
{   
  if ( i >= vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_getReactionName:\t"
		      "Requested Value %d not found in varySettings"
		      " # parameters: %d",
		      i, vs->nrparams);
    return NULL;
  }
  return((const char *) vs->rid[i]);
}



/** Set the id (SBML ID) of the ith parameter;
    returns 1 for success and returns for 0 for failure,
    please check SolverError for errors.

    Passed values: \n
    0 <= i < nrparams  \n
    as used for varySettings_allocate.  \n
    `rid' is the SBML reaction id, if a local parameter shall
    be varied. For global parameters rid must be passed as NULL.
*/

SBML_ODESOLVER_API int VarySettings_setName(varySettings_t *vs, int i, char *id, char *rid)
{

  if ( i >= vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_setName:\t"
		      "Requested value %d not found in varySettings"
		      " # parameters: %d. ID %s (reaction %s) can't be set.",
		      i, vs->nrparams, id, rid);
    return 0;
  }

  /* free if parameter nr. i has already been set */
  if ( vs->id[i] != NULL )  free(vs->id[i]);
  if ( vs->rid[i] != NULL ) free(vs->rid[i]);
  
  /* setting parameter reaction id: local parameters will be
     `globalized' in the input model */
  if ( rid != NULL  && strlen(rid) > 0 )
  {
    ASSIGN_NEW_MEMORY_BLOCK(vs->rid[i], strlen(rid)+1, char, 0);
    sprintf(vs->rid[i], "%s", rid);    
  }
  else vs->rid[i] = NULL;

  ASSIGN_NEW_MEMORY_BLOCK(vs->id[i], strlen(id)+1, char, 0);
  sprintf(vs->id[i], "%s", id);
  
  return(1);
}

/** Get the jth value of the ith parameter;
    WARNING: returns 0 for failure,
    please check SolverError for errors.

    Passed values: \n
    0 <= i < nrdesignpoints \n
    0 <= j < nrparams  and \n
    as used for varySettings_allocate
*/

SBML_ODESOLVER_API double VarySettings_getValue(varySettings_t *vs, int i, int j)
{
  if ( i >= vs->nrdesignpoints )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_getValue:\t"
		      "Requested design points #%d not found in varySettings"
		      " # design points: %d",
		      i, vs->nrdesignpoints);
    return 0;
  }

  if ( j >= vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_setValue:\t"
		      "Requested value #%d not found in varySettings"
		      " # parameters: %d",
		      i, vs->nrparams);
   return 0;
  }
  
  return(vs->params[i][j]);
}


/** Set the jth value of the ith parameter,
    returns 1 for success, 0 for failure

    where \n
    0 <= i < nrdesignpoints \n
    0 <= j < nrparams  and \n
    as used for varySettings_allocate
*/

SBML_ODESOLVER_API int VarySettings_setValue(varySettings_t *vs, int i, int j, double value)
{
  if ( i >= vs->nrdesignpoints )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_setValue:\t"
		      "Requested design points #%d not found in varySettings"
		      " # design points: %d",
		      i, vs->nrdesignpoints);
    return 0;
  }

  if ( j >= vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_setValue:\t"
		      "Requested value %d not found in varySettings"
		      " # parameters: %d",
		      i, vs->nrparams);
    return 0;
  }
  vs->params[i][j] = value;
  return 1;
}
/** Get the jth value of the ith parameter,
    WARNING: returns 0 for failure,
    please check SolverError for errors

    where \n
    0 <= i < nrdesignpoints \n
    0 <= j < nrparams  and \n
    as used for varySettings_allocate
*/

SBML_ODESOLVER_API double VarySettings_getValueByID(varySettings_t *vs, int i, char *id, char *rid)
{
  int j;
  if ( i >= vs->nrdesignpoints )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_getValueByID:\t"
		      "Requested design points #%d not found in varySettings"
		      " # design points: %d",
		      i, vs->nrdesignpoints);
    return 0;
  }
  
  for ( j=0; j<vs->nrparams; j++ )
  {
    if ( !strcmp(id, vs->id[j]) && !strcmp(rid, vs->rid[j]) )
      break;
  }
  if ( j == vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_getValueByID:\t"
		      "Requested ID %s (reaction %s) not found in varySettings",
		      id, rid);
    return 0;
  }
  return(vs->params[i][j]);
}


/** Set the jth value of the ith parameter,
    returns 1 for success, 0 for failure

    Passed values: \n
    0 <= i < nrdesignpoints \n
    as used for varySettings_allocate and \n
    id:  SBML ID of parameter \n
    rid: SBML ID of reaction for local parameters\n
*/

SBML_ODESOLVER_API int VarySettings_setValueByID(varySettings_t *vs, int i, char *id, char *rid, double value)
{
  int j;
  if ( i >= vs->nrdesignpoints )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_setValueByID:\t"
		      "Requested design points #%d not found in varySettings"
		      " # design points: %d",
		      i, vs->nrdesignpoints);
    return 0;
  }
  
  for ( j=0; j<vs->nrparams; j++ )
  {
    if ( !strcmp(id, vs->id[j]) && !strcmp(rid, vs->rid[j]) )
      break;
  }
  if ( j == vs->nrparams )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_VARY_SETTINGS,
		      "VarySettings_setValueByID:\t"
		      "Requested ID %s (reaction %s) not found in varySettings",
		      id, rid);
    return 0;
  }
  vs->params[i][j] = value;
  return 1;
}



/** Print all parameters and their values in varySettings
*/

SBML_ODESOLVER_API void VarySettings_dump(varySettings_t *vs)
{
  int i, j;
  printf("\n");
  printf("Design points for batch integration (#params=%i, #points=%i):\n",
	 vs->nrparams, vs->nrdesignpoints);

  printf("Run");
  for ( j=0; j<vs->nrparams; j++ )
  {
    printf("\t%s", vs->id[j]);
  }
  printf("\n");
  
  for ( i=0; i<vs->nrdesignpoints; i++ )
  {
    printf("#%d:", i);
    for ( j=0; j<vs->nrparams; j++ )
    {
       printf("\t%.3f", vs->params[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}


/** Frees varySettings structure
*/
SBML_ODESOLVER_API void VarySettings_free(varySettings_t *vs)
{
  int i;
  
  for ( i=0; i<vs->nrparams; i++ )
  {
    free(vs->id[i]);
    free(vs->rid[i]);
  }
  free(vs->id);
  free(vs->rid);
  
  for ( i=0; i<vs->nrdesignpoints; i++ )
  {
    free(vs->params[i]);
  }
  free(vs->params);
  free(vs);    
}





/*\@} */



/* End of file */
