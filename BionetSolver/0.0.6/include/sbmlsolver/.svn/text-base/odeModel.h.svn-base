/*
  Last changed Time-stamp: <2009-02-12 18:15:12 raim>
  $Id: odeModel.h,v 1.55 2009/03/27 15:55:03 fbergmann Exp $ 
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
 *     Andrew Finney
 *
 * Contributor(s):
 *     Rainer Machne     
 */

#ifndef _ODEMODEL_H_
#define _ODEMODEL_H_

typedef struct odeModel odeModel_t;
typedef struct odeSense odeSense_t;
typedef struct objFunc objFunc_t;
typedef struct nonzeroElem nonzeroElem_t;
typedef int (*EventFn)(void *, int *); /* RM: replaced cvodeData_t
					    pointer with void pointer
					    because of dependency
					    problems */

#include <cvodes/cvodes.h>
#include <cvodes/cvodes_dense.h>
#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

#include "sbmlsolver/exportdefs.h"
#include "sbmlsolver/interpol.h"
#include "sbmlsolver/integratorSettings.h"
#include "sbmlsolver/compiler.h"
#include "sbmlsolver/arithmeticCompiler.h"
#include "sbmlsolver/variableIndex.h"

/** The internal ODE Model as constructed in odeModel.c from an SBML
    input file, that only contains rate rules (constructed from
    reaction network in odeConstruct.c)
*/
struct odeModel
{
  SBMLDocument_t *d; /**< not-NULL only if the odeModel was directly
			created from file */
  Model_t *m;        /**< the input SBML reaction network */
  Model_t *simple;   /**< the derived SBML with rate rules */
  double *values;    /**< input initial conditions and parameter values
			(alternative to SBML!) */
  
  /** ODE SYSTEM */
  /** All names, i.e. ODE variables, assigned parameters, and constant
      parameters */
  char **names;
  /** matrix of the DAG of all variable/parameter
      of assignment and initial assignment rules */
  int **dependencyMatrix;
  int hasCycle;
  
  int nconst; /**< number of constant parameters */
  
  /** Assigned variables: stores species, compartments and parameters,
      that are set by an assignment rule */
  int nass;   /**< number of assigned variables (nass) */
  ASTNode_t **assignment;
  directCode_t **assignmentcode;
  
  /** topological order of assignments */
  nonzeroElem_t **assignmentOrder;     /* size nass */
  /** subset of rules required before ODE evaluation, see
      'discontinuities' for temporal subsets of rules before event
      evaluation */
  int nassbeforeodes;    
  nonzeroElem_t **assignmentsBeforeODEs; 
  
  /** The Ordinary Differential Equation System (ODE)s: f(x,p,t) = dx/dt */
  int neq;    /**< number of ODEs */
  ASTNode_t **ode; 
  directCode_t **odecode;
  
  /** JACOBI MATRIX df(x)/dx of the ODE system 
      neq x neq */
  ASTNode_t ***jacob;
  directCode_t ***jacobcode;
  
  /** List of non-zero elements i,j of the Jacobi matrix. 
      Contains indices i and j, as well as direct pointers to
      the ASTNode in the full matrix construct, and (optionally)
      compiled versions of the ASTNode equations */
  nonzeroElem_t **jacobSparse; /**< array of non-zero elements */
  int sparsesize; /**< number of non-zero elements */
  
  /** was the jacobian matrix constructed ? */
  int jacobian;
    /** flag indicating that jacobian matrix construction had failed for
	this model already and does not need to be tried again */
  int jacobianFailed;
  
  
  /** DISCONTINUITIES : piecewise, events, initial assignments */
  /** PIECEWISE: piecewise expressions as well as events can lead to
      problems and will result in CVODES running in CV_NORMAL_TSTOP
      mode which avoids that the solver internally integrates beyond
      the next requested timestep */
    
  int npiecewise;  /**< number of piecewise expression in equations */

  /** INITIAL ASSIGNMENTS: only evaluated at time <= 0, only
      ODE variables and constants can be affected */
  int *indexInit;  /**< index map from om->names to initial assignments */
  int ninitAss;    /**< number of initial assignments */
  int *initIndex;  /**< index map from initial assignments to om->names */
  ASTNode_t **initAssignment;
  directCode_t **initAssignmentcode;
    
    
  /** EVENTS */
  int nevents;     /**< number of model events */
  ASTNode_t **event;
  directCode_t **eventcode;
  int *neventAss;     /**< number of event assignments per event */
  int **eventIndex;  /**< index map from event assignments to om->names */
  ASTNode_t ***eventAssignment;
  directCode_t ***eventAssignmentcode;
  /** topological order of event assignments incl. other assignments */
  nonzeroElem_t **eventAssignmentOrder; /* size : nIass + nass */
 

  /** topological order of assignments and initial assignments,
      and assignments required before event evaluation */
  nonzeroElem_t **initAssignmentOrder; /* size : nass + ninitAss */
  int nassbeforeevents;
  nonzeroElem_t **assignmentsBeforeEvents;
    
  /** DAE SYSTEMS : NOT USED */
  /** Algebraic Rules (constraints) as used for DAE systems */
  int nalg;   /**< number of algebraic rules */ 
  ASTNode_t **algebraic;
  directCode_t **algebraiccode;
    

  /* COMPILED CODE OBJECTS */
  /** compiled code containing compiled ODE and Jacobian functions */
  compiled_code_t *compiledCVODEFunctionCode; 

  /** CVODE rhs function created by compiling code generated from model */
  CVRhsFn compiledCVODERhsFunction;
  /** CVODE jacobian function created by compiling
      code generated from model */
  CVDenseJacFn compiledCVODEJacobianFunction;

  /** Event function created by compiling code generated from model */
  EventFn compiledEventFunction; 


  /* compilation of adjoint functions */
  /** CVODE adjoint rhs function created by compiling
      code generated from model */
  CVRhsFnB compiledCVODEAdjointRhsFunction;
  /* remember which function is used (compiled or hard-coded) */
  CVRhsFnB current_AdjRHS; 
  /** CVODE adjoint jacobian function created by compiling code
      generated from model */
  CVDenseJacFnB compiledCVODEAdjointJacobianFunction;
  /* remember which function is used (compiled or hard-coded) */
  CVDenseJacFnB current_AdjJAC;
    

  /* ADJOINT */
  /* Adjoint: Given a parameter to observation map F(p),
     computes the adjoint operator applied to the vector v, F'*(p)v.
     v is given by a symbolic expression involving x and observation data. */

  /*!!!: TODO : move objective function and data to separate
    structure for multi-threaded use */
  int discrete_observation_data;    /**< 0: data observed is of
				       continuous type (i.e., interpolated)
				       1: data observed is of
				       discrete type  */

  int compute_vector_v;            /*  if evaluateAST is called to
				       computed vector_v  */

  time_series_t *time_series;  /**< time series of observation data
				  or of vector v */

  ASTNode_t **vector_v;     /**< the vector v, expressiing linear
			       objective used in sensitivity solvers */
  ASTNode_t *ObjectiveFunction;  /**< expression for a general (nonlinear)
				    objective function */
};

struct odeSense
{
  odeModel_t *om;    /**< odeModel_t structure from which sensitivity
			structures where derived */    
  /* forward sensitivity analysis structure, neq x nsens */
  int neq;                 /**< number of variables for sens. analysis,
			      equals NEQ of odeModel */
  int nsens;               /**< number of parameters and initial conditions
			      for sens. analysis, nsens = nsensP + nsensIC */
  int *index_sens;         /**< map from sensitivity parameters and
			      init.cond.  to om->names and
			      data->values, the main ID and data
			      storage in odeModel_t and
			      cvodeData_t */
  int nsensP;
  int *index_sensP;        /**< indices of sensitivity parameters in the
			      sensitivity matrix (or -1 variables) */
  ASTNode_t ***sens;       /**< sensitivity matrix: df(x)/dp, neq x nsensP */
  directCode_t ***senscode;/**< compiled sensitivity matrix */
  int sensitivity;         /**< was the sensitivity matrix constructed ? */
  /** Non-zero elements i,j of the sensitivity matrix */
  int **sensLogic;         /**< logic matrix, indicating non-zero elements */
  nonzeroElem_t **sensSparse; /**< array of non-zero elements */
  int sparsesize;          /**< number of non-zero elements */

    
  /** compiled code containing compiled sensitivity functions */
  compiled_code_t *compiledCVODESensitivityCode; 

  /** flag that indicates whether compilation is required,
      upon first request or when required parameters have
      changed since last compilation*/
  int recompileSensitivity;

  /** Sensitivity function created by compiling code generated from model */
  CVSensRhs1Fn compiledCVODESenseFunction;
    
  /* compilation of adjoint functions */
  /** CVODE adjoint quadrature function */
  CVQuadRhsFnB compiledCVODEAdjointQuadFunction;
  /* remember which function is used (compiled or hard-coded) */
  CVQuadRhsFnB current_AdjQAD;
};
  
struct objFunc
{
  /* adjoint */
  /* Adjoint: Given a parameter to observation map F(p),
     computes the adjoint operator applied to the vector v, F'*(p)v.
     v is given by a symbolic expression involving x and observation data. */


  int discrete_observation_data;    /**< 0: data observed is of
				       continuous type (i.e., interpolated)
				       1: data observed is of
				       discrete type  */

  int compute_vector_v;            /*  if evaluateAST is called to
				       computed vector_v  */

  time_series_t *time_series;  /**< time series of observation data
				  or of vector v */

  ASTNode_t **vector_v;     /**< the vector v, expressiing linear
			       objective used in sensitivity solvers */
  ASTNode_t *ObjectiveFunction;  /**< expression for a general (nonlinear)
				    objective function */
 
};

/** Stores a variable index and equations defining this index.
    Arrays of this structure are used to specify subsets
    of equations (in the correct order) for evaluation.
*/
struct nonzeroElem
{
  int i, j;
  ASTNode_t *ij;
  directCode_t *ijcode;
};

#ifdef __cplusplus
extern "C" {
#endif

  /* ODE creation */
  SBML_ODESOLVER_API odeModel_t *ODEModel_createFromFile(const char *);
  SBML_ODESOLVER_API odeModel_t *ODEModel_createFromSBML2(SBMLDocument_t *);
  SBML_ODESOLVER_API odeModel_t *ODEModel_create(Model_t *);
  SBML_ODESOLVER_API odeModel_t *ODEModel_createFromODEs(ASTNode_t **, int neq, int nass, int nconst, char **, double *, Model_t *);
  SBML_ODESOLVER_API void ODEModel_free(odeModel_t *);

  /* ODE variables and parameters */
  SBML_ODESOLVER_API const Model_t *ODEModel_getModel(odeModel_t *);
  SBML_ODESOLVER_API const Model_t *ODEModel_getOdeSBML(odeModel_t *);
  SBML_ODESOLVER_API const Model_t *ODEModel_getInputSBML(odeModel_t *);

  SBML_ODESOLVER_API int ODEModel_getNumValues(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNeq(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNalg(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNumAssignments(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNumConstants(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_hasVariable(odeModel_t *, const char *);
  SBML_ODESOLVER_API variableIndex_t *ODEModel_getVariableIndexByNum(odeModel_t *, int);
  SBML_ODESOLVER_API variableIndex_t *ODEModel_getOdeVariableIndex(odeModel_t *, int);
  SBML_ODESOLVER_API variableIndex_t *ODEModel_getAssignedVariableIndex(odeModel_t *, int);
  SBML_ODESOLVER_API variableIndex_t *ODEModel_getConstantIndex(odeModel_t *, int);
  SBML_ODESOLVER_API variableIndex_t *ODEModel_getVariableIndex(odeModel_t *, const char *);
  SBML_ODESOLVER_API void ODEModel_dumpNames(odeModel_t *);

  /* Topological sorting */
  SBML_ODESOLVER_API int ODEModel_hasCycle(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNumAssignmentsBeforeODEs(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNumAssignmentsBeforeEvents(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNumJacobiElements(odeModel_t *);
  SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getAssignmentOrder(odeModel_t *, int);
  SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getAssignmentBeforeODEs(odeModel_t *, int);
  SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getAssignmentBeforeEvents(odeModel_t *, int);
  SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getJacobiElement(odeModel_t *, int);
  
  /* Variable Index */
  /*outdated */ const char *ODEModel_getVariableName(odeModel_t *, variableIndex_t *);
  SBML_ODESOLVER_API const char *VariableIndex_getName(variableIndex_t *, odeModel_t *);
  SBML_ODESOLVER_API int VariableIndex_getIndex(variableIndex_t *);
  SBML_ODESOLVER_API void VariableIndex_free(variableIndex_t *);
  /* Evaluation elements */
  /*!!! TODO : fuse nonzeroElem with variableIndex and use list of variableIndex instead */
  SBML_ODESOLVER_API const ASTNode_t *NonzeroElement_getEquation(nonzeroElem_t *);
  SBML_ODESOLVER_API const char *NonzeroElement_getVariableName(nonzeroElem_t *, odeModel_t*);
  SBML_ODESOLVER_API const char *NonzeroElement_getVariable2Name(nonzeroElem_t *, odeModel_t*);
  SBML_ODESOLVER_API variableIndex_t *NonzeroElement_getVariableIndex(nonzeroElem_t *, odeModel_t*);
  SBML_ODESOLVER_API variableIndex_t *NonzeroElement_getVariable2Index(nonzeroElem_t *, odeModel_t*);

  
  /* ODEs and assignments */
  SBML_ODESOLVER_API const ASTNode_t *ODEModel_getOde(odeModel_t *, variableIndex_t *);
  SBML_ODESOLVER_API const ASTNode_t *ODEModel_getAssignment(odeModel_t *, variableIndex_t *);


  /* diverse other useful functions */
  SBML_ODESOLVER_API List_t *topoSort(int **matrix, int n, int *changed, int*required);

  /* Discontinuities */
  SBML_ODESOLVER_API int ODEModel_getNumPiecewise(odeModel_t *);
  SBML_ODESOLVER_API int ODEModel_getNumEvents(odeModel_t *);
  SBML_ODESOLVER_API const ASTNode_t *ODEModel_getEventTrigger(odeModel_t *, int);
  SBML_ODESOLVER_API const ASTNode_t *ODEModel_getEventAssignment(odeModel_t *, int, int);
  

  /* ODE Jacobi matrix */
  SBML_ODESOLVER_API int ODEModel_constructJacobian(odeModel_t *);
  SBML_ODESOLVER_API void ODEModel_freeJacobian(odeModel_t *);
  SBML_ODESOLVER_API const ASTNode_t *ODEModel_getJacobianIJEntry(odeModel_t *, int i, int j);
  SBML_ODESOLVER_API const ASTNode_t *ODEModel_getJacobianEntry(odeModel_t *, variableIndex_t *, variableIndex_t *);
  SBML_ODESOLVER_API ASTNode_t *ODEModel_constructDeterminant(odeModel_t *);

  /* Sensitivity Model */
  SBML_ODESOLVER_API odeSense_t *ODEModel_constructSensitivity(odeModel_t *);
  SBML_ODESOLVER_API odeSense_t *ODESense_create(odeModel_t *, cvodeSettings_t *);
  SBML_ODESOLVER_API void ODESense_free(odeSense_t *);
  SBML_ODESOLVER_API variableIndex_t *ODESense_getSensParamIndexByNum(odeSense_t *, int);
  SBML_ODESOLVER_API int ODESense_getNeq(odeSense_t *);
  SBML_ODESOLVER_API int ODESense_getNsens(odeSense_t *);
  SBML_ODESOLVER_API const ASTNode_t *ODESense_getSensIJEntry(odeSense_t *, int i, int j);
  SBML_ODESOLVER_API const ASTNode_t *ODESense_getSensEntry(odeSense_t *, variableIndex_t *, variableIndex_t *);

  /* ODEModel compilation */
  SBML_ODESOLVER_API int ODEModel_compileCVODEFunctions(odeModel_t *);
  SBML_ODESOLVER_API int ODESense_compileCVODESenseFunctions(odeSense_t *);
  SBML_ODESOLVER_API CVRhsFn ODEModel_getCompiledCVODERHSFunction(odeModel_t *);
  SBML_ODESOLVER_API CVDenseJacFn ODEModel_getCompiledCVODEJacobianFunction(odeModel_t *);
  SBML_ODESOLVER_API CVRhsFnB ODEModel_getCompiledCVODEAdjointRHSFunction(odeModel_t *);
  SBML_ODESOLVER_API CVDenseJacFnB ODEModel_getCompiledCVODEAdjointJacobianFunction(odeModel_t *);
  SBML_ODESOLVER_API CVQuadRhsFnB ODESense_getCompiledCVODEAdjointQuadFunction(odeSense_t *);
  SBML_ODESOLVER_API CVSensRhs1Fn ODESense_getCompiledCVODESenseFunction(odeSense_t *);

#ifdef __cplusplus
};
#endif

/* internal functions, not be used by calling applications */  
int ODEModel_getVariableIndexFields(odeModel_t *, const char *SBML_ID);
int ODESense_constructMatrix(odeSense_t *, odeModel_t *);
void ODESense_freeMatrix(odeSense_t *);
void ODESense_freeStructures(odeSense_t *);
#endif
