<<<<<<< .mine


=======

>>>>>>> .r1534
/*
  Last changed Time-stamp: <2010-04-12 10:24:04 raim>
  $Id: odeModel.c,v 1.130 2010/04/12 08:29:38 raimc Exp $
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
 *     Andrew M. Finney
 */

/* System specific definitions,
   created by configure script */
#ifndef WIN32
#include "config.h"
#else
#include "stdio.h"
#endif

#include <string.h>
#include <stdlib.h>

#include "sbmlsolver/sbml.h"
#include "sbmlsolver/odeConstruct.h"
#include "sbmlsolver/processAST.h"
#include "sbmlsolver/solverError.h"
#include "sbmlsolver/modelSimplify.h"
#include "sbmlsolver/odeModel.h"
#include "sbmlsolver/variableIndex.h"
#include "sbmlsolver/compiler.h"
#include "sbmlsolver/arithmeticCompiler.h"

#include <sbml/util/List.h>

#define COMPILED_RHS_FUNCTION_NAME "ode_f"
#define COMPILED_ADJOINT_RHS_FUNCTION_NAME "adjode_f"
#define COMPILED_JACOBIAN_FUNCTION_NAME "jacobi_f"
#define COMPILED_ADJOINT_JACOBIAN_FUNCTION_NAME "adj_jacobi_f"
#define COMPILED_EVENT_FUNCTION_NAME "event_f"
#define COMPILED_SENSITIVITY_FUNCTION_NAME "sense_f"
#define COMPILED_ADJOINT_QUAD_FUNCTION_NAME "adj_quad"


/* model allocation */
static odeModel_t *ODEModel_fillStructures(Model_t *);
static odeModel_t *ODEModel_allocate(int neq, int nconst, int nass, int nalg);
static int ODEModel_allocateDiscontinuities(odeModel_t *om, int nvalues,
					    int nevents,int neventAss,
					    int ninitAss);
static int ODEModel_setDiscontinuities(odeModel_t *om, Model_t *ode);
static int ODEModel_freeDiscontinuities(odeModel_t *);
static void ODEModel_initializeValuesFromSBML(odeModel_t *, Model_t *);

/* rule sorting */
typedef struct assignmentStage assignmentStage_t ;
static nonzeroElem_t *copyNonzeroElem(nonzeroElem_t *);
static int ODEModel_topologicalRuleSort(odeModel_t *);
static void List_append(List_t *, List_t *);



/*! \defgroup odeModel ODE Model: f(x,p,t) = dx/dt
  \ingroup symbolic
  \brief This module contains all functions to create and interface
  the internal ODE Model it's Jacobian matrix and other derivatives

  The internal ODE Model (structure odeModel) can be interfaced for
  analytical purposes. All formulae can be retrieved as libSBML
  Abstract Syntax Trees (AST).
*/
/*@{*/


/** \brief Create internal model odeModel from a reaction network,
    represented as libSBML's Model_t structure.

    The input model must be of SBML level2!
    The function at first, attempts to construct a simplified SBML
    model, that contains all compartments, species, parameters, events
    and rules of the input model, and constructs new ODEs as SBML
    RateRules from the reaction network of the input model.  The
    function then creates the structure odeModel which contains
    variable, parameter and constant names and all formulas (ODEs and
    assignments) as indexed AST (iAST). This structure can be used to
    initialize and run several integration runs, each associated with
    initial conditions in cvodeData_t. Alternatively I.1a - I.1c
    allow to construct odeModel from higher-level data (a file, an
    SBML document or a reaction network model, respectively).
*/
SBML_ODESOLVER_API odeModel_t *ODEModel_create(Model_t *m)
{
  Model_t *ode = 0;
  odeModel_t *om = 0;

    //printf("ODEModel_create was called...\n");

  /* NULL SBML model passed ? */
  if ( m == NULL )
  {
    printf("NULL Model_t passed to ODEModel_create. Returning NULL value for odeModel_t*...\n");
    return NULL;
  } // else {
  //  printf("Valid Model_t* passed as argument...\n");
  //}
    
  //printf("Calling Model_reduceToOdes...\n");
  
  ode = Model_reduceToOdes(m);
  /* if errors occured, free SBML and return NULL */
  if ( ode == NULL ) {
    printf("Model_reduceToOdes was not successful. NULL was returned. Now returning NULL for odeModel_t*...\n");
    return NULL;
  } //else {
  //  printf("Model_reduceToOdes was successful. New Model_t* successfully created...\n");
  //}

  /* check if compatible */
  /* algebraic rules */
  if ( SBase_isSetNotes((SBase_t *)ode) ) {
    if ( strcmp(SBase_getNotesString((SBase_t *)ode),"<notes>DAE model</notes>") == 0 )
    {
      printf( "Results of strcmp is 0. Cannot proceed. Freeing Model_t* and returning NULL for odeModel_t*...\n" );
      Model_free(ode);
      return NULL;
    }
  }

  //printf("Calling ODEModel_fillStructures with Model_t* as argument...\n");
  /* CREATE odeModel equations */
  om = ODEModel_fillStructures(ode);
  /* if om is NULL a memory allocation failure has occured, pass on NULL */
  if ( om == NULL ) { 
    printf("ODEModel_fillStructures returned NULL value for odeModel_t*. Returning NULL for odeModel_t*...\n");
    return NULL;
  } //else {
  //  printf("Valid odeModel_t* returned from ODEModel_fillStructures function call...\n");
  //}

  om->m = m;
  om->d = NULL;  /* will be set and free'd if created from file */

  //printf("Calling ODEModel_topologicalRuleSort to generate ordered list of assignment rules...\n");
  /* generate ordered list of assignment rules for evaluation order */
  /* -1 for memory allocation failures */
  om->hasCycle = ODEModel_topologicalRuleSort(om);

  printf("Returning valid odeModel_t*...\n");
  return om;
}

/* given the DAG of equation dependencies,
   labels all equations required for equation with index 'start' */
static void searchPath(int n, int **matrix, int start, int *required)
{
  int i;

  for ( i=0; i<n; i++ )
    if ( matrix[start][i] ) /* if r.h.s. 'start' depends on l.h.s 'i' */
      if ( !required[i] )/* if 'i' is not tagged already */
      {
	searchPath(n, matrix, i, required); /* ... PROCEED SEARCH */
	required[i] = 1; /* tag 'i' as required (upstream have been tagged) */
      }
}
/** Does the ODEModel's set of assignments (initial assignments,
    assignments, kinetic law) contain a dependency cycle?

    If the model  contains a cycle in its  rules it is underdetermined
    and  can not  be solved.  IntegratorInstance_create will  fail for
    this model, however it's  equations can be inspected and evaluated
    with initial condition data, using CvodeData.
*/
SBML_ODESOLVER_API int ODEModel_hasCycle(odeModel_t *om)
{
  return om->hasCycle;
}
SBML_ODESOLVER_API int ODEModel_getNumAssignmentsBeforeODEs(odeModel_t *om)
{
  return om->nassbeforeodes;
}
SBML_ODESOLVER_API int ODEModel_getNumAssignmentsBeforeEvents(odeModel_t *om)
{
  return om->nassbeforeevents;
}
SBML_ODESOLVER_API int ODEModel_getNumJacobiElements(odeModel_t *om)
{
  return om->sparsesize;
}

SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getAssignmentOrder(odeModel_t *om, int i)
{
  if ( i >= om->nass ) return NULL;
  return om->assignmentOrder[i];
}
SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getAssignmentBeforeODEs(odeModel_t *om, int i)
{
  if ( i >= om->nassbeforeodes ) return NULL;
  return om->assignmentsBeforeODEs[i];
}
SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getAssignmentBeforeEvents(odeModel_t *om, int i)
{
  if ( i >= om->nassbeforeevents ) return NULL;
  return om->assignmentsBeforeEvents[i];
}
SBML_ODESOLVER_API const nonzeroElem_t *ODEModel_getJacobiElement(odeModel_t *om, int i)
{
  if ( i >= om->sparsesize ) return NULL;
  return om->jacobSparse[i];
}
/* Evaluation elements */
SBML_ODESOLVER_API const ASTNode_t *NonzeroElement_getEquation(nonzeroElem_t *nonzero)
{
  return nonzero->ij;
}
SBML_ODESOLVER_API const char *NonzeroElement_getVariableName(nonzeroElem_t *nonzero, odeModel_t*om)
{
  if ( nonzero->i != -1 ) return om->names[nonzero->i];
  else return om->names[nonzero->j];
}
SBML_ODESOLVER_API const char *NonzeroElement_getVariable2Name(nonzeroElem_t *nonzero, odeModel_t*om)
{
  if ( nonzero->i == -1 /*|| nonzero->i == -1*/ ) return NULL;
  else return om->names[nonzero->j];
}

SBML_ODESOLVER_API variableIndex_t *NonzeroElement_getVariable(nonzeroElem_t *nonzero, odeModel_t*om)
{
  int index;
  if ( nonzero->i != -1 ) index = nonzero->i;
  else index = nonzero->j;
  return ODEModel_getVariableIndexByNum(om, index);
}
SBML_ODESOLVER_API variableIndex_t *NonzeroElement_getVariable2(nonzeroElem_t *nonzero, odeModel_t*om)
{
  int index;
  if ( nonzero->j == -1 || nonzero->i == -1 ) return NULL;
  else index = nonzero->j;
  return ODEModel_getVariableIndexByNum(om, index);
}

/** Topological sort:

    The input matrix of size n x n is a boolean matrix, representing the
    dependency graph of a set of simple assignment rules. Rows are equations
    and columns are parameters appearing in each equation. The
    function returns an ordered list of size n, representing the
    (non-unique) ordering of evaluation. The function returns NULL and issues
    a correpsonding SolverError_error if cycles are detected in the
    dependency matrix. The sorting algorithm has been taken from
    http://en.wikipedia.org/wiki/Topological_sorting */

SBML_ODESOLVER_API List_t *topoSort(int **inMatrix, int n, int *changed, int*required)
{
  int i, j, ins, **matrix, allchanged, allrequired;
  List_t *sorted;   /* L : Empty list where we put the sorted elements */
  List_t *noincome; /* Q : Set of all nodes with no incoming edges */

  noincome = List_create();
  sorted = List_create();

  /* copy matrix */
  ASSIGN_NEW_MEMORY_BLOCK(matrix, n, int *, 0);
  for ( i=0; i<n; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(matrix[i], n, int, 0);
    for ( j=0; j<n; j++ )
      matrix[i][j] = inMatrix[i][j];
  }

  /* default: all changed, all required */
  allchanged = 0;
  if ( changed == NULL )
  {
    allchanged = 1;
    ASSIGN_NEW_MEMORY_BLOCK(changed, n, int, NULL);
    for ( i=0; i<n; i++ ) changed[i] = 1;
  }
  allrequired = 0;
  if ( required == NULL )
  {
    allrequired = 1;
    ASSIGN_NEW_MEMORY_BLOCK(required, n, int, NULL);
    for ( i=0; i<n; i++ ) required[i] = 1;
  }

#ifdef _DEBUG
/*   for ( i=0; i<n; i++ )  */
/*     printf("value %i is required? %s\n", i, required[i] ? "yes" : "no");    */
#endif
  /* REQUIRED : label all edges upstream by DFS */
  /* remove all edges in rows of non-requested nodes */

  /* http://en.wikipedia.org/wiki/Topological_sorting */
  /* L : Empty list where we put the sorted elements */
  /* Q : Set of all nodes with no incoming edges */
  /* while Q is non-empty do */
  /*     remove a node n from Q */
  /*     insert n into L */ /* IFF N IS CHANGED AND REQUIRED ! */
  /*     for each node m with an edge e from n to m do */
  /*         remove edge e from the graph */ /* AND INHERIT CHANGE STATUS */
  /*         if m has no other incoming edges then */
  /*             insert m into Q */
  /* if graph has edges then */
  /*     output error message (graph has a cycle) */
  /* else  */
  /*     output message (proposed topologically sorted order: L) */

  /*OR INSTEAD: generate DAG and use directly for evaluation
   * (e.g. AST derived tree) for recursive evaluation `evaluateDAG' */

  /* generate Q : Set of all nodes with no incoming edges */

  for ( i=0; i<n; i++ )
  {
    ins = 0;
    for  ( j=0; j<n; j++ )
      ins += matrix[i][j];

    if ( !ins )
    {
      int *idx;
      ASSIGN_NEW_MEMORY(idx, int, NULL);
      *idx = i;
      List_add(noincome, idx);
    }
  }

  /* setup changed array */

  /* generate L (sorted elements) until Q is empty */
  while( List_size(noincome) )
  {
    int *idx; int current;
    idx = List_remove(noincome, 0); /* remove node n from Q */
    current = *idx;
    if ( required[current] && changed[current] ) /* iff n is changed and required ... */
      List_add(sorted, idx);   /* ... insert n into L */
    else {
	  /* free not required */
      free(idx);
	  idx = 0;
	}

    for ( i=0; i<n; i++ )
    {
      if ( matrix[i][current] ) /* for each node m with an edge e from n to m do */
      {
	matrix[i][current] = 0; /* remove edge e from the graph */
	if ( changed[current] )
	  changed[i] = 1; /* ... and inherit change status */

	ins = 0;
	for ( j=0; j<n; j++ )
	  ins += matrix[i][j];

	if ( !ins ) /* if m has no other incoming edges then */
	{
	  //int *idx;
	  ASSIGN_NEW_MEMORY(idx, int, NULL);
	  *idx = i;
	  List_add(noincome, idx);     /* insert m into Q */
	}
      }
    }
  }

  /* free sorting list */
  List_freeItems(noincome, free, int); /* only in case of cyclic graphs ?*/
  List_free(noincome); /* free Q */

  /* check whether any edges remain ... */
  ins = 0;
  for ( i=0; i<n; i++ )
    for ( j=0; j<n; j++ )
      ins += matrix[i][j];

#ifdef _DEBUG
/*   fprintf(stderr, "MATRIX:\n"); */
/*   for ( i=0; i<n; i++ ) */
/*   { */
/*     for ( j=0; j<n; j++ ) */
/*       fprintf(stderr, "\t%d", matrix[i][j]); */
/*     fprintf(stderr, "\n"); */
/*   } */
#endif


  /* if graph has edges then */
  if ( ins )
  {
    int *idx;
#ifdef _DEBUG
    fprintf(stderr, "ERROR: Cyclic dependency found in topological sorting.\n");
    fprintf(stderr, "MATRIX:\n");
    for ( i=0; i<n; i++ )
      for ( j=0; j<n; j++ )
	if ( matrix[i][j] )
	  fprintf(stderr, "%d -> %d\n", j, i);
    fprintf(stderr, "\n");
#endif
    /* output error message (graph has a cycle) */
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_ODE_MODEL_CYCLIC_DEPENDENCY_IN_RULES,
		      "Cyclic dependency found in topological sorting.");
    List_freeItems(sorted, free, int);
    List_free(sorted);
    sorted = List_create();

    ASSIGN_NEW_MEMORY(idx, int, NULL);
    *idx = -1;
    List_add(sorted, idx);
    /*!!! TODO : this could return the remaining edges, if this is meaningful*/
  }

  /* free dependency matrix */
  for ( i=0; i<n; i++ )
    free(matrix[i]);
  free(matrix);

  /* free helper arrays for default use */
  if ( allchanged ) free(changed);
  if ( allrequired ) free(required);

  return sorted;   /* proposed topologically sorted order: L */
}

static nonzeroElem_t *copyNonzeroElem(nonzeroElem_t *source)
{
  nonzeroElem_t *target;

  if ( source == NULL )
    return NULL;

  ASSIGN_NEW_MEMORY(target, nonzeroElem_t, NULL);
  target->i = source->i;
  target->j = source->j;
  target->ij = source->ij;
  target->ijcode = source->ijcode;
  return target;
}

/* generates dependency graph (matrix) from the odeModel's assignment
   rules, calls topological sorting and generates the ordering of
   assignment rule evaluation, used during solving.
   Returns 1 if the model's assignments contain a cycle (algebraic
   loop, 0 if not, and -1 for memory allocation errors. */
static int ODEModel_topologicalRuleSort(odeModel_t *om)
{
  unsigned int ui;
  int i, j, k, l, nvalues, **matrix, *tmpIndex, *idx,
    *changedBySolver, *requiredForODEs, *requiredForEvents;
  List_t *dependencyList;
  ASTNode_t *math;
  int hasCycle = 0;

  //printf("ODEModel_topologicalRuleSort checkpoint #1...\n");

  nvalues = om->neq + om->nass + om->nconst;
  om->initAssignmentOrder = NULL;
  om->assignmentOrder = NULL;
  om->assignmentsBeforeODEs = NULL;
  om->assignmentsBeforeEvents = NULL;
  
  //printf("ODEModel_topologicalRuleSort checkpoint #2...\n");
  
  /* 1: GENERATE DEPENDENCY MATRIX for complete asssignment set */

  /*!!! TODO : use -1/NULL as return value for failed alloc
    everywhere in sosLib? */
  ASSIGN_NEW_MEMORY_BLOCK(matrix, nvalues, int *, -1);
  for ( i=0; i<nvalues; i++ )
  {
    /* assignment rules */
    if ( i >= om->neq && i < om->neq + om->nass )
    {
      matrix[i] = ASTNode_getIndexArray(om->assignment[i - om->neq], nvalues);
    }
    /*  init. and event assignments for ODE variables and constants */
    else
    {
      math = NULL;
      if ( om->indexInit[i] != -1 ) /* check whether initial assignment exist */
	math = om->initAssignment[om->indexInit[i]];
      matrix[i] = ASTNode_getIndexArray(math, nvalues);

      /* memory error */
      /* if ( matrix[i] == NULL ) */
      /*	return -1; */

      /*!!! TODO  ALSO ADD EVENTS HERE ?? */

    }
  }

  //printf("ODEModel_topologicalRuleSort checkpoint #3...\n");

  /* attach dependency matrix to odeModel */
  om->dependencyMatrix = matrix;


  /* 2: ORDERING OF COMPLETE ASSIGNMENT SET, all changed, all required */
  dependencyList = topoSort(matrix, nvalues, NULL, NULL);


  //printf("ODEModel_topologicalRuleSort checkpoint #4...\n");

  /* generate ordered array of complete rule set and assignment subset */

  k = 0;
  l = 0;
  for ( i=0; i<List_size(dependencyList); i++ )
  {
    idx = List_get(dependencyList, i);
    /* issue solver error and return if topo. sort was unsuccessful */
    if ( *idx == -1 )
    {
      SolverError_error(ERROR_ERROR_TYPE,
			SOLVER_ERROR_ODE_MODEL_RULE_SORTING_FAILED,
			"Topological sorting failed for complete rule set "
			"(initial assignments, assignments and kinetic laws) "
			"Found cyclic dependency in rules. ");
      /* AS -1 error is passed as single element no array elements
	 have been allocated and can be simply freed */
      List_freeItems(dependencyList, free, int);
      List_free(dependencyList);
      hasCycle = 1;
      return hasCycle;
    }

    if ( i == 0 ) /* create structures */
    {
      ASSIGN_NEW_MEMORY_BLOCK(om->assignmentOrder, om->nass,
			      nonzeroElem_t *, -1);
      ASSIGN_NEW_MEMORY_BLOCK(om->initAssignmentOrder, om->nass+om->ninitAss,
			      nonzeroElem_t *, -1);
    }

    if ( *idx >= om->neq && *idx < om->neq + om->nass ) /* assignments */
    {

      nonzeroElem_t *ordered;
      ASSIGN_NEW_MEMORY(ordered, nonzeroElem_t, -1);
      ordered->i = *idx;
      ordered->j = -1;   /* used for initial assignments, see below */
      ordered->ij = om->assignment[ *idx - om->neq ];
      ordered->ijcode = om->assignmentcode[ *idx - om->neq ];
      om->initAssignmentOrder[k] = ordered;
      k++;
      om->assignmentOrder[l] = copyNonzeroElem(ordered);
      l++;
    }
    else if ( om->indexInit[*idx] != -1 ) /* initial assignments */
    {
      nonzeroElem_t *ordered;
      ASSIGN_NEW_MEMORY(ordered, nonzeroElem_t, -1);
      ordered->i = -1;   /* used for assignments, see above */
      ordered->j = *idx;
      ordered->ij = om->initAssignment[ om->indexInit[*idx] ];
      ordered->ijcode = om->initAssignmentcode[ om->indexInit[*idx] ];
      om->initAssignmentOrder[k] = ordered;
      k++;
    }
  }
  
  //printf("ODEModel_topologicalRuleSort checkpoint #5...\n");

#ifdef _DEBUG
  printf("COMPLETE RULE SET:\n");
  for ( i=0; i<om->nass+om->ninitAss; i++ )
  {
    char *eq;
    nonzeroElem_t *ordered = om->initAssignmentOrder[i];
    int idxI = ordered->i;
    int idxJ = ordered->j;
    printf("rule %d: ", i);
    if ( idxI == -1 )
      printf("init.ass: %s = ", om->names[ idxJ ]);
    else
      printf("norm.ass: %s = ", om->names[ idxI ]);

    eq = SBML_formulaToString(ordered->ij);
    printf("%s\n", eq);
    free(eq);
  }
  printf("\n");
#endif

  /* free dependency list */
  List_freeItems(dependencyList, free, int);
  List_free(dependencyList);

  /* 3: ORDERING OF ASSIGNMENT SETS, for evaluation stages */

  /* set changed variables : same for ODEs and Events */
  ASSIGN_NEW_MEMORY_BLOCK(changedBySolver, nvalues, int , -1);
  for ( i=0; i<om->neq; i++ ) /* all variables have changed */
    changedBySolver[i] = 1;
  for ( i=om->neq; i<nvalues; i++ )
    changedBySolver[i] = 0;

  /* add time dependencies */
  for ( i=0; i<om->nass; i++ )
    if ( ASTNode_containsTime(om->assignment[i]) )
      changedBySolver[ om->neq + i ] = 1;


  //printf("ODEModel_topologicalRuleSort checkpoint #6...\n");

  /* RULES TO BE EVALUATED BEFORE ODEs: ODE variables and TIME have changed */

  /* get required rules */
  ASSIGN_NEW_MEMORY_BLOCK(requiredForODEs, nvalues, int , -1);
  for ( j=0; j<nvalues; j++ ) requiredForODEs[j] = 0;

  /* required for ODE evaluation  */
  for ( i=0; i<om->neq; i++ )
  {
    tmpIndex = ASTNode_getIndexArray(om->ode[i], nvalues);
    for ( j=0; j<nvalues; j++ )
      if ( !requiredForODEs[j] && tmpIndex[j] )
      {
	searchPath(nvalues, matrix, j, requiredForODEs);
	requiredForODEs[j] = 1;
      }
    free(tmpIndex);
  }


  //printf("ODEModel_topologicalRuleSort checkpoint #7...\n");

  /* calculate TOPOLOGICAL SORTING of dependency matrix */
  dependencyList = topoSort(matrix, nvalues, changedBySolver, requiredForODEs);

  /* generate ordered array of rule set before ODEs */
  k = 0;
  /* count assignment rules */
  for ( ui=0; ui<List_size(dependencyList); ui++ )
  {
    idx = List_get(dependencyList, ui);
    /* issue solver error and return if topo. sort was unsuccessful */
    /* topo sort was tested on global matrix, so no errors should occur
       when we are here !*/
    if ( *idx == -1 )
    {
      SolverError_error(ERROR_ERROR_TYPE,
			SOLVER_ERROR_ODE_MODEL_RULE_SORTING_FAILED,
			"Topological sorting failed for complete rule set "
			"(assignments and kinetic laws required before ODE "
			"evaluation). Found cyclic dependency in rules. ");

      List_freeItems(dependencyList, free, int);
      List_free(dependencyList);
      hasCycle = 1;
      return hasCycle;
    }
    else if ( *idx >= om->neq && *idx < om->neq + om->nass ) /* assignments */
      k++;
  }
  
  //printf("ODEModel_topologicalRuleSort checkpoint #8...\n");
  
  om->nassbeforeodes = k;
  ASSIGN_NEW_MEMORY_BLOCK(om->assignmentsBeforeODEs, k, nonzeroElem_t *, -1);

  /*!!! TODO : instead of creating new nonzeroElem's the array could point
	into the global ordering assignmentOrder */
  k = 0;
  for ( ui=0; ui<List_size(dependencyList); ui++ )
  {
    idx = List_get(dependencyList, ui);
    if ( *idx >= om->neq && *idx < om->neq + om->nass ) /* assignments */
    {
      nonzeroElem_t *ordered;
      ASSIGN_NEW_MEMORY(ordered, nonzeroElem_t, 0);
      ordered->i = *idx;
      ordered->j = -1; /* not used */
      ordered->ij = om->assignment[ *idx - om->neq ];
      ordered->ijcode = om->assignmentcode[ *idx - om->neq ];
      om->assignmentsBeforeODEs[k] = ordered;
      k++;
    }
  }

  //printf("ODEModel_topologicalRuleSort checkpoint #9...\n");


#ifdef _DEBUG
  printf("DEPENDENCY LIST: %d\n", List_size(dependencyList));
  for ( i=0; i<List_size(dependencyList); i++ )
  {
    idx = List_get(dependencyList, i);
    printf(" %d %s\n", *idx, om->names[*idx]);
  }
#endif
#ifdef _DEBUG
  printf("BEFORE ODEs RULE SET:\n");
  for ( i=0; i<om->nassbeforeodes; i++ )
  {
    char *eq;
    nonzeroElem_t *ordered = om->assignmentsBeforeODEs[i];
    int idxI = ordered->i;
    int idxJ = ordered->j;
    printf("rule %d: ", i);
    if ( idxI == -1 )
      printf("INIT.ass: %s = ", om->names[ idxJ ]);
    else
      printf("norm.ass: %s = ", om->names[ idxI ]);

    eq = SBML_formulaToString(ordered->ij);
    printf("%s\n", eq);
    free(eq);
  }
  printf("\n");
#endif

  /* free dependency list */
  List_freeItems(dependencyList, free, int);
  List_free(dependencyList);

  /* RULES TO BE EVALUATED BEFORE EVENTs: ODE variables and TIME have changed */
  /* get required rules */
  ASSIGN_NEW_MEMORY_BLOCK(requiredForEvents, nvalues, int , -1);

  for ( j=0; j<nvalues; j++ )
    requiredForEvents[j] = 0;


  //printf("ODEModel_topologicalRuleSort checkpoint #10...\n");

  /* required for ODEs  */
  for ( i=0; i<om->nevents; i++ )
  {
    tmpIndex = ASTNode_getIndexArray(om->event[i], nvalues);
    for ( j=0; j<nvalues; j++ )
      if ( !requiredForEvents[j] && tmpIndex[j] )
      {
	searchPath(nvalues, matrix, j, requiredForEvents);
	requiredForEvents[j] = 1;
      }
    free(tmpIndex);

    /* event assignments */
    for ( j=0; j<om->neventAss[i]; j++ )
    {
      tmpIndex = ASTNode_getIndexArray(om->eventAssignment[i][j], nvalues);
      for ( k=0; k<nvalues; k++ )
	if ( !requiredForEvents[k] && tmpIndex[k] )
	{
	  searchPath(nvalues, matrix, k, requiredForEvents);
	  requiredForEvents[k] = 1;
	}
      free(tmpIndex);

      /* set changed variable */
      /* changedByEvents[om->eventIndex[i][j]] = 1; */
    }
  }


  //printf("ODEModel_topologicalRuleSort checkpoint #11...\n");

  /* calculate TOPOLOGICAL SORTING of dependency matrix */
  dependencyList = topoSort(matrix, nvalues, changedBySolver,
			    requiredForEvents);

  /* generate ordered array of rule set before ODEs */
  k = 0;
  /* count assignment rules */
  for ( i=0; i<List_size(dependencyList); i++ )
  {
    idx = List_get(dependencyList, i);
    /* issue solver error and return if topo. sort was unsuccessful */
    if ( *idx == -1 )
    {
      SolverError_error(ERROR_ERROR_TYPE,
			SOLVER_ERROR_ODE_MODEL_RULE_SORTING_FAILED,
			"Topological sorting failed for Event rule set "
			"(assignments and kinetic laws required BEFORE event "
			"evaluation).  Found cyclic dependency in rules.");
      List_freeItems(dependencyList, free, int);
      List_free(dependencyList);
      free(requiredForODEs);
      free(changedBySolver);
      free(requiredForEvents);
      hasCycle = 1;
      return hasCycle;
    }
    else if ( *idx >= om->neq && *idx < om->neq + om->nass ) /* assignments */
      k++;
  }


  //printf("ODEModel_topologicalRuleSort checkpoint #12...\n");

  om->nassbeforeevents = k;
  ASSIGN_NEW_MEMORY_BLOCK(om->assignmentsBeforeEvents, k, nonzeroElem_t *, -1);

  k = 0;
  for ( i=0; i<List_size(dependencyList); i++ )
  {
    idx = List_get(dependencyList, i);
    if ( *idx >= om->neq && *idx < om->neq + om->nass ) /* assignments */
    {
      nonzeroElem_t *ordered;
      ASSIGN_NEW_MEMORY(ordered, nonzeroElem_t, -1);
      ordered->i = *idx;
      ordered->j = -1; /* not used */
      ordered->ij = om->assignment[ *idx - om->neq ];
      ordered->ijcode = om->assignmentcode[ *idx - om->neq ];
      om->assignmentsBeforeEvents[k] = ordered;
      k++;
    }
  }

 
  //printf("ODEModel_topologicalRuleSort checkpoint #13...\n");

#ifdef _DEBUG
  printf("DEPENDENCY LIST: %d\n", List_size(dependencyList));
  for ( i=0; i<List_size(dependencyList); i++ )
  {
    idx = List_get(dependencyList, i);
    printf(" %d %s\n", *idx, om->names[*idx]);
  }
#endif
#ifdef _DEBUG
  printf("BEFORE EVENTs RULE SET:\n");
  for ( i=0; i<om->nassbeforeevents; i++ )
  {
    char * eq;
    nonzeroElem_t *ordered = om->assignmentsBeforeEvents[i];
    int idxI = ordered->i;
    int idxJ = ordered->j;
    printf("rule %d: ", i);
    if ( idxI == -1 )
      printf("INIT.ass: %s = ", om->names[ idxJ ]);
    else
      printf("norm.ass: %s = ", om->names[ idxI ]);

    eq = SBML_formulaToString(ordered->ij);
    printf("%s\n", eq);
    free(eq);
  }
  printf("\n");
#endif

  /* free dependency list */
  List_freeItems(dependencyList, free, int);
  List_free(dependencyList);

  /* free boolean arrays */
  free(changedBySolver);
  free(requiredForODEs);
  free(requiredForEvents);

  //printf("ODEModel_topologicalRuleSort checkpoint #14...\n");

  return hasCycle;
}


/* adds the contents of 'source' to the end of 'target'.
   List items are only shallow copied. */
static void List_append(List_t *target, List_t *source)
{
  int i;

  for (i = 0; i != List_size(source); i++)
    List_add(target, List_get(source, i));
}

/* allocates memory for substructures of a new odeModel:
   1) ODEs: writes variable and parameter names, creates equation,
   2) DISCONTINUITIES: writes equations for events and initial assignments
   and returns a pointer to the
   the newly created odeModel. */
static odeModel_t *ODEModel_fillStructures(Model_t *ode)
{
  int i, j, found, flag, nvalues, neq, nalg, nconst, nass, npiecewise;
  Compartment_t *c;
  Parameter_t *p;
  Species_t *s;
  Rule_t *rl;
  SBMLTypeCode_t type;
  ASTNode_t *math;
  odeModel_t *om;

  /* 1: ODE SYSTEM */

  /* size of ODE/DAE system */
  neq     = 0;
  nalg    = 0;
  nconst  = 0;
  nass    = 0;
  nvalues = 0;
  found   = 0;

  /* counting number of equations (ODEs/rateRules and assignment Rules)
     to initialize cvodeData structure. Any other occuring values are
     stored as parameters. */

  for ( j=0; j<Model_getNumRules(ode); j++ )
  {
    rl = Model_getRule(ode,j);
    type = SBase_getTypeCode((SBase_t *)rl);
    if ( type == SBML_RATE_RULE ) neq++;
    if ( type == SBML_ALGEBRAIC_RULE ) nalg++;
    if ( type == SBML_ASSIGNMENT_RULE ) nass++;
  }

  nvalues = Model_getNumCompartments(ode) + Model_getNumSpecies(ode) +
    Model_getNumParameters(ode);

  nconst = nvalues - nass - neq - nalg;


  om = ODEModel_allocate(neq, nconst, nass, nalg);
  /* if om is NULL memory allocation failed, just pass on NULL */
  if ( om == NULL ) return om;


  /* filling the IDs and inital values
     of all rate rules (ODEs) and assignment rules the ODE model */

  neq  = 0;
  nass = 0;
  nalg = 0;

  for ( j=0; j<Model_getNumRules(ode); j++ )
  {
    rl = Model_getRule(ode,j);
    type = SBase_getTypeCode((SBase_t *)rl);

    if ( type == SBML_RATE_RULE )
    {
      ASSIGN_NEW_MEMORY_BLOCK(om->names[neq],
			      strlen(Rule_getVariable(rl))+1,
			      char, NULL);
      sprintf(om->names[neq], "%s", Rule_getVariable(rl));
      neq++;
    }
    else if ( type == SBML_ASSIGNMENT_RULE )
    {
      ASSIGN_NEW_MEMORY_BLOCK(om->names[om->neq+nass],
			      strlen(Rule_getVariable(rl))+1,
			      char, NULL);
      sprintf(om->names[om->neq + nass], "%s", Rule_getVariable(rl));
      nass++;
    }
    else if ( type == SBML_ALGEBRAIC_RULE )
    {
      /* find variables defined by algebraic rules here! */
/*       ASSIGN_NEW_MEMORY_BLOCK(om->names[nvalues + nalg], */
/*			      strlen("tmp")+3, char, NULL); */
/*       sprintf(om->names[om->neq+om->nass+om->nconst+ nalg], "tmp%d", nalg); */
      /* printf("tmp%d \n", nalg); */
      nalg++;
    }
  }



  /* filling constants, i.e. all values in the model, that are not
     defined by an assignment or rate rule */

  nconst = 0;
  for ( i=0; i<Model_getNumCompartments(ode); i++ )
  {
    found = 0;
    c = Model_getCompartment(ode, i);

    for ( j=0; j<neq+nass; j++ )
      if ( strcmp(Compartment_getId(c), om->names[j]) == 0 )
	found ++;

    if ( !found )
    {
      ASSIGN_NEW_MEMORY_BLOCK(om->names[neq+nass+nconst],
			      strlen(Compartment_getId(c))+1, char, NULL);
      sprintf(om->names[neq+nass+nconst], "%s", Compartment_getId(c));
      nconst++;
    }
  }
  for ( i=0; i<Model_getNumSpecies(ode); i++ )
  {
    found = 0;
    s = Model_getSpecies(ode, i);

    for ( j=0; j<neq+nass; j++ )
      if ( strcmp(Species_getId(s), om->names[j]) == 0 )
	found ++;

    if ( !found )
    {
      ASSIGN_NEW_MEMORY_BLOCK(om->names[neq+nass+nconst],
			      strlen(Species_getId(s))+1, char, NULL);
      sprintf(om->names[neq+nass+nconst], "%s", Species_getId(s));
      nconst++;
    }
  }
  for ( i=0; i<Model_getNumParameters(ode); i++ )
  {
    found = 0;
    p = Model_getParameter(ode, i);
    for ( j=0; j<neq+nass; j++ )
      if ( strcmp(Parameter_getId(p), om->names[j]) == 0 )
	found ++;

    if ( !found )
    {
      ASSIGN_NEW_MEMORY_BLOCK(om->names[neq+nass+nconst],
			      strlen(Parameter_getId(p))+1, char, NULL);
      sprintf(om->names[neq+nass+nconst], "%s", Parameter_getId(p));
      nconst++;
    }
  }

  /* Writing and Indexing Formulas: Indexing rate rules and assignment
     rules, using the string array created above and writing the
     indexed formulas to the CvodeData structure. These AST are used
     for evaluation during the integration routines!!
     At the same time, check whether the ASTs contain piecewise functions,
     which can lead to discontinuities in the RHS side of the ODE System
     and require special CVODES solver mode CV_NORMAL_TSTOP */

  neq = 0;
  nass = 0;
  nalg = 0;

  npiecewise = 0;

  for ( j=0; j<Model_getNumRules(ode); j++ )
  {
    rl = Model_getRule(ode, j);
    type = SBase_getTypeCode((SBase_t *)rl);

    if ( type == SBML_RATE_RULE )
    {
      math = indexAST(Rule_getMath(rl), nvalues, om->names);
      om->ode[neq] = math;
#ifdef ARITHMETIC_TEST
      ASSIGN_NEW_MEMORY(om->odecode[neq], directCode_t, NULL);
      om->odecode[neq]->eqn = math;
      generateFunction(om->odecode[neq], math);
#endif
      npiecewise += ASTNode_containsPiecewise(math);
      neq++;
    }
    else if ( type == SBML_ASSIGNMENT_RULE )
    {
      math = indexAST(Rule_getMath(rl), nvalues, om->names);
      om->assignment[nass] = math;
#ifdef ARITHMETIC_TEST
      ASSIGN_NEW_MEMORY(om->assignmentcode[nass], directCode_t, NULL);
      om->assignmentcode[nass]->eqn = math;
      generateFunction(om->assignmentcode[nass], math);
#endif
      npiecewise += ASTNode_containsPiecewise(math);
      nass++;
    }
    else if ( type == SBML_ALGEBRAIC_RULE )
    {
      math = indexAST(Rule_getMath(rl), nvalues, om->names);
      om->algebraic[nalg] = math;
      npiecewise += ASTNode_containsPiecewise(math);
      nalg++;
    }
  }

  om->simple = ode;
  /* set assignment order to NULL: done later */
  om->assignmentOrder = NULL;
  om->initAssignmentOrder = NULL;
  /* set Jacobi to NULL: done later */
  om->jacob = NULL;
  om->jacobcode = NULL;
  om->jacobSparse = NULL;
  /* set construction flag to zero: done later */
  om->jacobian = 0;
  /* set failed flag to zero: done later */
  om->jacobianFailed = 0;


  /* set counted piecewise expressions */
  om->npiecewise = npiecewise;

  /* ... finally, retrieve values (initial conditions and parameters )
     from input model */
  ODEModel_initializeValuesFromSBML(om, ode);



  /* 2: DISCONTINUITIES */

  flag = ODEModel_setDiscontinuities(om, ode);
  if ( flag == -1 ) /* -1 memory allocation failures */
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_ODE_MODEL_SET_DISCONTINUITIES_FAILED,
		      "setting discontinuity structures (initial assignments," \
		      "events) failed.");
    ODEModel_freeDiscontinuities(om);
  }

  return om;
}

/* returns 1 for success or -1 for memory allocation failure */
static int ODEModel_setDiscontinuities(odeModel_t *om, Model_t *ode)
{
  int i, j, flag;
  int nvalues, nevents, ninitAss, neventAss;
  ASTNode_t *math;

  nvalues = om->neq + om->nass + om->nconst;

  /* size of discontinuities */
  nevents = 0;
  neventAss = 0;
  ninitAss = 0;
  if ( ode != NULL )
  {
    nevents = Model_getNumEvents(ode);
    for ( i=0; i<nevents; i++ )
      neventAss += Event_getNumEventAssignments(Model_getEvent(ode, i));
    ninitAss = Model_getNumInitialAssignments(ode);
  }
  /* allocate basic structures */
  flag = ODEModel_allocateDiscontinuities(om, nvalues,
					  nevents, neventAss, ninitAss);
  if ( flag == -1 ) return -1; /* pass on memory failure signal -1 */

  for ( i=0; i<nvalues; i++ ) /* initialize map to -1 */
    om->indexInit[i] = -1;


  /* allocate and fill equations */
  for ( i=0; i<ninitAss; i++ )
  {
    const InitialAssignment_t *init = Model_getInitialAssignment(ode, i);
    const char *id = InitialAssignment_getSymbol(init);
    int idx = ODEModel_getVariableIndexFields(om, id);
    om->initIndex[i] = idx; /* map from om->initAssignment to om->names */
    om->indexInit[idx] = i; /* map from om->names to om->initAssignemnt */
    math = indexAST(InitialAssignment_getMath(init), nvalues, om->names);
    om->initAssignment[i] = math;
#ifdef ARITHMETIC_TEST
    ASSIGN_NEW_MEMORY(om->initAssignmentcode[i], directCode_t, -1);
    om->initAssignmentcode[i]->eqn = math;
    generateFunction(om->initAssignmentcode[i], math);
#endif
  }

  for ( i=0; i<nevents; i++ )
  {
    int nea;
    Event_t *e = Model_getEvent(ode, i);
    math = indexAST(Trigger_getMath(Event_getTrigger(e)),
		    nvalues, om->names);
    om->event[i] = math;
#ifdef ARITHMETIC_TEST
    ASSIGN_NEW_MEMORY(om->eventcode[i], directCode_t, -1);
    om->eventcode[i]->eqn = math;
    generateFunction(om->eventcode[i], math);
#endif
    /* event assignments */
    nea = Event_getNumEventAssignments(e);
    om->neventAss[i] = nea;
    ASSIGN_NEW_MEMORY_BLOCK(om->eventIndex[i], nea, int, -1);
    ASSIGN_NEW_MEMORY_BLOCK(om->eventAssignment[i], nea, ASTNode_t *, -1);
    ASSIGN_NEW_MEMORY_BLOCK(om->eventAssignmentcode[i], nea, directCode_t *,-1);

    for ( j=0; j<nea; j++ )
    {
      const EventAssignment_t *ea = Event_getEventAssignment(e, j);
      om->eventIndex[i][j] =
	ODEModel_getVariableIndexFields(om, EventAssignment_getVariable(ea));
      math = indexAST(EventAssignment_getMath(ea), nvalues, om->names);
      om->eventAssignment[i][j] = math;

#ifdef ARITHMETIC_TEST
      ASSIGN_NEW_MEMORY(om->eventAssignmentcode[i][j], directCode_t, -1);
      om->eventAssignmentcode[i][j]->eqn = math;
      generateFunction(om->eventAssignmentcode[i][j], math);
#endif
    }
  }
  return 1;
}

/* initializes values in odeModel from SBML file */
static void  ODEModel_initializeValuesFromSBML(odeModel_t *om, Model_t *ode)
{
  int i, nvalues;
  nvalues = om->neq + om->nass + om->nconst;
  /* initial conditions */
  for ( i=0; i<om->neq; i++ )
    om->values[i] = Model_getValueById(ode, om->names[i]);

  /* parameters */
  for ( i=(om->neq+om->nass); i<nvalues; i++ )
    om->values[i] = Model_getValueById(ode, om->names[i]);
}

/* free discontinuities */
static int ODEModel_freeDiscontinuities(odeModel_t *om)
{
  int i, j;

  /* initial assignments */
  for ( i=0; i<om->ninitAss; i++ )
    ASTNode_free(om->initAssignment[i]);
  free(om->initAssignment);
  free(om->indexInit);
  free(om->initIndex);

#ifdef ARITHMETIC_TEST
  for ( i=0; i<om->ninitAss; i++ )
  {
    destructFunction(om->initAssignmentcode[i]);
    free(om->initAssignmentcode[i]);
  }
#endif
  free(om->initAssignmentcode);

  /* free global assignment ordering via init array */
  for ( i=0; i<(om->nass + om->ninitAss); i++ )
  {
    if ( om->initAssignmentOrder != NULL )
      free(om->initAssignmentOrder[i]);
  }
  /* free init. ass. array */
  if ( om->initAssignmentOrder != NULL )
    free(om->initAssignmentOrder);

  /* events */
#ifdef ARITHMETIC_TEST
  for ( i=0; i<om->nevents; i++ )
  {
    destructFunction(om->eventcode[i]);
    free(om->eventcode[i]);
    for ( j=0; j<om->neventAss[i]; j++ )
    {
      destructFunction(om->eventAssignmentcode[i][j]);
      free(om->eventAssignmentcode[i][j]);
    }
  }
#endif
  for ( i=0; i<om->nevents; i++ )
  {
    ASTNode_free(om->event[i]);
    for ( j=0; j<om->neventAss[i]; j++ )
      ASTNode_free(om->eventAssignment[i][j]);
    free(om->eventIndex[i]);
    free(om->eventAssignment[i]);
    free(om->eventAssignmentcode[i]);
  }
  free(om->event);
  free(om->eventcode);
  free(om->neventAss);
  free(om->eventIndex);
  free(om->eventAssignment);
  free(om->eventAssignmentcode);

  /* rule ordering */
  for ( i=0; i<om->nassbeforeevents; i++ )
    free(om->assignmentsBeforeEvents[i]);
  free(om->assignmentsBeforeEvents);

  return 1;
}

/* allocates memory for a new odeModel structure and returns
   a pointer to it, returns 1 for success and -1 for memory
   allocation failure */
static int ODEModel_allocateDiscontinuities(odeModel_t *om, int nvalues,
					    int nevents, int neventAss,
					    int ninitAss)
{
  /* initial assignments */
  om->ninitAss  = ninitAss;
  ASSIGN_NEW_MEMORY_BLOCK(om->indexInit, nvalues, int, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->initIndex, ninitAss, int, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->initAssignment, ninitAss, ASTNode_t *, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->initAssignmentcode, ninitAss, directCode_t *, -1);

  /* events and event assignments */
  om->nevents = nevents;
  ASSIGN_NEW_MEMORY_BLOCK(om->event, nevents, ASTNode_t *, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->neventAss, nevents, int, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->eventIndex, nevents, int *, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->eventcode, nevents, directCode_t *, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->eventAssignment, nevents, ASTNode_t **, -1);
  ASSIGN_NEW_MEMORY_BLOCK(om->eventAssignmentcode, nevents, directCode_t **,-1);

  return 1;
}

/* allocates memory for a new odeModel structure and returns
   a pointer to it */
static odeModel_t *ODEModel_allocate(int neq, int nconst, int nass, int nalg)
{
  odeModel_t *om;
  int nvalues;

  ASSIGN_NEW_MEMORY(om, odeModel_t, NULL);
  /* init. to 0 first */
  om->neq    = 0;
  om->nconst = 0;
  om->nass   = 0;
  om->nalg   = 0;

  nvalues = neq + nalg + nass + nconst;

  /* names */
  ASSIGN_NEW_MEMORY_BLOCK(om->names, nvalues, char *, NULL);

  /* values : used for storing initial values only */
  ASSIGN_NEW_MEMORY_BLOCK(om->values, nvalues, realtype, NULL);

  /* equations */
  ASSIGN_NEW_MEMORY_BLOCK(om->ode, neq, ASTNode_t *, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(om->assignment, nass, ASTNode_t *, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(om->algebraic, nalg, ASTNode_t *, NULL);

  /* compiled equations */
  ASSIGN_NEW_MEMORY_BLOCK(om->odecode, neq, directCode_t *, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(om->assignmentcode, nass, directCode_t *, NULL);

/*   ASSIGN_NEW_MEMORY_BLOCK(om->algebraiccode, nalg, directCode_t *, NULL); */

  om->neq    = neq;
  om->nconst = nconst;
  om->nass   = nass;
  om->nalg   = nalg; /*!!! this causes crash at the moment, because
		      ODEs have been constructed for that
		      should be defined by alg. rules */

  /* set discontinuities to 0 */
  om->nevents = 0;
  om->neventAss = 0;
  om->ninitAss  = 0;

  /* set compiled function pointers to NULL */
  om->compiledCVODEFunctionCode = NULL;
  om->compiledCVODEJacobianFunction = NULL;
  om->compiledCVODERhsFunction = NULL;
  om->compiledCVODEAdjointRhsFunction = NULL;
  om->compiledCVODEAdjointJacobianFunction = NULL;

  /* objective function */
  /*!!!TODO : move to separate structure */
  om->vector_v = NULL;
  om->ObjectiveFunction = NULL;
  om->discrete_observation_data = 0;
  om->compute_vector_v = 0;
  om->time_series = NULL;

  return om ;
}


/** \brief Create internal model odeModel from an SBML file, that
    contains level 1 or level 2 SBML.

    Conversion of level 1 to level 2 models is done internally.
*/

SBML_ODESOLVER_API odeModel_t *ODEModel_createFromFile(const char *sbmlFileName)
{

  SBMLDocument_t *d;
  odeModel_t *om;
    
    printf("ODEModel_createFromFile was called... %s\n",sbmlFileName);
    
  d =  parseModel((char *)sbmlFileName,
		  0 /* print message */,
		  0 /* don't validate */);

  if ( d == NULL ) {
    printf("Null SBMLDocument_t obtained in ODEModel_createFromFile...\n");
    return NULL;
  } //else {
  //  printf("SBMLDocument_t successfully created in ODEModel_createFromFile...\n");
  //}

  om = ODEModel_createFromSBML2(d);
  if ( om == NULL ) { 
    printf("Null odeModel_t obtained in ODEModel_createFromFile...\n");
    return NULL;
  } //else {
  //  printf("odeModel_t successfully created in ODEModel_createFromFile...\n");
  //}

  /* remember for freeing afterwards */
  om->d = d;

  return om;
}


/** \brief Create internal model odeModel_t from SBMLDocument containing
    a level 2 SBML model.
*/

SBML_ODESOLVER_API odeModel_t *ODEModel_createFromSBML2(SBMLDocument_t *d)
{
  Model_t *m = NULL;
  odeModel_t *om = NULL;
  
  //printf("Called ODEModel_createFromSBML2...\n");
  
  if ( SBMLDocument_getLevel(d) == 1 )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_DOCUMENTLEVEL_ONE,
		      "SBML Level %d cannot be processed with function"
		      " ODEModel_createFromSBML2",
		      SBMLDocument_getLevel(d));
    return NULL;
  } //else {
  //  printf("SBML document is SBML level 2 or greater...\n");
  //}

  m = SBMLDocument_getModel(d);
  
  if( m == NULL ){
    printf("Model_t* is NULL. Returning NULL pointer in ODEModel_createFromSBML2...\n");
    return NULL;
  } //else {
  //  printf("Model_t* successfully created. Now calling ODEModel_create with Model_t*...\n");
  //}

  om = ODEModel_create(m);
  /* if om is NULL a memory allocation failure has occured, pass on NULL */
  if ( om == NULL ) {
    printf("Memory allocation failure in ODEModel_createFromSBML2. Returning NULL pointer...\n");
    return NULL;
  } //else {
   // printf("odeModel_t * successfully created in ODEModel_createFromSBML2...\n");
  //}

  return om;
}


/** Create odeModel_t directly:
    This function allows to create the internal odeModel_t structure
    independently from SBML. This structure can then be used to create
    and run integratorInstance_t, including all sensitivity analysis
    features.

    The formulae, both ODEs and assignments, can be passed as an array
    `f' of libSBML ASTs. `neq' is the number of ODEs, `nass' is
    the number of assignments and the passed array `f' contains both
    ODEs and assignments in this order. Assignment rules must currently
    occur in correct order, i.e. an assignment rule MUST NOT DEPEND on a
    subsequent assignment rule! See SBML Level 2 Version 1 specification
    for details on this restriction on assignment rules.
    The passed `names' and `values' arrays are of size neq+nass+nconst and
    contain names and values of ODE variables, assigned variables and
    model parameters in this order and in the same order as ASTs in `f'.
*/

SBML_ODESOLVER_API odeModel_t *ODEModel_createFromODEs(ASTNode_t **f, int neq, int nass, int nconst, char **names, realtype *values, Model_t *events)
{
  int i, nvalues, flag;
  odeModel_t *om;

  nvalues = neq + nass + nconst;

  /* allocate odeModel structure and set values */
  om = ODEModel_allocate(neq, nconst, nass, 0);
  /* if om is NULL memory allocation failed, just pass on NULL */
  if ( om == NULL ) return om;


  /* set SBML input to NULL */
  om->d = NULL;
  om->m = NULL;
  /* set optional SBML model containing events */
  om->simple = events;

  /* set ODEs with indexed ASTs */
  for ( i=0; i<neq; i++ )
    om->ode[i] = indexAST(f[i], nvalues, names);

  /* set assignments */
  for ( i=0; i<nass; i++ )
    om->assignment[i] = indexAST(f[neq+i], nvalues, names);

  /* set names and values */
  for ( i=0; i<neq+nass+nconst; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(om->names[i], strlen(names[i]) + 1, char, NULL);
    strcpy(om->names[i], names[i]);
  }

  /* set discontinuities from input modelevents, initial assignments */
  flag = ODEModel_setDiscontinuities(om, events);
  if ( flag == -1 ) /* -1 memory allocation failures */
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_ODE_MODEL_SET_DISCONTINUITIES_FAILED,
		      "setting discontinuity structures (initial assignments," \
		      "events) failed");
    ODEModel_freeDiscontinuities(om);
  }

  /* set values: initial conditions and parameters */
  for ( i=0; i<neq+nass+nconst; i++ )
    om->values[i] = values[i];


  /* generate ordered list of assignment rules for evaluation order */
  /* -1 memory allocation failures */
  om->hasCycle = ODEModel_topologicalRuleSort(om);

  return om;

}

/** \brief Frees the odeModel structures
 */
SBML_ODESOLVER_API void ODEModel_free(odeModel_t *om)
{

  int i;

  if (om == NULL) return;

  for ( i=0; i<om->neq+om->nass+om->nconst; i++ )
  {
    free(om->names[i]);
    free(om->dependencyMatrix[i]);
  }
  free(om->names);
  free(om->dependencyMatrix);

  /* free ODEs */
  for ( i=0; i<om->neq; i++ )
    ASTNode_free(om->ode[i]);
  free(om->ode);

  /* free compiled ODEs */
#ifdef ARITHMETIC_TEST
  for ( i=0; i<om->neq; i++ )
  {
    destructFunction(om->odecode[i]);
    free(om->odecode[i]);
  }
#endif
  free(om->odecode);



  /* free global assignment ordering via init array */
  for ( i=0; i<om->nass; i++ )
  {
    if ( om->assignmentOrder != NULL )
      free(om->assignmentOrder[i]);
  }
  /* free ass. array */
  if ( om->assignmentOrder != NULL )
    free(om->assignmentOrder);

  /* free assignments */
  for ( i=0; i<om->nass; i++ )
  {
    ASTNode_free(om->assignment[i]);
  }
  free(om->assignment);

#ifdef ARITHMETIC_TEST
  for ( i=0; i<om->nass; i++ )
  {
    destructFunction(om->assignmentcode[i]);
    free(om->assignmentcode[i]);
  }
#endif
  free(om->assignmentcode);


  /* free algebraic rules */
  for ( i=0; i<om->nalg; i++ )
    ASTNode_free(om->algebraic[i]);
  free(om->algebraic);

  /* free Jacobian matrix, if it has been constructed */
  ODEModel_freeJacobian(om);

  /* free discontinuities */
  ODEModel_freeDiscontinuities(om);


  /* free objective function AST if it has been constructed */
  if ( om->ObjectiveFunction != NULL )
    ASTNode_free(om->ObjectiveFunction);
  om->ObjectiveFunction = NULL;

  /* free linear objective function AST's if constructed */
  if ( om->vector_v != NULL )
    for ( i=0; i<om->neq; i++ )
      ASTNode_free(om->vector_v[i]);
  free(om->vector_v);

  /* free time_series, if present */
  if ( om->time_series != NULL )
    free_data( om->time_series );

  /* free simplified ODE model */
  if ( om->simple != NULL ) Model_free(om->simple);

  /* free document, if model was constructed from file */
  if ( om->d != NULL ) SBMLDocument_free(om->d);

  /* free values structure from SBML independent odeModel */
  if ( om->values != NULL ) free(om->values);

  /* free compiled code */
  if ( om->compiledCVODEFunctionCode != NULL )
  {
    CompiledCode_free(om->compiledCVODEFunctionCode);
    om->compiledCVODEFunctionCode = NULL;
  }

  /* free assignment evaulation ordering */
  for ( i=0; i<om->nassbeforeodes; i++ )
    free(om->assignmentsBeforeODEs[i]);
  free(om->assignmentsBeforeODEs);

  /* free model structure */
  free(om);
}

/** \brief Returns 1 if a variable or parameter with the SBML id
    exists in the ODEModel.
*/

SBML_ODESOLVER_API int ODEModel_hasVariable(odeModel_t *model, const char *symbol)
{
  return ODEModel_getVariableIndexFields(model, symbol) != -1;
}


/** \brief Returns the total number of values in odeModel, equivalent
    to ODEModel_getNeq + ODEModel_getNumAssignments +
    ODEModel_getNumConstants
*/

SBML_ODESOLVER_API int ODEModel_getNumValues(odeModel_t *om)
{
  return om->neq + om->nass + om->nconst + om->nalg ;
}


/** \brief Returns the number of ODEs (number of equations) in the
    odeModel
*/

SBML_ODESOLVER_API int ODEModel_getNeq(odeModel_t *om)
{
  return om->neq;
}


/** \brief Returns the number parameters for which sensitivity
    analysis might be requested

*/

SBML_ODESOLVER_API int ODESense_getNsens(odeSense_t *os)
{
  return os->nsens;
}

/** \brief Returns the number variables for which sensitivity
    analysis might be requested, equals NEQ of odeModel

*/
SBML_ODESOLVER_API int ODESense_getNeq(odeSense_t *os)
{
  return os->neq;
}

/** \brief Returns the ODE from the odeModel for the variable with
    variableIndex vi.

    The ODE is returned as an `indexed abstract syntax tree' (iAST),
    which is an extension to the usual libSBML AST. Every AST_NAME
    type node in the tree has been replaced by an ASTIndexNameNode,
    that allows a O(1) retrieval of values for this node from an array
    of all values of the odeModel.
*/

SBML_ODESOLVER_API const ASTNode_t *ODEModel_getOde(odeModel_t *om, variableIndex_t *vi)
{
  if ( 0 <= vi->index && vi->index < om->neq )
    return (const ASTNode_t *) om->ode[vi->index];
  else return NULL;
}



/** \brief Returns the number of variable assignments in the odeModel
 */

SBML_ODESOLVER_API int ODEModel_getNumAssignments(odeModel_t *om)
{
  return om->nass;
}


/** \brief Returns the assignment formula from the odeModel for the
    variable with variableIndex vi

    The ODE is returned as an `indexed abstract syntax tree' (iAST),
    which is an extension to the usual libSBML AST. Every AST_NAME
    type node in the tree has been replaced by an ASTIndexNameNode,
    that allows a O(1) retrieval of value for this node from an array
    of all values of the odeModel.
*/

SBML_ODESOLVER_API const ASTNode_t *ODEModel_getAssignment(odeModel_t *om, variableIndex_t *vi)
{
  if (  0 <= vi->type_index && vi->type_index < om->nass )
    return (const ASTNode_t *) om->assignment[vi->type_index];
  else return NULL;
}

/** \brief Returns the number of constant parameters of the odeModel
 */

SBML_ODESOLVER_API int ODEModel_getNumConstants(odeModel_t *om)
{
  return om->nconst;
}


/** \brief Returns the number variables that are defined by
    an algebraic rule.

    As SBML Algebraic Rules and ODE models with algebraic constraints (DAE
    models) can currently not be handled, this function is
    of no use.
*/

SBML_ODESOLVER_API int ODEModel_getNalg(odeModel_t *om)
{
  return om->nalg;
}


/** \brief Prints the names (SBML IDs) of all model variables
    and parameters
*/

SBML_ODESOLVER_API void ODEModel_dumpNames(odeModel_t *om)
{
  int i;
  for ( i=0; i<(om->neq+om->nass+om->nconst+om->nalg); i++ )
    printf("%s ", om->names[i]);
  printf("\n");
}


/** \brief Returns the SBML model that has been extracted from the input
    SBML model's reaction network and structures

    The model contains only compartments, species, parameters and SBML
    Rules. Changes to this model will have no effect on odeModel or
    integratorInstance.
*/

SBML_ODESOLVER_API const Model_t *ODEModel_getOdeSBML(odeModel_t *om)
{
  return (const Model_t *) om->simple;
}

/** deprecated, please use ODEModel_getOdeSBML instead */
SBML_ODESOLVER_API const Model_t *ODEModel_getModel(odeModel_t *om)
{
  return ODEModel_getOdeSBML(om);
}

/** \brief Returns a pointer to the SBML input model from which the
    ODE system has been created.


*/

SBML_ODESOLVER_API const Model_t *ODEModel_getInputSBML(odeModel_t *om)
{
  return (const Model_t *) om->m;
}

/** @} */


/*! \defgroup jacobian Jacobian Matrix: J = df(x)/dx
  \ingroup odeModel
  \brief Constructing and Interfacing the Jacobian matrix of an ODE
  system

  as used for CVODES and IDA Dense Solvers
*/
/*@{*/

/** \brief Construct Jacobian Matrix for ODEModel.

Once an ODE system has been constructed from an SBML model, this
function calculates the derivative of each species' ODE with respect
to all other species for which an ODE exists, i.e. it constructs the
jacobian matrix of the ODE system. At the moment this matrix is
freed together with the ODE model. A separate function will be available
soon.\n
Returns 1 if successful, 0 otherwise, and -1 for memory allocation failures
*/

SBML_ODESOLVER_API int ODEModel_constructJacobian(odeModel_t *om)
{
  unsigned int uk;
  int i, j, failed, nvalues;
  double val;
  ASTNode_t *fprime, *simple, *index, *ode;
  List_t *names, *sparse;

  if ( om == NULL ) return 0;

  /******************** Calculate Jacobian ************************/

  failed = 0;
  nvalues = om->neq + om->nass + om->nconst;

  ASSIGN_NEW_MEMORY_BLOCK(om->jacob, om->neq, ASTNode_t **, -1);
  /* compiled equations */
  ASSIGN_NEW_MEMORY_BLOCK(om->jacobcode, om->neq, directCode_t **, -1);
  for ( i=0; i<om->neq; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(om->jacob[i], om->neq, ASTNode_t *, -1);
    ASSIGN_NEW_MEMORY_BLOCK(om->jacobcode[i], om->neq, directCode_t *, -1);
  }

  /* create list to remember non-zero elements of the Jacobi matrix */
  sparse = List_create();
  om->sparsesize = 0;
/*   fprintf(stderr, "GENERATING JACOBI: neq = %d\n", om->neq); */

  for ( i=0; i<om->neq; i++ )
  {
    ode = copyAST(om->ode[i]);

    /* assignment rule replacement: reverse to satisfy
       SBML specifications that variables defined by
       an assignment rule can appear in rules declared afterwards */
    for ( j=om->nass-1; j>=0; j-- )
      AST_replaceNameByFormula(ode,
			       om->names[om->neq + j], om->assignment[j]);


    for ( j=0; j<om->neq; j++ )
    {
      /* 1: differentiate ODE */
      fprime = differentiateAST(ode, om->names[j]);
      simple = simplifyAST(fprime);
      ASTNode_free(fprime);
      index = indexAST(simple, nvalues, om->names);
      ASTNode_free(simple);
      om->jacob[i][j] = index;

      /* 2: generate list of non-zero Jacobi elements */

      /* check whether jacobian is 0 */
      val = 1;
      if ( ASTNode_isInteger(index) )
	val = (double) ASTNode_getInteger(index) ;
      if ( ASTNode_isReal(index) )
	val = ASTNode_getReal(index) ;

      if ( val != 0.0 )
      {

	/* 3: generate compiled ODE */
#ifdef ARITHMETIC_TEST
	ASSIGN_NEW_MEMORY(om->jacobcode[i][j], directCode_t, -1);
	om->jacobcode[i][j]->eqn = index;
	generateFunction(om->jacobcode[i][j], index);
#endif

	/* 4: generate sparse list */
	nonzeroElem_t *nonzero;
	ASSIGN_NEW_MEMORY(nonzero, nonzeroElem_t, -1);
	nonzero->i = i;
	nonzero->j = j;
	nonzero->ij = om->jacob[i][j];
	nonzero->ijcode = om->jacobcode[i][j];

	List_add(sparse, nonzero);
	om->sparsesize++;
      }
      else
	{
#ifdef ARITHMETIC_TEST
	  om->jacobcode[i][j] = NULL;
#endif

	/* can be optionally done to save memory for large models */
/*      ASTNode_free(om->jacob[i][j]); /\* free 0 elements *\/ */
/*      om->jacob[i][j] = NULL; */
	}

      /* 5: check if the AST contains a failure notice */
      names = ASTNode_getListOfNodes(index ,
				     (ASTNodePredicate) ASTNode_isName);
      for ( uk=0; uk<List_size(names); uk++ )
	if ( strcmp(ASTNode_getName(List_get(names,uk)),
		    "differentiation_failed") == 0 )
	  failed++;
      List_free(names);
    }

    ASTNode_free(ode);
  }

  if ( failed != 0 )
  {
    /** if Jacobi matrix construction failed, the equations are still
	kept for users to check which equation were non-differentiable
	(by SOSlib ;)). To save memory the matrix can however be freed
	for following integration runs, by calling
	ODEModel_freeJacobian(om) */
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_ENTRIES_OF_THE_JACOBIAN_MATRIX_COULD_NOT_BE_CONSTRUCTED,
		      "%d entries of the Jacobian matrix could not be "
		      "constructed, due to failure of differentiation. "
		      "Cvode will use internal approximation of the "
		      "Jacobian instead.", failed);
    om->jacobian = 0;
  }
  else om->jacobian = 1;

  om->jacobianFailed = failed;


  /* 6: generate non-zero element array from list */
/*   fprintf(stderr,"USING SPARSE JACOBI: %d of %d elements are non-zero ...", */
/*   List_size(sparse), om->neq*om->neq); */

  ASSIGN_NEW_MEMORY_BLOCK(om->jacobSparse, om->sparsesize, nonzeroElem_t *, -1);
  for ( i=0; i<om->sparsesize; i++ )
    om->jacobSparse[i] = List_get(sparse, i);
  List_free(sparse);
  /*   fprintf(stderr,"... finished\n"); */

  return om->jacobian;
}


/** \brief Free the Jacobian matrix of the ODEModel.
 */

SBML_ODESOLVER_API void ODEModel_freeJacobian(odeModel_t *om)
{
  int i, j;
  if ( om->jacob != NULL )
  {

    /* free compiled function via array of non-zero entries */
#ifdef ARITHMETIC_TEST
    /* free compiledCode function */
    for ( i=0; i<om->sparsesize; i++ )
    {
      nonzeroElem_t *nonzero = om->jacobSparse[i];
      destructFunction(nonzero->ijcode);
    }
#endif

    /* free full matrix */
    for ( i=0; i<om->neq; i++ )
    {
      for ( j=0; j<om->neq; j++ )
      {
	ASTNode_free(om->jacob[i][j]);
#ifdef ARITHMETIC_TEST
	free(om->jacobcode[i][j]);
#endif
      }
      free(om->jacob[i]);
      free(om->jacobcode[i]);
    }
    free(om->jacob);
    free(om->jacobcode);
    om->jacob = NULL;

    /* free  array of non-zero entries */
    for ( i=0; i<om->sparsesize; i++ )
    {
      free(om->jacobSparse[i]);
    }
    free(om->jacobSparse);
  }
  om->jacobian = 0;
}

/**  \brief Returns the ith/jth entry of the jacobian matrix

     Returns NULL if either the jacobian has not been constructed yet,
     or if i or j are >neq. Ownership remains within the odeModel
     structure.
*/

SBML_ODESOLVER_API const ASTNode_t *ODEModel_getJacobianIJEntry(odeModel_t *om, int i, int j)
{
  if ( om->jacob == NULL ) return NULL;
  if ( i >= om->neq || j >= om->neq ) return NULL;
  return (const ASTNode_t *) om->jacob[i][j];
}


/** \brief Returns the entry (d(vi1)/dt)/d(vi2) of the jacobian matrix.

    Returns NULL if either the jacobian has not been constructed yet,
    or if the v1 or vi2 are not ODE variables. Ownership remains
    within the odeModel structure.
*/

SBML_ODESOLVER_API const ASTNode_t *ODEModel_getJacobianEntry(odeModel_t *om, variableIndex_t *vi1, variableIndex_t *vi2)
{
  return ODEModel_getJacobianIJEntry(om, vi1->index, vi2->index);
}


/** \brief Constructs and returns the determinant of the jacobian matrix.

    The calling application takes ownership of the returned ASTNode_t
    and must free it, if not required.
*/

SBML_ODESOLVER_API ASTNode_t *ODEModel_constructDeterminant(odeModel_t *om)
{
  if ( om->jacob != NULL && om->jacobian == 1 )
    return determinantNAST(om->jacob, om->neq);
  else
    return NULL;
}


/** @} */

/************* SENSITIVITY *****************/

int ODESense_constructMatrix(odeSense_t *os, odeModel_t *om)
{
  unsigned int uk;
  int i, j, l, nvalues, failed;
  double val;
  ASTNode_t *ode, *fprime, *simple, *index;
  List_t *names, *sparse;

  ASSIGN_NEW_MEMORY_BLOCK(os->sens, om->neq, ASTNode_t **, -1);
  /* compiled equations */
  ASSIGN_NEW_MEMORY_BLOCK(os->senscode, om->neq, directCode_t **, -1);
  /* simple logic vector of non-zero elements */
  ASSIGN_NEW_MEMORY_BLOCK(os->sensLogic, om->neq, int *, -1);

  /* if only init.cond. sensitivities, nsensP will be 0
     and the matrix will essentially be empty (NULL) */
  for ( i=0; i<om->neq; i++ )
  {
    ASSIGN_NEW_MEMORY_BLOCK(os->sens[i], os->nsensP, ASTNode_t *, -1);
    ASSIGN_NEW_MEMORY_BLOCK(os->senscode[i], os->nsensP, directCode_t *, -1);
    ASSIGN_NEW_MEMORY_BLOCK(os->sensLogic[i], os->nsensP, int, -1);
  }

  /* create list to remember non-zero elements of the Jacobi matrix */
  sparse = List_create();
  os->sparsesize = 0;
  /*   fprintf(stderr, "GENERATING PARAMETER MATRIX: neq=%d * nsens=%d\n", */
  /*      om->neq, os->nsensP); */

  nvalues = om->neq + om->nass + om->nalg + om->nconst;
  failed = 0;
  for ( i=0; i<om->neq; i++ )
  {
    ode = copyAST(om->ode[i]);
    /* assignment rule replacement: reverse to satisfy
       SBML specifications that variables defined by
       an assignment rule can appear in rules declared afterwards */
    for ( j=om->nass-1; j>=0; j-- )
      AST_replaceNameByFormula(ode, om->names[om->neq+j], om->assignment[j]);

    l = 0;
    for ( j=0; j<os->nsens; j++ )
    {
      /* skip species sens. */
      if ( !(os->index_sens[j] < om->neq) )
      {
	/* differentiate d(dYi/dt) / dPj */
	fprime = differentiateAST(ode, om->names[os->index_sens[j]]);
	simple =  simplifyAST(fprime);
	ASTNode_free(fprime);
	index = indexAST(simple, nvalues, om->names);
	ASTNode_free(simple);
	os->sens[i][l] = index;

	/* check whether matrix element is 0 */
	val = 1;
	if ( ASTNode_isInteger(index) )
	  val = (double) ASTNode_getInteger(index) ;
	if ( ASTNode_isReal(index) )
	  val = ASTNode_getReal(index) ;

	if ( val != 0.0 )
	{
#ifdef ARITHMETIC_TEST
	  ASSIGN_NEW_MEMORY(os->senscode[i][l], directCode_t, -1);
	  os->senscode[i][l]->eqn = index;
	  generateFunction(os->senscode[i][l], index);
#endif
	  /* generate sparse list */
	  nonzeroElem_t *nonzero;
	  ASSIGN_NEW_MEMORY(nonzero, nonzeroElem_t, -1);
	  nonzero->i = i;
	  nonzero->j = j;
	  nonzero->ij = os->sens[i][l];
	  nonzero->ijcode = os->senscode[i][l];

	  List_add(sparse, nonzero);
	  os->sparsesize++;

	  /* fill sparse logic */
	  os->sensLogic[i][l] = 1;
	}
	else
	{
	  os->sensLogic[i][l] = 0;
#ifdef ARITHMETIC_TEST
	  os->senscode[i][l] = NULL;
#endif
	}
	/* increase sparse matrix counter */
	l++;

	/* check if the AST contains a failure notice */
	names = ASTNode_getListOfNodes(index,
				       (ASTNodePredicate) ASTNode_isName);

	for ( uk=0; uk<List_size(names); uk++ )
	  if ( strcmp(ASTNode_getName(List_get(names,uk)),
		      "differentiation_failed") == 0 )
	    failed++;
	List_free(names);

      }
    }
    ASTNode_free(ode);
  }


  if ( failed != 0 )
  {
    SolverError_error(WARNING_ERROR_TYPE,
		      SOLVER_ERROR_ENTRIES_OF_THE_PARAMETRIC_MATRIX_COULD_NOT_BE_CONSTRUCTED,
		      "%d entries of the parametric `Jacobian' matrix "
		      "could not be constructed, due to failure of "
		      "differentiation. Cvode will use internal "
		      "approximation instead.",
		      failed);
    /* ODESense_freeMatrix(os); */
    return 0;
  }

  /*  generate non-zero element array from list */
/*   fprintf(stderr,"USING SPARSE PARAM: %d of %d elements are non-zero ...", */
/*     os->sparsesize, om->neq*os->nsensP); */

  ASSIGN_NEW_MEMORY_BLOCK(os->sensSparse, os->sparsesize, nonzeroElem_t *, -1);
  for ( i=0; i<os->sparsesize; i++ )
    os->sensSparse[i] = List_get(sparse, i);
  List_free(sparse);
  /*   fprintf(stderr,"... finished\n"); */


  return 1;

}

void ODESense_freeMatrix(odeSense_t *os)
{
  int i, j;

  if ( os == NULL )
    return;

  /* free parametric matrix, if it has been constructed */
  if ( os->sens != NULL )
  {
    /* free compiled function via array of non-zero entries */
#ifdef ARITHMETIC_TEST
    /* free compiledCode function */
    for ( i=0; i<os->sparsesize; i++ )
    {
      nonzeroElem_t *nonzero = os->sensSparse[i];
      destructFunction(nonzero->ijcode);
    }
#endif

    for ( i=0; i<os->om->neq; i++ )
    {
      for ( j=0; j<os->nsensP; j++ )
      {
	ASTNode_free(os->sens[i][j]);
#ifdef ARITHMETIC_TEST
	free(os->senscode[i][j]);
#endif
      }
      free(os->sens[i]);
      free(os->senscode[i]);
      free(os->sensLogic[i]);
    }
    free(os->sens);
    free(os->senscode);
    os->sens = NULL;

    free(os->sensLogic);
    /* free  array of non-zero entries */
    for ( i=0; i<os->sparsesize; i++ )
    {
      free(os->sensSparse[i]);
    }
    free(os->sensSparse);
  }
}

void ODESense_freeStructures(odeSense_t *os)
{
  if ( os->index_sens != NULL )
    free(os->index_sens);
  if ( os->index_sensP != NULL )
    free(os->index_sensP);
  os->index_sens = NULL;
  os->index_sensP = NULL;
}


/*! \defgroup parametric Sensitivity Matrix: P = df(x)/dp
  \ingroup odeModel
  \brief Constructing and Interfacing the sensitivity matrix of
  an ODE system

  as used for CVODES sensitivity analysis
*/
/*@{*/


/** Construct Sensitivity Matrix for ODEModel.

    Once an ODE system has been constructed from an SBML model, this
    function calculates the derivative of each species' ODE with respect
    to all (global) constants of the SBML model.

    To calculate a matrix w.r.t. only a subset of model constants, the
    SBML IDs must by additionally passed via the function
    ODESense_create(odeModel_t *om, cvodeSettings_t *opt). */
SBML_ODESOLVER_API odeSense_t *ODEModel_constructSensitivity(odeModel_t *om)
{
  return ODESense_create(om, NULL);
}

/** Construct Sensitivity Matrix for ODEModel for selected parameters

    Once an ODE system has been constructed from an SBML model, this
    function calculates the derivative of each species' ODE with respect
    to model constants passed as SBML IDs via cvodeSettings_t structure. */
/*!!! TODO : pass (int nsens, char **opt->sensIDs directly) instead of opt
  and ask for adjoint and jacobian from where this is called internally!!!*/
SBML_ODESOLVER_API odeSense_t *ODESense_create(odeModel_t *om, cvodeSettings_t *opt)
{
  int i, k, nsens, all, construct;
  odeSense_t *os;

  ASSIGN_NEW_MEMORY(os, odeSense_t, NULL);

  all = 0;
  construct = 0;

  /* catch default case, no parameters/variables selected */
  /* for calls independent of integrator */
  if ( opt == NULL )
  {
    all = 1;
    construct = 1;
  }
  /* for calls with the integration */
  else
  {
    if ( opt->sensIDs == NULL )
      all = 1;
    else
      all = 0;
    /* check whether jacobian is present or adjoint is requested,
       to indicate whether the sensitivity matrix shall be constructed */
    if ( opt->DoAdjoint || om->jacobian )
      construct = 1;
  }

  if ( all )
    nsens = om->nconst;
  else
    nsens = opt->nsens;

  ASSIGN_NEW_MEMORY_BLOCK(os->index_sens, nsens, int, NULL);
  ASSIGN_NEW_MEMORY_BLOCK(os->index_sensP, nsens, int, NULL);

  os->om = om;
  os->neq = om->neq;
  os->nsens = nsens;

  /* fill with default parameters if none specified */
  if ( all )
  {
    for ( i=0; i<os->nsens; i++ )
    {
      /* index_sens: map between cvodeSettings and os->index_sens */
      os->index_sens[i] = om->neq + om->nass + i;
      /* index_sensP: set index for the optional sensitivity matrix */
      os->index_sensP[i] = i;
    }
    os->nsensP = om->nconst;
  }
  /* map input setting parameters */
  else
  {
    k = 0;
    for ( i=0; i<os->nsens; i++ )
    {
      /* index_sens: map between cvodeSettings and os->index_sens */
      os->index_sens[i] =
	ODEModel_getVariableIndexFields(om, opt->sensIDs[i]);
      /* indes_sensP:
	 map sensitivities for parameters to sensitivity matrix */
      /* distinguish between parameters and variables */
      if ( os->index_sens[i] < om->neq )
	/* set to -1 for variable sensitivities */
	os->index_sensP[i] = -1;
      else
      {
	/* index_sensP: set index for the optional sensitivity matrix */
	os->index_sensP[i] = k;
	k++;
      }
    }
    /* store number of parameter sensitivities */
    os->nsensP = k;
  }

  /* only required if either jacobian has been constructed, or adjoint
     solver is requested */
  if ( construct  )
    os->sensitivity = ODESense_constructMatrix(os, om);
  else
    os->sensitivity = 0;

  /* set flag for recompilation */
  os->recompileSensitivity = 1;
  return os;
}


/** Free Sensitivity Functions
 */

SBML_ODESOLVER_API void ODESense_free(odeSense_t *os)
{
  if ( os != NULL )
  {
    ODESense_freeMatrix(os);
    ODESense_freeStructures(os);
    /* free compiled code */
    if ( os->compiledCVODESensitivityCode != NULL )
    {
      CompiledCode_free(os->compiledCVODESensitivityCode);
      os->compiledCVODESensitivityCode = NULL;
    }
    free(os);
    os = NULL;
  }
}


/**  Returns an AST of the ith/jth entry of the parametric matrix

     Returns NULL if either the parametric has not been constructed yet,
     or if i > neq or j > nsens. Ownership remains within the odeModel
     structure. */
SBML_ODESOLVER_API const ASTNode_t *ODESense_getSensIJEntry(odeSense_t *os, int i, int j)
{
  if ( os->sens == NULL ) return NULL;
  if ( i >= os->om->neq || j >= os->nsens ) return NULL;
  return (const ASTNode_t *) os->sens[i][j];
}


/** \brief Returns an AST for the entry (d(vi1)/dt)/d(vi2) of the
    parametric matrix.

    Returns NULL if either the parametric matrix has not been constructed
    yet, or if vi1 is not an ODE variable or vi2 is not a parameter or
    variable for which sensitivity analysis was requested.
    Ownership remains within the odeModel structure.
*/

SBML_ODESOLVER_API const ASTNode_t *ODESense_getSensEntry(odeSense_t *os, variableIndex_t *vi1, variableIndex_t *vi2)
{
  int i;
  /* find sensitivity parameter/variable */
  for ( i=0; i<os->nsens && !(os->index_sens[i] == vi2->index); i++ );

  if ( i == os->nsens ) return NULL;
  return ODESense_getSensIJEntry(os, vi1->index, i);
}


/** \brief Returns the variableIndex for the jth parameter for
    which sensitivity analysis was requested, where
    0 < j < ODEModel_getNsens;

    Returns NULL if either the parametric matrix has not been constructed
    yet, or if j => ODEModel_getNsens;
*/

SBML_ODESOLVER_API variableIndex_t *ODESense_getSensParamIndexByNum(odeSense_t *os, int j)
{
  if ( j < os->nsens )
    return ODEModel_getVariableIndexByNum(os->om, os->index_sens[j]);
  else
    return NULL;
}


/** @} */

/************VARIABLE INTERFACE*************/

/* searches for the string "symbol" in the odeModel's names array
   and returns its index number, or -1 if it doesn't exist */
int ODEModel_getVariableIndexFields(odeModel_t *om, const char *symbol)
{
  int i, nvalues;

  nvalues = om->neq + om->nass + om->nconst + om->nalg;

  for ( i=0; i<nvalues && strcmp(symbol, om->names[i]); i++ );
  if (i<nvalues)
    return i;
  return -1;
}


int VariableIndex_getIndex(variableIndex_t *vi)
{
  return vi->index ;
}

/*! \defgroup variableIndex Variables + Parameters
  \ingroup odeModel
  \brief Getting the variableIndex structure

  The variableIndex can be used to retrieve formulae from odeModel,
  and to get and set current values in the integratorInstance.
*/
/*@{*/

/** \brief Creates and returns a variable index for ith variable

Returns NULL if i > nvalues. This functions works for all types of
variables (ODE_VARIABLE, ASSIGNED_VARIABLE, ALGEBRAIC_VARIABLE and
CONSTANT). This variableIndex can be used to get and set values
during an integration run with IntegratorInstance_getVariable and
IntegratorInstance_setVariable, respectively. The variableIndex
must be freed by the calling application.
*/

SBML_ODESOLVER_API variableIndex_t *ODEModel_getVariableIndexByNum(odeModel_t *om, int i)
{
  variableIndex_t *vi;

  if ( i > ODEModel_getNumValues(om) )
  {
    SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_SYMBOL_IS_NOT_IN_MODEL,
		      "Requested variable is not in the model. "
		      "Index larger then number of variables and "
		      "paramaters");
    return NULL;
  }
  else
  {
    ASSIGN_NEW_MEMORY(vi, variableIndex_t, NULL);
    vi->index = i;

    if ( i<om->neq )
    {
      vi->type = ODE_VARIABLE;
      vi->type_index = vi->index;
    }
    else if ( i < om->neq + om->nass )
    {
      vi->type = ASSIGNMENT_VARIABLE;
      vi->type_index = i - om->neq;
    }
    else if ( i < om->neq + om->nass + om->nconst )
    {
      vi->type = CONSTANT;
      vi->type_index = i - om->neq - om->nass;
    }
    else
    {
      vi->type = ALGEBRAIC_VARIABLE;
      vi->type_index = i - om->neq - om->nass - om->nconst;
    }
  }

  return vi;
}


/** \brief Creates and returns the variableIndex for the string "symbol"

where `symbol' is the ID (corresponding to the SBML ID in the input
model) of one of the models variables (ODE_VARIABLE,
ASSIGNED_VARIABLE, ALGEBRAIC_VARIABLE and CONSTANT) or NULL if the
symbol was not found. The variableIndex must be freed by the
calling application.
*/

SBML_ODESOLVER_API variableIndex_t *ODEModel_getVariableIndex(odeModel_t *om, const char *symbol)
{

  int index;

  if ( symbol == NULL )
  {
    SolverError_error(ERROR_ERROR_TYPE, SOLVER_ERROR_SYMBOL_IS_NOT_IN_MODEL,
		      "NULL string passed to ODEModel_getVariableIndex", symbol);

    return NULL;
  }

  index = ODEModel_getVariableIndexFields(om, symbol);

  if ( index == -1 )
  {
    SolverError_error(ERROR_ERROR_TYPE, SOLVER_ERROR_SYMBOL_IS_NOT_IN_MODEL,
		      "Symbol %s is not in the model", symbol);

    return NULL;
  }

  return ODEModel_getVariableIndexByNum(om, index);
}



/** \brief  Creates and returns a variable index for ith ODE variable.

    Returns NULL if not existing (i > ODEModel_getNeq(om)). The
    variableIndex must be freed by the calling application.
*/

SBML_ODESOLVER_API variableIndex_t *ODEModel_getOdeVariableIndex(odeModel_t *om, int i)
{
  if ( i < om->neq )
    return ODEModel_getVariableIndexByNum(om, i);
  else
    return NULL;
}


/** \brief Creates and returns a variable index for ith assigned variable.

Returns NULL if not existing (i > ODEModel_getNumAssignments(om)).
The variableIndex must be freed by the calling application.
*/

SBML_ODESOLVER_API variableIndex_t *ODEModel_getAssignedVariableIndex(odeModel_t *om, int i)
{
  if ( i < om->nass )
    return ODEModel_getVariableIndexByNum(om, i + om->neq);
  else
    return NULL;
}

/**\brief  Creates and returns a variable index for ith constant.

Returns NULL if not existing (i > ODEModel_getNumConstants(om)).
The variableIndex must be freed by the calling application.
*/

SBML_ODESOLVER_API variableIndex_t *ODEModel_getConstantIndex(odeModel_t *om, int i)
{
  if ( i < om->nconst )
    return ODEModel_getVariableIndexByNum(om, i + om->neq + om->nass);
  else
    return NULL;
}

/** \brief Returns the name of the variable corresponding to passed
    variableIndex. The returned string (const char *) may NOT be
    changed or freed by calling applications.
*/

SBML_ODESOLVER_API const char *VariableIndex_getName(variableIndex_t *vi, odeModel_t *om)
{
  return (const char*) om->names[vi->index];
}

/* outdated */
const char *ODEModel_getVariableName(odeModel_t *om, variableIndex_t *vi)
{
  return VariableIndex_getName(vi, om);
}

/** \brief  Frees a variableIndex structure
 */

SBML_ODESOLVER_API void VariableIndex_free(variableIndex_t *vi)
{
  free(vi);
}

/** @} */


/****************COMPILATION*******************/

/* appends a compilable expression for the given AST to the given buffer
   assuming that the node has not been indexed */
void ODEModel_generateASTWithoutIndex(odeModel_t *om,
				      charBuffer_t *buffer,
				      ASTNode_t *node)
{
  ASTNode_t *index = indexAST(node, om->neq + om->nass +
			      om->nconst, om->names);
  generateAST(buffer, index);
  ASTNode_free(index);
}

/* appends a compilable assignment to the buffer.
   The assignment is made to the 'value' array item indexed by 'index'.
   The value assigned is computed from the given AST. */
void ODEModel_generateAssignmentCode(odeModel_t *om, int index, ASTNode_t *node,
				     charBuffer_t *buffer)
{
  CharBuffer_append(buffer, "value[");
  CharBuffer_appendInt(buffer, index);
  CharBuffer_append(buffer, "] = ");
  generateAST(buffer, node);
  CharBuffer_append(buffer, ";\n");
}

/* appends compiled code for a set of assignment rules to the gievn buffer.
   The assignments generated are taken from the assignment rules in the
   given model however the set generated is determined by the
   given 'requiredAssignments' boolean array which is indexed in the same
   order as the 'assignment' array on the given model. */
void ODEModel_generateAssignmentRuleCode(odeModel_t *om, int nass,
					 nonzeroElem_t **orderedList,
					 charBuffer_t *buffer)
{
  int i ;

  for ( i=0; i<nass; i++ )
  {
    nonzeroElem_t *ordered = orderedList[i];
    ODEModel_generateAssignmentCode(om, ordered->i, ordered->ij, buffer);
  }
}

void ODEModel_generateAssignmentRuleCodeOUTDATED(odeModel_t *om,
						 int *requiredAssignments,
						 charBuffer_t *buffer)
{
  int i ;

  for ( i=0; i<om->nass; i++ )
  {
    nonzeroElem_t *ordered = om->assignmentOrder[i];
    if ( !requiredAssignments || requiredAssignments[ordered->i - om->neq] )
      ODEModel_generateAssignmentCode(om, ordered->i, ordered->ij, buffer);
  }
}

/** appends compiled code to the given buffer for the function called by
    the value of 'COMPILED_EVENT_FUNCTION_NAME' which implements the
    evaluation event triggers and assignment rules required for event
    triggers and event assignments.
*/
void ODEModel_generateEventFunction(odeModel_t *om, charBuffer_t *buffer)
{
  int i, j, idx;
  ASTNode_t *trigger, *assignment;

  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_EVENT_FUNCTION_NAME);
  CharBuffer_append(buffer,"(cvodeData_t *data, int *engineIsValid)\n"\
		    "{\n"\
		    "    realtype *value = data->value;\n"\
		    "    int fired = 0;\n"\
		    "    int *trigger = data->trigger;\n");

  ODEModel_generateAssignmentRuleCode(om, om->nassbeforeevents,
				      om->assignmentsBeforeEvents, buffer);

  for ( i=0; i<om->nevents; i++ )
  {
    int setIsValidFalse = 0;

    trigger = (ASTNode_t *) om->event[i];

    CharBuffer_append(buffer, "if ((trigger[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] == 0) && (");
    generateAST(buffer, trigger);
    CharBuffer_append(buffer, "))\n"		\
		      "{\n"			\
		      "    fired++;\n"		\
		      "    trigger[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = 1;\n");

    for ( j=0; j<om->neventAss[i]; j++ )
    {
      /* generate event assignment */
      assignment = om->eventAssignment[i][j];
      idx = om->eventIndex[i][j];
      CharBuffer_append(buffer, "    ");
      ODEModel_generateAssignmentCode(om, idx, assignment, buffer);

      /* identify cases which modify variables computed by solver which
	 set the solver into an invalid state : NOT CORRECT : solver
	 should always be reinitialized ! */
      if ( /* idx < om->neq && */ !setIsValidFalse )
      {
	CharBuffer_append(buffer, "    *engineIsValid = 0;\n");
	setIsValidFalse = 1 ;
      }
    }

    
    CharBuffer_append(buffer, "}\n"		\
		      "else {\n"		\
		      "    trigger[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = 0;\n"	\
		      
		      "}\n");
  }

  /* NOT REQUIRED : complete rule set evaluated where required */
  /*   CharBuffer_append(buffer, "if ( fired )\n{\n"); */
  /*   ODEModel_generateAssignmentRuleCode(om, om->nassafterevents, */
  /*			      om->assignmentsAfterEvents, buffer); */
  /*   CharBuffer_append(buffer, "\n}\n"); */

  CharBuffer_append(buffer, "return fired;\n}\n");
}

/* appends compiled code to the given buffer for the function called
   by the value of 'COMPILED_RHS_FUNCTION_NAME' which calculates the
   right hand side ODE values for the set of ODEs being solved. */
void ODEModel_generateCVODERHSFunction(odeModel_t *om, charBuffer_t *buffer)
{
  int i ;

  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_RHS_FUNCTION_NAME);
  CharBuffer_append(buffer,
		    "(realtype t, N_Vector y, N_Vector ydot, void *f_data)\n"\
		    "{\n"\
		    "    int i;\n"\
		    "    realtype *ydata, *dydata;\n"\
		    "    cvodeData_t *data;\n"\
		    "    realtype *value ;\n"\
		    "    data = (cvodeData_t *) f_data;\n"\
		    "    value = data->value;\n"\
		    "    ydata = NV_DATA_S(y);\n"\
		    "    dydata = NV_DATA_S(ydot);\n");

  /* update time  */
  CharBuffer_append(buffer, "data->currenttime = t;\n");

  /* UPDATE ODE VARIABLES from CVODE */
  for ( i=0; i<om->neq; i++ )
  {
    CharBuffer_append(buffer, "value[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = ydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "];\n");
  }
  /* negative state detection */
  CharBuffer_append(buffer, "if ( data->opt->DetectNegState  )\n");
  CharBuffer_append(buffer, "  for ( i=0; i<data->model->neq; i++ )\n");
  CharBuffer_append(buffer, "    if (data->value[i] < 0) return (1);\n");

  /* UPDATE ASSIGNMENT RULES */
  /* in case sensitivity or jacobi matrix is not available */
  CharBuffer_append(buffer,
		    "if ( data->use_p )\n"\
		    "{\n"\
		    "  for ( i=0; i<data->nsens; i++ )\n"\
		    "    value[data->os->index_sens[i]] = data->p[i];\n");

  ODEModel_generateAssignmentRuleCode(om, om->nass,
				      om->assignmentOrder, buffer);

  /* in case sensitivity or jacobi matrix are available */
  /* CharBuffer_append(buffer, "\n printf(\"HALLO\\n\");\n"); */
  CharBuffer_append(buffer, "\n}\nelse\n{\n");
  ODEModel_generateAssignmentRuleCode(om, om->nassbeforeodes,
				      om->assignmentsBeforeODEs, buffer);
  CharBuffer_append(buffer, "}\n");


  /* EVALUATE ODEs f(x,p,t) = dx/dt */
  for ( i=0; i<om->neq; i++ )
  {
    CharBuffer_append(buffer, "dydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = ");
    generateAST(buffer, om->ode[i]);
    CharBuffer_append(buffer, ";\n");
  }
  /* reset parameters for printout etc. */
  CharBuffer_append(buffer,
		    "if ( data->use_p )\n"\
		    "{"\
		    "  for ( i=0; i<data->nsens; i++ )\n"\
		    "    value[data->os->index_sens[i]] = data->p_orig[i];\n");

  ODEModel_generateAssignmentRuleCode(om, om->nass,
				      om->assignmentOrder, buffer);
  CharBuffer_append(buffer, "}\n");

  CharBuffer_append(buffer, "return (0);\n");
  CharBuffer_append(buffer, "}\n\n");
}


/* appends compiled code to the given buffer for the function called
   by the value of 'COMPILED_ADJRHS_FUNCTION_NAME' which calculates the
   right hand side ODE values for the adjoint ODEs being solved. */
void ODEModel_generateCVODEAdjointRHSFunction(odeModel_t *om, charBuffer_t *buffer)
{
  int i,j ;
  ASTNode_t *jacob_ji;
  double val;
  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_ADJOINT_RHS_FUNCTION_NAME);
  CharBuffer_append(buffer,
		    "(realtype t, N_Vector y, N_Vector yA, "\
		    " N_Vector yAdot, void *fA_data)\n"\
		    "{\n"\
		    "    int i;\n"\
		    "    realtype *ydata, *yAdata, *dyAdata;\n"\
		    "    cvodeData_t *data;\n"\
                    "    realtype *value ;\n"\
		    "    data = (cvodeData_t *) fA_data;\n"\
                    "    value = data->value;\n"\
                    "    ydata = NV_DATA_S(y);\n"\
		    "    yAdata = NV_DATA_S(yA);\n"\
		    "    dyAdata = NV_DATA_S(yAdot);\n" );


  /*  update ODE variables from CVODE */
  for ( i=0; i<om->neq; i++ )
  {
    CharBuffer_append(buffer,  "value[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer,  "] = ydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer,  "];\n" );
  }

  /* update time  */
  CharBuffer_append(buffer, "data->currenttime = t;\n");


  /*  evaluate adjoint sensitivity RHS: -[df/dx]^T * yA + v */
  for ( i=0; i<om->neq; i++ )
  {
    CharBuffer_append(buffer, "dyAdata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = 0.0;\n");
    for ( j=0; j<om->neq; j++ )
    {
      jacob_ji = om->jacob[j][i];
      /*  check whether jacobian is 0  */
      val = 1;
      if ( ASTNode_isInteger(jacob_ji) )
	val = (double) ASTNode_getInteger(jacob_ji) ;
      if ( ASTNode_isReal(jacob_ji) )
	val = ASTNode_getReal(jacob_ji) ;

      /* write Jacobi evaluation only if entry is not 0 */
      if ( val != 0.0 )
      {
	CharBuffer_append(buffer, "dyAdata[");
	CharBuffer_appendInt(buffer, i);
	CharBuffer_append(buffer, "]");
	CharBuffer_append(buffer, "-= ( ");
	generateAST(buffer, jacob_ji);
	CharBuffer_append(buffer, " ) * yAdata[");
	CharBuffer_appendInt(buffer, j);
	CharBuffer_append(buffer, "];\n");
      }
    }

    CharBuffer_append(buffer,
		      "if (data->model->discrete_observation_data == 0)\n ");
    CharBuffer_append(buffer, "dyAdata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] +=");
    CharBuffer_append(buffer, " evaluateAST( data->model->vector_v[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "], data);\n");

  }

  CharBuffer_append(buffer, "return (0);\n");

  CharBuffer_append(buffer, "}\n\n");
}

/* appends compiled code to the given buffer for the function called by
   the value of 'COMPILED_JACOBIAN_FUNCTION_NAME' which
   calculates the Jacobian for the set of ODEs being solved. */
void ODEModel_generateCVODEJacobianFunction(odeModel_t *om,
					    charBuffer_t *buffer)
{
  int i, j ;
  ASTNode_t *jacob_ij;
  float val;

  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_JACOBIAN_FUNCTION_NAME);
  CharBuffer_append(buffer,
		    "(long int N, DenseMat J, realtype t,\n"\
		    "    N_Vector y, N_Vector fy, void *jac_data,\n"\
		    "    N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)\n"\
		    "{\n"\
		    "  \n"\
		    "int i;\n"\
		    "realtype *ydata;\n"\
		    "cvodeData_t *data;\n"\
		    "realtype *value;\n"\
		    "data  = (cvodeData_t *) jac_data;\n"\
		    "value = data->value ;\n"\
		    "ydata = NV_DATA_S(y);\n"\
		    "data->currenttime = t;\n"\
		    "\n"\
		    "if (  (data->opt->Sensitivity && data->os ) &&"\
		    " (!data->os->sensitivity || !data->model->jacobian))\n"\
		    "    for ( i=0; i<data->nsens; i++ )\n"\
		    "        value[data->os->index_sens[i]] = "\
		    "data->p[i];\n\n");


  /** update ODE variables from CVODE */
  for ( i=0; i<om->neq; i++ )
  {
    CharBuffer_append(buffer, "value[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = ydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "];\n");
  }

  /** evaluate Jacobian J = df/dx */
  for ( i=0; i<om->neq; i++ )
  {
    for ( j=0; j<om->neq; j++ )
    {
      jacob_ij = om->jacob[i][j];
      /*  check whether jacobian is 0  */
      val = 1;
      if ( ASTNode_isInteger(jacob_ij) )
	val = (float) ASTNode_getInteger(jacob_ij) ;
      if ( ASTNode_isReal(jacob_ij) )
	val = (float) ASTNode_getReal(jacob_ij) ;

      /* write Jacobi evaluation only if entry is not 0 */
      if ( val != 0.0 )
      {
	CharBuffer_append(buffer, "DENSE_ELEM(J,");
	CharBuffer_appendInt(buffer, i);
	CharBuffer_append(buffer, ",");
	CharBuffer_appendInt(buffer, j);
	CharBuffer_append(buffer, ") = ");
	generateAST(buffer, jacob_ij);
	CharBuffer_append(buffer, ";\n");
      }
    }
  }
  /* reset parameters for printout etc. */
  CharBuffer_append(buffer,
		    "if (  (data->opt->Sensitivity && data->os ) &&"\
		    " (!data->os->sensitivity || !data->model->jacobian))\n"\
		    "    for ( i=0; i<data->nsens; i++ )\n"\
		    "        value[data->os->index_sens[i]] = "\
		    "data->p_orig[i];\n\n");

  /* CharBuffer_append(buffer, "printf(\"J\");"); */
  CharBuffer_append(buffer, "return (0);\n");
  CharBuffer_append(buffer, "}\n");
}

/* appends compiled code to the given buffer for the function called by
   the value of 'COMPILED_JACOBIAN_FUNCTION_NAME' which
   calculates the Jacobian for the set of ODEs being solved. */
void ODEModel_generateCVODEAdjointJacobianFunction(odeModel_t *om,
						   charBuffer_t *buffer)
{
  int i, j ;
  ASTNode_t *jacob_ji;
  float val;

  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_ADJOINT_JACOBIAN_FUNCTION_NAME);
  CharBuffer_append(buffer,
		    "(long int NB, DenseMat JB, realtype t, N_Vector y,\n" \
		    "    N_Vector yB,  N_Vector fyB, void *jac_dataB,\n" \
		    "    N_Vector tmpB, N_Vector tmp2B, N_Vector tmp3B)\n" \
		    "{\n"						\
		    "  \n"						\
		    "int i;\n"						\
		    "realtype *ydata;\n"				\
		    "cvodeData_t *data;\n"				\
		    "realtype *value;\n"				\
		    "data  = (cvodeData_t *) jac_dataB;\n"		\
		    "value = data->value ;\n"				\
		    "ydata = NV_DATA_S(y);\n"				\
		    "data->currenttime = t;\n"				\
		    "\n");

  /** update ODE variables from CVODE */
  for ( i=0; i<om->neq; i++ )
  {
    CharBuffer_append(buffer, "value[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = ydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "];\n");
  }

  /** evaluate Jacobian J = df/dx */
  for ( i=0; i<om->neq; i++ )
  {
    for ( j=0; j<om->neq; j++ )
    {
      jacob_ji = om->jacob[j][i];
      /*  check whether jacobian is 0  */
      val = 1;
      if ( ASTNode_isInteger(jacob_ji) )
	val = (float) ASTNode_getInteger(jacob_ji) ;
      if ( ASTNode_isReal(jacob_ji) )
	val = (float) ASTNode_getReal( jacob_ji ) ;

      /* write Jacobi evaluation only if entry is not 0 */
      if ( val != 0.0 )
      {
	CharBuffer_append(buffer, "DENSE_ELEM(JB,");
	CharBuffer_appendInt(buffer, i);
	CharBuffer_append(buffer, ",");
	CharBuffer_appendInt(buffer, j);
	CharBuffer_append(buffer, ") = - (");
	generateAST(buffer, jacob_ji);
	CharBuffer_append(buffer, ");\n");
      }
    }
  }
  /* CharBuffer_append(buffer, "printf(\"JA\");"); */
  CharBuffer_append(buffer, "return (0);\n");

  CharBuffer_append(buffer, "}\n");
}

/* appends compiled code to the given buffer for the function called
   by the value of 'COMPILED_SENSITIVITY_FUNCTION_NAME' which
   calculates the sensitivities (derived from Jacobian and parametrix
   matrices) for the set of ODEs being solved. */
void ODESense_generateCVODESensitivityFunction(odeSense_t *os,
					       charBuffer_t *buffer)
{
  int i, j, k;
  double val;
  ASTNode_t *jacob_ij, *sens_ik;

  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_SENSITIVITY_FUNCTION_NAME);
  CharBuffer_append(buffer,
		    "(int Ns, realtype t, N_Vector y, N_Vector ydot,\n"\
		    " int iS, N_Vector yS, N_Vector ySdot, \n"
		    " void *fs_data, N_Vector tmp1, N_Vector tmp2)\n"\
		    "{\n"\
		    "  \n"\
		    "realtype *ydata, *ySdata, *dySdata;\n"\
		    "cvodeData_t *data;\n"\
		    "realtype *value;\n"\
		    "data = (cvodeData_t *) fs_data;\n"\
		    "value = data->value ;\n"\
		    "ydata = NV_DATA_S(y);\n"\
		    "ySdata = NV_DATA_S(yS);\n"\
		    "dySdata = NV_DATA_S(ySdot);\n"\
		    "data->currenttime = t;\n");

  /** update ODE variables from CVODE */
  for ( i=0; i<os->om->neq; i++ )
  {
    CharBuffer_append(buffer, "value[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = ydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "];\n\n");
  }

  /** evaluate sensitivity RHS: df/dx * s + df/dp for one p */
  for ( i=0; i<os->om->neq; i++ )
  {
    CharBuffer_append(buffer, "dySdata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = 0.0;\n");
    for (j=0; j<os->om->neq; j++)
    {
      /* only non-zero Jacobi elements */
      jacob_ij = os->om->jacob[i][j];
      /*  check whether jacobian is 0  */
      val = 1;
      if ( ASTNode_isInteger(jacob_ij) )
	val = (double) ASTNode_getInteger(jacob_ij) ;
      if ( ASTNode_isReal(jacob_ij) )
	val = ASTNode_getReal(jacob_ij) ;

      /* write Jacobi evaluation only if entry is not 0 */
      if ( val != 0.0 )
      {
	CharBuffer_append(buffer, "dySdata[");
	CharBuffer_appendInt(buffer, i);
	CharBuffer_append(buffer, "] += ( ");
	generateAST(buffer, jacob_ij);
	CharBuffer_append(buffer, ") * ySdata[");
	CharBuffer_appendInt(buffer, j);
	CharBuffer_append(buffer, "]; ");
	CharBuffer_append(buffer, " /* om->jacob[");
	CharBuffer_appendInt(buffer, i);
	CharBuffer_append(buffer, "][");
	CharBuffer_appendInt(buffer, j);
	CharBuffer_append(buffer, "]  */ \n");
      }
    }

    for ( k=0; k<os->nsens; k++ )
    {
      if ( os->index_sensP[k] != -1 )
      {
	/* only non-zero Jacobi elements */
	sens_ik = os->sens[i][os->index_sensP[k]];
	/*  check whether jacobian is 0  */
	val = 1;
	if ( ASTNode_isInteger(sens_ik) )
	  val = (double) ASTNode_getInteger(sens_ik) ;
	if ( ASTNode_isReal(sens_ik) )
	  val = ASTNode_getReal(sens_ik) ;

	if ( val != 0.0 )
	{
	  CharBuffer_append(buffer, "if ( ");
	  CharBuffer_appendInt(buffer, k);
	  CharBuffer_append(buffer, " == iS ) ");
	  CharBuffer_append(buffer, "dySdata[");
	  CharBuffer_appendInt(buffer, i);
	  CharBuffer_append(buffer, "] += ");
	  generateAST(buffer, sens_ik);
	  CharBuffer_append(buffer, "; ");
	  CharBuffer_append(buffer, " /* om->sens[");
	  CharBuffer_appendInt(buffer, i);
	  CharBuffer_append(buffer, "][");
	  CharBuffer_appendInt(buffer,os->index_sensP[k]);
	  CharBuffer_append(buffer, "]  */ \n");
	}
      }
    }
  }
  /* CharBuffer_append(buffer, "printf(\"S\");"); */
  CharBuffer_append(buffer, "return (0);\n");

  CharBuffer_append(buffer, "}\n\n");
}


/* appends compiled code to the given buffer for the function called
   by the value of 'COMPILED_ADJOINT_QUAD_FUNCTION_NAME' */
void ODESense_generateCVODEAdjointQuadFunction(odeSense_t *os,
					       charBuffer_t *buffer)
{
  int i, k;
  double val;
  ASTNode_t *sens_ik;

  CharBuffer_append(buffer,"DLL_EXPORT int ");
  CharBuffer_append(buffer,COMPILED_ADJOINT_QUAD_FUNCTION_NAME);
  CharBuffer_append(buffer,
		    "(realtype t, N_Vector y, N_Vector yA,\n"	\
		    " N_Vector qAdot, void *fA_data)\n"		\
		    "{\n"					\
		    "  \n"					\
		    "realtype *ydata, *yAdata, *dqAdata;\n"	\
		    "cvodeData_t *data;\n"\
		    "realtype *value;\n"\
		    "data = (cvodeData_t *) fA_data;\n"\
		    "value = data->value ;\n"\
		    "ydata = NV_DATA_S(y);\n"\
		    "yAdata = NV_DATA_S(yA);\n"\
		    "dqAdata = NV_DATA_S(qAdot);\n"\
		    "data->currenttime = t;\n");

  /** update ODE variables from CVODE */
  for ( i=0; i<os->om->neq; i++ )
  {
    CharBuffer_append(buffer, "value[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "] = ydata[");
    CharBuffer_appendInt(buffer, i);
    CharBuffer_append(buffer, "];\n\n");
  }

  /** evaluate quadrature integrand: yA^T * df/dp */
  for ( k=0; k<os->nsens; k++ )
  {
    CharBuffer_append(buffer, "dqAdata[");
    CharBuffer_appendInt(buffer, k);
    CharBuffer_append(buffer, "] = 0.0;\n");

    for ( i=0; i<os->om->neq; i++ )
    {
      if ( os->index_sensP[k] != -1 )
      {
	/* only non-zero param matrix elements */
	sens_ik = os->sens[i][os->index_sensP[k]];

	/*  check whether element is 0  */
	val = 1;
	if ( ASTNode_isInteger(sens_ik) )
	  val = (double) ASTNode_getInteger(sens_ik) ;
	if ( ASTNode_isReal(sens_ik) )
	  val = ASTNode_getReal(sens_ik) ;

	if ( val != 0.0 )
	{
	  CharBuffer_append(buffer, "dqAdata[");
	  CharBuffer_appendInt(buffer, k);
	  CharBuffer_append(buffer, "] += ");
	  CharBuffer_append(buffer, "yAdata[");
	  CharBuffer_appendInt(buffer, i);
	  CharBuffer_append(buffer, "] * ( ");
	  generateAST(buffer, sens_ik);
	  CharBuffer_append(buffer, " ); /* om->sens[");
	  CharBuffer_appendInt(buffer, i);
	  CharBuffer_append(buffer, "][");
	  CharBuffer_appendInt(buffer, os->index_sensP[k]);
	  CharBuffer_append(buffer, "]  */ \n");
	}
      }
    }
  }

  CharBuffer_append(buffer, "return (0);\n");

  /* CharBuffer_append(buffer, "printf(\"qa\");"); */
  CharBuffer_append(buffer, "}\n\n");
}

/* dynamically generates and complies the ODE RHS, Jacobian and
   Events handling functions for the given model.
   The jacobian function is not generated if the jacobian AST
   expressions have not been generated.
   Returns 1 if successful, 0 otherwise
*/
int ODEModel_compileCVODEFunctions(odeModel_t *om)
{
  charBuffer_t *buffer = CharBuffer_create();


  /* if available, the whole code needs recompilation, can happen
     for subsequent runs with new sensitivity settings */
  if ( om->compiledCVODEFunctionCode != NULL )
  {
    CompiledCode_free(om->compiledCVODEFunctionCode);
    om->compiledCVODEFunctionCode = NULL;
  }

#ifdef WIN32
  CharBuffer_append(buffer,
		    "#include <windows.h>\n"\
		    "#include <math.h>\n"\
		    "#include <sbmlsolver/sundialstypes.h>\n"\
		    "#include <sbmlsolver/nvector.h>\n"\
		    "#include <sbmlsolver/nvector_serial.h>\n"\
		    "#include <sbmlsolver/dense.h>\n"\
		    "#include <sbmlsolver/cvodes.h>\n"\
		    "#include <sbmlsolver/cvodea.h>\n"\
		    "#include <sbmlsolver/cvdense.h>\n"\
		    "#include <sbmlsolver/cvodeData.h>\n"\
		    "#include <sbmlsolver/cvodeSettings.h>\n"\
		    "#include <sbmlsolver/odeModel.h>\n"\
		    "#define DLL_EXPORT __declspec(dllexport)\n");

#else
  CharBuffer_append(buffer,
		    "#include <math.h>\n"
		    "#include \"cvodes/cvodes.h\"\n"
		    "#include \"cvodes/cvodes_dense.h\"\n"
		    "#include \"nvector/nvector_serial.h\"\n"
		    "#include \"sbmlsolver/cvodeData.h\"\n"
		    "#include \"sbmlsolver/processAST.h\"\n"
		    "#define DLL_EXPORT\n\n");
#endif

  generateMacros(buffer);

  if ( om->jacobian )
  {
    ODEModel_generateCVODEJacobianFunction(om, buffer);
    ODEModel_generateCVODEAdjointJacobianFunction(om, buffer);
    ODEModel_generateCVODEAdjointRHSFunction(om, buffer);
  }

  ODEModel_generateEventFunction(om, buffer);
  ODEModel_generateCVODERHSFunction(om, buffer);


#ifdef _DEBUG /* write out source file for debugging*/
  {
    FILE *src;
    char *srcname =  "rhsfunctions.c";
    src = fopen(srcname, "w");
    fprintf(src, CharBuffer_getBuffer(buffer));
    fclose(src);
  }
#endif

  /* now all required sourcecode is in `buffer' and can be sent
     to the compiler */
  om->compiledCVODEFunctionCode =
    Compiler_compile(CharBuffer_getBuffer(buffer));


  if ( om->compiledCVODEFunctionCode == NULL )
  {
    CharBuffer_free(buffer);
    return 0;
  }

  CharBuffer_free(buffer);

  /* attach pointers */
  om->compiledCVODERhsFunction =
    CompiledCode_getFunction(om->compiledCVODEFunctionCode,
			     COMPILED_RHS_FUNCTION_NAME);

  om->compiledEventFunction =
    CompiledCode_getFunction(om->compiledCVODEFunctionCode,
			     COMPILED_EVENT_FUNCTION_NAME);


  if ( om->jacobian )
  {
    om->compiledCVODEJacobianFunction =
      CompiledCode_getFunction(om->compiledCVODEFunctionCode,
			       COMPILED_JACOBIAN_FUNCTION_NAME);


    om->compiledCVODEAdjointJacobianFunction =
      CompiledCode_getFunction(om->compiledCVODEFunctionCode,
			       COMPILED_ADJOINT_JACOBIAN_FUNCTION_NAME);

    om->compiledCVODEAdjointRhsFunction =
      CompiledCode_getFunction(om->compiledCVODEFunctionCode,
			       COMPILED_ADJOINT_RHS_FUNCTION_NAME);

  }
  return 1;
}


/* dynamically generates and compiles the ODE Sensitivity RHS
   for the given model */
int ODESense_compileCVODESenseFunctions(odeSense_t *os)
{
  charBuffer_t *buffer = CharBuffer_create();

#ifdef WIN32
  CharBuffer_append(buffer,
		    "#include <windows.h>\n"\
		    "#include <math.h>\n"\
		    "#include <sbmlsolver/sundialstypes.h>\n"\
		    "#include <sbmlsolver/nvector.h>\n"\
		    "#include <sbmlsolver/nvector_serial.h>\n"\
		    "#include <sbmlsolver/dense.h>\n" 
		    "#include <sbmlsolver/cvodes.h>\n"\
		    "#include <sbmlsolver/cvodea.h>\n"\
		    "#include <sbmlsolver/cvdense.h>\n"\
		    "#include <sbmlsolver/cvodeData.h>\n"\
		    "#include <sbmlsolver/cvodeSettings.h>\n"\
		    "#include <sbmlsolver/processAST.h>\n"\
		    "#include <sbmlsolver/odeModel.h>\n"\
		    "#define DLL_EXPORT __declspec(dllexport)\n");
#else
  CharBuffer_append(buffer,
		    "#include <math.h>\n"
		    "#include \"cvodes/cvodes.h\"\n"
		    "#include \"cvodes/cvodes_dense.h\"\n"
		    "#include \"nvector/nvector_serial.h\"\n"
		    "#include \"sbmlsolver/cvodeData.h\"\n"
		    "#include \"sbmlsolver/processAST.h\"\n"
		    "#define DLL_EXPORT\n\n");
#endif

  generateMacros(buffer);

  ODESense_generateCVODESensitivityFunction(os, buffer);
  ODESense_generateCVODEAdjointQuadFunction(os, buffer);

#ifdef _DEBUG /* write out source file for debugging*/
  {
    FILE *src;
    char *srcname =  "sensfunctions.c";
    src = fopen(srcname, "w");
    fprintf(src, CharBuffer_getBuffer(buffer));
    fclose(src);
  }
#endif

  /* now all required sourcecode is in `buffer' and can be sent
     to the compiler */
  os->compiledCVODESensitivityCode =
    Compiler_compile(CharBuffer_getBuffer(buffer));

  if ( os->compiledCVODESensitivityCode == NULL )
  {
    CharBuffer_free(buffer);
    return 0;
  }

  CharBuffer_free(buffer);

  os->compiledCVODESenseFunction =
    CompiledCode_getFunction(os->compiledCVODESensitivityCode,
			     COMPILED_SENSITIVITY_FUNCTION_NAME);

  os->compiledCVODEAdjointQuadFunction =
    CompiledCode_getFunction(os->compiledCVODESensitivityCode,
			     COMPILED_ADJOINT_QUAD_FUNCTION_NAME);

  os->recompileSensitivity = 0;

  return 1;
}


/** returns the compiled RHS ODE function for the given model */
SBML_ODESOLVER_API CVRhsFn ODEModel_getCompiledCVODERHSFunction(odeModel_t *om)
{
  if ( !om->compiledCVODERhsFunction )
    if ( !ODEModel_compileCVODEFunctions(om) )
      return NULL;

  return om->compiledCVODERhsFunction;
}

/** returns the compiled Jacobian function for the given model */
SBML_ODESOLVER_API CVDenseJacFn ODEModel_getCompiledCVODEJacobianFunction(odeModel_t *om)
{
  if ( !om->jacobian )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_CANNOT_COMPILE_JACOBIAN_NOT_COMPUTED,
		      "Attempting to compile jacobian before the jacobian "\
		      "is computed\n"\
		      "Call ODEModel_constructJacobian before calling\n"\
		      "ODEModel_getCompiledCVODEJacobianFunction or "\
		      "ODEModel_getCompiledCVODERHSFunction\n");
    return NULL;
  }

  if ( !om->compiledCVODEJacobianFunction )
    /* only for calling independent of solver!!
       function should have been compiled already */
    if ( !ODEModel_compileCVODEFunctions(om) )
      return NULL;


  return om->compiledCVODEJacobianFunction;
}

/** returns the compiled Sensitivity function for the given model */
SBML_ODESOLVER_API CVSensRhs1Fn ODESense_getCompiledCVODESenseFunction(odeSense_t *os)
{
  if ( !os->sensitivity )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_CANNOT_COMPILE_SENSITIVITY_NOT_COMPUTED,
		      "Attempting to compile sensitivity matrix before "\
		      "the matrix is computed\n"\
		      "Call ODESense_constructSensitivity before calling\n"\
		      "ODESense_getCompiledCVODESenseFunction\n");
    return NULL;
  }

  if ( !os->compiledCVODESenseFunction || os -> recompileSensitivity )
  {
    /*!!! currently not used: if TCC multiple states become possible,
      until then this must have been compiled already within the main
      compiled code structure */
    /* only for calling independent of solver!!
       function should have been compiled already */
    if ( !ODESense_compileCVODESenseFunctions(os) )
      return NULL;
  }

  return os->compiledCVODESenseFunction;
}

/** returns the compiled adjoint RHS ODE function for the given model */
SBML_ODESOLVER_API CVRhsFnB ODEModel_getCompiledCVODEAdjointRHSFunction(odeModel_t *om)
{
  if ( !om->jacobian )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_CANNOT_COMPILE_SENSITIVITY_NOT_COMPUTED,
		      "Attempting to compile adjoint RHS before " \
		      "the Jacobian matrix is computed\n"		\
		      "Call ODEModel_constructJacobian before calling\n" \
		      "ODEModel_getCompiledCVODEAdjointJacobianFunction or "\
		      "ODEModel_getCompiledCVODEAdjointRHSFunction\n");
    return NULL;
  }

  if ( !om->compiledCVODEAdjointRhsFunction  )
    if ( !ODEModel_compileCVODEFunctions(om) )
      return NULL;

  return om->compiledCVODEAdjointRhsFunction;
}

/** returns the compiled adjoint jacobian function for the given model */
SBML_ODESOLVER_API CVDenseJacFnB ODEModel_getCompiledCVODEAdjointJacobianFunction(odeModel_t *om)
{
  if ( !om->jacobian )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_CANNOT_COMPILE_JACOBIAN_NOT_COMPUTED,
		      "Attempting to compile adjoint jacobian before "\
		      "the jacobian is computed\n"\
		      "Call ODEModel_constructJacobian before calling\n"\
		      "ODEModel_getCompiledCVODEAdjointJacobianFunction or "\
		      "ODEModel_getCompiledCVODERHSFunction\n");
    return NULL;
  }

  if ( !om->compiledCVODEAdjointJacobianFunction )
    /* only for calling independent of solver!!
       function should have been compiled already */
    if ( !ODEModel_compileCVODEFunctions(om) )
      return NULL;

  return om->compiledCVODEAdjointJacobianFunction;
}

/** returns the compiled adjoint quadrature function for the given model */
SBML_ODESOLVER_API CVQuadRhsFnB ODESense_getCompiledCVODEAdjointQuadFunction(odeSense_t *os)
{
  if ( !os->sensitivity )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_CANNOT_COMPILE_SENSITIVITY_NOT_COMPUTED,
		      "Attempting to compile adjoint quadrature before " \
		      "the parametric matrix is computed\n"		\
		      "Call ODESense_constructSensitivity before calling\n" \
		      "ODESense_getCompiledCVODEAdjointQuadFunction\n");
    return NULL;
  }

  if ( !os->compiledCVODEAdjointQuadFunction  || os->recompileSensitivity )
    /*!!! currently not used: if TCC multiple states become possible,
      until then this must have been compiled already within the main
      compiled code structure, or main must be recompiled here on
      second integrator runs when sens was switched on */
    /* only for calling independent of solver!!
       function should have been compiled already */
    if ( !ODESense_compileCVODESenseFunctions(os) )
      return NULL;


  return os->compiledCVODEAdjointQuadFunction;
}

/** @} */



/* End of file */
