/*
  Last changed Time-stamp: <2008-10-16 17:40:12 raim>
  $Id: odeConstruct.c,v 1.41 2008/10/16 17:27:50 raimc Exp $
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

/*! \defgroup symbolic Symbolic Analysis */
/*! \defgroup odeConstruct ODE Construction: SBML reactions -> SBML rate rules
  \ingroup symbolic
  \brief This module contains all functions to condense
  the reaction network of an input models to an output model only
  consisting of SBML rate rules, representing an ODE system f(x,p,t) = dx/dt 
    
  The ODE construction currently can't handle SBML algebraic rules.
  As soon as this is done correctly, the resulting SBML model can be
  used to initialize SUNDIALS IDA Solver for Differential Algebraic
  Equation (DAE) Systems, which are ODE systems with additional
  algebraic constraints.
*/
/*@{*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

#include "sbmlsolver/cvodeData.h"
#include "sbmlsolver/odeConstruct.h"
#include "sbmlsolver/modelSimplify.h"
#include "sbmlsolver/processAST.h"
#include "sbmlsolver/solverError.h"

static void ODE_replaceFunctionDefinitions(Model_t *m);
static Model_t *Model_copyInits(Model_t *old);
static int Model_createOdes(Model_t *m, Model_t*ode);
static void Model_copyOdes(Model_t *m, Model_t*ode);
static void Model_copyEvents(Model_t *m, Model_t*ode);
static int Model_copyAlgebraicRules(Model_t *m, Model_t*ode);
static void Model_copyAssignmentRules(Model_t *m, Model_t*ode);
static void Model_copyInitialAssignmentRules(Model_t *m, Model_t*ode);


/** C: Construct an ODE system from a Reaction Network
    
constructs an ODE systems of the reaction network of the passed
model `m' and returns a new SBML model `ode', that only consists
of RateRules, representing ODEs.  See comments at steps C.1-C.5
for details
*/

SBML_ODESOLVER_API Model_t*Model_reduceToOdes(Model_t *m)
{
  int errors;
  Model_t *ode;

  errors = 0;

  /** C.1: Create and initialize a new model */  
  ode = Model_copyInits(m);

  /** C.2: Copy predefined ODES (RateRules) */
  Model_copyOdes(m, ode);

  /**!!! TODO : SBML L2V2 - constraints */
  /**!!! TODO : supress ODE construction for algebraic rule
         defined variables !!! */
  
  
  /** C.3: Copy Assignment Rules to new model */
  Model_copyAssignmentRules(m, ode);
    
  /** C.4: Copy InitialAssignmentRules to new model */
  Model_copyInitialAssignmentRules(m, ode);

  /** C.5: Create ODEs from reactions */
  errors = Model_createOdes(m, ode);

  if ( errors>0 )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_ODE_MODEL_COULD_NOT_BE_CONSTRUCTED,
		      "ODE construction failed for %d variables.", errors);
    Model_free(ode);
    return NULL;
  }
  
  /** Copy incompatible SBML structures     
      The next steps will copy remaining definitions that can't be
      simplified, i.e. expressed in a system of ODEs, to the new model.
      They will also store warning and error messages, if these
      definitions cannot be interpreted correctly by the current state
      of the SBML_odeSolver. Additionally all assignment rules are
      copied, only for printing out results.
  */
  
  /** C.6a: Copy events to new model and create warning */
  Model_copyEvents(m, ode);

  /** C.6b: Copy AlgebraicRules to new model and create error */
  errors = Model_copyAlgebraicRules(m, ode);
  if ( errors>0 )
  {
    SolverError_error(ERROR_ERROR_TYPE,
		      SOLVER_ERROR_ODE_MODEL_COULD_NOT_BE_CONSTRUCTED,
		      "Model contains %d algebraic rules.", errors);
    SBase_setNotesString ((SBase_t *)ode, "DAE model");
  }

  /** C.8: replace function definitions in all formulas */
  ODE_replaceFunctionDefinitions(ode);
    
  return ode;
}


/* Initialize a new model from an input model
   Creates a new SBML model and copies
   compartments, species and parameters from the passed model,
   species are converted to concentration units, unless the
   species has only substance units or its compartment has
   spatial dimension 0 */ 
static Model_t *Model_copyInits(Model_t *old)
{
  int i;
  Model_t *new;
  Compartment_t *c;
  Species_t *s, *s_new;

  new = Model_create();

  if ( Model_isSetId(old) )
    Model_setId(new, Model_getId(old));
  else if ( Model_isSetName(old) ) /* problematic? necessary ?*/
    Model_setId(new, Model_getName(old));

  if ( Model_isSetName(old) )
    Model_setName(new, Model_getName(old));
  else if ( Model_isSetId(old) )
    Model_setName(new, Model_getId(old));

  for ( i=0; i<Model_getNumCompartments(old); i++)
    Model_addCompartment(new, Model_getCompartment(old, i));
  
  for ( i=0; i<Model_getNumParameters(old); i++)
    Model_addParameter(new, Model_getParameter(old, i));
  
  for ( i=0; i<Model_getNumSpecies(old); i++)
  {
    s = Model_getSpecies(old, i);
    s_new = Species_clone(s);

    /* convert initial amount to concentration, unless species has only
       substance units */
    if ( Species_isSetInitialAmount(s_new) &&
	 !Species_getHasOnlySubstanceUnits(s_new) )
    {
      c = Model_getCompartmentById(new, Species_getCompartment(s_new));
      if (Compartment_getSpatialDimensions(c) != 0 )
      {
	Species_setInitialConcentration(s_new, Species_getInitialAmount(s)/
					Compartment_getSize(c));
      }    
    }
    Model_addSpecies(new, s_new);
    Species_free(s_new);
  }
  
  /* Function Definitions  */
  for ( i=0; i<Model_getNumFunctionDefinitions(old); i++ )
    Model_addFunctionDefinition(new, Model_getFunctionDefinition(old, i));

  return(new);
}

/* C.2: Copy predefined ODEs (RateRules) from `m' to `ode'
   identifies all predefined ODEs in `m' (RateRules) and
   adds them as RateRules to the model `ode' */
static void Model_copyOdes(Model_t *m, Model_t*ode )
{  
  Rule_t *rl;
  Rule_t *rr, *rl_new;
  SBMLTypeCode_t type;
  ASTNode_t *math;
  
  int j;

  math   = NULL;
  
  for ( j=0; j<Model_getNumRules(m); j++ )
  {
    rl = Model_getRule(m,j);
    type = SBase_getTypeCode((SBase_t *)rl);
    if ( type == SBML_RATE_RULE )
    {
      rr = rl;
      if ( Rule_isSetMath(rl) && Rule_isSetVariable(rr) )
      {
	math = copyAST(Rule_getMath(rl));
	rl_new = Rule_createRate();
	Rule_setVariable(rl_new, Rule_getVariable(rr));
	Rule_setMath((Rule_t *)rl_new, math);
	Model_addRule(ode, (Rule_t *)rl_new);
	Rule_free(rl_new);
	ASTNode_free(math);
      }
    }
  }
}


/* C.3: Create ODEs from Reactions
   for each species in model `m' an ODE is constructed from its
   reactions in `m' and added as a RateRule to `ode'
*/
static int Model_createOdes(Model_t *m, Model_t *ode)
{
  Species_t *s;
  Rule_t *rule, *rule_new;
  SBMLTypeCode_t type;
  ASTNode_t *math;

  int i, j, errors, found;

  errors = 0;
  found  = 0;
  s      = NULL;
  math   = NULL;


  /* C.3: Construct ODEs from Reactions construct ODEs for
     non-constant and non-boundary species from reactions, if they
     have not already been set by a rate or an assignment rule in the
     old model. Local parameters of the kinetic laws are replaced on
     the fly for each reaction.  Rate rules have been added to new
     model in C.2 and assignment rule will be handled later in step
     C.5. */

  /* The species vector should later be returned by a
     function that finds mass-conservation relations and reduces the
     number of independent variables by matrix operations as developed
     in Metabolic Control Analysis. (eg. Reder 1988, Sauro 2003). */

  for ( i=0; i<Model_getNumSpecies(m); i++ )
  {
    s = Model_getSpecies(m, i);

    found = 0;
    /* search `m' if a rule exists for this species */
    for ( j=0; j<Model_getNumRules(m); j++ )
    {
      rule = Model_getRule(m, j);
      type = SBase_getTypeCode((SBase_t *)rule);
      if ( type == SBML_RATE_RULE )
      {
	if ( strcmp(Species_getId(s), Rule_getVariable(rule)) == 0 )
	  found = 1;
      }
      else if ( type == SBML_ASSIGNMENT_RULE )
      {
	if ( strcmp(Species_getId(s), Rule_getVariable(rule)) == 0 ) 
	  found = 1;	
      }
    }

    /* if no rule exists and the species is not defined as a constant
       or a boundary condition of the model, construct and ODE from
       the reaction in `m' where the species is reactant or product,
       and add it as a RateRule to the new model `ode' */
    if ( found == 0 )
    {
      if ( !Species_getConstant(s) && !Species_getBoundaryCondition(s) )
      {
	math = Species_odeFromReactions(s, m);

	if ( math == NULL )
	{
	  errors++;
	  SolverError_error(ERROR_ERROR_TYPE,
			    SOLVER_ERROR_ODE_COULD_NOT_BE_CONSTRUCTED_FOR_SPECIES,
			    "ODE could not be constructed for species %s",
			    Species_getId(s));
	}
	else if (( ASTNode_getType(math) == AST_REAL &&
		   ASTNode_getReal(math) == 0.0 ) ||
		 ( ASTNode_getType(math) == AST_INTEGER &&
		   ASTNode_getInteger(math) == 0 ) )
	  /* don't create redundant rate rules for species */
	  ASTNode_free(math);
	else
	{
	  rule_new = Rule_createRate();
	  Rule_setVariable(rule_new, Species_getId(s));
	  Rule_setMath((Rule_t *)rule_new, math);
	  Model_addRule(ode, (Rule_t *)rule_new);
	  ASTNode_free(math);
	  Rule_free(rule_new);
	}
      }
    }
  }

  for ( i=0; i != Model_getNumReactions(m); i++ )
  {
    Reaction_t *reaction =
      (Reaction_t *)ListOf_get(Model_getListOfReactions(m), i);
    KineticLaw_t *kineticLaw = Reaction_getKineticLaw(reaction);
    Parameter_t *parameter = Parameter_create();

    Parameter_setId(parameter, Reaction_getId(reaction));
    Parameter_setConstant(parameter, 0);
    Model_addParameter(ode, parameter);
    Parameter_free(parameter);

    if ( kineticLaw )
    {
      rule = Rule_createAssignment();
      Rule_setVariable(rule, Reaction_getId(reaction));
      math = copyAST(KineticLaw_getMath(kineticLaw));
      AST_replaceNameByParameters(math,
				  KineticLaw_getListOfParameters(kineticLaw));
      Rule_setMath(rule, math);
      Model_addRule(ode, rule);
      Rule_free(rule);
      ASTNode_free(math);
    }
  }

  return errors;
}


/** Creates an ODE for a species from its reactions
    
This function takes a species and a model, constructs and ODE for
that species from all the reaction it appears in as either reactant
or product. Local kinetic Law parameters are replaced by their value.
It directly constructs an Abstract Syntax Tree (AST) and returns a
pointer to it. 

In one of the next releases this function will take additional
arguments:

a listOfParameter, which will allow to `globalize' local
parameters (e.g. for sensitivity analysis for local parameters)

and

a listOfRules, which will allow to convert kineticLaws to assignment
rules. The ODEs will then only consist of stoichiometry and a
parameter. As a kinetic Law appears in all ODEs of the reaction,
this will significantly reduce the time for numerical evalution of
the ODE system and thus improve the performance of integration routines.
*/

SBML_ODESOLVER_API ASTNode_t *Species_odeFromReactions(Species_t *s, Model_t *m)
{

  int j, k, errors;
  Reaction_t *r;
  KineticLaw_t *kl;
  SpeciesReference_t *sref;
  Compartment_t *c;
  ASTNode_t *simple, *ode, *tmp, *reactant, *reactionSymbol;

  errors = 0;
  ode = NULL;

  /* search for the species in all reactions, and
     add up the kinetic laws * stoichiometry for
     all consuming and producing reactions to
     an ODE */

  for ( j=0; j<Model_getNumReactions(m); j++ )
  {
    r = Model_getReaction(m,j);

    reactionSymbol = ASTNode_createWithType(AST_NAME);
    ASTNode_setName(reactionSymbol, Reaction_getId(r));
   /*  ASTNode_setType(reactionSymbol, AST_NAME); */

    kl = Reaction_getKineticLaw(r);

    if (!kl)
    {
      SolverError_error(ERROR_ERROR_TYPE,
			SOLVER_ERROR_NO_KINETIC_LAW_FOUND_FOR_REACTION,
			"The model has no kinetic law for reaction %s",
			Reaction_getId(r));
      ++errors;
    }

    for ( k=0; k<Reaction_getNumReactants(r); k++ )
    {
      sref = Reaction_getReactant(r,k);
      if ( strcmp(SpeciesReference_getSpecies(sref), Species_getId(s)) == 0 )
      {
	/* Construct expression for reactant by multiplying the
	   kinetic law with stoichiometry (math) and putting a
	   minus in front of it
	*/
	if ( SpeciesReference_isSetStoichiometryMath(sref) )
	{
	  reactant = ASTNode_create();
	  ASTNode_setCharacter(reactant, '*');
	  ASTNode_addChild(reactant,
			   copyAST(StoichiometryMath_getMath(SpeciesReference_getStoichiometryMath(sref))));
	  ASTNode_addChild(reactant, copyAST(reactionSymbol));
	}
	else
	{
	  if ( SpeciesReference_getStoichiometry(sref) == 1. )	  
	    reactant = copyAST(reactionSymbol);	  
	  else
	  {
	    reactant = ASTNode_create();
	    ASTNode_setCharacter(reactant, '*');
	    ASTNode_addChild(reactant, ASTNode_create());
	    ASTNode_setReal(ASTNode_getChild(reactant,0), 
                            SpeciesReference_getStoichiometry(sref));
	    ASTNode_addChild(reactant, copyAST(reactionSymbol));
	  }
	}

	/* replace local parameters by their value,
	   before adding to ODE */
	if (kl)
	  AST_replaceNameByParameters(reactant,
				      KineticLaw_getListOfParameters(kl));

	/* Add reactant expression to ODE  */
	if ( ode == NULL )
	{
	  ode = ASTNode_create();
	  ASTNode_setCharacter(ode,'-');
	  ASTNode_addChild(ode, reactant);
	}
	else
	{
	  tmp = copyAST(ode);
	  ASTNode_free(ode);
	  ode = ASTNode_create();
	  ASTNode_setCharacter(ode, '-');
	  ASTNode_addChild(ode, tmp);
	  ASTNode_addChild(ode, reactant);
	}
      }
    }

    for ( k=0; k<Reaction_getNumProducts(r); k++ )
    {
      sref = Reaction_getProduct(r,k);
      if ( strcmp(SpeciesReference_getSpecies(sref), Species_getId(s)) == 0 )
      {
	reactant = ASTNode_create();
	ASTNode_setCharacter(reactant, '*');

	if ( SpeciesReference_isSetStoichiometryMath(sref) ) 
	  ASTNode_addChild(reactant,
			   copyAST(StoichiometryMath_getMath(SpeciesReference_getStoichiometryMath(sref))));	
	else
	{
	  ASTNode_addChild(reactant, ASTNode_create());
	  ASTNode_setReal(ASTNode_getChild(reactant,0),
			  SpeciesReference_getStoichiometry(sref));
	}
	ASTNode_addChild(reactant, copyAST(reactionSymbol));

	/* replace local parameters by their value,
	   before adding to ODE */
	if (kl)
	  AST_replaceNameByParameters(reactant,
				      KineticLaw_getListOfParameters(kl));
	/* Add reactant expression to ODE  */
	if ( ode == NULL ) 
	  ode = reactant;
	else
	{
	  tmp = copyAST(ode);
	  ASTNode_free(ode);
	  ode = ASTNode_create();
	  ASTNode_setCharacter(ode, '+');
	  ASTNode_addChild(ode, tmp);
	  ASTNode_addChild(ode, reactant);
	}	  
      }
    }
    
    ASTNode_free(reactionSymbol);
  }

  /* Divide ODE by Name of the species' compartment.
     If formula is empty skip division by compartment and set formula
     to 0.  The latter case can happen, if a species is neither
     constant nor a boundary condition but appears only as a modifier
     in reactions. The rate for such species is set to 0. */

  c = Model_getCompartmentById(m, Species_getCompartment(s)); 

  if( ode != NULL )
  {
    if ( !Species_getHasOnlySubstanceUnits(s) &&
	 Compartment_getSpatialDimensions(c) !=0 )
    {
      tmp = copyAST(ode);
      ASTNode_free(ode);
      ode = ASTNode_create();
      ASTNode_setCharacter(ode, '/');
      ASTNode_addChild(ode, tmp);
      ASTNode_addChild(ode, ASTNode_create());
      ASTNode_setName(ASTNode_getChild(ode,1), Species_getCompartment(s));
    }
  }
  else
  {
    /*
      for modifier species that never appear as products or reactants
      but are not defined as constant or boundarySpecies, set ODE to 0.
    */
    ode = ASTNode_create();
    ASTNode_setInteger(ode, 0); /*  !!! change for DAE models should
                                    be defined by algebraic rule!*/
  }

  simple = simplifyAST(ode);
  ASTNode_free(ode);

  if ( errors>0 )
  {
    ASTNode_free(simple);
    return NULL;
  }
  else
    return simple;
}

/** C.4a: Copy Events
    copy events to new model and print warning */
static void Model_copyEvents(Model_t *m, Model_t*ode)
{
  int i;
    
  for ( i=0; i<Model_getNumEvents(m); i++ )
  {
    Model_addEvent(ode, Model_getEvent(m, i));
    
    if ( !i )
      SolverError_error(WARNING_ERROR_TYPE,
			SOLVER_ERROR_THE_MODEL_CONTAINS_EVENTS,
			"The model contains events. "
			"The SBML_odeSolver implementation of events "
			"is not fully SBML conformant. Results will "
			"depend on the simulation duration and the "
			"number of output steps.");
  }
}


/* C.4.b: Copy Algebraic Rules
   copy algebraic rules to new model and create error
   message, return number of AlgebraicRules */
static int Model_copyAlgebraicRules(Model_t *m, Model_t*ode)
{
  int i, j;
  Rule_t *rl, *alr_new;
  SBMLTypeCode_t type;
  ASTNode_t *math;
  int errors;
  
  errors = 0;
  
  for ( j=i=0; i<Model_getNumRules(m); i++ )
  {
    rl = Model_getRule(m, i);
    type = SBase_getTypeCode((SBase_t *)rl);
    if ( type == SBML_ALGEBRAIC_RULE )
    {
      if ( Rule_isSetMath(rl) )
      {
	math = copyAST(Rule_getMath(rl));
	alr_new = Rule_createAlgebraic();
	Rule_setMath(alr_new, math);
	Model_addRule(ode, alr_new);
	ASTNode_free(math);
	Rule_free(alr_new);
	errors++;
	if ( !j ) 
	  SolverError_error(ERROR_ERROR_TYPE,
			    SOLVER_ERROR_THE_MODEL_CONTAINS_ALGEBRAIC_RULES,
			    "The model contains Algebraic Rules. "
			    "SBML_odeSolver is unable to solve "
			    "models of this type.");
	j++;
      }
    }
  }
  
  return errors;
}

/* Copy InitialAssignment Rules  */

static void Model_copyInitialAssignmentRules(Model_t *m, Model_t*ode)
{
  int i;
  for ( i=0; i<Model_getNumInitialAssignments(m); i++ )
  {  
    Model_addInitialAssignment(ode, Model_getInitialAssignment(m, i));
  }
}

/* Copy Assignment Rules  */
static void Model_copyAssignmentRules(Model_t *m, Model_t*ode)
{
  int i;
  Rule_t *rl, *ar_new;
  SBMLTypeCode_t type;
  ASTNode_t *math;  

  for ( i=0; i<Model_getNumRules(m); i++ )
  {
    rl = Model_getRule(m,i);
    type = SBase_getTypeCode((SBase_t *)rl);

    if ( type == SBML_ASSIGNMENT_RULE )
    {
      if ( Rule_isSetMath(rl) && Rule_isSetVariable(rl) )
      {
	math = copyAST(Rule_getMath(rl));
	ar_new = Rule_createAssignment();
	Rule_setVariable(ar_new, Rule_getVariable(rl));
	Rule_setMath((Rule_t *)ar_new, math);
	Model_addRule(ode, (Rule_t *)ar_new);
	ASTNode_free(math);
	Rule_free(ar_new);
      }
    }
  }
}  
  
/** Function Definition Replacement: replaces all occurences of a user
    defined function by their function definition
*/
static void ODE_replaceFunctionDefinitions(Model_t *m)
{
  int i, j, k;
  Rule_t *rl_new;
  FunctionDefinition_t *f;
  Event_t *e;
  Trigger_t *tr;
  EventAssignment_t *ea;
  ASTNode_t *math;
  
 
  /** Step C.2: replace Function Definitions
      All Function Definitions will be replaced
      by the full expression
      in ODEs (rate rules), Algebraic Rules and Events
      of the ode model.
  */
  
  for ( i=0; i<Model_getNumFunctionDefinitions(m); i++ )
  {
    f = Model_getFunctionDefinition(m, i);
    /*
      replacing functions in ODEs (rate rules), assignment rules and
      algebraic rules of the ode model
    */
    for ( j=0; j<Model_getNumRules(m); j++ )
    {
      rl_new = Model_getRule(m, j);
      math = copyAST(Rule_getMath(rl_new));
      AST_replaceFunctionDefinition(math,
				    FunctionDefinition_getId(f),
				    FunctionDefinition_getMath(f));
      Rule_setMath(rl_new, math);
      ASTNode_free(math);
    }
    /*
      replacing functions in all events
      and event assignments of the ode model
    */	
    for ( j=0; j<Model_getNumEvents(m); j++ )
    {
      e = Model_getEvent(m, j);
      for ( k=0; k<Event_getNumEventAssignments(e); k++ )
      {
	ea = Event_getEventAssignment(e, k);
	math = copyAST(EventAssignment_getMath(ea));
	AST_replaceFunctionDefinition(math,
				      FunctionDefinition_getId(f),
				      FunctionDefinition_getMath(f));
	EventAssignment_setMath(ea, math);
	ASTNode_free(math);
      }

      /*??? problem: returned Trigger structure is const ???*/
      tr = (Trigger_t *)Event_getTrigger(e);
      math = copyAST(Trigger_getMath(tr));
      AST_replaceFunctionDefinition(math,
				    FunctionDefinition_getId(f),
				    FunctionDefinition_getMath(f));
      Trigger_setMath(tr, math);
      ASTNode_free(math);
    }
  }
}



/** Returns the value of a compartment, species or parameter
    with the passed ID. Note that species values are always
    returned as concentrations, unless the species has only substance
    units or its compartments has spatial dimension 0.
*/

SBML_ODESOLVER_API double Model_getValueById(Model_t *m, const char *id)
{

  Species_t *s;
  Parameter_t *p;
  Compartment_t *c;

  if ( (p = Model_getParameterById(m, id)) !=NULL ) 
    if ( Parameter_isSetValue(p) ) 
      return Parameter_getValue(p);    
  
  if ( (c = Model_getCompartmentById(m, id)) !=NULL ) 
    if ( Compartment_isSetSize(c) ) 
      return Compartment_getSize(c);  

  if ( (s = Model_getSpeciesById(m, id)) !=NULL )
  {
    if ( Species_isSetInitialConcentration(s) ) 
      return Species_getInitialConcentration(s);    
    else if ( Species_isSetInitialAmount(s) )
    {
      c = Model_getCompartmentById(m, Species_getCompartment(s));
      if ( Compartment_getSpatialDimensions(c) != 0 &&
	   !Species_getHasOnlySubstanceUnits(s) ) 
	return Species_getInitialAmount(s) / Compartment_getSize(c);
      else
	return Species_getInitialAmount(s);
    }
  }

  SolverError_error(ERROR_ERROR_TYPE,
		    SOLVER_ERROR_REQUESTED_PARAMETER_NOT_FOUND,
		    "SBML Model doesn't provide a value " \
		    "for SBML ID %s, value defaults to 0!", id);
  
/*   fprintf(stderr, "Value for %s not found!", id);  */
/*   fprintf(stderr, "Defaults to 0. Please check model!");  */
  return (0.0);
}


/** Sets the value of a compartment, species or parameter
    with the passed ID. For species it depends on the current
    state of the species whether concentrations or substance units
    are set.
*/

SBML_ODESOLVER_API int Model_setValue(Model_t *m, const char *id, const char *rid, double value)
{
  int i;
  Compartment_t *c;
  Species_t *s;
  Parameter_t *p;
  Reaction_t *r;
  KineticLaw_t *kl;

  if ( (r = Model_getReactionById(m, rid)) != NULL )
  {
    kl = Reaction_getKineticLaw(r);
    for ( i=0; i<KineticLaw_getNumParameters(kl); i++ )
    {
      p = KineticLaw_getParameter(kl, i);
      if ( strcmp(id, Parameter_getId(p)) == 0 )
      {
	Parameter_setValue(p, value);
	return 1;
      }
    }
  }
  if ( (c = Model_getCompartmentById(m, id)) != NULL )
  {
    Compartment_setSize(c, value);
    return 1;
  }
  if ( (s = Model_getSpeciesById(m, id)) != NULL )
  {
    if ( Species_isSetInitialAmount(s) ) 
      Species_setInitialAmount(s, value);    
    else 
      Species_setInitialConcentration(s, value);    
    return 1;
  }
  if ( (p = Model_getParameterById(m, id)) != NULL )
  {
    Parameter_setValue(p, value);
    return 1;
  }
  return 0;  
}

/** @} */
/* End of file */
