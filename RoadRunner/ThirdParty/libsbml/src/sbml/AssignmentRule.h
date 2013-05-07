/**
 * @file    AssignmentRule.h
 * @brief   Definitions of AssignmentRule.
 * @author  Ben Bornstein
 *
 *
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2009-2012 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 * 
 * @class AssignmentRule
 * @brief LibSBML implementation of %SBML's %AssignmentRule construct.
 *
 * The rule type AssignmentRule is derived from the parent class Rule.  It
 * is used to express equations that set the values of variables.  The
 * left-hand side (the attribute named "variable") of an assignment rule
 * can refer to the identifier of a Species, SpeciesReference (in SBML
 * Level&nbsp;3), Compartment, or Parameter object in the model (but not a
 * Reaction).  The entity identified must have its "constant" attribute set
 * to @c false.  The effects of an AssignmentRule are in general terms the
 * same, but differ in the precise details depending on the type of
 * variable being set: <ul>

 * <li> <em>In the case of a species</em>, an AssignmentRule sets the
 * referenced species' quantity (whether a "concentration" or "amount") to
 * the value determined by the formula in the MathML subelement "math".
 * The unit associated with the value produced by the "math" formula @em
 * should (in SBML Level&nbsp;2 Version&nbsp;4 and in SBML Level&nbsp;3) or @em must (in
 * SBML releases prior to Level&nbsp;2 version&nbsp;4) be equal to the unit
 * associated with the species' quantity.  <em>Restrictions</em>: There
 * must not be both an AssignmentRule "variable" attribute and a
 * SpeciesReference "species" attribute having the same value, unless the
 * referenced Species object has its "boundaryCondition" attribute set to
 * @c true.  In other words, an assignment rule cannot be defined for a
 * species that is created or destroyed in a reaction unless that species
 * is defined as a boundary condition in the model.
 *
 * <li> (For SBML Level&nbsp;3 only) <em>In the case of a species
 * reference</em>, an AssignmentRule sets the stoichiometry of the
 * referenced reactant or product to the value determined by the formula in
 * "math".  The unit associated with the value produced by the "math"
 * formula should be consistent with the unit "dimensionless", because
 * reactant and product stoichiometries in reactions are dimensionless
 * quantities.
  *
 * <li> <em>In the case of a compartment</em>, an AssignmentRule sets the
 * referenced compartment's size to the value determined by the formula in
 * the "math" subelement of the AssignmentRule object.  The overall units
 * of the formula in "math" @em should (in SBML Level&nbsp;2 Version&nbsp;4
 * and in SBML Level&nbsp;3) or @em must (in SBML releases prior to Level&nbsp;2
 * version&nbsp;4) be the same as the units of the size of the compartment.
 *
 * <li> <em>In the case of a parameter</em>, an AssignmentRule sets the
 * referenced parameter's value to that determined by the formula in the
 * "math" subelement of the AssignmentRule object.  The overall units of
 * the formula in the "math" subelement @em should (in SBML Level&nbsp;2
 * Version&nbsp;4 and in SBML Level&nbsp;3) or @em must (in SBML releases prior to
 * Level&nbsp;2 version&nbsp;4) be the same as the units defined for the
 * parameter.  </ul>
 * 
 * In the context of a simulation, assignment rules are in effect at all
 * times, <em>t</em> \f$\geq\f$ <em>0</em>.  For purposes of evaluating
 * expressions that involve the <em>delay</em> "csymbol" (see the SBML
 * Level&nbsp;2 specification), assignment rules are considered to apply
 * also at <em>t</em> \f$\leq\f$ <em>0</em>.  Please consult the relevant
 * SBML specification for additional information about the semantics of
 * assignments, rules, and entity values for simulation time <em>t</em>
 * \f$\leq\f$ <em>0</em>.
 *
 * A model must not contain more than one AssignmentRule or RateRule
 * object having the same value of "variable"; in other words, in the set
 * of all assignment rules and rate rules in an SBML model, each variable
 * appearing in the left-hand sides can only appear once.  This simply
 * follows from the fact that an indeterminate system would result if a
 * model contained more than one assignment rule for the same variable or
 * both an assignment rule and a rate rule for the same variable.
 *
 * Similarly, a model must also not contain <em>both</em> an AssignmentRule
 * and an InitialAssignment for the same variable, because both kinds of
 * constructs apply prior to and at the start of simulation time, i.e.,
 * <em>t</em> \f$\leq\f$ <em>0</em>.  If a model contained both an initial
 * assignment and an assignment rule for the same variable, an
 * indeterminate system would result.
 *
 * The value calculated by an AssignmentRule object overrides the value
 * assigned to the given symbol by the object defining that symbol.  For
 * example, if a Compartment object's "size" attribute value is set in its
 * definition, and the model also contains an AssignmentRule object having
 * that compartment's "id" as its "variable" value, then the "size"
 * assigned in the Compartment object definition is ignored and the value
 * assigned based on the computation defined in the AssignmentRule.  This
 * does <em>not</em> mean that a definition for a given symbol can be
 * omitted if there is an AssignmentRule object for it.  For example, there
 * must be a Parameter definition for a given parameter if there is an
 * AssignmentRule for that parameter.  It is only a question of which value
 * definition takes precedence.
 * 
 * @section general General summary of SBML rules
 *
 * @htmlinclude rules-general-summary.html
 * 
 * @section additional-restrictions Additional restrictions on SBML rules
 * 
 * @htmlinclude rules-additional-restrictions.html
 * 
 * @section RuleType_t Rule types for SBML Level 1
 *
 * SBML Level 1 uses a different scheme than SBML Level 2 and Level 3 for
 * distinguishing rules; specifically, it uses an attribute whose value is
 * drawn from an enumeration of 3 values.  LibSBML supports this using methods
 * that work @if clike a libSBML enumeration type, RuleType_t, whose values
 * are @else with the enumeration values @endif@~ listed below.
 *
 * @li @link RuleType_t#RULE_TYPE_RATE RULE_TYPE_RATE@endlink: Indicates
 * the rule is a "rate" rule.
 * @li @link RuleType_t#RULE_TYPE_SCALAR RULE_TYPE_SCALAR@endlink:
 * Indicates the rule is a "scalar" rule.
 * @li @link RuleType_t#RULE_TYPE_INVALID RULE_TYPE_INVALID@endlink:
 * Indicates the rule type is unknown or not yet set.
 *
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 


#ifndef AssignmentRule_h
#define AssignmentRule_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>



#ifdef __cplusplus


#include <string>

#include <sbml/Rule.h>
#include <sbml/SBMLVisitor.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLNamespaces;

class LIBSBML_EXTERN AssignmentRule : public Rule
{
public:

  /**
   * Creates a new AssignmentRule using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this AssignmentRule
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * AssignmentRule
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of an AssignmentRule object to an SBMLDocument
   * (e.g., using&nbsp; @if java Model::addRule(Rule r)@else Model::addRule()@endif, the SBML Level, SBML Version
   * and XML namespace of the document @em override the values used
   * when creating the AssignmentRule object via this constructor.  This is
   * necessary to ensure that an SBML document is a consistent structure.
   * Nevertheless, the ability to supply the values at the time of creation
   * of a AssignmentRule is an important aid to producing valid SBML.
   * Knowledge of the intented SBML Level and Version determine whether it
   * is valid to assign a particular value to an attribute, or whether it
   * is valid to add an object to an existing SBMLDocument.
   */
  AssignmentRule (unsigned int level, unsigned int version);


  /**
   * Creates a new AssignmentRule using the given SBMLNamespaces object
   * @p sbmlns.
   *
   * The SBMLNamespaces object encapsulates SBML Level/Version/namespaces
   * information.  It is used to communicate the SBML Level, Version, and
   * (in Level&nbsp;3) packages used in addition to SBML Level&nbsp;3 Core.
   * A common approach to using this class constructor is to create an
   * SBMLNamespaces object somewhere in a program, once, then pass it to
   * object constructors such as this one when needed.
   *
   * @param sbmlns an SBMLNamespaces object.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   *
   * @note Upon the addition of a AssignmentRule object to an SBMLDocument
   * (e.g., using&nbsp; @if java Model::addRule(Rule r)@else Model::addRule()@endif, the SBML XML namespace of
   * the document @em overrides the value used when creating the
   * AssignmentRule object via this constructor.  This is necessary to
   * ensure that an SBML document is a consistent structure.  Nevertheless,
   * the ability to supply the values at the time of creation of a
   * AssignmentRule is an important aid to producing valid SBML.  Knowledge
   * of the intented SBML Level and Version determine whether it is valid
   * to assign a particular value to an attribute, or whether it is valid
   * to add an object to an existing SBMLDocument.
   */
  AssignmentRule (SBMLNamespaces* sbmlns);


  /**
   * Destroys this AssignmentRule.
   */
  virtual ~AssignmentRule ();


  /**
   * Creates and returns a deep copy of this Rule.
   * 
   * @return a (deep) copy of this Rule.
   */
  virtual AssignmentRule* clone () const;


  /**
   * Accepts the given SBMLVisitor for this instance of AssignmentRule.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next AssignmentRule object
   * in the list of rules within which @em the present object is embedded.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Predicate returning @c true if
   * all the required attributes for this AssignmentRule object
   * have been set.
   *
   * @note In SBML Levels&nbsp;2&ndash;3, the only required attribute for
   * an AssignmentRule object is "variable".  For Level&nbsp;1, where the
   * equivalent attribute is known by different names ("compartment",
   * "species", or "name", depending on the type of object), there is an
   * additional required attribute called "formula".
   * 
   * @return @c true if the required attributes have been set, @c false
   * otherwise.
   */
  virtual bool hasRequiredAttributes() const ;


  /**
   * Renames all the SIdRef attributes on this element, including any found in MathML
   */
  virtual void renameSIdRefs(std::string oldid, std::string newid);


protected:
  /** @cond doxygen-libsbml-internal */

  /* the validator classes need to be friends to access the 
   * protected constructor that takes no arguments
   */
  friend class Validator;
  friend class ConsistencyValidator;
  friend class IdentifierConsistencyValidator;
  friend class InternalConsistencyValidator;
  friend class L1CompatibilityValidator;
  friend class L2v1CompatibilityValidator;
  friend class L2v2CompatibilityValidator;
  friend class L2v3CompatibilityValidator;
  friend class L2v4CompatibilityValidator;
  friend class MathMLConsistencyValidator;
  friend class ModelingPracticeValidator;
  friend class OverdeterminedValidator;
  friend class SBOConsistencyValidator;
  friend class UnitConsistencyValidator;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/* ----------------------------------------------------------------------------
 * See the .cpp file for the documentation of the following functions.
 * --------------------------------------------------------------------------*/


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG  */
#endif  /* AssignmentRule_h */

