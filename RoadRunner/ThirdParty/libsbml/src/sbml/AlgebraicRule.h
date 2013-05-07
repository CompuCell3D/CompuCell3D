/**
 * @file    AlgebraicRule.h
 * @brief   Definitions of AlgebraicRule.
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
 * @class AlgebraicRule
 * @brief LibSBML implementation of %SBML's %AlgebraicRule construct.
 *
 * The rule type AlgebraicRule is derived from the parent class Rule.  It
 * is used to express equations that are neither assignments of model
 * variables nor rates of change.  AlgebraicRule does not add any
 * attributes to the basic Rule; its role is simply to distinguish this
 * case from the other cases.
 *
 * In the context of a simulation, algebraic rules are in effect at all
 * times, <em>t</em> \f$\geq\f$ <em>0</em>.  For purposes of evaluating
 * expressions that involve the delay "csymbol" (see the SBML
 * specification), algebraic rules are considered to apply also at
 * <em>t</em> \f$\leq\f$ <em>0</em>.  Please consult the relevant SBML
 * specification for additional information about the semantics of
 * assignments, rules, and entity values for simulation time <em>t</em>
 * \f$\leq\f$ <em>0</em>.
 *
 * An SBML model must not be overdetermined.  The ability to define
 * arbitrary algebraic expressions in an SBML model introduces the
 * possibility that a model is mathematically overdetermined by the overall
 * system of equations constructed from its rules, reactions and events.
 * Therefore, if an algebraic rule is introduced in a model, for at least
 * one of the entities referenced in the rule's "math" element the value of
 * that entity must not be completely determined by other constructs in the
 * model.  This means that at least this entity must not have the attribute
 * "constant"=@c true and there must also not be a rate rule or assignment
 * rule for it.  Furthermore, if the entity is a Species object, its value
 * must not be determined by reactions, which means that it must either
 * have the attribute "boundaryCondition"=@c true or else not be involved
 * in any reaction at all.  These restrictions are explained in more detail
 * in the SBML specification documents.
 *
 * In SBML Levels 2 and&nbsp;3, Reaction object identifiers can be
 * referenced in the "math" expression of an algebraic rule, but reaction
 * rates can never be <em>determined</em> by algebraic rules.  This is true
 * even when a reaction does not contain a KineticLaw element.  (In such
 * cases of missing KineticLaw elements, the model is valid but incomplete;
 * the rates of reactions lacking kinetic laws are simply undefined, and
 * not determined by the algebraic rule.)
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


#ifndef AlgebraicRule_h
#define AlgebraicRule_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>



#ifdef __cplusplus


#include <string>

#include <sbml/Rule.h>
#include <sbml/SBMLVisitor.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLNamespaces;

class LIBSBML_EXTERN AlgebraicRule : public Rule
{
public:

  /**
   * Creates a new AlgebraicRule using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this AlgebraicRule
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * AlgebraicRule
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of an AlgebraicRule object to an SBMLDocument
   * (e.g., using&nbsp; @if java Model::addRule(Rule r)@else Model::addRule()@endif), the SBML Level, SBML Version
   * and XML namespace of the document @em override the values used
   * when creating the AlgebraicRule object via this constructor.  This is
   * necessary to ensure that an SBML document is a consistent structure.
   * Nevertheless, the ability to supply the values at the time of creation
   * of a AlgebraicRule is an important aid to producing valid SBML.
   * Knowledge of the intented SBML Level and Version determine whether it
   * is valid to assign a particular value to an attribute, or whether it
   * is valid to add an object to an existing SBMLDocument.
   */
  AlgebraicRule (unsigned int level, unsigned int version);


  /**
   * Creates a new AlgebraicRule using the given SBMLNamespaces object
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
   * @throws @if python ValueError @else SBMLConstructorException @endif
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   *
   * @note Upon the addition of a AlgebraicRule object to an SBMLDocument
   * (e.g., using&nbsp; @if java Model::addRule(Rule r)@else Model::addRule()@endif, the SBML XML namespace of the
   * document @em overrides the value used when creating the AlgebraicRule
   * object via this constructor.  This is necessary to ensure that an SBML
   * document is a consistent structure.  Nevertheless, the ability to
   * supply the values at the time of creation of a AlgebraicRule is an
   * important aid to producing valid SBML.  Knowledge of the intented SBML
   * Level and Version determine whether it is valid to assign a particular
   * value to an attribute, or whether it is valid to add an object to an
   * existing SBMLDocument.
   */
  AlgebraicRule (SBMLNamespaces* sbmlns);


  /**
   * Destroys this AlgebraicRule.
   */
  virtual ~AlgebraicRule ();


  /**
   * Creates and returns a deep copy of this Rule.
   * 
   * @return a (deep) copy of this Rule.
   */
  virtual AlgebraicRule* clone () const;


  /**
   * Accepts the given SBMLVisitor for this instance of AlgebraicRule.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next AlgebraicRule object
   * in the list of rules within which @em the present object is embedded.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /** @cond doxygen-libsbml-internal */

  /**
   * sets the mInternalIdOnly flag
   */

  void setInternalIdOnly();
  bool getInternalIdOnly() const;
  
  /** @endcond */


  /**
   * Predicate returning @c true if
   * all the required attributes for this AlgebraicRule object
   * have been set.
   *
   * @note In SBML Levels&nbsp;2&ndash;3, there is no required attribute
   * for an AlgebraicRule object.  For Level&nbsp;1, the only required
   * attribute is "formula".
   * 
   * @return @c true if the required attributes have been set, @c false
   * otherwise.
   */
  virtual bool hasRequiredAttributes() const ;


protected:
  /** @cond doxygen-libsbml-internal */

  bool mInternalIdOnly;

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
#endif  /* AlgebraicRule_h */

