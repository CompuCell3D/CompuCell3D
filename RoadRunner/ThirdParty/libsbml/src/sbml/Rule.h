/**
 * @file    Rule.h
 * @brief   Definitions of Rule and ListOfRules.
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
 * @class Rule
 * @brief LibSBML implementation of %SBML's %Rule construct.
 *
 * In SBML, @em rules provide additional ways to define the values of
 * variables in a model, their relationships, and the dynamical behaviors
 * of those variables.  They enable encoding relationships that cannot be
 * expressed using Reaction nor InitialAssignment objects alone.
 *
 * The libSBML implementation of rules mirrors the SBML Level&nbsp;3
 * Version&nbsp;1 Core definition (which is in turn is very similar to the
 * Level&nbsp;2 Version&nbsp;4 definition), with Rule being the parent
 * class of three subclasses as explained below.  The Rule class itself
 * cannot be instantiated by user programs and has no constructor; only the
 * subclasses AssignmentRule, AlgebraicRule and RateRule can be
 * instantiated directly.
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
/**
 * @class ListOfRules
 * @brief LibSBML implementation of SBML's %ListOfRules construct.
 * 
 * The various ListOf___ classes in %SBML are merely containers used for
 * organizing the main components of an %SBML model.  All are derived from
 * the abstract class SBase, and inherit the various attributes and
 * subelements of SBase, such as "metaid" as and "annotation".  The
 * ListOf___ classes do not add any attributes of their own.
 *
 * The relationship between the lists and the rest of an %SBML model is
 * illustrated by the following (for SBML Level&nbsp;3 and later versions
 * of SBML Level&nbsp;2 as well):
 *
 * @image html listof-illustration.jpg "ListOf___ elements in an SBML Model"
 * @image latex listof-illustration.jpg "ListOf___ elements in an SBML Model"
 *
 * Readers may wonder about the motivations for using the ListOf___
 * containers.  A simpler approach in XML might be to place the components
 * all directly at the top level of the model definition.  The choice made
 * in SBML is to group them within XML elements named after
 * ListOf<em>Classname</em>, in part because it helps organize the
 * components.  More importantly, the fact that the container classes are
 * derived from SBase means that software tools can add information @em about
 * the lists themselves into each list container's "annotation".
 *
 * @see ListOfFunctionDefinitions
 * @see ListOfUnitDefinitions
 * @see ListOfCompartmentTypes
 * @see ListOfSpeciesTypes
 * @see ListOfCompartments
 * @see ListOfSpecies
 * @see ListOfParameters
 * @see ListOfInitialAssignments
 * @see ListOfRules
 * @see ListOfConstraints
 * @see ListOfReactions
 * @see ListOfEvents
 */


#ifndef Rule_h
#define Rule_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>


BEGIN_C_DECLS

/**
 * @enum RuleType_t
 */
typedef enum
{
    RULE_TYPE_RATE
  , RULE_TYPE_SCALAR
  , RULE_TYPE_INVALID
} RuleType_t;

END_C_DECLS


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/ExpectedAttributes.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/ListOf.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class ASTNode;
class ListOfRules;
class SBMLNamespaces;


class LIBSBML_EXTERN Rule : public SBase
{
public:

  /**
   * Destroys this Rule.
   */
  virtual ~Rule ();


  /**
   * Copy constructor; creates a copy of this Rule.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  Rule (const Rule& orig);


  /**
   * Assignment operator for Rule.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  Rule& operator=(const Rule& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of Rule.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next Rule object in the
   * list of rules within which @em the present object is embedded.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this Rule.
   * 
   * @return a (deep) copy of this Rule.
   */
  virtual Rule* clone () const;


  /**
   * Returns the mathematical expression of this Rule in text-string form.
   *
   * The text string is produced by
   * @if java <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode)">libsbml.formulaToString()</a></code>@else SBML_formulaToString()@endif; please consult
   * the documentation for that function to find out more about the format
   * of the text-string formula.
   * 
   * @return the formula text string for this Rule.
   *
   * @note The attribute "formula" is specific to SBML Level&nbsp;1; in
   * higher Levels of SBML, it has been replaced with a subelement named
   * "math".  However, libSBML provides a unified interface to the
   * underlying math expression and this method can be used for models
   * of all Levels of SBML.
   *
   * @see getMath()
   */
  const std::string& getFormula () const;


  /**
   * Get the mathematical formula of this Rule as an ASTNode tree.
   *
   * @return an ASTNode, the value of the "math" subelement of this Rule.
   *
   * @note The subelement "math" is present in SBML Levels&nbsp;2
   * and&nbsp;3.  In SBML Level&nbsp;1, the equivalent construct is the
   * attribute named "formula".  LibSBML provides a unified interface to
   * the underlying math expression and this method can be used for models
   * of all Levels of SBML.
   *
   * @see getFormula()
   */
  const ASTNode* getMath () const;


  /**
   * Get the value of the "variable" attribute of this Rule object.
   *
   * In SBML Level&nbsp;1, the different rule types each have a different
   * name for the attribute holding the reference to the object
   * constituting the left-hand side of the rule.  (E.g., for
   * SBML Level&nbsp;1's SpeciesConcentrationRule the attribute is "species", for
   * CompartmentVolumeRule it is "compartment", etc.)  In SBML
   * Levels&nbsp;2 and&nbsp;3, the only two types of Rule objects with a
   * left-hand side object reference are AssignmentRule and RateRule, and
   * both of them use the same name for attribute: "variable".  In order to
   * make it easier for application developers to work with all Levels of
   * SBML, libSBML uses a uniform name for all of such attributes, and it
   * is "variable", regardless of whether Level&nbsp;1 rules or
   * Level&nbsp;2&ndash;3 rules are being used.
   * 
   * @return the identifier string stored as the "variable" attribute value
   * in this Rule, or @c NULL if this object is an AlgebraicRule object.
   */
  const std::string& getVariable () const;


  /**
   * Returns the units for the
   * mathematical formula of this Rule.
   * 
   * @return the identifier of the units for the expression of this Rule.
   *
   * @note The attribute "units" exists on SBML Level&nbsp;1 ParameterRule
   * objects only.  It is not present in SBML Levels&nbsp;2 and&nbsp;3.
   */
  const std::string& getUnits () const;


  /**
   * Predicate returning @c true if this
   * Rule's mathematical expression is set.
   * 
   * This method is equivalent to isSetMath().  This version is present for
   * easier compatibility with SBML Level&nbsp;1, in which mathematical
   * formulas were written in text-string form.
   * 
   * @return @c true if the mathematical formula for this Rule is
   * set, @c false otherwise.
   *
   * @note The attribute "formula" is specific to SBML Level&nbsp;1; in
   * higher Levels of SBML, it has been replaced with a subelement named
   * "math".  However, libSBML provides a unified interface to the
   * underlying math expression and this method can be used for models
   * of all Levels of SBML.
   *
   * @see isSetMath()
   */
  bool isSetFormula () const;


  /**
   * Predicate returning @c true if this
   * Rule's mathematical expression is set.
   *
   * This method is equivalent to isSetFormula().
   * 
   * @return @c true if the formula (or equivalently the math) for this
   * Rule is set, @c false otherwise.
   *
   * @note The subelement "math" is present in SBML Levels&nbsp;2
   * and&nbsp;3.  In SBML Level&nbsp;1, the equivalent construct is the
   * attribute named "formula".  LibSBML provides a unified interface to
   * the underlying math expression and this method can be used for models
   * of all Levels of SBML.
   *
   * @see isSetFormula()
   */
  bool isSetMath () const;


  /**
   * Predicate returning @c true if this
   * Rule's "variable" attribute is set.
   *
   * In SBML Level&nbsp;1, the different rule types each have a different
   * name for the attribute holding the reference to the object
   * constituting the left-hand side of the rule.  (E.g., for
   * SBML Level&nbsp;1's SpeciesConcentrationRule the attribute is "species", for
   * CompartmentVolumeRule it is "compartment", etc.)  In SBML
   * Levels&nbsp;2 and&nbsp;3, the only two types of Rule objects with a
   * left-hand side object reference are AssignmentRule and RateRule, and
   * both of them use the same name for attribute: "variable".  In order to
   * make it easier for application developers to work with all Levels of
   * SBML, libSBML uses a uniform name for all such attributes, and it is
   * "variable", regardless of whether Level&nbsp;1 rules or
   * Level&nbsp;2&ndash;3 rules are being used.
   *
   * @return @c true if the "variable" attribute value of this Rule is
   * set, @c false otherwise.
   */
  bool isSetVariable () const;


  /**
   * Predicate returning @c true
   * if this Rule's "units" attribute is set.
   *
   * @return @c true if the units for this Rule is set, @c false
   * otherwise
   *
   * @note The attribute "units" exists on SBML Level&nbsp;1 ParameterRule
   * objects only.  It is not present in SBML Levels&nbsp;2 and&nbsp;3.
   */
  bool isSetUnits () const;


  /**
   * Sets the "math" subelement of this Rule to an expression in
   * text-string form.
   *
   * This is equivalent to setMath(const ASTNode* math).  The provision of
   * using text-string formulas is retained for easier SBML Level&nbsp;1
   * compatibility.  The formula is converted to an ASTNode internally.
   *
   * @param formula a mathematical formula in text-string form.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   *
   * @note The attribute "formula" is specific to SBML Level&nbsp;1; in
   * higher Levels of SBML, it has been replaced with a subelement named
   * "math".  However, libSBML provides a unified interface to the
   * underlying math expression and this method can be used for models
   * of all Levels of SBML.
   *
   * @see setMath(const ASTNode* math)
   */
  int setFormula (const std::string& formula);


  /**
   * Sets the "math" subelement of this Rule to a copy of the given
   * ASTNode.
   *
   * @param math the ASTNode structure of the mathematical formula.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   *
   * @note The subelement "math" is present in SBML Levels&nbsp;2
   * and&nbsp;3.  In SBML Level&nbsp;1, the equivalent construct is the
   * attribute named "formula".  LibSBML provides a unified interface to
   * the underlying math expression and this method can be used for models
   * of all Levels of SBML.
   *
   * @see setFormula(const std::string& formula)
   */
  int setMath (const ASTNode* math);


  /**
   * Sets the "variable" attribute value of this Rule object.
   *
   * In SBML Level&nbsp;1, the different rule types each have a different
   * name for the attribute holding the reference to the object
   * constituting the left-hand side of the rule.  (E.g., for
   * SBML Level&nbsp;1's SpeciesConcentrationRule the attribute is "species", for
   * CompartmentVolumeRule it is "compartment", etc.)  In SBML
   * Levels&nbsp;2 and&nbsp;3, the only two types of Rule objects with a
   * left-hand side object reference are AssignmentRule and RateRule, and
   * both of them use the same name for attribute: "variable".  In order to
   * make it easier for application developers to work with all Levels of
   * SBML, libSBML uses a uniform name for all such attributes, and it is
   * "variable", regardless of whether Level&nbsp;1 rules or
   * Level&nbsp;2&ndash;3 rules are being used.
   * 
   * @param sid the identifier of a Compartment, Species or Parameter
   * elsewhere in the enclosing Model object.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   */
  int setVariable (const std::string& sid);


  /**
   * Sets the units for this Rule.
   *
   * @param sname the identifier of the units
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   *
   * @note The attribute "units" exists on SBML Level&nbsp;1 ParameterRule
   * objects only.  It is not present in SBML Levels&nbsp;2 and&nbsp;3.
   */
  int setUnits (const std::string& sname);


  /**
   * Unsets the "units" for this Rule.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   *
   * @note The attribute "units" exists on SBML Level&nbsp;1 ParameterRule
   * objects only.  It is not present in SBML Levels&nbsp;2 and&nbsp;3.
   */
  int unsetUnits ();


  /**
   * Calculates and returns a UnitDefinition that expresses the units of
   * measurement assumed for the "math" expression of this Rule.
   *
   * The units are calculated based on the mathematical expression in the
   * Rule and the model quantities referenced by <code>&lt;ci&gt;</code>
   * elements used within that expression.  The getDerivedUnitDefinition()
   * method returns the calculated units.
   *
   * Note that the functionality that facilitates unit analysis depends 
   * on the model as a whole.  Thus, in cases where the object has not 
   * been added to a model or the model itself is incomplete,
   * unit analysis is not possible and this method will return @c NULL.
   *
   * @warning Note that it is possible the "math" expression in the Rule
   * contains pure numbers or parameters with undeclared units.  In those
   * cases, it is not possible to calculate the units of the overall
   * expression without making assumptions.  LibSBML does not make
   * assumptions about the units, and getDerivedUnitDefinition() only
   * returns the units as far as it is able to determine them.  For
   * example, in an expression <em>X + Y</em>, if <em>X</em> has
   * unambiguously-defined units and <em>Y</em> does not, it will return
   * the units of <em>X</em>.  <strong>It is important that callers also
   * invoke the method</strong>
   * @if java Rule::containsUndeclaredUnits()@else containsUndeclaredUnits()@endif@~
   * <strong>to determine whether this situation holds</strong>.  Callers may
   * wish to take suitable actions in those scenarios.
   * 
   * @return a UnitDefinition that expresses the units of the math 
   * expression of this Rule, or @c NULL if one cannot be constructed.
   *
   * @see containsUndeclaredUnits()
   */
  UnitDefinition * getDerivedUnitDefinition();


  /**
   * Calculates and returns a UnitDefinition that expresses the units of
   * measurement assumed for the "math" expression of this Rule.
   *
   * The units are calculated based on the mathematical expression in the
   * Rule and the model quantities referenced by <code>&lt;ci&gt;</code>
   * elements used within that expression.  The getDerivedUnitDefinition()
   * method returns the calculated units.
   *
   * Note that the functionality that facilitates unit analysis depends 
   * on the model as a whole.  Thus, in cases where the object has not 
   * been added to a model or the model itself is incomplete,
   * unit analysis is not possible and this method will return @c NULL.
   *
   * @warning Note that it is possible the "math" expression in the Rule
   * contains pure numbers or parameters with undeclared units.  In those
   * cases, it is not possible to calculate the units of the overall
   * expression without making assumptions.  LibSBML does not make
   * assumptions about the units, and getDerivedUnitDefinition() only
   * returns the units as far as it is able to determine them.  For
   * example, in an expression <em>X + Y</em>, if <em>X</em> has
   * unambiguously-defined units and <em>Y</em> does not, it will return
   * the units of <em>X</em>.  <strong>It is important that callers also
   * invoke the method</strong>
   * @if java Rule::containsUndeclaredUnits()@else containsUndeclaredUnits()@endif@~
   * <strong>to determine whether this situation holds</strong>.  Callers
   * may wish to take suitable actions in those scenarios.
   * 
   * @return a UnitDefinition that expresses the units of the math 
   * expression of this Rule, or @c NULL if one cannot be constructed.
   *
   * @see containsUndeclaredUnits()
   */
  const UnitDefinition * getDerivedUnitDefinition() const;


  /**
   * Predicate returning @c true if 
   * the math expression of this Rule contains
   * parameters/numbers with undeclared units.
   * 
   * @return @c true if the math expression of this Rule
   * includes parameters/numbers 
   * with undeclared units, @c false otherwise.
   *
   * @note A return value of @c true indicates that the UnitDefinition
   * returned by getDerivedUnitDefinition() may not accurately represent
   * the units of the expression.
   *
   * @see getDerivedUnitDefinition()
   */
  bool containsUndeclaredUnits();


  /**
   * Predicate returning @c true if 
   * the math expression of this Rule contains
   * parameters/numbers with undeclared units.
   * 
   * @return @c true if the math expression of this Rule
   * includes parameters/numbers 
   * with undeclared units, @c false otherwise.
   *
   * @note A return value of @c true indicates that the UnitDefinition
   * returned by getDerivedUnitDefinition() may not accurately represent
   * the units of the expression.
   *
   * @see getDerivedUnitDefinition()
   */
  bool containsUndeclaredUnits() const;


  /**
   * Get the type of rule this is.
   * 
   * @return the rule type (a value drawn from the enumeration <a
   * class="el" href="#RuleType_t">RuleType_t</a>) of this Rule.  The value
   * will be either @link RuleType_t#RULE_TYPE_RATE RULE_TYPE_RATE@endlink
   * or @link RuleType_t#RULE_TYPE_SCALAR RULE_TYPE_SCALAR@endlink.
   *
   * @note The attribute "type" on Rule objects is present only in SBML
   * Level&nbsp;1.  In SBML Level&nbsp;2 and later, the type has been
   * replaced by subclassing the Rule object.
   */
  RuleType_t getType () const;


  /**
   * Predicate returning @c true if this
   * Rule is an AlgebraicRule.
   * 
   * @return @c true if this Rule is an AlgebraicRule, @c false otherwise.
   */
  bool isAlgebraic () const;


  /**
   * Predicate returning @c true if this
   * Rule is an AssignmentRule.
   * 
   * @return @c true if this Rule is an AssignmentRule, @c false otherwise.
   */
  bool isAssignment () const;


  /**
   * Predicate returning @c true if this Rule is an CompartmentVolumeRule
   * or equivalent.
   *
   * This libSBML method works for SBML Level&nbsp;1 models (where there is
   * such a thing as an explicit CompartmentVolumeRule), as well as other Levels of
   * SBML.  For Levels above Level&nbsp;1, this method checks the symbol
   * being affected by the rule, and returns @c true if the symbol is the
   * identifier of a Compartment object defined in the model.
   *
   * @return @c true if this Rule is a CompartmentVolumeRule, @c false
   * otherwise.
   */
  bool isCompartmentVolume () const;


  /**
   * Predicate returning @c true if this Rule is an ParameterRule or
   * equivalent.
   *
   * This libSBML method works for SBML Level&nbsp;1 models (where there is
   * such a thing as an explicit ParameterRule), as well as other Levels of
   * SBML.  For Levels above Level&nbsp;1, this method checks the symbol
   * being affected by the rule, and returns @c true if the symbol is the
   * identifier of a Parameter object defined in the model.
   *
   * @return @c true if this Rule is a ParameterRule, @c false
   * otherwise.
   */
  bool isParameter () const;


  /**
   * Predicate returning @c true if this Rule
   * is a RateRule (SBML Levels&nbsp;2&ndash;3) or has a "type" attribute
   * value of @c "rate" (SBML Level&nbsp;1).
   *
   * @return @c true if this Rule is a RateRule (Level&nbsp;2) or has
   * type "rate" (Level&nbsp;1), @c false otherwise.
   */
  bool isRate () const;


  /**
   * Predicate returning @c true if this Rule
   * is an AssignmentRule (SBML Levels&nbsp;2&ndash;3) or has a "type"
   * attribute value of @c "scalar" (SBML Level&nbsp;1).
   *
   * @return @c true if this Rule is an AssignmentRule (Level&nbsp;2) or has
   * type "scalar" (Level&nbsp;1), @c false otherwise.
   */
  bool isScalar () const;


  /**
   * Predicate returning @c true if this Rule is a
   * SpeciesConcentrationRule or equivalent.
   *
   * This libSBML method works for SBML Level&nbsp;1 models (where there is
   * such a thing as an explicit SpeciesConcentrationRule), as well as
   * other Levels of SBML.  For Levels above Level&nbsp;1, this method
   * checks the symbol being affected by the rule, and returns @c true if
   * the symbol is the identifier of a Species object defined in the model.
   *
   * @return @c true if this Rule is a SpeciesConcentrationRule, @c false
   * otherwise.
   */
  bool isSpeciesConcentration () const;


  /**
   * Returns the libSBML type code for this %SBML object.
   * 
   * @if clike LibSBML attaches an identifying code to every kind of SBML
   * object.  These are known as <em>SBML type codes</em>.  The set of
   * possible type codes is defined in the enumeration #SBMLTypeCode_t.
   * The names of the type codes all begin with the characters @c
   * SBML_. @endif@if java LibSBML attaches an identifying code to every
   * kind of SBML object.  These are known as <em>SBML type codes</em>.  In
   * other languages, the set of type codes is stored in an enumeration; in
   * the Java language interface for libSBML, the type codes are defined as
   * static integer constants in the interface class {@link
   * libsbmlConstants}.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if python LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the Python language interface for libSBML, the type
   * codes are defined as static integer constants in the interface class
   * @link libsbml@endlink.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if csharp LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the C# language interface for libSBML, the type codes
   * are defined as static integer constants in the interface class @link
   * libsbmlcs.libsbml@endlink.  The names of the type codes all begin with
   * the characters @c SBML_. @endif@~
   *
   * @return the SBML type code for this object, or @link
   * SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Returns the SBML Level&nbsp;1 type code for this Rule object.
   *
   * This method only applies to SBML Level&nbsp;1 model objects.  If this
   * is not an SBML Level&nbsp;1 rule object, this method will return @link
   * SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink.
   * 
   * @return the SBML Level&nbsp;1 type code for this Rule (namely, @link
   * SBMLTypeCode_t#SBML_COMPARTMENT_VOLUME_RULE
   * SBML_COMPARTMENT_VOLUME_RULE@endlink, @link
   * SBMLTypeCode_t#SBML_PARAMETER_RULE SBML_PARAMETER_RULE@endlink, @link
   * SBMLTypeCode_t#SBML_SPECIES_CONCENTRATION_RULE
   * SBML_SPECIES_CONCENTRATION_RULE@endlink, or @link
   * SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink).
   */
  int getL1TypeCode () const;


  /**
   * Returns the XML element name of this object
   *
   * The returned value can be any of a number of different strings,
   * depending on the SBML Level in use and the kind of Rule object this
   * is.  The rules as of libSBML version @htmlinclude libsbml-version.html
   * are the following:
   * <ul>
   * <li> (Level&nbsp;2 and&nbsp;3) RateRule: returns @c "rateRule"
   * <li> (Level&nbsp;2 and&nbsp;3) AssignmentRule: returns @c "assignmentRule" 
   * <li> (Level&nbsp;2 and&nbsp;3) AlgebraicRule: returns @c "algebraicRule"
   * <li> (Level&nbsp;1 Version&nbsp;1) SpecieConcentrationRule: returns @c "specieConcentrationRule"
   * <li> (Level&nbsp;1 Version&nbsp;2) SpeciesConcentrationRule: returns @c "speciesConcentrationRule"
   * <li> (Level&nbsp;1) CompartmentVolumeRule: returns @c "compartmentVolumeRule"
   * <li> (Level&nbsp;1) ParameterRule: returns @c "parameterRule"
   * <li> Unknown rule type: returns @c "unknownRule"
   * </ul>
   *
   * Beware that the last (@c "unknownRule") is not a valid SBML element
   * name.
   * 
   * @return the name of this element
   */
  virtual const std::string& getElementName () const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Subclasses should override this method to write out their contained
   * SBML objects as XML elements.  Be sure to call your parents
   * implementation of this method as well.
   */
  virtual void writeElements (XMLOutputStream& stream) const;
  /** @endcond */


  /**
   * Sets the SBML Level&nbsp;1 type code for this Rule.
   *
   * @param type the SBML Level&nbsp;1 type code for this Rule. The
   * allowable values are @link SBMLTypeCode_t#SBML_COMPARTMENT_VOLUME_RULE
   * SBML_COMPARTMENT_VOLUME_RULE@endlink, @link
   * SBMLTypeCode_t#SBML_PARAMETER_RULE SBML_PARAMETER_RULE@endlink, and
   * @link SBMLTypeCode_t#SBML_SPECIES_CONCENTRATION_RULE
   * SBML_SPECIES_CONCENTRATION_RULE@endlink.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE@endlink
   * if given @p type value is not one of the above.
   */
  int setL1TypeCode (int type);


  /**
   * Predicate returning @c true if all the
   * required elements for this Rule object have been set.
   *
   * The only required element for a Rule object is the "math" subelement.
   *
   * @return a boolean value indicating whether all the required
   * elements for this object have been defined.
   */
  virtual bool hasRequiredElements() const ;


  /**
   * Predicate returning @c true if all the
   * required attributes for this Rule object have been set.
   *
   * The required attributes for a Rule object depend on the type of Rule
   * it is.  For AssignmentRule and RateRule objects (and SBML
   * Level&nbsp1's SpeciesConcentrationRule, CompartmentVolumeRule, and
   * ParameterRule objects), the required attribute is "variable"; for
   * AlgebraicRule objects, there is no required attribute.
   *
   * @return a boolean value indicating whether all the required
   * elements for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const ;


  /**
   * Renames all the SIdRef attributes on this element, including any found in MathML
   */
  virtual void renameSIdRefs(std::string oldid, std::string newid);


  /**
   * Renames all the UnitSIdRef attributes on this element
   */
  virtual void renameUnitSIdRefs(std::string oldid, std::string newid);



  /** @cond doxygen-libsbml-internal */

  /* function to set/get an identifier for unit checking */
  std::string getInternalId() const { return mInternalId; };
  void setInternalId(std::string id) { mInternalId = id; };
  /** @endcond */

  
  /*
   * Return the variable attribute of this object.
   *
   * @note This function is an alias of getVariable() function.
   *       (id attribute is not defined in Rule element.)
   *
   * @return the string of variable attribute of this object.
   *
   * @see getVariable()
   */
  virtual const std::string& getId() const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Replace all nodes with the name 'id' from the child 'math' object with the provided function. 
   *
   */
  virtual void replaceSIDWithFunction(const std::string& id, const ASTNode* function);
  /** @endcond */


  /** @cond doxygen-libsbml-internal */
  /**
   * If this rule assigns a value or a change to the 'id' element, replace the 'math' object with the function (existing/function). 
   */
  virtual void divideAssignmentsToSIdByFunction(const std::string& id, const ASTNode* function);
  /** @endcond */

  /** @cond doxygen-libsbml-internal */
  /**
   * If this assignment assigns a value to the 'id' element, replace the 'math' object with the function (existing*function). 
   */
  virtual void multiplyAssignmentsToSIdByFunction(const std::string& id, const ASTNode* function);
  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * Only subclasses may create Rules.
   */
  Rule (  int      type
        , unsigned int        level
        , unsigned int        version );

  Rule (  int      type
        , SBMLNamespaces *    sbmlns );


  /**
   * Subclasses should override this method to read (and store) XHTML,
   * MathML, etc. directly from the XMLInputStream.
   *
   * @return true if the subclass read from the stream, false otherwise.
   */
  virtual bool readOtherXML (XMLInputStream& stream);


  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);


  /**
   * Subclasses should override this method to read values from the given
   * XMLAttributes set into their specific fields.  Be sure to call your
   * parents implementation of this method as well.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);

  void readL1Attributes (const XMLAttributes& attributes);

  void readL2Attributes (const XMLAttributes& attributes);
  
  void readL3Attributes (const XMLAttributes& attributes);


  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;




  std::string mVariable;
  mutable std::string  mFormula;
  mutable ASTNode*     mMath;
  std::string          mUnits;

  int mType;
  int mL1Type;


  /* internal id used by unit checking */
  std::string mInternalId;

  friend class ListOfRules;

  /** @endcond */
};

class LIBSBML_EXTERN ListOfRules : public ListOf
{
public:

  /**
   * Creates a new ListOfRules object.
   *
   * The object is constructed such that it is valid for the given SBML
   * Level and Version combination.
   *
   * @param level the SBML Level
   * 
   * @param version the Version within the SBML Level
   */
  ListOfRules (unsigned int level, unsigned int version);
          

  /**
   * Creates a new ListOfRules object.
   *
   * The object is constructed such that it is valid for the SBML Level and
   * Version combination determined by the SBMLNamespaces object in @p
   * sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object that is used to determine the
   * characteristics of the ListOfRules object to be created.
   */
  ListOfRules (SBMLNamespaces* sbmlns);


  /**
   * Creates and returns a deep copy of this ListOfRules instance.
   *
   * @return a (deep) copy of this ListOfRules.
   */
  virtual ListOfRules* clone () const;


  /**
   * Returns the libSBML type code for this %SBML object.
   * 
   * @if clike LibSBML attaches an identifying code to every kind of SBML
   * object.  These are known as <em>SBML type codes</em>.  The set of
   * possible type codes is defined in the enumeration #SBMLTypeCode_t.
   * The names of the type codes all begin with the characters @c
   * SBML_. @endif@if java LibSBML attaches an identifying code to every
   * kind of SBML object.  These are known as <em>SBML type codes</em>.  In
   * other languages, the set of type codes is stored in an enumeration; in
   * the Java language interface for libSBML, the type codes are defined as
   * static integer constants in the interface class {@link
   * libsbmlConstants}.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if python LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the Python language interface for libSBML, the type
   * codes are defined as static integer constants in the interface class
   * @link libsbml@endlink.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if csharp LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the C# language interface for libSBML, the type codes
   * are defined as static integer constants in the interface class @link
   * libsbmlcs.libsbml@endlink.  The names of the type codes all begin with
   * the characters @c SBML_. @endif@~
   *
   * @return the SBML type code for this object, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const { return SBML_LIST_OF; };


  /**
   * Returns the libSBML type code for the objects contained in this ListOf
   * (i.e., Rule objects, if the list is non-empty).
   * 
   * @if clike LibSBML attaches an identifying code to every kind of SBML
   * object.  These are known as <em>SBML type codes</em>.  The set of
   * possible type codes is defined in the enumeration #SBMLTypeCode_t.
   * The names of the type codes all begin with the characters @c
   * SBML_. @endif@if java LibSBML attaches an identifying code to every
   * kind of SBML object.  These are known as <em>SBML type codes</em>.  In
   * other languages, the set of type codes is stored in an enumeration; in
   * the Java language interface for libSBML, the type codes are defined as
   * static integer constants in the interface class {@link
   * libsbmlConstants}.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if python LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the Python language interface for libSBML, the type
   * codes are defined as static integer constants in the interface class
   * @link libsbml@endlink.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if csharp LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the C# language interface for libSBML, the type codes
   * are defined as static integer constants in the interface class @link
   * libsbmlcs.libsbml@endlink.  The names of the type codes all begin with
   * the characters @c SBML_. @endif@~
   * 
   * @return the SBML type code for the objects contained in this ListOf
   * instance, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getItemTypeCode () const;


  /**
   * Returns the XML element name of this object.
   *
   * For ListOfRules, the XML element name is @c "listOfRules".
   * 
   * @return the name of this element, i.e., @c "listOfRules".
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a Rule from the ListOfRules.
   *
   * @param n the index number of the Rule to get.
   * 
   * @return the nth Rule in this ListOfRules.
   *
   * @see size()
   */
  virtual Rule * get(unsigned int n); 


  /**
   * Get a Rule from the ListOfRules.
   *
   * @param n the index number of the Rule to get.
   * 
   * @return the nth Rule in this ListOfRules.
   *
   * @see size()
   */
  virtual const Rule * get(unsigned int n) const; 


  /**
   * Get a Rule from the ListOfRules
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Rule to get.
   * 
   * @return Rule in this ListOfRules
   * with the given id or @c NULL if no such
   * Rule exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual Rule* get (const std::string& sid);


  /**
   * Get a Rule from the ListOfRules
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Rule to get.
   * 
   * @return Rule in this ListOfRules
   * with the given id or @c NULL if no such
   * Rule exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const Rule* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfRules items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual Rule* remove (unsigned int n);


  /**
   * Returns the first child element found that has the given id in the model-wide SId namespace, or NULL if no such object is found.  Note that AssignmentRules and RateRules do not actually have IDs, but the libsbml interface pretends that they do:  no assignment rule or rate rule is returned by this function.
   *
   * @param id string representing the id of objects to find
   *
   * @return pointer to the first element found with the given id.
   */
  virtual SBase* getElementBySId(std::string id);
  
  
  /**
   * Removes item in this ListOfRules items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual Rule* remove (const std::string& sid);


  /** @cond doxygen-libsbml-internal */

  /**
   * Get the ordinal position of this element in the containing object
   * (which in this case is the Model object).
   *
   * The ordering of elements in the XML form of %SBML is generally fixed
   * for most components in %SBML.
   *
   * @return the ordinal position of the element with respect to its
   * siblings, or @c -1 (default) to indicate the position is not significant.
   */
  virtual int getElementPosition () const;

  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or @c NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);

  virtual bool isValidTypeForList(SBase * item);
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

LIBSBML_EXTERN
Rule_t *
Rule_createAlgebraic (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Rule_t *
Rule_createAlgebraicWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
Rule_t *
Rule_createAssignment (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Rule_t *
Rule_createAssignmentWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
Rule_t *
Rule_createRate (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Rule_t *
Rule_createRateWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
Rule_free (Rule_t *r);


LIBSBML_EXTERN
Rule_t *
Rule_clone (const Rule_t *r);


LIBSBML_EXTERN
const XMLNamespaces_t *
Rule_getNamespaces(Rule_t *r);


LIBSBML_EXTERN
const char *
Rule_getFormula (const Rule_t *r);


LIBSBML_EXTERN
const ASTNode_t *
Rule_getMath (const Rule_t *r);


LIBSBML_EXTERN
RuleType_t
Rule_getType (const Rule_t *r);


LIBSBML_EXTERN
const char *
Rule_getVariable (const Rule_t *r);


LIBSBML_EXTERN
const char *
Rule_getUnits (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isSetFormula (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isSetMath (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isSetVariable (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isSetUnits (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_setFormula (Rule_t *r, const char *formula);


LIBSBML_EXTERN
int
Rule_setMath (Rule_t *r, const ASTNode_t *math);


LIBSBML_EXTERN
int
Rule_setVariable (Rule_t *r, const char *sid);


LIBSBML_EXTERN
int
Rule_setUnits (Rule_t *r, const char *sname);


LIBSBML_EXTERN
int
Rule_unsetUnits (Rule_t *r);


LIBSBML_EXTERN
int
Rule_isAlgebraic (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isAssignment (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isCompartmentVolume (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isParameter (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isRate (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isScalar (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_isSpeciesConcentration (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_getTypeCode (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_getL1TypeCode (const Rule_t *r);


LIBSBML_EXTERN
int
Rule_setL1TypeCode (Rule_t *r, int L1Type);


LIBSBML_EXTERN
UnitDefinition_t * 
Rule_getDerivedUnitDefinition(Rule_t *ia);


LIBSBML_EXTERN
int 
Rule_containsUndeclaredUnits(Rule_t *ia);

LIBSBML_EXTERN
Rule_t *
ListOfRules_getById (ListOf_t *lo, const char *sid);


LIBSBML_EXTERN
Rule_t *
ListOfRules_removeById (ListOf_t *lo, const char *sid);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG  */

#ifndef LIBSBML_USE_STRICT_INCLUDES
#include <sbml/AlgebraicRule.h>
#include <sbml/AssignmentRule.h>
#include <sbml/RateRule.h>
#endif

#endif  /* Rule_h */

