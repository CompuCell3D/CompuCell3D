/**
 * @file    Parameter.h
 * @brief   Definitions of Parameter and ListOfParameters.
 * @author  Ben Bornstein
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 * 
 * @class Parameter.
 * @brief LibSBML implementation of SBML's %Parameter construct.
 *
 * A Parameter is used in SBML to define a symbol associated with a value;
 * this symbol can then be used in mathematical formulas in a model.  By
 * default, parameters have constant value for the duration of a
 * simulation, and for this reason are called @em parameters instead of @em
 * variables in SBML, although it is crucial to understand that <em>SBML
 * parameters represent both concepts</em>.  Whether a given SBML
 * parameter is intended to be constant or variable is indicated by the
 * value of its "constant" attribute.
 * 
 * SBML's Parameter has a required attribute, "id", that gives the
 * parameter a unique identifier by which other parts of an SBML model
 * definition can refer to it.  A parameter can also have an optional
 * "name" attribute of type @c string.  Identifiers and names must be used
 * according to the guidelines described in the SBML specifications.
 * 
 * The optional attribute "value" determines the value (of type @c double)
 * assigned to the parameter.  A missing value for "value" implies that
 * the value either is unknown, or to be obtained from an external source,
 * or determined by an initial assignment.  The unit of measurement
 * associated with the value of the parameter can be specified using the
 * optional attribute "units".  Here we only mention briefly some notable
 * points about the possible unit choices, but readers are urged to consult
 * the SBML specification documents for more information:
 * <ul>
 *
 * <li> In SBML Level&nbsp;3, there are no constraints on the units that
 * can be assigned to parameters in a model; there are also no units to
 * inherit from the enclosing Model object (unlike the case for, e.g.,
 * Species and Compartment).
 *
 * <li> In SBML Level&nbsp;2, the value assigned to the parameter's "units"
 * attribute must be chosen from one of the following possibilities: one of
 * the base unit identifiers defined in SBML; one of the built-in unit
 * identifiers @c "substance", @c "time", @c "volume", @c "area" or @c
 * "length"; or the identifier of a new unit defined in the list of unit
 * definitions in the enclosing Model structure.  There are no constraints
 * on the units that can be chosen from these sets.  There are no default
 * units for parameters.
 * </ul>
 *
 * The Parameter structure has another boolean attribute named "constant"
 * that is used to indicate whether the parameter's value can vary during a
 * simulation.  (In SBML Level&nbsp;3, the attribute is mandatory and must
 * be given a value; in SBML Levels below Level&nbsp;3, the attribute is
 * optional.)  A value of @c true indicates the parameter's value cannot be
 * changed by any construct except InitialAssignment.  Conversely, if the
 * value of "constant" is @c false, other constructs in SBML, such as rules
 * and events, can change the value of the parameter.
 *
 * SBML Level&nbsp;3 uses a separate object class, LocalParameter, for
 * parameters that are local to a Reaction's KineticLaw.  In Levels prior
 * to SBML Level&nbsp;3, the Parameter class is used both for definitions
 * of global parameters, as well as reaction-local parameters stored in a
 * list within KineticLaw objects.  Parameter objects that are local to a
 * reaction (that is, those defined within the KineticLaw structure of a
 * Reaction) cannot be changed by rules and therefore are <em>implicitly
 * always constant</em>; consequently, in SBML Level&nbsp;2, parameter
 * definitions within Reaction structures should @em not have their
 * "constant" attribute set to @c false.
 * 
 * What if a global parameter has its "constant" attribute set to @c false,
 * but the model does not contain any rules, events or other constructs
 * that ever change its value over time?  Although the model may be
 * suspect, this situation is not strictly an error.  A value of @c false
 * for "constant" only indicates that a parameter @em can change value, not
 * that it @em must.
 *
 * As with all other major SBML components, Parameter is derived from
 * SBase, and the methods defined on SBase are available on Parameter.
 * 
 * @note The use of the term @em parameter in SBML sometimes leads to
 * confusion among readers who have a particular notion of what something
 * called "parameter" should be.  It has been the source of heated debate,
 * but despite this, no one has yet found an adequate replacement term that
 * does not have different connotations to different people and hence leads
 * to confusion among @em some subset of users.  Perhaps it would have been
 * better to have two constructs, one called @em constants and the other
 * called @em variables.  The current approach in SBML is simply more
 * parsimonious, using a single Parameter construct with the boolean flag
 * "constant" indicating which flavor it is.  In any case, readers are
 * implored to look past their particular definition of a @em parameter and
 * simply view SBML's Parameter as a single mechanism for defining both
 * constants and (additional) variables in a model.  (We write @em
 * additional because the species in a model are usually considered to be
 * the central variables.)  After all, software tools are not required to
 * expose to users the actual names of particular SBML constructs, and
 * thus tools can present to their users whatever terms their designers
 * feel best matches their target audience.
 *
 * @see ListOfParameters
 *
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 
/**
 * @class ListOfParameters
 * @brief LibSBML implementation of SBML's %ListOfParameters construct.
 * 
 * The various ListOf___ classes in SBML are merely containers used for
 * organizing the main components of an SBML model.  All are derived from
 * the abstract class SBase, and inherit the various attributes and
 * subelements of SBase, such as "metaid" as and "annotation".  The
 * ListOf___ classes do not add any attributes of their own.
 *
 * The relationship between the lists and the rest of an SBML model is
 * illustrated by the following (for SBML Level&nbsp;2 Version&nbsp;4):
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

#ifndef Parameter_h
#define Parameter_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>


#ifdef __cplusplus


#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLVisitor;


class LIBSBML_EXTERN Parameter : public SBase
{
public:

  /**
   * Creates a new Parameter using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this Parameter
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * Parameter
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of a Parameter object to an SBMLDocument
   * (e.g., using Model::addParameter(@if java Parameter p@endif)), the SBML Level, SBML Version
   * and XML namespace of the document @em override the values used
   * when creating the Parameter object via this constructor.  This is
   * necessary to ensure that an SBML document is a consistent structure.
   * Nevertheless, the ability to supply the values at the time of creation
   * of a Parameter is an important aid to producing valid SBML.  Knowledge
   * of the intented SBML Level and Version determine whether it is valid
   * to assign a particular value to an attribute, or whether it is valid
   * to add an object to an existing SBMLDocument.
   */
  Parameter (unsigned int level, unsigned int version);


  /**
   * Creates a new Parameter using the given SBMLNamespaces object
   * @p sbmlns.
   *
   * The SBMLNamespaces object encapsulates SBML Level/Version/namespaces
   * information.  It is used to communicate the SBML Level, Version, and
   * (in Level&nbsp;3) packages used in addition to SBML Level&nbsp;3 Core.
   * A common approach to using this class constructor is to create an
   * SBMLNamespaces object somewhere in a program, once, then pass it to
   * object constructors such as this one when needed.
   *
   * It is worth emphasizing that although this constructor does not take
   * an identifier argument, in SBML Level&nbsp;2 and beyond, the "id"
   * (identifier) attribute of a Parameter is required to have a value.
   * Thus, callers are cautioned to assign a value after calling this
   * constructor if no identifier is provided as an argument.  Setting the
   * identifier can be accomplished using the method
   * @if java setId(String id)@else setId()@endif.
   *
   * @param sbmlns an SBMLNamespaces object.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   *
   * @note Upon the addition of a Parameter object to an SBMLDocument
   * (e.g., using Model::addParameter(@if java Parameter p@endif)), the SBML XML namespace of the
   * document @em overrides the value used when creating the Parameter
   * object via this constructor.  This is necessary to ensure that an SBML
   * document is a consistent structure.  Nevertheless, the ability to
   * supply the values at the time of creation of a Parameter is an
   * important aid to producing valid SBML.  Knowledge of the intented SBML
   * Level and Version determine whether it is valid to assign a particular
   * value to an attribute, or whether it is valid to add an object to an
   * existing SBMLDocument.
   */
  Parameter (SBMLNamespaces* sbmlns);


  /**
   * Destroys this Parameter.
   */
  virtual ~Parameter ();


  /**
   * Copy constructor; creates a copy of a Parameter.
   * 
   * @param orig the Parameter instance to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  Parameter(const Parameter& orig);


  /**
   * Assignment operator for Parameter.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  Parameter& operator=(const Parameter& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of Parameter.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, indicating
   * whether the Visitor would like to visit the next Parameter object in
   * the list of parameters within which @em the present object is
   * embedded.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this Parameter.
   * 
   * @return a (deep) copy of this Parameter.
   */
  virtual Parameter* clone () const;


  /**
   * Initializes the fields of this Parameter object to "typical" defaults
   * values.
   *
   * The SBML Parameter component has slightly different aspects and
   * default attribute values in different SBML Levels and Versions.  Many
   * SBML object classes defined by libSBML have an initDefaults() method
   * to set the values to certain common defaults, based mostly on what
   * they are in SBML Level&nbsp;2.  In the case of Parameter, this method
   * only sets the value of the "constant" attribute to @c true.
   *
   * @see getConstant()
   * @see isSetConstant()
   * @see setConstant(@if java boolean flag@endif)
   */
  void initDefaults ();

  
  /**
   * Returns the value of the "id" attribute of this Parameter.
   * 
   * @return the id of this Parameter.
   */
  virtual const std::string& getId () const;


  /**
   * Returns the value of the "name" attribute of this Parameter.
   * 
   * @return the name of this Parameter.
   */
  virtual const std::string& getName () const;


  /**
   * Gets the numerical value of this Parameter.
   * 
   * @return the value of the "value" attribute of this Parameter, as a
   * number of type @c double.
   *
   * @note <b>It is crucial</b> that callers not blindly call
   * Parameter::getValue() without first using Parameter::isSetValue() to
   * determine whether a value has ever been set.  Otherwise, the value
   * return by Parameter::getValue() may not actually represent a value
   * assigned to the parameter.  The reason is simply that the data type
   * @c double in a program always has @em some value.  A separate test is
   * needed to determine whether the value is a true model value, or
   * uninitialized data in a computer's memory location.
   * 
   * @see isSetValue()
   * @see setValue(double value)
   * @see getUnits()
   */
  double getValue () const;


  /**
   * Gets the units defined for this Parameter.
   *
   * The value of an SBML parameter's "units" attribute establishes the
   * unit of measurement associated with the parameter's value.
   *
   * @return the value of the "units" attribute of this Parameter, as a
   * string.  An empty string indicates that no units have been assigned.
   *
   * @note @htmlinclude unassigned-units-are-not-a-default.html
   * 
   * @see isSetUnits()
   * @see setUnits(@if java String units@endif)
   * @see getValue()
   */
  const std::string& getUnits () const;


  /**
   * Gets the value of the "constant" attribute of this Parameter instance.
   * 
   * @return @c true if this Parameter is declared as being constant,
   * @c false otherwise.
   *
   * @note Readers who view the documentation for LocalParameter may be
   * confused about the presence of this method.  LibSBML derives
   * LocalParameter from Parameter; however, this does not precisely match
   * the object hierarchy defined by SBML Level&nbsp;3, where
   * LocalParameter is derived directly from SBase and not Parameter.  We
   * believe this arrangement makes it easier for libSBML users to program
   * applications that work with both SBML Level&nbsp;2 and SBML
   * Level&nbsp;3, but programmers should also keep in mind this difference
   * exists.  A side-effect of libSBML's scheme is that certain methods on
   * LocalParameter that are inherited from Parameter do not actually have
   * relevance to LocalParameter objects.  An example of this is the
   * methods pertaining to Parameter's attribute "constant" (i.e.,
   * isSetConstant(), setConstant(), and getConstant()).
   *
   * @see isSetConstant()
   * @see setConstant(@if java boolean flag@endif)
   */
  bool getConstant () const;


  /**
   * Predicate returning @c true if this
   * Parameter's "id" attribute is set.
   *
   * @return @c true if the "id" attribute of this Parameter is
   * set, @c false otherwise.
   */
  virtual bool isSetId () const;


  /**
   * Predicate returning @c true if this
   * Parameter's "name" attribute is set.
   *
   * @return @c true if the "name" attribute of this Parameter is
   * set, @c false otherwise.
   */
  virtual bool isSetName () const;


  /**
   * Predicate returning @c true if the
   * "value" attribute of this Parameter is set.
   *
   * In SBML definitions after SBML Level&nbsp;1 Version&nbsp;1,
   * parameter values are optional and have no defaults.  If a model read
   * from a file does not contain a setting for the "value" attribute of a
   * parameter, its value is considered unset; it does not default to any
   * particular value.  Similarly, when a Parameter object is created in
   * libSBML, it has no value until given a value.  The
   * Parameter::isSetValue() method allows calling applications to
   * determine whether a given parameter's value has ever been set.
   *
   * In SBML Level&nbsp;1 Version&nbsp;1, parameters are required to have
   * values and therefore, the value of a Parameter <b>should always be
   * set</b>.  In Level&nbsp;1 Version&nbsp;2 and beyond, the value is
   * optional and as such, the "value" attribute may or may not be set.
   *
   * @return @c true if the value of this Parameter is set,
   * @c false otherwise.
   *
   * @see getValue()
   * @see setValue(double value)
   */
  bool isSetValue () const;


  /**
   * Predicate returning @c true if the
   * "units" attribute of this Parameter is set.
   *
   * @return @c true if the "units" attribute of this Parameter is
   * set, @c false otherwise.
   *
   * @note @htmlinclude unassigned-units-are-not-a-default.html
   */
  bool isSetUnits () const;


  /**
   * Predicate returning @c true if the
   * "constant" attribute of this Parameter is set.
   *
   * @return @c true if the "constant" attribute of this Parameter is
   * set, @c false otherwise.
   *
   * @note Readers who view the documentation for LocalParameter may be
   * confused about the presence of this method.  LibSBML derives
   * LocalParameter from Parameter; however, this does not precisely match
   * the object hierarchy defined by SBML Level&nbsp;3, where
   * LocalParameter is derived directly from SBase and not Parameter.  We
   * believe this arrangement makes it easier for libSBML users to program
   * applications that work with both SBML Level&nbsp;2 and SBML
   * Level&nbsp;3, but programmers should also keep in mind this difference
   * exists.  A side-effect of libSBML's scheme is that certain methods on
   * LocalParameter that are inherited from Parameter do not actually have
   * relevance to LocalParameter objects.  An example of this is the
   * methods pertaining to Parameter's attribute "constant" (i.e.,
   * isSetConstant(), setConstant(), and getConstant()).
   *
   * @see getConstant()
   * @see setConstant(@if java boolean flag@endif)
   */
  bool isSetConstant () const;


  /**
   * Sets the value of the "id" attribute of this Parameter.
   *
   * The string @p sid is copied.  Note that SBML has strict requirements
   * for the syntax of identifiers.  @htmlinclude id-syntax.html
   *
   * @param sid the string to use as the identifier of this Parameter
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setId (const std::string& sid);


  /**
   * Sets the value of the "name" attribute of this Parameter.
   *
   * The string in @p name is copied.
   *
   * @param name the new name for the Parameter
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  int setName (const std::string& name);


  /**
   * Sets the "value" attribute of this Parameter to the given @c double
   * value and marks the attribute as set.
   *
   * @param value a @c double, the value to assign
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int setValue (double value);


  /**
   * Sets the "units" attribute of this Parameter to a copy of the given
   * units identifier @p units.
   *
   * @param units a string, the identifier of the units to assign to this
   * Parameter instance
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  int setUnits (const std::string& units);


  /**
   * Sets the "constant" attribute of this Parameter to the given boolean
   * @p flag.
   *
   * @param flag a boolean, the value for the "constant" attribute of this
   * Parameter instance
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   *
   * @note Readers who view the documentation for LocalParameter may be
   * confused about the presence of this method.  LibSBML derives
   * LocalParameter from Parameter; however, this does not precisely match
   * the object hierarchy defined by SBML Level&nbsp;3, where
   * LocalParameter is derived directly from SBase and not Parameter.  We
   * believe this arrangement makes it easier for libSBML users to program
   * applications that work with both SBML Level&nbsp;2 and SBML
   * Level&nbsp;3, but programmers should also keep in mind this difference
   * exists.  A side-effect of libSBML's scheme is that certain methods on
   * LocalParameter that are inherited from Parameter do not actually have
   * relevance to LocalParameter objects.  An example of this is the
   * methods pertaining to Parameter's attribute "constant" (i.e.,
   * isSetConstant(), setConstant(), and getConstant()).
   *
   * @see getConstant()
   * @see isSetConstant()
   */
  int setConstant (bool flag);


  /**
   * Unsets the value of the "name" attribute of this Parameter.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int unsetName ();


  /**
   * Unsets the "value" attribute of this Parameter instance.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   *
   * In SBML Level&nbsp;1 Version&nbsp;1, parameters are required to have
   * values and therefore, the value of a Parameter <b>should always be
   * set</b>.  In SBML Level&nbsp;1 Version&nbsp;2 and beyond, the value
   * is optional and as such, the "value" attribute may or may not be set.
   */
  int unsetValue ();


  /**
   * Unsets the "units" attribute of this Parameter instance.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetUnits ();


  /**
   * Constructs and returns a UnitDefinition that corresponds to the units
   * of this Parameter's value.
   *
   * Parameters in SBML have an attribute ("units") for declaring the units
   * of measurement intended for the parameter's value.  <b>No defaults are
   * defined</b> by SBML in the absence of a definition for "units".  This
   * method returns a UnitDefinition object based on the units declared for
   * this Parameter using its "units" attribute, or it returns @c NULL if
   * no units have been declared.
   *
   * Note that unit declarations for Parameter objects are specified in
   * terms of the @em identifier of a unit (e.g., using setUnits()), but
   * @em this method returns a UnitDefinition object, not a unit
   * identifier.  It does this by constructing an appropriate
   * UnitDefinition.For SBML Level&nbsp;2 models, it will do this even when
   * the value of the "units" attribute is one of the special SBML
   * Level&nbsp;2 unit identifiers @c "substance", @c "volume", @c "area",
   * @c "length" or @c "time".  Callers may find this useful in conjunction
   * with the helper methods provided by the UnitDefinition class for
   * comparing different UnitDefinition objects.
   *
   * @return a UnitDefinition that expresses the units of this 
   * Parameter, or @c NULL if one cannot be constructed.
   *
   * @note The libSBML system for unit analysis depends on the model as a
   * whole.  In cases where the Parameter object has not yet been added to
   * a model, or the model itself is incomplete, unit analysis is not
   * possible, and consequently this method will return @c NULL.
   *
   * @see isSetUnits()
   */
  UnitDefinition * getDerivedUnitDefinition();


  /**
   * Constructs and returns a UnitDefinition that corresponds to the units
   * of this Parameter's value.
   *
   * Parameters in SBML have an attribute ("units") for declaring the units
   * of measurement intended for the parameter's value.  <b>No defaults are
   * defined</b> by SBML in the absence of a definition for "units".  This
   * method returns a UnitDefinition object based on the units declared for
   * this Parameter using its "units" attribute, or it returns @c NULL if
   * no units have been declared.
   *
   * Note that unit declarations for Parameter objects are specified in
   * terms of the @em identifier of a unit (e.g., using setUnits()), but
   * @em this method returns a UnitDefinition object, not a unit
   * identifier.  It does this by constructing an appropriate
   * UnitDefinition.  For SBML Level&nbsp;2 models, it will do this even
   * when the value of the "units" attribute is one of the predefined SBML
   * units @c "substance", @c "volume", @c "area", @c "length" or @c
   * "time".  Callers may find this useful in conjunction with the helper
   * methods provided by the UnitDefinition class for comparing different
   * UnitDefinition objects.
   *
   * @return a UnitDefinition that expresses the units of this 
   * Parameter, or @c NULL if one cannot be constructed.
   *
   * @note The libSBML system for unit analysis depends on the model as a
   * whole.  In cases where the Parameter object has not yet been added to
   * a model, or the model itself is incomplete, unit analysis is not
   * possible, and consequently this method will return @c NULL.
   * 
   * @see isSetUnits()
   */
  const UnitDefinition * getDerivedUnitDefinition() const;


  /**
   * Returns the libSBML type code for this SBML object.
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
   * @return the SBML type code for this object, or
   * @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Returns the XML element name of this object, which for Parameter, is
   * always @c "parameter".
   * 
   * @return the name of this element, i.e., @c "parameter".
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
   * Predicate returning @c true if
   * all the required attributes for this Parameter object
   * have been set.
   *
   * @note The required attributes for a Parameter object are:
   * @li "id" (or "name" in SBML Level&nbsp;1)
   * @li "value" (required in Level&nbsp;1, optional otherwise)
   *
   * @return a boolean value indicating whether all the required
   * attributes for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const ;


  /**
   * Renames all the UnitSIdRef attributes on this element
   */
  virtual void renameUnitSIdRefs(std::string oldid, std::string newid);


protected:
  /** @cond doxygen-libsbml-internal */


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

  bool isExplicitlySetConstant() const 
                            { return mExplicitlySetConstant; } ;

  std::string  mId;
  std::string  mName;
  double       mValue;
  std::string  mUnits;
  bool         mConstant;

  bool mIsSetValue;
  bool mIsSetConstant;

  bool  mExplicitlySetConstant;

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


class LIBSBML_EXTERN ListOfParameters : public ListOf
{
public:

  /**
   * Creates a new ListOfParameters object.
   *
   * The object is constructed such that it is valid for the given SBML
   * Level and Version combination.
   *
   * @param level the SBML Level
   * 
   * @param version the Version within the SBML Level
   */
  ListOfParameters (unsigned int level, unsigned int version);
          

  /**
   * Creates a new ListOfParameters object.
   * 
   * The object is constructed such that it is valid for the SBML Level and
   * Version combination determined by the SBMLNamespaces object in @p
   * sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object that is used to determine the
   * characteristics of the ListOfParameters object to be created.
   */
  ListOfParameters (SBMLNamespaces* sbmlns);


  /**
   * Creates and returns a deep copy of this ListOfParameters instance.
   *
   * @return a (deep) copy of this ListOfParameters.
   */
  virtual ListOfParameters* clone () const;


  /**
   * Returns the libSBML type code for this SBML object.
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
  virtual int getTypeCode () const { return SBML_LIST_OF; };


  /**
   * Returns the libSBML type code for the objects contained in this ListOf
   * (i.e., Parameter objects, if the list is non-empty).
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
   * instance, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink
   * (default).
   *
   * @see getElementName()
   */
  virtual int getItemTypeCode () const;


  /**
   * Returns the XML element name of this object.
   *
   * For ListOfParameters, the XML element name is @c "listOfParameters".
   * 
   * @return the name of this element, i.e., @c "listOfParameters".
   */
  virtual const std::string& getElementName () const;


  /**
   * Returns the Parameter object located at position @p n within this
   * ListOfParameters instance.
   *
   * @param n the index number of the Parameter to get.
   * 
   * @return the nth Parameter in this ListOfParameters.  If the index @p n
   * is out of bounds for the length of the list, then @c NULL is returned.
   *
   * @see size()
   * @see get(const std::string& sid)
   */
  virtual Parameter * get(unsigned int n); 


  /**
   * Returns the Parameter object located at position @p n within this
   * ListOfParameters instance.
   *
   * @param n the index number of the Parameter to get.
   * 
   * @return the nth Parameter in this ListOfParameters.  If the index @p n
   * is out of bounds for the length of the list, then @c NULL is returned.
   *
   * @see size()
   * @see get(const std::string& sid)
   */
  virtual const Parameter * get(unsigned int n) const; 


  /**
   * Returns the first Parameter object matching the given identifier.
   *
   * @param sid a string, the identifier of the Parameter to get.
   * 
   * @return the Parameter object found.  The caller owns the returned
   * object and is responsible for deleting it.  If none of the items have
   * an identifier matching @p sid, then @c NULL is returned.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual Parameter* get (const std::string& sid);


  /**
   * Returns the first Parameter object matching the given identifier.
   *
   * @param sid a string representing the identifier of the Parameter to
   * get.
   * 
   * @return the Parameter object found.  The caller owns the returned
   * object and is responsible for deleting it.  If none of the items have
   * an identifier matching @p sid, then @c NULL is returned.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const Parameter* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfParameters, and returns a pointer
   * to it.
   *
   * @param n the index of the item to remove
   *
   * @return the item removed.  The caller owns the returned object and is
   * responsible for deleting it.  If the index number @p n is out of
   * bounds for the length of the list, then @c NULL is returned.
   *
   * @see size()
   */
  virtual Parameter* remove (unsigned int n);


  /**
   * Removes the first Parameter object in this ListOfParameters
   * matching the given identifier, and returns a pointer to it.
   *
   * @param sid the identifier of the item to remove.
   *
   * @return the item removed.  The caller owns the returned object and is
   * responsible for deleting it.  If none of the items have an identifier
   * matching @p sid, then @c NULL is returned.
   */
  virtual Parameter* remove (const std::string& sid);


  /** @cond doxygen-libsbml-internal */

  /**
   * Gets the ordinal position of this element in the containing object
   * (which in this case is the Model object).
   *
   * The ordering of elements in the XML form of SBML is generally fixed
   * for most components in SBML.  So, for example, the ListOfParameters
   * in a model is (in SBML Level&nbsp;2 Version&nbsp;4) the seventh
   * ListOf___.  (However, it differs for different Levels and Versions of
   * SBML.)
   *
   * @return the ordinal position of the element with respect to its
   * siblings, or @c -1 (default) to indicate the position is not significant.
   */
  virtual int getElementPosition () const;

  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * Create a ListOfParameters object corresponding to the next token in
   * the XML input stream.
   * 
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream, or @c NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);

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

typedef Parameter Parameter_t;

LIBSBML_EXTERN
Parameter_t *
Parameter_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Parameter_t *
Parameter_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
Parameter_free (Parameter_t *p);


LIBSBML_EXTERN
Parameter_t *
Parameter_clone (const Parameter_t *p);


LIBSBML_EXTERN
void
Parameter_initDefaults (Parameter_t *p);


LIBSBML_EXTERN
const XMLNamespaces_t *
Parameter_getNamespaces(Parameter_t *c);


LIBSBML_EXTERN
const char *
Parameter_getId (const Parameter_t *p);


LIBSBML_EXTERN
const char *
Parameter_getName (const Parameter_t *p);


LIBSBML_EXTERN
double
Parameter_getValue (const Parameter_t *p);


LIBSBML_EXTERN
const char *
Parameter_getUnits (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_getConstant (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_isSetId (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_isSetName (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_isSetValue (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_isSetUnits (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_isSetConstant (const Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_setId (Parameter_t *p, const char *sid);


LIBSBML_EXTERN
int
Parameter_setName (Parameter_t *p, const char *name);


LIBSBML_EXTERN
int
Parameter_setValue (Parameter_t *p, double value);


LIBSBML_EXTERN
int
Parameter_setUnits (Parameter_t *p, const char *units);


LIBSBML_EXTERN
int
Parameter_setConstant (Parameter_t *p, int value);


LIBSBML_EXTERN
int
Parameter_unsetName (Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_unsetValue (Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_unsetUnits (Parameter_t *p);


LIBSBML_EXTERN
UnitDefinition_t * 
Parameter_getDerivedUnitDefinition(Parameter_t *p);


LIBSBML_EXTERN
int
Parameter_hasRequiredAttributes (Parameter_t *p);


LIBSBML_EXTERN
Parameter_t *
ListOfParameters_getById (ListOf_t *lo, const char *sid);


LIBSBML_EXTERN
Parameter_t *
ListOfParameters_removeById (ListOf_t *lo, const char *sid);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* Parameter_h */

