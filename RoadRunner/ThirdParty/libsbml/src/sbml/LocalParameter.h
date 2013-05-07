/**
 * @file    LocalParameter.h
 * @brief   Definitions of LocalParameter and ListOfLocalParameters.
 * @author  Sarah Keating
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
 * @class LocalParameter.
 * @brief LibSBML implementation of SBML Level&nbsp;3's %LocalParameter construct.
 *
 * LocalParameter has been introduced in SBML Level&nbsp;3 to serve as the
 * object class for parameter definitions that are intended to be local to
 * a Reaction.  Objects of class LocalParameter never appear at the Model
 * level; they are always contained within ListOfLocalParameters lists
 * which are in turn contained within KineticLaw objects.
 *
 * Like its global Parameter counterpart, the LocalParameter object class
 * is used to define a symbol associated with a value; this symbol can then
 * be used in a model's mathematical formulas (and specifically, for
 * LocalParameter, reaction rate formulas).  Unlike Parameter, the
 * LocalParameter class does not have a "constant" attribute: local
 * parameters within reactions are @em always constant.
 * 
 * LocalParameter has one required attribute, "id", to give the
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
 * inherit from the enclosing Model object.
 *
 * <li> In SBML Level&nbsp;2, the value assigned to the parameter's "units"
 * attribute must be chosen from one of the following possibilities: one of
 * the base unit identifiers defined in SBML; one of the built-in unit
 * identifiers @c "substance", @c "time", @c "volume", @c "area" or @c
 * "length"; or the identifier of a new unit defined in the list of unit
 * definitions in the enclosing Model structure.  There are no constraints
 * on the units that can be chosen from these sets.  There are no default
 * units for local parameters.
 * </ul>
 *
 * As with all other major SBML components, LocalParameter is derived from
 * SBase, and the methods defined on SBase are available on LocalParameter.
 * 
 * @warning LibSBML derives LocalParameter from Parameter; however, this
 * does not precisely match the object hierarchy defined by SBML
 * Level&nbsp;3, where LocalParameter is derived directly from SBase and not
 * Parameter.  We believe this arrangement makes it easier for libSBML
 * users to program applications that work with both SBML Level&nbsp;2 and
 * SBML Level&nbsp;3, but programmers should also keep in mind this
 * difference exists.  A side-effect of libSBML's scheme is that certain
 * methods on LocalParameter that are inherited from Parameter do not
 * actually have relevance to LocalParameter objects.  An example of this
 * is the methods pertaining to Parameter's attribute "constant"
 * (i.e., isSetConstant(), setConstant(), and getConstant()).
 *
 * @see ListOfLocalParameters
 * @see KineticLaw
 * 
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 
/**
 * @class ListOfLocalParameters.
 * @brief LibSBML implementation of SBML Level&nbsp;3's %ListOfLocalParameters construct.
 * 
 * The various ListOf___ classes in SBML are merely containers used for
 * organizing the main components of an SBML model.  All are derived from
 * the abstract class SBase, and inherit the various attributes and
 * subelements of SBase, such as "metaid" as and "annotation".  The
 * ListOf___ classes do not add any attributes of their own.
 *
 * ListOfLocalParameters is a subsidiary object class used only within
 * KineticLaw in SBML Level&nbsp;3.  It is not defined in SBML Levels
 * 1&ndash;2.  In Level&nbsp;3, a KineticLaw object can have a single
 * object of class ListOfLocalParameters containing a set of local
 * parameters used in that kinetic law definition.
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

#ifndef LocalParameter_h
#define LocalParameter_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/Parameter.h>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>


#ifdef __cplusplus


#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLVisitor;


class LIBSBML_EXTERN LocalParameter : public Parameter
{
public:

  /**
   * Creates a new LocalParameter object with the given SBML @p level and
   * @p version values.
   *
   * @param level an unsigned int, the SBML Level to assign to this
   * LocalParameter.
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * LocalParameter.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of a LocalParameter object to an SBMLDocument
   * (e.g., using KineticLaw::addLocalParameter(@if java LocalParameter p@endif)), the SBML Level, SBML
   * Version and XML namespace of the document @em override the
   * values used when creating the LocalParameter object via this
   * constructor.  This is necessary to ensure that an SBML document is a
   * consistent structure.  Nevertheless, the ability to supply the values
   * at the time of creation of a LocalParameter is an important aid to
   * producing valid SBML.  Knowledge of the intented SBML Level and
   * Version determine whether it is valid to assign a particular value to
   * an attribute, or whether it is valid to add an object to an existing
   * SBMLDocument.
   */
  LocalParameter (unsigned int level, unsigned int version);


  /**
   * Creates a new LocalParameter object with the given SBMLNamespaces
   * object @p sbmlns.
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
   * (identifier) attribute of a LocalParameter is required to have a value.
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
   * @note Upon the addition of a LocalParameter object to an SBMLDocument
   * (e.g., using KineticLaw::addLocalParameter(@if java LocalParameter p@endif)), the SBML XML namespace of
   * the document @em overrides the value used when creating the
   * LocalParameter object via this constructor.  This is necessary to
   * ensure that an SBML document is a consistent structure.  Nevertheless,
   * the ability to supply the values at the time of creation of a
   * LocalParameter is an important aid to producing valid SBML.  Knowledge
   * of the intented SBML Level and Version determine whether it is valid
   * to assign a particular value to an attribute, or whether it is valid
   * to add an object to an existing SBMLDocument.
   */
  LocalParameter (SBMLNamespaces* sbmlns);


  /**
   * Destroys this LocalParameter.
   */
  virtual ~LocalParameter ();


  /**
   * Copy constructor; creates a copy of a given LocalParameter object.
   * 
   * @param orig the LocalParameter instance to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  LocalParameter(const LocalParameter& orig);


  /**
   * Copy constructor; creates a LocalParameter object by copying
   * the attributes of a given Parameter object.
   * 
   * @param orig the Parameter instance to copy.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  LocalParameter(const Parameter& orig);


  /**
   * Assignment operator for LocalParameter.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  LocalParameter& operator=(const LocalParameter& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of LocalParameter.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next LocalParameter in the list
   * of parameters within which this LocalParameter is embedded (i.e., either
   * the list of parameters in the parent Model or the list of parameters
   * in the enclosing KineticLaw).
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this LocalParameter.
   * 
   * @return a (deep) copy of this LocalParameter.
   */
  virtual LocalParameter* clone () const;


  /**
   * Constructs and returns a UnitDefinition that corresponds to the units
   * of this LocalParameter's value.
   *
   * LocalParameters in SBML have an attribute ("units") for declaring the
   * units of measurement intended for the parameter's value.  <b>No
   * defaults are defined</b> by SBML in the absence of a definition for
   * "units".  This method returns a UnitDefinition object based on the
   * units declared for this LocalParameter using its "units" attribute, or
   * it returns @c NULL if no units have been declared.
   *
   * Note that unit declarations for LocalParameter objects are specified
   * in terms of the @em identifier of a unit (e.g., using setUnits()), but
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
   * LocalParameter, or @c NULL if one cannot be constructed.
   *
   * @note The libSBML system for unit analysis depends on the model as a
   * whole.  In cases where the LocalParameter object has not yet been
   * added to a model, or the model itself is incomplete, unit analysis is
   * not possible, and consequently this method will return @c NULL.
   *
   * @see isSetUnits()
   */
  UnitDefinition * getDerivedUnitDefinition();


  /**
   * Constructs and returns a UnitDefinition that corresponds to the units
   * of this LocalParameter's value.
   *
   * LocalParameters in SBML have an attribute ("units") for declaring the
   * units of measurement intended for the parameter's value.  <b>No
   * defaults are defined</b> by SBML in the absence of a definition for
   * "units".  This method returns a UnitDefinition object based on the
   * units declared for this LocalParameter using its "units" attribute, or
   * it returns @c NULL if no units have been declared.
   *
   * Note that unit declarations for LocalParameter objects are specified
   * in terms of the @em identifier of a unit (e.g., using setUnits()), but
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
   * LocalParameter, or @c NULL if one cannot be constructed.
   *
   * @note The libSBML system for unit analysis depends on the model as a
   * whole.  In cases where the LocalParameter object has not yet been
   * added to a model, or the model itself is incomplete, unit analysis is
   * not possible, and consequently this method will return @c NULL.
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
   * @return the SBML type code for this object, or @link
   * SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Returns the XML element name of this object, which for LocalParameter,
   * is always @c "localParameter".
   * 
   * @return the name of this element, i.e., @c "localParameter".
   */
  virtual const std::string& getElementName () const;


  /**
   * Predicate returning @c true if
   * all the required attributes for this LocalParameter object
   * have been set.
   *
   * @note The required attributes for a LocalParameter object are:
   * @li "id"
   * @li "value"
   *
   * @return a boolean value indicating whether all the required
   * attributes for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const ;


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

  void readL3Attributes (const XMLAttributes& attributes);


  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;

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


class LIBSBML_EXTERN ListOfLocalParameters : public ListOfParameters
{
public:

  /**
   * Creates a new ListOfLocalParameters object.
   *
   * The object is constructed such that it is valid for the given SBML
   * Level and Version combination.
   *
   * @param level the SBML Level
   * 
   * @param version the Version within the SBML Level
   */
  ListOfLocalParameters (unsigned int level, unsigned int version);
          

  /**
   * Creates a new ListOfLocalParameters object.
   *
   * The object is constructed such that it is valid for the SBML Level and
   * Version combination determined by the SBMLNamespaces object in @p
   * sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object that is used to determine the
   * characteristics of the ListOfLocalParameters object to be created.
   */
  ListOfLocalParameters (SBMLNamespaces* sbmlns);


  /**
   * Creates and returns a deep copy of this ListOfLocalParameters object.
   *
   * @return a (deep) copy of this ListOfLocalParameters.
   */
  virtual ListOfLocalParameters* clone () const;


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
   * (i.e., LocalParameter objects, if the list is non-empty).
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
   * For ListOfLocalParameters, the XML element name is @c "listOfLocalParameters".
   * 
   * @return the name of this element, i.e., @c "listOfLocalParameters".
   */
  virtual const std::string& getElementName () const;


  /**
   * Returns the LocalParameter object located at position @p n within this
   * ListOfLocalParameters instance.
   *
   * @param n the index number of the LocalParameter to get.
   * 
   * @return the nth LocalParameter in this ListOfLocalParameters.  If the
   * index @p n is out of bounds for the length of the list, then @c NULL
   * is returned.
   *
   * @see size()
   * @see get(const std::string& sid)
   */
  virtual LocalParameter * get (unsigned int n); 


  /**
   * Returns the LocalParameter object located at position @p n within this
   * ListOfLocalParameters instance.
   *
   * @param n the index number of the LocalParameter to get.
   * 
   * @return the item at position @p n.  The caller owns the returned
   * object and is responsible for deleting it.  If the index number @p n
   * is out of bounds for the length of the list, then @c NULL is returned.
   *
   * @see size()
   * @see get(const std::string& sid)
   */
  virtual const LocalParameter * get (unsigned int n) const; 


  /**
   * Returns the first LocalParameter object matching the given identifier.
   *
   * @param sid a string, the identifier of the LocalParameter to get.
   * 
   * @return the LocalParameter object found.  The caller owns the returned
   * object and is responsible for deleting it.  If none of the items have
   * an identifier matching @p sid, then @c NULL is returned.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual LocalParameter* get (const std::string& sid);


  /**
   * Returns the first LocalParameter object matching the given identifier.
   *
   * @param sid a string representing the identifier of the LocalParameter
   * to get.
   * 
   * @return the LocalParameter object found.  The caller owns the returned
   * object and is responsible for deleting it.  If none of the items have
   * an identifier matching @p sid, then @c NULL is returned.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const LocalParameter* get (const std::string& sid) const;


   /**
   * Returns the first child element found that has the given id in the model-wide SId namespace, or NULL if no such object is found.  Note that LocalParameters, while they use the SId namespace, are not in the model-wide SId namespace, so no LocalParameter object will be returned from this function (and is the reason we override the base ListOf::getElementBySId function here).
   *
   * @param id string representing the id of objects to find
   *
   * @return pointer to the first element found with the given id.
   */
  virtual SBase* getElementBySId(std::string id);
  
  
 /**
   * Removes the nth item from this ListOfLocalParameters, and returns a
   * pointer to it.
   *
   * @param n the index of the item to remove.  
   *
   * @return the item removed.  The caller owns the returned object and is
   * responsible for deleting it.  If the index number @p n is out of
   * bounds for the length of the list, then @c NULL is returned.
   *
   * @see size()
   * @see remove(const std::string& sid)
   */
  virtual LocalParameter* remove (unsigned int n);


  /**
   * Removes the first LocalParameter object in this ListOfLocalParameters
   * matching the given identifier, and returns a pointer to it.
   *
   * @param sid the identifier of the item to remove.
   *
   * @return the item removed.  The caller owns the returned object and is
   * responsible for deleting it.  If none of the items have an identifier
   * matching @p sid, then @c NULL is returned.
   */
  virtual LocalParameter* remove (const std::string& sid);


  /** @cond doxygen-libsbml-internal */

  /**
   * Get the ordinal position of this element in the containing object
   * (which in this case is the Model object).
   *
   * The ordering of elements in the XML form of SBML is generally fixed
   * for most components in SBML.  So, for example, the ListOfLocalParameters
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
   * Create a ListOfLocalParameters object corresponding to the next token in
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


LIBSBML_EXTERN
LocalParameter_t *
LocalParameter_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
LocalParameter_t *
LocalParameter_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
LocalParameter_free (LocalParameter_t *p);


LIBSBML_EXTERN
LocalParameter_t *
LocalParameter_clone (const LocalParameter_t *p);


LIBSBML_EXTERN
void
LocalParameter_initDefaults (LocalParameter_t *p);


LIBSBML_EXTERN
const XMLNamespaces_t *
LocalParameter_getNamespaces(LocalParameter_t *c);


LIBSBML_EXTERN
const char *
LocalParameter_getId (const LocalParameter_t *p);


LIBSBML_EXTERN
const char *
LocalParameter_getName (const LocalParameter_t *p);


LIBSBML_EXTERN
double
LocalParameter_getValue (const LocalParameter_t *p);


LIBSBML_EXTERN
const char *
LocalParameter_getUnits (const LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_getConstant (const LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_isSetId (const LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_isSetName (const LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_isSetValue (const LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_isSetUnits (const LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_setId (LocalParameter_t *p, const char *sid);


LIBSBML_EXTERN
int
LocalParameter_setName (LocalParameter_t *p, const char *name);


LIBSBML_EXTERN
int
LocalParameter_setValue (LocalParameter_t *p, double value);


LIBSBML_EXTERN
int
LocalParameter_setUnits (LocalParameter_t *p, const char *units);


LIBSBML_EXTERN
int
LocalParameter_setConstant (LocalParameter_t *p, int value);


LIBSBML_EXTERN
int
LocalParameter_unsetName (LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_unsetValue (LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_unsetUnits (LocalParameter_t *p);


LIBSBML_EXTERN
int
LocalParameter_hasRequiredAttributes (LocalParameter_t *p);


LIBSBML_EXTERN
UnitDefinition_t * 
LocalParameter_getDerivedUnitDefinition(LocalParameter_t *p);


LIBSBML_EXTERN
LocalParameter_t *
ListOfLocalParameters_getById (ListOf_t *lo, const char *sid);


LIBSBML_EXTERN
LocalParameter_t *
ListOfLocalParameters_removeById (ListOf_t *lo, const char *sid);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* LocalParameter_h */
