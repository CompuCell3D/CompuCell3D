/**
 * @file    CompartmentType.h
 * @brief   Definitions of CompartmentType and ListOfCompartmentTypes.
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
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 * 
 * @class CompartmentType.
 * @brief LibSBML implementation of SBML's Level&nbsp;2's %CompartmentType construct.
 *
 * SBML Level&nbsp;2 Versions&nbsp;2&ndash;4 provide the <em>compartment
 * type</em> as a grouping construct that can be used to establish a
 * relationship between multiple Compartment objects.  A CompartmentType
 * object only has an identity, and this identity can only be used to
 * indicate that particular Compartment objects in the model belong to this
 * type.  This may be useful for conveying a modeling intention, such as
 * when a model contains many similar compartments, either by their
 * biological function or the reactions they carry.  Without a compartment
 * type construct, it would be impossible within SBML itself to indicate
 * that all of the compartments share an underlying conceptual relationship
 * because each SBML compartment must be given a unique and separate
 * identity.  Compartment types have no mathematical meaning in
 * SBML&mdash;they have no effect on a model's mathematical interpretation.
 * Simulators and other numerical analysis software may ignore
 * CompartmentType definitions and references to them in a model.
 * 
 * There is no mechanism in SBML Level 2 for representing hierarchies of
 * compartment types.  One CompartmentType instance cannot be the subtype
 * of another CompartmentType instance; SBML provides no means of defining
 * such relationships.
 * 
 * As with other major structures in SBML, CompartmentType has a mandatory
 * attribute, "id", used to give the compartment type an identifier.  The
 * identifier must be a text %string conforming to the identifer syntax
 * permitted in SBML.  CompartmentType also has an optional "name"
 * attribute, of type @c string.  The "id" and "name" must be used
 * according to the guidelines described in the SBML specification (e.g.,
 * Section 3.3 in the Level 2 Version 4 specification).
 *
 * CompartmentType was introduced in SBML Level 2 Version 2.  It is not
 * available in SBML Level&nbsp;1 nor in Level&nbsp;3.
 *
 * @see Compartment
 * @see ListOfCompartmentTypes
 * @see SpeciesType
 * @see ListOfSpeciesTypes
 * 
 * 
 * @class ListOfCompartmentTypes.
 * @brief LibSBML implementation of SBML's %ListOfCompartmentTypes construct.
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

#ifndef CompartmentType_h
#define CompartmentType_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>


#ifdef __cplusplus


#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLVisitor;


class LIBSBML_EXTERN CompartmentType : public SBase
{
public:

  /**
   * Creates a new CompartmentType using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this CompartmentType
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * CompartmentType
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of a CompartmentType object to an SBMLDocument
   * (e.g., using Model::addCompartmentType(@if java CompartmentType ct@endif)), the SBML Level, SBML
   * Version and XML namespace of the document @em override the
   * values used when creating the CompartmentType object via this
   * constructor.  This is necessary to ensure that an SBML document is a
   * consistent structure.  Nevertheless, the ability to supply the values
   * at the time of creation of a CompartmentType is an important aid to
   * producing valid SBML.  Knowledge of the intented SBML Level and
   * Version determine whether it is valid to assign a particular value to
   * an attribute, or whether it is valid to add an object to an existing
   * SBMLDocument.
   */
  CompartmentType (unsigned int level, unsigned int version);


  /**
   * Creates a new CompartmentType using the given SBMLNamespaces object
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
   * (identifier) attribute of a CompartmentType is required to have a value.
   * Thus, callers are cautioned to assign a value after calling this
   * constructor.  Setting the identifier can be accomplished using the
   * method setId(@if java String id@endif).
   *
   * @param sbmlns an SBMLNamespaces object.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   *
   * @note Upon the addition of a CompartmentType object to an SBMLDocument
   * (e.g., using Model::addCompartmentType(@if java CompartmentType ct@endif)), the SBML XML namespace of
   * the document @em overrides the value used when creating the
   * CompartmentType object via this constructor.  This is necessary to
   * ensure that an SBML document is a consistent structure.  Nevertheless,
   * the ability to supply the values at the time of creation of a
   * CompartmentType is an important aid to producing valid SBML.
   * Knowledge of the intented SBML Level and Version determine whether it
   * is valid to assign a particular value to an attribute, or whether it
   * is valid to add an object to an existing SBMLDocument.
   */
  CompartmentType (SBMLNamespaces* sbmlns);


  /**
   * Destroys this CompartmentType.
   */
  virtual ~CompartmentType ();


  /**
   * Copy constructor; creates a copy of this CompartmentType.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  CompartmentType(const CompartmentType& orig);


  /**
   * Assignment operator for CompartmentType.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  CompartmentType& operator=(const CompartmentType& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of CompartmentType.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next CompartmentType in
   * the list of compartment types.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this CompartmentType.
   * 
   * @return a (deep) copy of this CompartmentType.
   */
  virtual CompartmentType* clone () const;


  /**
   * Returns the value of the "id" attribute of this CompartmentType.
   * 
   * @return the id of this CompartmentType.
   */
  virtual const std::string& getId () const;


  /**
   * Returns the value of the "name" attribute of this CompartmentType.
   * 
   * @return the name of this CompartmentType.
   */
  virtual const std::string& getName () const;


  /**
   * Predicate returning @c true if this
   * CompartmentType's "id" attribute is set.
   *
   * @return @c true if the "id" attribute of this CompartmentType is
   * set, @c false otherwise.
   */
  virtual bool isSetId () const;


  /**
   * Predicate returning @c true if this
   * CompartmentType's "name" attribute is set.
   *
   * @return @c true if the "name" attribute of this CompartmentTypeType is
   * set, @c false otherwise.
   */
  virtual bool isSetName () const;


  /**
   * Sets the value of the "id" attribute of this CompartmentType.
   *
   * The string @p sid is copied.  Note that SBML has strict requirements
   * for the syntax of identifiers.  @htmlinclude id-syntax.html
   *
   * @param sid the string to use as the identifier of this CompartmentType
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setId (const std::string& sid);


  /**
   * Sets the value of the "name" attribute of this CompartmentType.
   *
   * The string in @p name is copied.
   *
   * @param name the new name for the CompartmentType
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setName (const std::string& name);


  /**
   * Unsets the value of the "name" attribute of this CompartmentType.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int unsetName ();


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
   * @return the SBML type code for this object, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Returns the XML element name of this object, which for
   * CompartmentType, is always @c "compartmentType".
   * 
   * @return the name of this element, i.e., @c "compartmentType".
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
   * all the required attributes for this CompartmentType object
   * have been set.
   *
   * @note The required attributes for a CompartmentType object are:
   * @li "id"
   *
   * @return a boolean value indicating whether all the required
   * attributes for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const;


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
   *
   * @param attributes the XMLAttributes to use.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);


  void readL2Attributes (const XMLAttributes& attributes);
  

  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.
   *
   * @param stream the XMLOutputStream to use.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;

  std::string mId;
  std::string mName;

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



class LIBSBML_EXTERN ListOfCompartmentTypes : public ListOf
{
public:

  /**
   * Creates a new ListOfCompartmentTypes object.
   *
   * The object is constructed such that it is valid for the given SBML
   * Level and Version combination.
   *
   * @param level the SBML Level
   * 
   * @param version the Version within the SBML Level
   */
  ListOfCompartmentTypes (unsigned int level, unsigned int version);
          

  /**
   * Creates a new ListOfCompartmentTypes object.
   *
   * The object is constructed such that it is valid for the SBML Level and
   * Version combination determined by the SBMLNamespaces object in @p
   * sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object that is used to determine the
   * characteristics of the ListOfCompartmentTypes object to be created.
   */
  ListOfCompartmentTypes (SBMLNamespaces* sbmlns);


  /**
   * Creates and returns a deep copy of this ListOfCompartmentTypes instance.
   *
   * @return a (deep) copy of this ListOfCompartmentTypes.
   */
  virtual ListOfCompartmentTypes* clone () const;


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
   * @return the SBML type code for this object, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const { return SBML_LIST_OF; };


  /**
   * Returns the libSBML type code for the objects contained in this ListOf
   * (i.e., CompartmentType objects, if the list is non-empty).
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
   * libsbmlcs.libsbml libsbml@endlink.  The names of the type codes all begin with
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
   * For ListOfCompartmentTypes, the XML element name is @c
   * "listOfCompartmentTypes".
   * 
   * @return the name of this element, i.e., @c "listOfCompartmentTypes".
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a CompartmentType from the ListOfCompartmentTypes.
   *
   * @param n the index number of the CompartmentType to get.
   * 
   * @return the nth CompartmentType in this ListOfCompartmentTypes.
   *
   * @see size()
   */
  virtual CompartmentType * get(unsigned int n); 


  /**
   * Get a CompartmentType from the ListOfCompartmentTypes.
   *
   * @param n the index number of the CompartmentType to get.
   * 
   * @return the nth CompartmentType in this ListOfCompartmentTypes.
   *
   * @see size()
   */
  virtual const CompartmentType * get(unsigned int n) const; 

  /**
   * Get a CompartmentType from the ListOfCompartmentTypes
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the CompartmentType to get.
   * 
   * @return CompartmentType in this ListOfCompartmentTypes
   * with the given id or @c NULL if no such
   * CompartmentType exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual CompartmentType* get (const std::string& sid);


  /**
   * Get a CompartmentType from the ListOfCompartmentTypes
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the CompartmentType to get.
   * 
   * @return CompartmentType in this ListOfCompartmentTypes
   * with the given id or @c NULL if no such
   * CompartmentType exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const CompartmentType* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfCompartmentTypes items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual CompartmentType* remove (unsigned int n);


  /**
   * Removes item in this ListOfCompartmentTypes items with the given identifier.
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
  virtual CompartmentType* remove (const std::string& sid);


  /** @cond doxygen-libsbml-internal */

  /**
   * Get the ordinal position of this element in the containing object
   * (which in this case is the Model object).
   *
   * The ordering of elements in the XML form of SBML is generally fixed
   * for most components in SBML.  For example, the
   * ListOfCompartmentTypes in a model (in SBML Level 2 Version 4) is the
   * third ListOf___.  (However, it differs for different Levels and
   * Versions of SBML, so calling code should not hardwire this number.)
   *
   * @return the ordinal position of the element with respect to its
   * siblings, or @c -1 (default) to indicate the position is not significant.
   */
  virtual int getElementPosition () const;

  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * Create a ListOfCompartmentTypes object corresponding to the next token
   * in the XML input stream.
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

/*
LIBSBML_EXTERN
CompartmentType_t *
CompartmentType_createWithLevelVersionAndNamespaces (unsigned int level,
              unsigned int version, XMLNamespaces_t *xmlns);
*/

LIBSBML_EXTERN
CompartmentType_t *
CompartmentType_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
CompartmentType_t *
CompartmentType_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
CompartmentType_free (CompartmentType_t *ct);


LIBSBML_EXTERN
CompartmentType_t *
CompartmentType_clone (const CompartmentType_t *ct);


LIBSBML_EXTERN
const XMLNamespaces_t *
CompartmentType_getNamespaces(CompartmentType_t *c);


LIBSBML_EXTERN
const char *
CompartmentType_getId (const CompartmentType_t *ct);


LIBSBML_EXTERN
const char *
CompartmentType_getName (const CompartmentType_t *ct);


LIBSBML_EXTERN
int
CompartmentType_isSetId (const CompartmentType_t *ct);


LIBSBML_EXTERN
int
CompartmentType_isSetName (const CompartmentType_t *ct);


LIBSBML_EXTERN
int
CompartmentType_setId (CompartmentType_t *ct, const char *sid);


LIBSBML_EXTERN
int
CompartmentType_setName (CompartmentType_t *ct, const char *name);


LIBSBML_EXTERN
int
CompartmentType_unsetName (CompartmentType_t *ct);


LIBSBML_EXTERN
CompartmentType_t *
ListOfCompartmentTypes_getById (ListOf_t *lo, const char *sid);


LIBSBML_EXTERN
CompartmentType_t *
ListOfCompartmentTypes_removeById (ListOf_t *lo, const char *sid);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* CompartmentType_h */
