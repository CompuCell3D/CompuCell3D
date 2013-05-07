/**
 * @file    Delay.h
 * @brief   Definition of Delay.
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
 * @class Delay
 * @brief LibSBML implementation of %SBML's %Delay construct for %Event.
 *
 * An Event object defines when the event can occur, the variables that
 * are affected by the event, and how the variables are affected.  The
 * effect of the event can optionally be delayed after the occurrence of
 * the condition which invokes it.  An event delay is defined using an
 * object of class Delay.
 *
 * The object class Delay is derived from SBase and adds a single
 * subelement called "math".  This subelement is used to hold MathML
 * content.  The mathematical formula represented by "math" must evaluate
 * to a numerical value.  It is used as the length of time between when the
 * event is @em triggered and when the event's assignments are
 * actually @em executed.  If no delay is present on a given Event, a time
 * delay of zero is assumed.
 *
 * The expression in "math" must be evaluated at the time the event is @em
 * triggered.  The expression must always evaluate to a nonnegative number
 * (otherwise, a nonsensical situation could arise where an event is
 * defined to execute before it is triggered!).
 *
 * @section delay-units The units of the mathematical expression in a Delay
 *
 * In SBML Level&nbsp;2 versions before Version&nbsp;4, the units of the
 * numerical value computed by the Delay's "math" expression are @em
 * required to be in units of time, or the model is considered to have a
 * unit consistency error.  In Level&nbsp;2 Version&nbsp;4 as well as SBML
 * Level&nbsp;3 Version&nbsp;1 Core, this requirement is relaxed; these
 * specifications only stipulate that the units of the numerical value
 * computed by a Delay instance's "math" expression @em should match the
 * model's units of time (meaning the definition of the @c time units in
 * the model).  LibSBML respects these requirements, and depending on
 * whether an earlier Version of SBML Level&nbsp;2 is in use, libSBML may
 * or may not flag unit inconsistencies as errors or merely warnings.
 *
 * Note that <em>units are not predefined or assumed</em> for the contents
 * of "math" in a Delay object; rather, they must be defined explicitly for
 * each instance of a Delay object in a model.  This is an important point
 * to bear in mind when literal numbers are used in delay expressions.  For
 * example, the following Event instance would result in a warning logged
 * by SBMLDocument::checkConsistency() about the fact that libSBML cannot
 * verify the consistency of the units of the expression.  The reason is
 * that the formula inside the "math" element does not have any declared
 * units, whereas what is expected in this context is units of time:
 * @verbatim
<model>
    ...
    <listOfEvents>
        <event useValuesFromTriggerTime="true">
            ...
            <delay>
                <math xmlns="http://www.w3.org/1998/Math/MathML">
                    <cn> 1 </cn>
                </math>
            </delay>
            ...
        </event>
    </listOfEvents>
    ...
</model>
@endverbatim
 * 
 * The <code>&lt;cn&gt; 1 &lt;/cn&gt;</code> within the mathematical formula
 * of the @c delay above has <em>no units declared</em>.  To make the
 * expression have the needed units of time, literal numbers should be
 * avoided in favor of defining Parameter objects for each quantity, and
 * declaring units for the Parameter values.  The following fragment of
 * SBML illustrates this approach:
 * @verbatim
<model>
    ...
    <listOfParameters>
        <parameter id="transcriptionDelay" value="10" units="second"/>
    </listOfParameters>
    ...
    <listOfEvents>
        <event useValuesFromTriggerTime="true">
            ...
            <delay>
                <math xmlns="http://www.w3.org/1998/Math/MathML">
                    <ci> transcriptionDelay </ci>
                </math>
            </delay>
            ...
        </event>
    </listOfEvents>
    ...
</model>
@endverbatim
 *
 * In SBML Level&nbsp;3, an alternative approach is available in the form
 * of the @c units attribute, which SBML Level&nbsp;3 allows to appear on
 * MathML @c cn elements.  The value of this attribute can be used to
 * indicate the unit of measurement to be associated with the number in the
 * content of a @c cn element.  The attribute is named @c units but,
 * because it appears inside MathML element (which is in the XML namespace
 * for MathML and not the namespace for SBML), it must always be prefixed
 * with an XML namespace prefix for the SBML Level&nbsp;3 Version&nbsp;1
 * namespace.  The following is an example of this approach:
 * @verbatim
<model timeUnits="second" ...>
    ...
    <listOfEvents>
        <event useValuesFromTriggerTime="true">
            ...
            <delay>
                <math xmlns="http://www.w3.org/1998/Math/MathML"
                      xmlns:sbml="http://www.sbml.org/sbml/level3/version1/core">
                    <cn sbml:units="second"> 10 </cn>
                </math>
            </delay>
            ...
        </event>
    </listOfEvents>
    ...
</model>
@endverbatim
 */

#ifndef Delay_h
#define Delay_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class ASTNode;
class SBMLVisitor;


class LIBSBML_EXTERN Delay : public SBase
{
public:

  /**
   * Creates a new Delay using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this Delay
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * Delay
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of a Delay object to an Event (e.g., using
   * Event::setDelay(@if java Delay d@endif)), the SBML Level, SBML Version
   * and XML namespace of the document @em override the values used when
   * creating the Delay object via this constructor.  This is necessary to
   * ensure that an SBML document is a consistent structure.  Nevertheless,
   * the ability to supply the values at the time of creation of a Delay is
   * an important aid to producing valid SBML.  Knowledge of the intented
   * SBML Level and Version determine whether it is valid to assign a
   * particular value to an attribute, or whether it is valid to add a
   * particular Delay object to an existing Event.
   */
  Delay (unsigned int level, unsigned int version);


  /**
   * Creates a new Delay using the given SBMLNamespaces object
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
   * @note Upon the addition of a Delay object to an Event (e.g., using
   * Event::setDelay(@if java Delay d@endif)), the SBML XML namespace of
   * the document @em overrides the value used when creating the Delay
   * object via this constructor.  This is necessary to ensure that an SBML
   * document is a consistent structure.  Nevertheless, the ability to
   * supply the values at the time of creation of a Delay is an important
   * aid to producing valid SBML.  Knowledge of the intented SBML Level and
   * Version determine whether it is valid to assign a particular value to
   * an attribute, or whether it is valid to add a particular Delay object
   * to an existing Event.
   */
  Delay (SBMLNamespaces* sbmlns);


  /**
   * Destroys this Delay.
   */
  virtual ~Delay ();


  /**
   * Copy constructor; creates a copy of this Delay.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  Delay (const Delay& orig);


  /**
   * Assignment operator
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  Delay& operator=(const Delay& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of Delay.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this Delay.
   *
   * @return a (deep) copy of this Delay.
   */
  virtual Delay* clone () const;


  /**
   * Get the mathematical formula for the delay and return it
   * as an AST.
   * 
   * @return the math of this Delay.
   */
  const ASTNode* getMath () const;


  /**
   * Predicate to test whether the formula for this delay is set.
   *
   * @return @c true if the formula (meaning the @c math subelement) of
   * this Delay is set, @c false otherwise.
   */
  bool isSetMath () const;


  /**
   * Sets the delay expression of this Delay instance to a copy of the given
   * ASTNode.
   *
   * @param math an ASTNode representing a formula tree.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t.  @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   */
  int setMath (const ASTNode* math);


  /**
   * Calculates and returns a UnitDefinition that expresses the units
   * of measurement assumed for the "math" expression of this Delay.
   *
   * Delay elements in SBML express a time delay for an Event.  Beginning
   * with SBML Level&nbsp;2 Version&nbsp;2, the units of that time are
   * calculated based on the mathematical expression and the model
   * quantities referenced by <code>&lt;ci&gt;</code> elements used within
   * that expression.  The Delay::getDerivedUnitDefinition() method returns
   * what libSBML computes the units to be, to the extent that libSBML can
   * compute them.
   *
   * (Prior to SBML Level&nbsp;2 Version&nbsp;2, there existed an attribute
   * on Event called "timeUnits".  This attribute could be used to set the
   * intended units of the delay expression.  For such models, this will
   * return a UnitDefinition object that corresponds to the declared
   * "timeUnits" units.)
   *
   * Note that the functionality that facilitates unit analysis depends 
   * on the model as a whole.  Thus, in cases where the object has not 
   * been added to a model or the model itself is incomplete,
   * unit analysis is not possible and this method will return @c NULL.
   *
   * @warning Note that it is possible the "math" expression in the Delay
   * contains literal numbers or parameters with undeclared units.  In
   * those cases, it is not possible to calculate the units of the overall
   * expression without making assumptions.  LibSBML does not make
   * assumptions about the units, and Delay::getDerivedUnitDefinition()
   * only returns the units as far as it is able to determine them.  For
   * example, in an expression <em>X + Y</em>, if <em>X</em> has
   * unambiguously-defined units and <em>Y</em> does not, it will return
   * the units of <em>X</em>.  When using this method, <strong>it is
   * critical that callers also invoke the method</strong>
   * Delay::containsUndeclaredUnits() <strong>to determine whether this
   * situation holds</strong>.  Callers should take suitable action in
   * those situations.
   * 
   * @return a UnitDefinition that expresses the units of the math 
   * expression of this Delay, or @c NULL if one cannot be constructed.
   *
   * @see containsUndeclaredUnits()
   */
  UnitDefinition * getDerivedUnitDefinition();


  /**
   * Calculates and returns a UnitDefinition that expresses the units
   * of measurement assumed for the "math" expression of this Delay.
   *
   * Delay elements in SBML express a time delay for an Event.  Beginning
   * with SBML Level&nbsp;2 Version&nbsp;2, the units of that time are
   * calculated based on the mathematical expression and the model
   * quantities referenced by <code>&lt;ci&gt;</code> elements used within
   * that expression.  The Delay::getDerivedUnitDefinition() method returns
   * what libSBML computes the units to be, to the extent that libSBML can
   * compute them.
   *
   * (Prior to SBML Level&nbsp;2 Version&nbsp;2, there existed an attribute
   * on Event called "timeUnits".  This attribute could be used to set the
   * intended units of the delay expression.  For such models, this will
   * return a UnitDefinition object that corresponds to the declared
   * "timeUnits" units.)
   *
   * Note that the functionality that facilitates unit analysis depends 
   * on the model as a whole.  Thus, in cases where the object has not 
   * been added to a model or the model itself is incomplete,
   * unit analysis is not possible and this method will return @c NULL.
   *
   * @warning Note that it is possible the "math" expression in the Delay
   * contains literal numbers or parameters with undeclared units.  In
   * those cases, it is not possible to calculate the units of the overall
   * expression without making assumptions.  LibSBML does not make
   * assumptions about the units, and Delay::getDerivedUnitDefinition()
   * only returns the units as far as it is able to determine them.  For
   * example, in an expression <em>X + Y</em>, if <em>X</em> has
   * unambiguously-defined units and <em>Y</em> does not, it will return
   * the units of <em>X</em>.  When using this method, <strong>it is
   * critical that callers also invoke the method</strong>
   * Delay::containsUndeclaredUnits() <strong>to determine whether this
   * situation holds</strong>.  Callers should take suitable action in
   * those situations.
   * 
   * @return a UnitDefinition that expresses the units of the math 
   * expression of this Delay, or @c NULL if one cannot be constructed.
   *
   * @see containsUndeclaredUnits()
   */
  const UnitDefinition * getDerivedUnitDefinition() const;


  /**
   * Predicate returning @c true if 
   * the "math" expression in this Delay instance contains
   * parameters with undeclared units or literal numbers.
   * 
   * Delay elements in SBML express a time delay for an Event.  Beginning
   * with SBML Level&nbsp;2 Version&nbsp;2, the units of that time are
   * calculated based on the mathematical expression and the model
   * quantities referenced by <code>&lt;ci&gt;</code> elements used within
   * that expression.  The Delay::getDerivedUnitDefinition() method returns
   * what libSBML computes the units to be, to the extent that libSBML can
   * compute them.  However, if the expression contains literal numbers or
   * parameters with undeclared units, libSBML may not be able to compute
   * the full units of the expression and will only return what it can
   * compute.  Callers should always use Delay::containsUndeclaredUnits()
   * when using Delay::getDerivedUnitDefinition() to decide whether the
   * returned units may be incomplete.
   * 
   * @return @c true if the math expression of this Delay includes
   * numbers/parameters with undeclared units, @c false otherwise.
   *
   * @note A return value of @c true indicates that the UnitDefinition
   * returned by Delay::getDerivedUnitDefinition() may not accurately
   * represent the units of the expression.
   *
   * @see getDerivedUnitDefinition()
   */
  bool containsUndeclaredUnits();


  /**
   * Predicate returning @c true if 
   * the "math" expression in this Delay instance contains
   * parameters with undeclared units or literal numbers.
   * 
   * Delay elements in SBML express a time delay for an Event.  Beginning
   * with SBML Level&nbsp;2 Version&nbsp;2, the units of that time are
   * calculated based on the mathematical expression and the model
   * quantities referenced by <code>&lt;ci&gt;</code> elements used within
   * that expression.  The Delay::getDerivedUnitDefinition() method returns
   * what libSBML computes the units to be, to the extent that libSBML can
   * compute them.  However, if the expression contains literal numbers or
   * parameters with undeclared units, libSBML may not be able to compute
   * the full units of the expression and will only return what it can
   * compute.  Callers should always use Delay::containsUndeclaredUnits()
   * when using Delay::getDerivedUnitDefinition() to decide whether the
   * returned units may be incomplete.
   * 
   * @return @c true if the math expression of this Delay includes
   * numbers/parameters with undeclared units, @c false otherwise.
   *
   * @note A return value of @c true indicates that the UnitDefinition
   * returned by Delay::getDerivedUnitDefinition() may not accurately
   * represent the units of the expression.
   *
   * @see getDerivedUnitDefinition()
   */
  bool containsUndeclaredUnits() const;


  /**
   * Returns the libSBML type code of this object instance.
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
   * Returns the XML element name of this object, which for Delay, is
   * always @c "delay".
   * 
   * @return the name of this element, i.e., @c "delay".
   *
   * @see getTypeCode()
   */
  virtual const std::string& getElementName () const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Returns the position of this element.
   * 
   * @return the ordinal position of the element with respect to its
   * siblings or -1 (default) to indicate the position is not significant.
   */
  virtual int getElementPosition () const;
  /** @endcond */


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
   * all the required elements for this Delay object
   * have been set.
   *
   * @note The required elements for a Delay object are:
   * @li "math"
   *
   * @return a boolean value indicating whether all the required
   * elements for this object have been defined.
   */
  virtual bool hasRequiredElements() const;


  /**
   * Finds this Delay's Event parent and calls unsetDelay() on it, indirectly deleting itself.  Overridden from the SBase function since the parent is not a ListOf.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int removeFromParentAndDelete();

  /**
   * Renames all the SIdRef attributes on this element, including any found in MathML
   */
  virtual void renameSIdRefs(std::string oldid, std::string newid);


  /**
   * Renames all the UnitSIdRef attributes on this element
   */
  virtual void renameUnitSIdRefs(std::string oldid, std::string newid);


  /** @cond doxygen-libsbml-internal */
  /**
   * Replace all nodes with the name 'id' from the child 'math' object with the provided function. 
   *
   */
  virtual void replaceSIDWithFunction(const std::string& id, const ASTNode* function);
  /** @endcond */


  /** @cond doxygen-libsbml-internal */
  /*
   * Function to set/get an identifier for unit checking
   */
  std::string getInternalId() const { return mInternalId; };
  void setInternalId(std::string id) { mInternalId = id; };
  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * Subclasses should override this method to read (and store) XHTML,
   * MathML, etc. directly from the XMLInputStream.
   *
   * @return true if the subclass read from the stream, false otherwise.
   */
  virtual bool readOtherXML (XMLInputStream& stream);

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or @c NULL if the token was not recognized.
   */
//  virtual SBase* createObject (XMLInputStream& stream);


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

  void readL2Attributes (const XMLAttributes& attributes);
  
  void readL3Attributes (const XMLAttributes& attributes);


  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;


  ASTNode*     mMath;

  /* internal id used by unit checking */
  std::string mInternalId;

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

/*
LIBSBML_EXTERN
Delay_t *
Delay_createWithLevelVersionAndNamespaces (unsigned int level,
              unsigned int version, XMLNamespaces_t *xmlns);
*/

LIBSBML_EXTERN
Delay_t *
Delay_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Delay_t *
Delay_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
Delay_free (Delay_t *d);


LIBSBML_EXTERN
Delay_t *
Delay_clone (const Delay_t *d);


LIBSBML_EXTERN
const XMLNamespaces_t *
Delay_getNamespaces(Delay_t *c);


LIBSBML_EXTERN
const ASTNode_t *
Delay_getMath (const Delay_t *d);


LIBSBML_EXTERN
int
Delay_isSetMath (const Delay_t *d);


LIBSBML_EXTERN
int
Delay_setMath (Delay_t *d, const ASTNode_t *math);


LIBSBML_EXTERN
UnitDefinition_t * 
Delay_getDerivedUnitDefinition(Delay_t *d);


LIBSBML_EXTERN
int 
Delay_containsUndeclaredUnits(Delay_t *d);

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* Delay_h */

