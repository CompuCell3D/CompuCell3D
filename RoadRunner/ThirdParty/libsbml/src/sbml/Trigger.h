/**
 * @file    Trigger.h
 * @brief   Definition of Trigger
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
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 *
 * @class Trigger
 * @brief LibSBML implementation of %SBML's %Trigger construct for %Event.
 *
 * An Event object defines when the event can occur, the variables that are
 * affected by the event, and how the variables are affected.  The Trigger
 * construct in SBML is used to define a mathematical expression that
 * determines when an Event is @em triggered.
 *
 * A Trigger object in SBML Level&nbsp;2 and Level&nbsp;3 contains one
 * subelement named "math" containing a MathML expression.  The expression
 * must evaluate to a value of type @c boolean.  The exact moment at which
 * the expression evaluates to @c true is the time point when the Event is
 * @em triggered.  In SBML Level&nbsp;3, Trigger has additional attributes
 * that must be assigned values; they are discussed in a separate section
 * below.
 * 
 * An event only @em triggers when its Trigger expression makes the
 * transition in value from @c false to @c true.  The event will also
 * trigger at any subsequent time points when the trigger makes this
 * transition; in other words, an event can be triggered multiple times
 * during a simulation if its trigger condition makes the transition from
 * @c false to @c true more than once.  In SBML Level&nbsp;3, the behavior
 * at the very start of simulation (i.e., at <em>t = 0</em>, where
 * <em>t</em> stands for time) is determined in part by the boolean flag
 * "initialValue".  This and other additional features introduced in SBML
 * Level&nbsp;3 are discussed further below.
 *
 * @section version-diffs Version differences
 *
 * SBML Level&nbsp;3 Version&nbsp;1 introduces two required attributes
 * on the Trigger object: "persistent" and "initialValue".  The rest of
 * this introduction describes these two attributes.
 *
 * @subsection trigger-persistent The "persistent" attribute on Trigger
 *
 * In the interval between when an Event object <em>triggers</em> (i.e.,
 * its Trigger object expression transitions in value from @c false to
 * @c true) and when its assignments are to be <em>executed</em>, conditions
 * in the model may change such that the trigger expression transitions
 * back from @c true to @c false.  Should the event's assignments still be
 * made if this happens?  Answering this question is the purpose of the
 * "persistent" attribute on Trigger.
 * 
 * If the boolean attribute "persistent" has a value of @c true, then once
 * the event is triggered, all of its assignments are always performed when
 * the time of execution is reached.  The name @em persistent is meant to
 * evoke the idea that the trigger expression does not have to be
 * re-checked after it triggers if "persistent"=@c true.  Conversely, if
 * the attribute value is @c false, then the trigger expression is not
 * assumed to persist: if the expression transitions in value back to @c
 * false at any time between when the event triggered and when it is to be
 * executed, the event is no longer considered to have triggered and its
 * assignments are not executed.  (If the trigger expression transitions
 * once more to @c true after that point, then the event is triggered, but
 * this then constitutes a whole new event trigger-and-execute sequence.)
 * 
 * The "persistent" attribute can be especially useful when Event objects
 * contain Delay objects, but it is relevant even in a model without delays
 * if the model contains two or more events.  As explained in the
 * introduction to this section, the operation of all events in SBML
 * (delayed or not) is conceptually divided into two phases,
 * <em>triggering</em> and <em>execution</em>; however, unless events have
 * priorities associated with them, SBML does not mandate a particular
 * ordering of event execution in the case of simultaneous events.  Models
 * with multiple events can lead to situations where the execution of one
 * event affects another event's trigger expression value.  If that other
 * event has "persistent"=@c false, and its trigger expression evaluates to
 * @c false before it is to be executed, the event must not be executed
 * after all.
 * 
 * @subsection trigger-initialvalue The "initialValue" attribute on Trigger
 * 
 * As mentioned above, an event <em>triggers</em> when the mathematical
 * expression in its Trigger object transitions in value from @c false to
 * @c true.  An unanswered question concerns what happens at the start of a
 * simulation: can event triggers make this transition at <em>t = 0</em>,
 * where <em>t</em> stands for time?
 * 
 * In order to determine whether an event may trigger at <em>t = 0</em>, it
 * is necessary to know what value the Trigger object's "math" expression
 * had immediately prior to <em>t = 0</em>.  This starting value of the
 * trigger expression is determined by the value of the boolean attribute
 * "initialValue".  A value of @c true means the trigger expression is
 * taken to have the value @c true immediately prior to <em>t = 0</em>.  In
 * that case, the trigger cannot transition in value from @c false to @c
 * true at the moment simulation begins (because it has the value @c true
 * both before and after <em>t = 0</em>), and can only make the transition
 * from @c false to @c true sometime <em>after</em> <em>t = 0</em>.  (To do
 * that, it would also first have to transition to @c false before it could
 * make the transition from @c false back to @c true.)  Conversely, if
 * "initialValue"=@c false, then the trigger expression is assumed to start
 * with the value @c false, and therefore may trigger at <em>t = 0</em> if
 * the expression evaluates to @c true at that moment.
 * 
 *
 * @see Event
 * @see Delay
 * @see EventAssignment
 */


#ifndef Trigger_h
#define Trigger_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class ASTNode;
class SBMLVisitor;


class LIBSBML_EXTERN Trigger : public SBase
{
public:

  /**
   * Creates a new Trigger using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this Trigger
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * Trigger
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   */
  Trigger (unsigned int level, unsigned int version);


  /**
   * Creates a new Trigger using the given SBMLNamespaces object
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
   */
  Trigger (SBMLNamespaces* sbmlns);


  /**
   * Destroys this Trigger.
   */
  virtual ~Trigger ();


  /**
   * Copy constructor; creates a copy of this Trigger.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  Trigger (const Trigger& orig);


  /**
   * Assignment operator
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  Trigger& operator=(const Trigger& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of Trigger.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this Trigger.
   *
   * @return a (deep) copy of this Trigger.
   */
  virtual Trigger* clone () const;


  /**
   * Get the mathematical formula for the trigger and return it
   * as an AST.
   * 
   * @return the math of this Trigger.
   */
  const ASTNode* getMath () const;


  /**
   * (SBML Level&nbsp;3 only) Get the value of the "initialValue" attribute
   * of this Trigger.
   * 
   * @return the boolean value stored as the "initialValue" attribute value
   * in this Trigger.
   * 
   * @note The attribute "initialValue" is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present in lower Levels of SBML.
   */
  bool getInitialValue () const;


  /**
   * (SBML Level&nbsp;3 only) Get the value of the "persistent" attribute
   * of this Trigger.
   * 
   * @return the boolean value stored as the "persistent" attribute value
   * in this Trigger.
   * 
   * @note The attribute "persistent" is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present in lower Levels of SBML.
   */
  bool getPersistent () const;


  /**
   * Predicate to test whether the math for this trigger is set.
   *
   * @return @c true if the formula (meaning the "math" subelement) of
   * this Trigger is set, @c false otherwise.
   */
  bool isSetMath () const;


  /**
   * (SBML Level&nbsp;3 only) Predicate to test whether the "initialValue"
   * attribute for this trigger is set.
   *
   * @return @c true if the initialValue attribute of
   * this Trigger is set, @c false otherwise.
   * 
   * @note The attribute "initialValue" is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present in lower Levels of SBML.
   */
  bool isSetInitialValue () const;


  /**
   * (SBML Level&nbsp;3 only) Predicate to test whether the "persistent"
   * attribute for this trigger is set.
   *
   * @return @c true if the persistent attribute of
   * this Trigger is set, @c false otherwise.
   * 
   * @note The attribute "persistent" is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present in lower Levels of SBML.
   */
  bool isSetPersistent () const;


  /**
   * Sets the trigger expression of this Trigger instance to a copy of the given
   * ASTNode.
   *
   * @param math an ASTNode representing a formula tree.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   */
  int setMath (const ASTNode* math);

 
  /**
   * (SBML Level&nbsp;3 only) Sets the "initialValue" attribute of this Trigger instance.
   *
   * @param initialValue a boolean representing the initialValue to be set.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * 
   * @note The attribute "initialValue" is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present in lower Levels of SBML.
   */
  int setInitialValue (bool initialValue);


  /**
   * (SBML Level&nbsp;3 only) Sets the "persistent" attribute of this Trigger instance.
   *
   * @param persistent a boolean representing the persistent value to be set.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * 
   * @note The attribute "persistent" is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present in lower Levels of SBML.
   */
  int setPersistent (bool persistent);

#if (0)
  /** @cond doxygen-libsbml-internal */

  /**
   * Sets the parent SBMLDocument of this SBML object.
   *
   * @param d the SBMLDocument to use.
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets the parent SBML object of this SBML object.
   *
   * @param sb the SBML object to use
   */
  virtual void setParentSBMLObject (SBase* sb);

  /** @endcond */
#endif

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
   * Returns the XML element name of this object, which for Trigger, is
   * always @c "trigger".
   * 
   * @return the name of this element, i.e., @c "trigger". 
   */
  virtual const std::string& getElementName () const;


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
   * all the required elements for this Trigger object
   * have been set.
   *
   * @note The required elements for a Trigger object are:
   * @li "math"
   *
   * @return a boolean value indicating whether all the required
   * elements for this object have been defined.
   */
  virtual bool hasRequiredElements() const ;


  /**
   * Predicate returning @c true if
   * all the required attributes for this Trigger object
   * have been set.
   *
   * @note The required attributes for a Trigger object are:
   * @li "persistent" (required in SBML Level&nbsp;3)
   * @li "initialValue" (required in SBML Level&nbsp;3)
   *
   * @return a boolean value indicating whether all the required
   * attributes for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const ;

  /**
   * Finds this Trigger's Event parent and calls unsetTrigger() on it, indirectly deleting itself.  Overridden from the SBase function since the parent is not a ListOf.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int removeFromParentAndDelete();


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

  bool mInitialValue;
  bool mPersistent;
  bool mIsSetInitialValue;
  bool mIsSetPersistent;

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


LIBSBML_EXTERN
Trigger_t *
Trigger_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Trigger_t *
Trigger_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
Trigger_free (Trigger_t *t);


LIBSBML_EXTERN
Trigger_t *
Trigger_clone (const Trigger_t *t);


LIBSBML_EXTERN
const XMLNamespaces_t *
Trigger_getNamespaces(Trigger_t *c);


LIBSBML_EXTERN
const ASTNode_t *
Trigger_getMath (const Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_getInitialValue (const Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_getPersistent (const Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_isSetMath (const Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_isSetInitialValue (const Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_isSetPersistent (const Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_setMath (Trigger_t *t, const ASTNode_t *math);


LIBSBML_EXTERN
int
Trigger_setInitialValue (Trigger_t *t, int initialValue);


LIBSBML_EXTERN
int
Trigger_setPersistent (Trigger_t *t, int persistent);


LIBSBML_EXTERN
int
Trigger_hasRequiredAttributes (Trigger_t *t);


LIBSBML_EXTERN
int
Trigger_hasRequiredElements (Trigger_t *t);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* Trigger_h */
