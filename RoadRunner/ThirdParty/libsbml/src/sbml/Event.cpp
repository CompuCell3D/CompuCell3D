/**
 * @file    Event.cpp
 * @brief   Implementations of Event and ListOfEvents.
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
 * ---------------------------------------------------------------------- -->*/

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/math/FormulaParser.h>
#include <sbml/math/MathML.h>
#include <sbml/math/ASTNode.h>

#include <sbml/SBO.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/SBMLDocument.h>
#include <sbml/SBMLError.h>
#include <sbml/Model.h>
#include <sbml/EventAssignment.h>
#include <sbml/Event.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

Event::Event (unsigned int level, unsigned int version) :
   SBase ( level, version )
 , mId                       ( ""   )
 , mName                     ( ""   )
 , mTrigger                  ( NULL    )
 , mDelay                    ( NULL    )
 , mPriority                 ( NULL    )
 , mUseValuesFromTriggerTime ( true )
 , mIsSetUseValuesFromTriggerTime ( false )
 , mExplicitlySetUVFTT (false )
 , mEventAssignments(level, version)
{
  mInternalIdOnly = false;
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();
  // before level 3 useValuesFromTriggerTime was set by default
  if (level < 3)
  {
    mIsSetUseValuesFromTriggerTime = true;
  }
  connectToChild();
}


Event::Event (SBMLNamespaces * sbmlns) :
   SBase                     ( sbmlns )
 , mId                       ( ""   )
 , mName                     ( ""   )
 , mTrigger                  ( NULL    )
 , mDelay                    ( NULL    )
 , mPriority                 ( NULL    )
 , mUseValuesFromTriggerTime ( true )
 , mIsSetUseValuesFromTriggerTime (false )
 , mExplicitlySetUVFTT (false )
 , mEventAssignments(sbmlns)
{
  mInternalIdOnly = false;

  if (!hasValidLevelVersionNamespaceCombination())
  {
    throw SBMLConstructorException(getElementName(), sbmlns);
  }

  if (sbmlns->getLevel() < 3)
  {
    mIsSetUseValuesFromTriggerTime = true;
  }
  connectToChild();
  loadPlugins(sbmlns);
}


/*
 * Destroys this Event.
 */
Event::~Event ()
{
  delete mTrigger;
  delete mDelay;
  delete mPriority;
}


/*
 * Copy constructor. Creates a copy of this Event.
 */
Event::Event (const Event& orig) :
   SBase                     ( orig )
 , mTrigger                  ( NULL    )
 , mDelay                    ( NULL    )
 , mPriority                 ( NULL    )
 , mEventAssignments         ( orig.mEventAssignments         )
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mId                            = orig.mId;  
    mName                          = orig.mName;
    mTimeUnits                     = orig.mTimeUnits;
    mUseValuesFromTriggerTime      = orig.mUseValuesFromTriggerTime ;
    mIsSetUseValuesFromTriggerTime = orig.mIsSetUseValuesFromTriggerTime ;
    mExplicitlySetUVFTT            = orig.mExplicitlySetUVFTT;
    mInternalIdOnly                = orig.mInternalIdOnly;
 
    if (orig.mTrigger != NULL) 
    {
      mTrigger = new Trigger(*orig.getTrigger());
    }
    if (orig.mDelay != NULL) 
    {
      mDelay = new Delay(*orig.getDelay());
    }
    if (orig.mPriority != NULL) 
    {
      mPriority = new Priority(*orig.getPriority());
    }
  }
  
  connectToChild();
}
 

/*
 * Assignment operator
 */
Event& Event::operator=(const Event& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    this->SBase::operator =(rhs);
   
    mId = rhs.mId;
    mName = rhs.mName;
    mTimeUnits        = rhs.mTimeUnits        ;
    mUseValuesFromTriggerTime = rhs.mUseValuesFromTriggerTime;
    mIsSetUseValuesFromTriggerTime = rhs.mIsSetUseValuesFromTriggerTime;
    mExplicitlySetUVFTT = rhs.mExplicitlySetUVFTT;
    mInternalIdOnly   = rhs.mInternalIdOnly   ;
    mEventAssignments = rhs.mEventAssignments ;

    delete mTrigger;
    if (rhs.mTrigger != NULL) 
    {
      mTrigger = new Trigger(*rhs.getTrigger());
    }
    else
    {
      mTrigger = NULL;
    }

    delete mDelay;
    if (rhs.mDelay != NULL) 
    {
      mDelay = new Delay(*rhs.getDelay());
    }
    else
    {
      mDelay = NULL;
    }

    delete mPriority;
    if (rhs.mPriority != NULL) 
    {
      mPriority = new Priority(*rhs.getPriority());
    }
    else
    {
      mPriority = NULL;
    }
  }

  connectToChild();

  return *this;
}


/*
 * Accepts the given SBMLVisitor.
 *
 * @return the result of calling <code>v.visit()</code>, which indicates
 * whether or not the Visitor would like to visit the Model's next Event
 * (if available).
 */
bool
Event::accept (SBMLVisitor& v) const
{
  bool result = v.visit(*this);

  if (mTrigger != NULL) mTrigger->accept(v);
  
  if (mDelay != NULL) mDelay->accept(v);

  if (mPriority != NULL) mPriority->accept(v);

  mEventAssignments.accept(v);

  return result;
}


/*
 * @return a (deep) copy of this Event.
 */
Event*
Event::clone () const
{
  return new Event(*this);
}


SBase*
Event::getElementBySId(std::string id)
{
  if (id.empty()) return NULL;
  SBase* obj = NULL;
  if (mTrigger != NULL) {
    if (mTrigger->getId() == id) return mTrigger;
    obj = mTrigger->getElementBySId(id);
    if (obj != NULL) return obj;
  }
  if (mDelay != NULL) {
    if (mDelay->getId() == id) return mDelay;
    obj = mDelay->getElementBySId(id);
    if (obj != NULL) return obj;
  }
  if (mPriority != NULL) {
    if (mPriority->getId() == id) return mPriority;
    obj = mPriority->getElementBySId(id);
    if (obj != NULL) return obj;
  }
  if (mEventAssignments.getId() == id) return &mEventAssignments;
  obj = mEventAssignments.getElementBySId(id);
  if (obj != NULL) return obj;
  return getElementFromPluginsBySId(id);
}


SBase*
Event::getElementByMetaId(std::string metaid)
{
  if (metaid.empty()) return NULL;
  SBase* obj = NULL;
  if (mTrigger != NULL) {
    if (mTrigger->getMetaId() == metaid) return mTrigger;
    obj = mTrigger->getElementByMetaId(metaid);
    if (obj != NULL) return obj;
  }
  if (mDelay != NULL) {
    if (mDelay->getMetaId() == metaid) return mDelay;
    obj = mDelay->getElementByMetaId(metaid);
    if (obj != NULL) return obj;
  }
  if (mPriority != NULL) {
    if (mPriority->getMetaId() == metaid) return mPriority;
    obj = mPriority->getElementByMetaId(metaid);
    if (obj != NULL) return obj;
  }
  if (mEventAssignments.getMetaId() == metaid) return &mEventAssignments;
  obj = mEventAssignments.getElementByMetaId(metaid);
  if (obj != NULL) return obj;
  return getElementFromPluginsByMetaId(metaid);
}

List*
Event::getAllElements()
{
  List* ret = new List();
  List* sublist = NULL;
  if (mTrigger != NULL) {
    ret->add(mTrigger);
    sublist = mTrigger->getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mDelay != NULL) {
    ret->add(mDelay);
    sublist = mDelay->getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mPriority != NULL) {
    ret->add(mPriority);
    sublist = mPriority->getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }
  if (mEventAssignments.size() > 0) {
    ret->add(&mEventAssignments);
    sublist = mEventAssignments.getAllElements();
    ret->transferFrom(sublist);
    delete sublist;
  }

  sublist = getAllElementsFromPlugins();
  ret->transferFrom(sublist);
  delete sublist;
  return ret;
}

/*
 * @return the id of this SBML object.
 */
const string&
Event::getId () const
{
  return mId;
}


/*
 * @return the name of this SBML object.
 */
const string&
Event::getName () const
{
  return (getLevel() == 1) ? mId : mName;
}


/*
 * @return the trigger of this Event.
 */
const Trigger*
Event::getTrigger () const
{
  return mTrigger;
}


/*
 * @return the trigger of this Event.
 */
Trigger*
Event::getTrigger ()
{
  return mTrigger;
}


/*
 * @return the delay of this Event.
 */
const Delay*
Event::getDelay () const
{
  return mDelay;
}


/*
 * @return the delay of this Event.
 */
Delay*
Event::getDelay ()
{
  return mDelay;
}


/*
 * @return the delay of this Event.
 */
const Priority*
Event::getPriority () const
{
  return mPriority;
}


/*
 * @return the delay of this Event.
 */
Priority*
Event::getPriority ()
{
  return mPriority;
}


/*
 * @return the timeUnits of this Event
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Versions 3 cannot contain it.
 */
const string&
Event::getTimeUnits () const
{
  return mTimeUnits;
}


/*
 * Returns the value of the "useValuesFromTriggerTime" attribute of this Event.
 */
bool 
Event::getUseValuesFromTriggerTime () const
{
  return mUseValuesFromTriggerTime;
}

  
/*
 * @return true if the id of this SBML object is set, false
 * otherwise.
 */
bool
Event::isSetId () const
{
  return (mId.empty() == false);
}


/*
 * @return true if the name of this SBML object is set, false
 * otherwise.
 */
bool
Event::isSetName () const
{
  return (getLevel() == 1) ? (mId.empty() == false) : 
                            (mName.empty() == false);
}


/*
 * @return true if the trigger of this Event is set, false otherwise.
 */
bool
Event::isSetTrigger () const
{
  return (mTrigger != NULL);
}


/*
 * @return true if the delay of this Event is set, false otherwise.
 */
bool
Event::isSetDelay () const
{
  return (mDelay != NULL);
}


/*
 * @return true if the priority of this Event is set, false otherwise.
 */
bool
Event::isSetPriority () const
{
  return (mPriority != NULL);
}


/*
 * @return true if the timeUnits of this Event is set, false
 * otherwise.
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Version 3 cannot contain it.
 */
bool
Event::isSetTimeUnits () const
{
  return (mTimeUnits.empty() == false);
}


/*
 * @return true if the mUseValuesFromTriggerTime of this Event is set, false otherwise.
 */
bool
Event::isSetUseValuesFromTriggerTime () const
{
  return mIsSetUseValuesFromTriggerTime;
}


/*
 * Sets the id of this SBML object to a copy of sid.
 */
int
Event::setId (const std::string& sid)
{
  /* since the setId function has been used as an
   * alias for setName we cant require it to only
   * be used on a L2 model
   */
/*  if (getLevel() == 1)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
*/
  if (&(sid) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (!(SyntaxChecker::isValidInternalSId(sid)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mId = sid;
    mInternalIdOnly = false;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the name of this SBML object to a copy of name.
 */
int
Event::setName (const std::string& name)
{
  /* if this is setting an L2 name the type is string
   * whereas if it is setting an L1 name its type is SId
   */
  if (&(name) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() == 1)
  {
    if (!(SyntaxChecker::isValidInternalSId(name)))
    {
      return LIBSBML_INVALID_ATTRIBUTE_VALUE;
    }
    else
    {
      mId = name;
      return LIBSBML_OPERATION_SUCCESS;
    }
  }
  else
  {
    mName = name;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the trigger of this Event to a copy of the given Trigger.
 */
int
Event::setTrigger (const Trigger* trigger)
{
  if (mTrigger == trigger)
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (trigger == NULL)
  {
    delete mTrigger;
    mTrigger = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (getLevel() != trigger->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != trigger->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else
  {
    delete mTrigger;
    mTrigger = (trigger != NULL) ? 
                static_cast<Trigger*>( trigger->clone() ) : NULL;

    if (mTrigger != NULL) mTrigger->connectToParent(this);
    
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the delay of this Event to a copy of the given Delay.
 */
int
Event::setDelay (const Delay* delay)
{
  if (mDelay == delay) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (delay == NULL)
  {
    delete mDelay;
    mDelay = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (getLevel() != delay->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != delay->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else
  {
    delete mDelay;
    mDelay = (delay != NULL) ? static_cast<Delay*>( delay->clone() ) : NULL;

    if (mDelay != NULL) mDelay->connectToParent(this);
    
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the prioirty of this Event to a copy of the given Priority.
 */
int
Event::setPriority (const Priority* priority)
{
  if (getLevel() < 3)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (mPriority == priority) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (priority == NULL)
  {
    delete mPriority;
    mPriority = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (getLevel() != priority->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != priority->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else
  {
    delete mPriority;
    mPriority = (priority != NULL) ? 
                 static_cast<Priority*>( priority->clone() ) : NULL;

    if (mPriority != NULL) mPriority->connectToParent(this);
    
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the timeUnits of this Event to a copy of sid.
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Version 3 cannot contain it.
 */
int
Event::setTimeUnits (const std::string& sid)
{
  if (&(sid) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else if (getLevel() == 2 && getVersion() > 2)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else if (!(SyntaxChecker::isValidInternalSId(sid)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mTimeUnits = sid;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Sets the "useValuesFromTriggerTime" attribute of this Event to a @p value.
 */
int 
Event::setUseValuesFromTriggerTime (bool value)
{
  if (getLevel() == 2 && getVersion() < 4)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
  else
  {
    mUseValuesFromTriggerTime = value;
    mIsSetUseValuesFromTriggerTime = true;
    mExplicitlySetUVFTT = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Unsets the id of this SBML object.
 */
int
Event::unsetId ()
{
  mId.erase();

  if (mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the name of this SBML object.
 */
int
Event::unsetName ()
{
  if (getLevel() == 1) 
  {
    mId.erase();
  }
  else 
  {
    mName.erase();
  }

  if (getLevel() == 1 && mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (mName.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the delay of this Event.
 */
int
Event::unsetDelay ()
{
  delete mDelay;
  mDelay = NULL;

  if (mDelay == NULL) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the priority of this Event.
 */
int
Event::unsetPriority ()
{
  delete mPriority;
  mPriority = NULL;

  if (mPriority == NULL) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the trigger of this Event.
 */
int
Event::unsetTrigger ()
{
  delete mTrigger;
  mTrigger = NULL;

  if (mTrigger == NULL) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Unsets the timeUnits of this Event.
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Version 3 cannot contain it.
 */
int
Event::unsetTimeUnits ()
{
  if (getLevel() == 2 && getVersion() > 2)
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }

  mTimeUnits.erase();
  
  if (mTimeUnits.empty()) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/*
 * Appends a copy of the given EventAssignment to this Event.
 */
int
Event::addEventAssignment (const EventAssignment* ea)
{
  if (ea == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!(ea->hasRequiredAttributes()) || !(ea->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != ea->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != ea->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (matchesRequiredSBMLNamespacesForAddition(static_cast<const SBase *>(ea)) == false)
  {
    return LIBSBML_NAMESPACES_MISMATCH;
  }
  else if (getEventAssignment(ea->getVariable()) != NULL)
  {
    // an eventAssignment for this variable already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mEventAssignments.append(ea);

    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * Creates a new EventAssignment, adds it to this Event's list of event
 * assignments and returns it.
 */
EventAssignment*
Event::createEventAssignment ()
{
  EventAssignment* ea = NULL;

  try
  {
    ea = new EventAssignment(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }
  
  if (ea != NULL) mEventAssignments.appendAndOwn(ea);

  return ea;
}


/*
 * Creates a new Trigger, adds it to this Event
 * and returns it.
 */
Trigger*
Event::createTrigger ()
{
  delete mTrigger;
  mTrigger = NULL;
  
  try
  {
    mTrigger = new Trigger(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }

  if (mTrigger != NULL)
  {
    mTrigger->connectToParent(this);
  }

  return mTrigger;
}


/*
 * Creates a new Delay, adds it to this Event
 * and returns it.
 */
Delay*
Event::createDelay ()
{
  delete mDelay;
  mDelay = 0;
  
  try
  {
    mDelay = new Delay(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }

  if (mDelay)
  {
    mDelay->connectToParent(this);
  }

  return mDelay;
}


/*
 * Creates a new Priority, adds it to this Event
 * and returns it.
 */
Priority*
Event::createPriority ()
{
  delete mPriority;
  mPriority = NULL;
  
  try
  {
    mPriority = new Priority(getSBMLNamespaces());
  }
  catch (...)
  {
    /* here we do not create a default object as the level/version must
     * match the parent object
     *
     * so do nothing
     */
  }

  if (mPriority != NULL)
  {
    mPriority->connectToParent(this);
  }

  return mPriority;
}


/*
 * @return the list of EventAssignments for this Event.
 */
const ListOfEventAssignments*
Event::getListOfEventAssignments () const
{
  return &mEventAssignments;
}


/*
 * @return the list of EventAssignments for this Event.
 */
ListOfEventAssignments*
Event::getListOfEventAssignments ()
{
  return &mEventAssignments;
}


/*
 * @return the nth EventAssignment of this Event.
 */
const EventAssignment*
Event::getEventAssignment (unsigned int n) const
{
  return static_cast<const EventAssignment*>( mEventAssignments.get(n) );
}


/*
 * @return the nth EventAssignment of this Event.
 */
EventAssignment*
Event::getEventAssignment (unsigned int n)
{
  return static_cast<EventAssignment*>( mEventAssignments.get(n) );
}


/*
 * @return the EventAssignment for the given variable, or @c NULL if no such
 * EventAssignment exits.
 */
const EventAssignment*
Event::getEventAssignment (const std::string& variable) const
{
  return (&variable != NULL ) ? 
  static_cast<const EventAssignment*>( mEventAssignments.get(variable) ) : NULL;
}


/*
 * @return the EventAssignment for the given variable, or @c NULL if no such
 * EventAssignment exits.
 */
EventAssignment*
Event::getEventAssignment (const std::string& variable)
{
  return (&variable != NULL ) ? 
    static_cast<EventAssignment*>( mEventAssignments.get(variable) ) : NULL;
}


/*
 * @return the number of EventAssignments in this Event.
 */
unsigned int
Event::getNumEventAssignments () const
{
  return mEventAssignments.size();
}


/**
 * Removes the nth EventAssignment object from this Event object and
 * returns a pointer to it.
 */
EventAssignment* 
Event::removeEventAssignment (unsigned int n)
{
  return mEventAssignments.remove(n);  
}


/**
 * Removes the EventAssignment object with the given "variable" attribute 
 * from this Event object and returns a pointer to it.
 */
EventAssignment* 
Event::removeEventAssignment (const std::string& variable)
{
  return (&variable != NULL) ? mEventAssignments.remove(variable) : NULL;  
}


/** @cond doxygen-libsbml-internal */

/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Event::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);

  mEventAssignments.setSBMLDocument(d);
  if (mTrigger != NULL) mTrigger->setSBMLDocument(d);
  if (mDelay != NULL) mDelay->setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
  */
void
Event::connectToChild()
{
  mEventAssignments.connectToParent(this);
  if (mTrigger != NULL) mTrigger->connectToParent(this);
  if (mDelay != NULL)   mDelay->connectToParent(this);
  if (mPriority != NULL)   mPriority->connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePackage function)
 */
void 
Event::enablePackageInternal(const std::string& pkgURI, const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mEventAssignments.enablePackageInternal(pkgURI,pkgPrefix,flag);
  if (mTrigger) mTrigger->enablePackageInternal(pkgURI,pkgPrefix,flag);
  if (mDelay)   mDelay->enablePackageInternal(pkgURI,pkgPrefix,flag);
  if (mPriority)   mPriority->enablePackageInternal(pkgURI,pkgPrefix,flag);
}


/** @endcond */


/*
 * @return the typecode (int) of this SBML object or SBML_UNKNOWN
 * (default).
 *
 * @see getElementName()
 */
int
Event::getTypeCode () const
{
  return SBML_EVENT;
}


/*
 * @return the name of this element ie "event".
 */
const string&
Event::getElementName () const
{
  static const string name = "event";
  return name;
}


bool 
Event::hasRequiredAttributes() const
{
  bool allPresent = true;

  /* required attributes for event: useValuesFromtriggerTime (L3 ->) */
  if (getLevel() > 2)
  {
    if(!isSetUseValuesFromTriggerTime())
      allPresent = false;
  }

  return allPresent;
}


bool 
Event::hasRequiredElements() const
{
  bool allPresent = true;

  /* required attributes for event: trigger; 
   * listOfEventAssignments (not L3)
  */

  if (!isSetTrigger())
    allPresent = false;

  if (getLevel() < 3 && getNumEventAssignments() == 0)
    allPresent = false;

  return allPresent;
}


/** @cond doxygen-libsbml-internal */

/*
 * sets the mInternalIdOnly flag
 */
void 
Event::setInternalIdOnly()
{
  mInternalIdOnly = true;
}

/*
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or @c NULL if the token was not recognized.
 */
SBase*
Event::createObject (XMLInputStream& stream)
{
  SBase* object = NULL;

  const string& name = stream.peek().getName();
  if (name == "listOfEventAssignments") 
  {
    if (mEventAssignments.size() != 0)
    {
      if (getLevel() < 3)
        logError(NotSchemaConformant, getLevel(), getVersion(),
	       "Only one <listOfEventAssignments> elements is permitted "
	       "in a single <event> element.");
      else
        logError(OneListOfEventAssignmentsPerEvent, getLevel(), getVersion());
    }
    object = &mEventAssignments;
  }
  else if (name == "trigger")
  {
    if (mTrigger != NULL)
    {
      if (getLevel() < 3)
        logError(NotSchemaConformant, getLevel(), getVersion(),
	       "Only one <trigger> elements is permitted "
	       "in a single <event> element.");
      else
        logError(MissingTriggerInEvent, getLevel(), getVersion());
    }

    delete mTrigger;

    try
    {
      mTrigger = new Trigger(getSBMLNamespaces());
    }
    catch (SBMLConstructorException*)
    {
      mTrigger = new Trigger(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    catch ( ... )
    {
      mTrigger = new Trigger(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    object = mTrigger;
  }
  else if (name == "delay")
  {
    if (mDelay != NULL)
    {
      if (getLevel() < 3)
        logError(NotSchemaConformant, getLevel(), getVersion(),
	       "Only one <delay> element is permitted in a single "
	       "<event> element.");
      else
        logError(OnlyOneDelayPerEvent, getLevel(), getVersion());
    }
    delete mDelay;

    try
    {
      mDelay = new Delay(getSBMLNamespaces());
    }
    catch (SBMLConstructorException*)
    {
      mDelay = new Delay(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    catch ( ... )
    {
      mDelay = new Delay(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    object = mDelay;
  }
  else if (name == "priority")
  {
    if (mPriority != NULL)
    {
      if (getLevel() < 3)
        logError(NotSchemaConformant, getLevel(), getVersion(),
	       "Priority is not a valid component for this level/version.");
      else 
      {
        logError(OnlyOnePriorityPerEvent, getLevel(), getVersion());
      }
      
    }
    delete mPriority;

    try
    {
      mPriority = new Priority(getSBMLNamespaces());
    }
    catch (SBMLConstructorException*)
    {
      mPriority = new Priority(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    catch ( ... )
    {
      mPriority = new Priority(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    object = mPriority;
  }

  return object;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */

/**
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
Event::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  const unsigned int level = getLevel();
  const unsigned int version = getVersion();

  switch (level)
  {
  case 2:
    attributes.add("name");
    attributes.add("id");
    if (version < 3)
    {
      attributes.add("timeUnits");
    }
    if (version == 2)
    {
      attributes.add("sboTerm");
    }
    if (version == 4)
    {
      attributes.add("useValuesFromTriggerTime");
    }
    break;
  case 3:
  default:
    attributes.add("name");
    attributes.add("id");
    attributes.add("useValuesFromTriggerTime");
    break;
  }
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Event::readAttributes (const XMLAttributes& attributes,
                       const ExpectedAttributes& expectedAttributes)
{
  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  SBase::readAttributes(attributes, expectedAttributes);

  switch (level)
  {
  case 1:
    logError(NotSchemaConformant, level, version,
	      "Event is not a valid component for this level/version.");
    break;
  case 2:
    readL2Attributes(attributes);
    break;
  case 3:
  default:
    readL3Attributes(attributes);
    break;
  }
}
/** @endcond */

/** @cond doxygen-libsbml-internal */

/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Event::readL2Attributes (const XMLAttributes& attributes)
{
  const unsigned int level = getLevel();
  const unsigned int version = getVersion();
  //
  // id: SId  { use="optional" }  (L2v1 ->)
  //
  bool assigned = attributes.readInto("id", mId, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mId.size() == 0)
  {
    logEmptyString("id", level, version, "<event>");
  }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);

  //
  // name: string  { use="optional" }  (L2v1 ->)
  //
  attributes.readInto("name", mName, getErrorLog(), false, getLine(), getColumn());

  //
  // timeUnits: SId  { use="optional" }  (L2v1, L2v2)
  // removed in l2v3
  //
  if (version < 3)
  {
    assigned = attributes.readInto("timeUnits", mTimeUnits, getErrorLog(), false, getLine(), getColumn());
    if (assigned && mTimeUnits.size() == 0)
    {
      logEmptyString("timeUnits", level, version, "<event>");
    }
    if (!SyntaxChecker::isValidInternalUnitSId(mTimeUnits))
    {
      logError(InvalidUnitIdSyntax);
    }
  }

  //
  // sboTerm: SBOTerm { use="optional" }  (L2v2 ->)
  //
  if (version == 2) 
    mSBOTerm = SBO::readTerm(attributes, this->getErrorLog(), level, version,
				getLine(), getColumn());

  //
  // useValuesFromTriggerTime: bool {use="optional" default="true"} (L2V4 ->)
  // useValuesFromTriggerTime: bool {use="optional" } (L3 ->)
  //
  if (version  == 4)
  {
    mExplicitlySetUVFTT = attributes.readInto("useValuesFromTriggerTime", 
                                                mUseValuesFromTriggerTime, getErrorLog(), false, getLine(), getColumn());
  }
}

/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Event::readL3Attributes (const XMLAttributes& attributes)
{
  const unsigned int level = getLevel();
  const unsigned int version = getVersion();

  //
  // id: SId  { use="optional" }  (L2v1 ->)
  //
  bool assigned = attributes.readInto("id", mId, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mId.size() == 0)
  {
    logEmptyString("id", level, version, "<event>");
  }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);

  //
  // name: string  { use="optional" }  (L2v1 ->)
  //
  attributes.readInto("name", mName, getErrorLog(), false, getLine(), getColumn());

  //
  //
  // useValuesFromTriggerTime: bool {use="required" } (L3 ->)
  //
  mIsSetUseValuesFromTriggerTime = attributes.readInto(
      "useValuesFromTriggerTime", mUseValuesFromTriggerTime, 
       getErrorLog(),false, getLine(), getColumn());
 
  if (!mIsSetUseValuesFromTriggerTime)
  {
    logError(AllowedAttributesOnEvent, level, version);
  }

}

/** @endcond */




/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write their XML attributes
 * to the XMLOutputStream.  Be sure to call your parents implementation
 * of this method as well.
 */
void
Event::writeAttributes (XMLOutputStream& stream) const
{
  const unsigned int level = getLevel();
  const unsigned int version = getVersion();

  /* invalid level/version */
  if (level < 2)
  {
    return;
  }

  SBase::writeAttributes(stream);

  //
  // sboTerm: SBOTerm { use="optional" }  (L2v2 ->)
  //
  // sboTerm for L2V3 or later is written in SBase::writeAttributes()
  //
  if ( (level == 2) && (version == 2) )
  {
    SBO::writeTerm(stream, mSBOTerm);
  }


  //
  //
  // id: SId  { use="optional" }  (L2v1 ->)
  //
  if (!mInternalIdOnly)
    stream.writeAttribute("id", mId);

  //
  // name: string  { use="optional" }  (L2v1->)
  //
  stream.writeAttribute("name", mName);

  if (level == 2 && version < 3)
  {
    //
    // timeUnits: SId  { use="optional" }  (L2v1, L2v2)
    // removed in l2v3
    //
    stream.writeAttribute("timeUnits", mTimeUnits);
  }


  // useValuesFromTriggerTime: bool {use="optional" default="true"} (L2V4 ->)
  // useValuesFromTriggerTime: bool {use="required"} (L3 ->)
  //
  if (level == 2 && version == 4)
  {
    if (isExplicitlySetUVFTT() || !mUseValuesFromTriggerTime)
      stream.writeAttribute("useValuesFromTriggerTime", 
                            mUseValuesFromTriggerTime);
  }
  else if (level > 2)
  {
    // in L3 only write it out if it has been set
    if (isSetUseValuesFromTriggerTime())
      stream.writeAttribute("useValuesFromTriggerTime", 
                          mUseValuesFromTriggerTime);
  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
Event::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  if (mTrigger != NULL)
  {
    mTrigger->write(stream);
  }

  if (mDelay != NULL)
  {
    mDelay->write(stream);
  }

  if (getLevel() > 2 && mPriority != NULL)
  {
    mPriority->write(stream);
  }

  if ( getNumEventAssignments() > 0 ) mEventAssignments.write(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}
/** @endcond */


/*
 * Creates a new ListOfEvents items.
 */
ListOfEvents::ListOfEvents (unsigned int level, unsigned int version)
  : ListOf(level,version)
{
}


/*
 * Creates a new ListOfEvents items.
 */
ListOfEvents::ListOfEvents (SBMLNamespaces* sbmlns)
  : ListOf(sbmlns)
{
  loadPlugins(sbmlns);
}


/*
 * @return a (deep) copy of this ListOfEvents.
 */
ListOfEvents*
ListOfEvents::clone () const
{
  return new ListOfEvents(*this);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfEvents::getItemTypeCode () const
{
  return SBML_EVENT;
}


/*
 * @return the name of this element ie "listOfEvents".
 */
const string&
ListOfEvents::getElementName () const
{
  static const string name = "listOfEvents";
  return name;
}


/* return nth item in list */
Event *
ListOfEvents::get(unsigned int n)
{
  return static_cast<Event*>(ListOf::get(n));
}


/* return nth item in list */
const Event *
ListOfEvents::get(unsigned int n) const
{
  return static_cast<const Event*>(ListOf::get(n));
}


/**
 * Used by ListOf::get() to lookup an SBase based by its id.
 */
struct IdEqE : public unary_function<SBase*, bool>
{
  const string& id;

  IdEqE (const string& id) : id(id) { }
  bool operator() (SBase* sb) 
       { return static_cast <Event *> (sb)->getId() == id; }
};


/* return item by id */
Event*
ListOfEvents::get (const std::string& sid)
{
  return (&sid != NULL) ? const_cast<Event*>( 
    static_cast<const ListOfEvents&>(*this).get(sid) ) : NULL;
}


/* return item by id */
const Event*
ListOfEvents::get (const std::string& sid) const
{
  if (&sid == NULL) return NULL;
  vector<SBase*>::const_iterator result;

  result = find_if( mItems.begin(), mItems.end(), IdEqE(sid) );
  return (result == mItems.end()) ? NULL : static_cast <Event*> (*result);
}


/* Removes the nth item from this list */
Event*
ListOfEvents::remove (unsigned int n)
{
   return static_cast<Event*>(ListOf::remove(n));
}


/* Removes item in this list by id */
Event*
ListOfEvents::remove (const std::string& sid)
{
  SBase* item = NULL;
  vector<SBase*>::iterator result;

  if (&(sid) != NULL)
  {
    result = find_if( mItems.begin(), mItems.end(), IdEqE(sid) );

    if (result != mItems.end())
    {
      item = *result;
      mItems.erase(result);
    }
  }

  return static_cast <Event*> (item);
}


/** @cond doxygen-libsbml-internal */
/*
 * @return the ordinal position of the element with respect to its siblings
 * or -1 (default) to indicate the position is not significant.
 */
int
ListOfEvents::getElementPosition () const
{
  return 12;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or @c NULL if the token was not recognized.
 */
SBase*
ListOfEvents::createObject (XMLInputStream& stream)
{
  const string& name   = stream.peek().getName();
  SBase*        object = NULL;


  if (name == "event")
  {
    try
    {
      object = new Event(getSBMLNamespaces());
    }
    catch (SBMLConstructorException*)
    {
      object = new Event(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    catch ( ... )
    {
      object = new Event(SBMLDocument::getDefaultLevel(),
        SBMLDocument::getDefaultVersion());
    }
    
    if (object != NULL) mItems.push_back(object);
  }

  return object;
}
/** @endcond */



/** @cond doxygen-c-only */


/**
 * Creates a new Event_t structure using the given SBML @p level
 * and @p version values.
 *
 * @param level an unsigned int, the SBML Level to assign to this
 * Event
 *
 * @param version an unsigned int, the SBML Version to assign to this
 * Event
 *
 * @return a pointer to the newly created Event_t structure.
 *
 * @note Once a Event has been added to an SBMLDocument, the @p
 * level and @p version for the document @em override those used to create
 * the Event.  Despite this, the ability to supply the values at
 * creation time is an important aid to creating valid SBML.  Knowledge of
 * the intended SBML Level and Version  determine whether it is valid to
 * assign a particular value to an attribute, or whether it is valid to add
 * an object to an existing SBMLDocument.
 */
LIBSBML_EXTERN
Event_t *
Event_create (unsigned int level, unsigned int version)
{
  try
  {
    Event* obj = new Event(level,version);
    return obj;
  }
  catch (SBMLConstructorException)
  {
    return NULL;
  }
}


/**
 * Creates a new Event_t structure using the given
 * SBMLNamespaces_t structure.
 *
 * @param sbmlns SBMLNamespaces, a pointer to an SBMLNamespaces structure
 * to assign to this Event
 *
 * @return a pointer to the newly created Event_t structure.
 *
 * @note Once a Event has been added to an SBMLDocument, the
 * @p sbmlns namespaces for the document @em override those used to create
 * the Event.  Despite this, the ability to supply the values at creation time
 * is an important aid to creating valid SBML.  Knowledge of the intended SBML
 * Level and Version determine whether it is valid to assign a particular value
 * to an attribute, or whether it is valid to add an object to an existing
 * SBMLDocument.
 */
LIBSBML_EXTERN
Event_t *
Event_createWithNS (SBMLNamespaces_t* sbmlns)
{
  try
  {
    Event* obj = new Event(sbmlns);
    return obj;
  }
  catch (SBMLConstructorException)
  {
    return NULL;
  }
}


/**
 * Frees the given Event_t structure.
 *
 * @param e the Event_t structure to free.
 */
LIBSBML_EXTERN
void
Event_free (Event_t *e)
{
  if (e != NULL)
  delete e;
}


/**
 * Returns a copy of the given Event_t structure.
 *
 * @param e the Event_t structure to copy.
 * 
 * @return a (deep) copy of the Event_t.
 */
LIBSBML_EXTERN
Event_t *
Event_clone (const Event_t *e)
{
  return (e != NULL) ? static_cast<Event_t*>( e->clone() ) : NULL;
}


/**
 * Returns a list of XMLNamespaces_t associated with this Event_t
 * structure.
 *
 * @param e the Event_t structure
 * 
 * @return pointer to the XMLNamespaces_t structure associated with 
 * this SBML object
 */
LIBSBML_EXTERN
const XMLNamespaces_t *
Event_getNamespaces(Event_t *e)
{
  return (e != NULL) ? e->getNamespaces() : NULL;
}


/**
 * Takes an Event_t structure and returns its identifier.
 *
 * @param p the Event_t structure whose identifier is sought
 * 
 * @return the identifier of this Event_t, as a pointer to a string.
 */
LIBSBML_EXTERN
const char *
Event_getId (const Event_t *e)
{
  return (e != NULL && e->isSetId()) ? e->getId().c_str() : NULL;
}


/**
 * Takes a Event_t structure and returns its name.
 *
 * @param p the Event_t whose name is sought.

 * @return the name of this Event_t, as a pointer to a string.
 */
LIBSBML_EXTERN
const char *
Event_getName (const Event_t *e)
{
  return (e != NULL && e->isSetName()) ? e->getName().c_str() : NULL;
}


/**
 * Takes an Event_t structure and returns its Trigger_t structure.
 *
 * @param e the Event_t structure whose trigger definition is sought.
 * 
 * @return the Trigger_t of this Event.
 */
LIBSBML_EXTERN
Trigger_t *
Event_getTrigger (Event_t *e)
{
  return (e != NULL) ? e->getTrigger() : NULL;
}


/**
 * Takes an Event_t structure and returns its Delay_t structure.
 *
 * @param e the Event_t structure whose delay definition is sought.
 * 
 * @return the Delay_t of this Event.
 */
LIBSBML_EXTERN
Delay_t *
Event_getDelay (Event_t *e)
{
  return (e != NULL) ? e->getDelay() : NULL;
}


/**
 * Takes an Event_t structure and returns its Priority_t structure.
 *
 * @param e the Event_t structure whose delay definition is sought.
 * 
 * @return the Priority_t of this Event.
 */
LIBSBML_EXTERN
Priority_t *
Event_getPriority (Event_t *e)
{
  return (e != NULL) ? e->getPriority() : NULL;
}


/**
 * Takes an Event_t structure and returns the value of its "timeUnits"
 * attribute.
 *
 * @param e the Event_t structure whose "timeUnits" value is sought
 * 
 * @return the timeUnits of this Event
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Versions 3 and 4 cannot contain it.
 */
LIBSBML_EXTERN
const char *
Event_getTimeUnits (const Event_t *e)
{
  return (e != NULL && e->isSetTimeUnits()) ? e->getTimeUnits().c_str() : NULL;
}


/**
 * Takes an Event_t structure and returns the value of its "useValuesFromTriggerTime"
 * attribute.
 *
 * @param e the Event_t structure whose "useValuesFromTriggerTime" value is sought
 * 
 * @return the useValuesFromTriggerTime of this Event
 */
LIBSBML_EXTERN
int
Event_getUseValuesFromTriggerTime (const Event_t *e)
{
  return (e != NULL) ? static_cast<int> (e->getUseValuesFromTriggerTime()) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's identifier is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if the "id" attribute of the given
 * Event_t structure is set, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
Event_isSetId (const Event_t *e)
{
  return (e != NULL) ? static_cast<int>( e->isSetId() ) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's name is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if the "name" attribute of the given
 * Event_t structure is set, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
Event_isSetName (const Event_t *e)
{
  return (e != NULL) ? static_cast<int>( e->isSetName() ) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's trigger is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if a Trigger_t structure is assigned to
 * the given Event_t structure, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
Event_isSetTrigger (const Event_t *e)
{
  return (e != NULL) ? static_cast<int>( e->isSetTrigger() ) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's delay is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if a Delay_t structure is assigned to
 * the given Event_t structure, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
Event_isSetDelay (const Event_t *e)
{
  return (e != NULL) ? static_cast<int>( e->isSetDelay() ) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's priority is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if a Priority_t structure is assigned to
 * the given Event_t structure, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
Event_isSetPriority (const Event_t *e)
{
  return (e != NULL) ? static_cast<int>( e->isSetPriority() ) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's "timeUnits" attribute is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if a value for the "timeUnits" attribute is
 * assigned in the given Event_t structure, zero (false) otherwise.
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Versions 3 and 4 cannot contain it.
 */
LIBSBML_EXTERN
int
Event_isSetTimeUnits (const Event_t *e)
{
  return (e != NULL) ? static_cast<int>( e->isSetTimeUnits() ) : 0;
}


/**
 * Predicate returning @c true or @c false depending on whether the given
 * Event_t structure's useValuesFromTriggerTime attribute is set.
 *
 * @param e the Event_t structure to query
 * 
 * @return @c non-zero (true) if the "useValuesFromTriggerTime" attribute of the given
 * Event_t structure is set, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
Event_isSetUseValuesFromTriggerTime (const Event_t *e)
{
  return (e != NULL) ? 
    static_cast<int>( e->isSetUseValuesFromTriggerTime() ) : 0;
}


/**
 * Assigns the identifier of an Event_t structure.
 *
 * This makes a copy of the string passed in the param @p sid.
 *
 * @param e the Event_t structure to set.
 * @param sid the string to use as the identifier.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with an id of NULL is equivalent to
 * unsetting the "id" attribute.
 */
LIBSBML_EXTERN
int
Event_setId (Event_t *e, const char *sid)
{
  if (e != NULL)
    return (sid == NULL) ? e->unsetId() : e->setId(sid);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Sets the name of this Event to a copy of @p name.
 *
 * @param e the Event_t structure to set
 * @param name the name to assign to this Event_t's "name" attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 *
 * @note Using this function with the name set to NULL is equivalent to
 * unsetting the "name" attribute.
 */
LIBSBML_EXTERN
int
Event_setName (Event_t *e, const char *name)
{
  if (e != NULL)
    return (name == NULL) ? e->unsetName() : e->setName(name);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Sets the trigger of this Event to a copy of the given Trigger.
 *
 * @param e the Event_t structure to set
 * @param trigger the Trigger_t structure to use.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 */
LIBSBML_EXTERN
int
Event_setTrigger (Event_t *e, const Trigger_t *trigger)
{
  if (e != NULL)
    return e->setTrigger(trigger);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Sets the delay of this Event to a copy of the given Delay.
 * 
 * @param e the Event_t structure to set
 * @param delay the Delay_t structure to use.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 */
LIBSBML_EXTERN
int
Event_setDelay (Event_t *e, const Delay_t *delay)
{
  if (e != NULL)
    return e->setDelay(delay);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Sets the priority of this Event to a copy of the given Priority.
 * 
 * @param e the Event_t structure to set
 * @param priority the Priority_t structure to use.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 */
LIBSBML_EXTERN
int
Event_setPriority (Event_t *e, const Priority_t *priority)
{
  if (e != NULL)
    return e->setPriority(priority);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Sets the "timeUnits" attribute of this Event to a copy of @p sid.
 * 
 * @param e the Event_t structure to set
 * @param sid the identifier of the units to use as the value of the
 * "timeUnits" attribute
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Versions 3 and 4 cannot contain it.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 *
 * @note Using this function with an id of NULL is equivalent to
 * unsetting the "timeUnits" attribute.
 */
LIBSBML_EXTERN
int
Event_setTimeUnits (Event_t *e, const char *sid)
{
  if (e != NULL)
    return (sid == NULL) ? e->unsetTimeUnits() : e->setTimeUnits(sid);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Sets the "useValuesFromTriggerTime" attribute of this Event to a @p value.
 * 
 * @param e the Event_t structure to set
 * @param value the value of the "useValuesFromTriggerTime" attribute
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 */
LIBSBML_EXTERN
int
Event_setUseValuesFromTriggerTime (Event_t *e, int value)
{
  if (e != NULL)
    return e->setUseValuesFromTriggerTime( static_cast<bool>(value) );
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "id" attribute of this Event_t structure.
 *
 * @param e the Event_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Event_unsetId (Event_t *e)
{
  if (e != NULL)
    return e->unsetId();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "name" attribute of this Event_t structure.
 *
 * @param e the Event_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Event_unsetName (Event_t *e)
{
  if (e != NULL)
    return e->unsetName();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the delay of this Event.
 *
 * @param e the Event_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Event_unsetDelay (Event_t *e)
{
  if (e != NULL)
    return e->unsetDelay();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the priority of this Event.
 *
 * @param e the Event_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Event_unsetPriority (Event_t *e)
{
  if (e != NULL)
    return e->unsetPriority();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Unsets the "timeUnits" attribute of this Event.
 *
 * @param e the Event_t structure to unset
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 *
 * @warning Definitions of Event in SBML Level 2 Versions 1 and 2
 * included the additional attribute called "timeUnits", but it was
 * removed in SBML Level 2 Version 3.  LibSBML supports this attribute
 * for compatibility with previous versions of SBML Level 2, but its use
 * is discouraged since models in Level 2 Versions 3 and 4 cannot contain it.
 */
LIBSBML_EXTERN
int
Event_unsetTimeUnits (Event_t *e)
{
  if (e != NULL)
    return e->unsetTimeUnits();
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
  * Predicate returning @c true or @c false depending on whether
  * all the required attributes for this Event object
  * have been set.
  *
  * @note The required attributes for a Event object are:
  * @li useValuesfromTriggerTime ( L3 onwards )
  */
LIBSBML_EXTERN
int
Event_hasRequiredAttributes (Event_t *e)
{
  return (e != NULL) ? static_cast <int> (e->hasRequiredAttributes()) : 0;
}



/**
  * Predicate returning @c true or @c false depending on whether
  * all the required elements for this Event object
  * have been set.
  *
  * @note The required elements for a Event object are:
  * @li trigger
  * @li listOfEventAssignments (requirement removed in L3)
  */
LIBSBML_EXTERN
int
Event_hasRequiredElements (Event_t *e)
{
  return (e != NULL) ? static_cast <int> (e->hasRequiredElements() ) : 0;
}



/**
 * Appends a copy of the given EventAssignment_t structure to this Event_t
 * structure.
 *
 * @param e the Event_t structure to which the event assignment should be
 * added
 *
 * @param ea an EventAssignment_t structure to add
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_LEVEL_MISMATCH
 * @li LIBSBML_VERSION_MISMATCH
 * @li LIBSBML_DUPLICATE_OBJECT_ID
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
Event_addEventAssignment (Event_t *e, const EventAssignment_t *ea)
{
  if (e != NULL)
    return e->addEventAssignment(ea);
  else
    return LIBSBML_INVALID_OBJECT;
}


/**
 * Creates a new, empty EventAssignment_t structure, adds it to this
 * Event's list of event assignments, and returns the EventAssignment_t.
 *
 * @param e the Event_t structure to which the event assignment should be
 * added
 */
LIBSBML_EXTERN
EventAssignment_t *
Event_createEventAssignment (Event_t *e)
{
  return (e != NULL) ? e->createEventAssignment() : NULL;
}


/**
 * Creates a new, empty Trigger_t structure, adds it to this
 * Event, and returns the Trigger_t.
 *
 * @param e the Event_t structure to which the trigger should be
 * added
 */
LIBSBML_EXTERN
Trigger_t *
Event_createTrigger (Event_t *e)
{
  return (e != NULL) ? e->createTrigger() : NULL;
}


/**
 * Creates a new, empty Delay_t structure, adds it to this
 * Event, and returns the Delay_t.
 *
 * @param e the Event_t structure to which the delay should be
 * added
 */
LIBSBML_EXTERN
Delay_t *
Event_createDelay (Event_t *e)
{
  return (e != NULL) ? e->createDelay() : NULL;
}


/**
 * Creates a new, empty Priority_t structure, adds it to this
 * Event, and returns the Priority_t.
 *
 * @param e the Event_t structure to which the priority should be
 * added
 */
LIBSBML_EXTERN
Priority_t *
Event_createPriority (Event_t *e)
{
  return (e != NULL) ? e->createPriority() : NULL;
}


/**
 * Get the list of EventAssignment_t structures from this Event_t
 * structure.
 *
 * @param e the Event_t structure to use.
 *
 * @return the list of EventAssignments for this Event.
 */
LIBSBML_EXTERN
ListOf_t *
Event_getListOfEventAssignments (Event_t *e)
{
  return (e != NULL) ? e->getListOfEventAssignments() : NULL;
}


/**
 * Return a specific EventAssignment_t structure of this Event_t.
 *
 * @param e the Event_t structure to use
 *
 * @param n an integer, the index of the EventAssignment_t structure to return
 * 
 * @return the nth EventAssignment_t of this Event.
 */
LIBSBML_EXTERN
EventAssignment_t *
Event_getEventAssignment (Event_t *e, unsigned int n)
{
  return (e != NULL) ? e->getEventAssignment(n) : NULL;
}


/**
 * Return the event assignment indicated by the given @p variable.
 *
 * @param e the Event_t structure to use
 *
 * @param variable a string, the identifier of the variable whose
 * EventAssignment_t is being sought.
 *
 * @return the EventAssignment_t for the given variable, or @c NULL if no such
 * EventAssignment_t exits.
 */
LIBSBML_EXTERN
EventAssignment_t *
Event_getEventAssignmentByVar (Event_t *e, const char *variable)
{
  return (e != NULL && variable != NULL) ? 
           e->getEventAssignment(variable) : NULL;
}


/**
 * Returns the number of EventAssignment_t objects attached to this
 * Event.
 *
 * @param e the Event_t structure to use
 * 
 * @return the number of EventAssignment_t structures in this Event.
 */
LIBSBML_EXTERN
unsigned int
Event_getNumEventAssignments (const Event_t *e)
{
  return (e != NULL) ? e->getNumEventAssignments() : SBML_INT_MAX;
}


/**
 * Removes the nth EventAssignment_t object from this Event_t object and
 * returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Event_t structure
 * @param n the integer index of the EventAssignment_t sought
 *
 * @return the EventAssignment_t object removed.  As mentioned above, 
 * the caller owns the returned item. @c NULL is returned if the given index 
 * is out of range.
 */
LIBSBML_EXTERN
EventAssignment_t *
Event_removeEventAssignment (Event_t *e, unsigned int n)
{
  return (e != NULL) ? e->removeEventAssignment(n) : NULL;
}


/**
 * Removes the EventAssignment_t object with the given "variable" attribute
 * from this Event_t object and returns a pointer to it.
 *
 * The caller owns the returned object and is responsible for deleting it.
 *
 * @param m the Event_t structure
 * @param sid the string of the "variable" attribute of the EventAssignment_t sought
 *
 * @return the EventAssignment_t object removed.  As mentioned above, the 
 * caller owns the returned object. @c NULL is returned if no EventAssignment_t
 * object with the "variable" attribute exists in this Event_t object.
 */
LIBSBML_EXTERN
EventAssignment_t *
Event_removeEventAssignmentByVar (Event_t *e, const char *variable)
{
  if (e != NULL)
    return (variable != NULL) ? e->removeEventAssignment(variable) : NULL;
  else
    return NULL;
}


/**
 * @return item in this ListOfEvent with the given id or @c NULL if no such
 * item exists.
 */
LIBSBML_EXTERN
Event_t *
ListOfEvents_getById (ListOf_t *lo, const char *sid)
{
  if (lo != NULL)
    return (sid != NULL) ? 
      static_cast <ListOfEvents *> (lo)->get(sid) : NULL;
  else
    return NULL;
}


/**
 * Removes item in this ListOf items with the given id or @c NULL if no such
 * item exists.  The caller owns the returned item and is responsible for
 * deleting it.
 */
LIBSBML_EXTERN
Event_t *
ListOfEvents_removeById (ListOf_t *lo, const char *sid)
{
  if (lo != NULL)
    return (sid != NULL) ? 
                         static_cast <ListOfEvents *> (lo)->remove(sid) : NULL;
  else
    return NULL;
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END


