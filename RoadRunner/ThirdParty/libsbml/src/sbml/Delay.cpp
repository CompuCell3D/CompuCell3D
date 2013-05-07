/**
 * @file    Delay.cpp
 * @brief   Implementation of Delay.
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
 * ---------------------------------------------------------------------- -->*/

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/math/FormulaFormatter.h>
#include <sbml/math/FormulaParser.h>
#include <sbml/math/MathML.h>
#include <sbml/math/ASTNode.h>

#include <sbml/SBO.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/SBMLDocument.h>
#include <sbml/SBMLError.h>
#include <sbml/Model.h>
#include <sbml/Parameter.h>
#include <sbml/Delay.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

Delay::Delay (unsigned int level, unsigned int version) :
   SBase ( level, version )
 , mMath      ( NULL              )
 , mInternalId ( "" )
{
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();
}


Delay::Delay (SBMLNamespaces * sbmlns) :
   SBase ( sbmlns )
 , mMath      ( NULL              )
 , mInternalId ( "" )
{
  if (!hasValidLevelVersionNamespaceCombination())
  {
    throw SBMLConstructorException(getElementName(), sbmlns);
  }

  loadPlugins(sbmlns);
}


/*
 * Destroys this Delay.
 */
Delay::~Delay ()
{
  delete mMath;
}


/*
 * Copy constructor. Creates a copy of this Delay.
 */
Delay::Delay (const Delay& orig) :
   SBase          ( orig )
 , mMath          ( NULL    )
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mInternalId = orig.mInternalId;
 
    if (orig.mMath != NULL) 
    {
      mMath = orig.mMath->deepCopy();
      mMath->setParentSBMLObject(this);
    }
  }
}


/*
 * Assignment operator
 */
Delay& Delay::operator=(const Delay& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    this->SBase::operator =(rhs);
    this->mInternalId = rhs.mInternalId;

    delete mMath;
    if (rhs.mMath != NULL) 
    {
      mMath = rhs.mMath->deepCopy();
      mMath->setParentSBMLObject(this);
    }
    else
    {
      mMath = NULL;
    }
  }

  return *this;
}


/*
 * Accepts the given SBMLVisitor.
 */
bool
Delay::accept (SBMLVisitor& v) const
{
  return v.visit(*this);
}


/*
 * @return a (deep) copy of this Delay.
 */
Delay*
Delay::clone () const
{
  return new Delay(*this);
}



/*
 * @return the math of this Delay.
 */
const ASTNode*
Delay::getMath () const
{
  return mMath;
}


/*
 * @return true if the math (or equivalently the formula) of this
 * Delay is set, false otherwise.
 */
bool
Delay::isSetMath () const
{
  return (mMath != NULL);
}


/*
 * Sets the math of this Delay to a copy of the given ASTNode.
 */
int
Delay::setMath (const ASTNode* math)
{
  if (mMath == math) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (math == NULL)
  {
    delete mMath;
    mMath = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (!(math->isWellFormedASTNode()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else
  {
    delete mMath;
    mMath = (math != NULL) ? math->deepCopy() : NULL;
    if (mMath != NULL) mMath->setParentSBMLObject(this);
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
  * Calculates and returns a UnitDefinition that expresses the units
  * returned by the math expression of this InitialAssignment.
  */
UnitDefinition * 
Delay::getDerivedUnitDefinition()
{
  if (!isSetMath())
    return NULL;
  /* if we have the whole model but it is not in a document
   * it is still possible to determine the units
   */
  Model * m = static_cast <Model *> (getAncestorOfType(SBML_MODEL));

  if (m != NULL)
  {
    if (!m->isPopulatedListFormulaUnitsData())
    {
      m->populateListFormulaUnitsData();
    }
    
    if (m->getFormulaUnitsData(getId(), SBML_EVENT))
    {
      return m->getFormulaUnitsData(getId(), SBML_EVENT)
                                             ->getUnitDefinition();
    }
    else
    {
      return NULL;
    } 
  }
  else
  {
    return NULL;
  }
}


/*
  * Constructs and returns a UnitDefinition that expresses the units of this 
  * Compartment.
  */
const UnitDefinition *
Delay::getDerivedUnitDefinition() const
{
  return const_cast <Delay *> (this)->getDerivedUnitDefinition();
}


/*
 * Predicate returning @c true or @c false depending on whether 
 * the math expression of this InitialAssignment contains
 * parameters/numbers with undeclared units that cannot be ignored.
 */
bool 
Delay::containsUndeclaredUnits()
{
  if (!isSetMath())
    return false;
  /* if we have the whole model but it is not in a document
   * it is still possible to determine the units
   */
  Model * m = static_cast <Model *> (getAncestorOfType(SBML_MODEL));

  if (m != NULL)
  {
    if (!m->isPopulatedListFormulaUnitsData())
    {
      m->populateListFormulaUnitsData();
    }
    
    if (m->getFormulaUnitsData(getId(), SBML_EVENT))
    {
      return m->getFormulaUnitsData(getId(), SBML_EVENT)
      ->getContainsUndeclaredUnits();
    }
    else
    {
      return false;
    }  
  }
  else
  {
    return false;
  }
}


/*
 * Predicate returning @c true if 
 * the math expression of this InitialAssignment contains
 * parameters/numbers with undeclared units that cannot be ignored.
 */
bool 
Delay::containsUndeclaredUnits() const
{
  return const_cast<Delay *> (this)->containsUndeclaredUnits();
}


/*
 * @return the typecode (int) of this SBML object or SBML_UNKNOWN
 * (default).
 *
 * @see getElementName()
 */
int
Delay::getTypeCode () const
{
  return SBML_DELAY;
}


/*
 * @return the name of this element ie "delay".
 */
const string&
Delay::getElementName () const
{
  static const string name = "delay";
  return name;
}


/** @cond doxygen-libsbml-internal */
/*
 * @return the ordinal position of the element with respect to its siblings
 * or -1 (default) to indicate the position is not significant.
 */
int
Delay::getElementPosition () const
{
  return 1;
}
/** @endcond */


bool 
Delay::hasRequiredElements() const
{
  bool allPresent = true;

  /* required attributes for delay: math */

  if (!isSetMath())
    allPresent = false;

  return allPresent;
}

int Delay::removeFromParentAndDelete()
{
  SBase* parent = getParentSBMLObject();
  if (parent==NULL) return LIBSBML_OPERATION_FAILED;
  Event* parentEvent = static_cast<Event*>(parent);
  if (parentEvent == NULL) return LIBSBML_OPERATION_FAILED;
  return parentEvent->unsetDelay();
}


void
Delay::renameSIdRefs(std::string oldid, std::string newid)
{
  if (isSetMath()) {
    mMath->renameSIdRefs(oldid, newid);
  }
}

void 
Delay::renameUnitSIdRefs(std::string oldid, std::string newid)
{
  if (isSetMath()) {
    mMath->renameUnitSIdRefs(oldid, newid);
  }
}

/** @cond doxygen-libsbml-internal */
void 
Delay::replaceSIDWithFunction(const std::string& id, const ASTNode* function)
{
  if (isSetMath()) {
    if (mMath->getType() == AST_NAME && mMath->getId() == id) {
      delete mMath;
      mMath = function->deepCopy();
    }
    else {
      mMath->replaceIDWithFunction(id, function);
    }
  }
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read (and store) XHTML,
 * MathML, etc. directly from the XMLInputStream.
 *
 * @return true if the subclass read from the stream, false otherwise.
 */
bool
Delay::readOtherXML (XMLInputStream& stream)
{
  bool          read = false;
  const string& name = stream.peek().getName();

  if (name == "math")
  {
    // if this is level 1 there shouldnt be any math!!!
    if (getLevel() == 1) 
    {
      logError(NotSchemaConformant, getLevel(), getVersion(),
	       "SBML Level 1 does not support MathML.");
      delete mMath;
      return false;
    }

    if (mMath != NULL)
    {
      if (getLevel() < 3) 
      {
        logError(NotSchemaConformant, getLevel(), getVersion(),
	        "Only one <math> element is permitted inside a "
	        "particular containing element.");
      }
      else
      {
        logError(OneMathPerDelay, getLevel(), getVersion());
      }
    }
    /* check for MathML namespace 
     * this may be explicitly declared here
     * or implicitly declared on the whole document
     */
    const XMLToken elem = stream.peek();
    const std::string prefix = checkMathMLNamespace(elem);

    delete mMath;
    mMath = readMathML(stream, prefix);
    if (mMath != NULL) mMath->setParentSBMLObject(this);
    read  = true;
  }

  /* ------------------------------
   *
   *   (EXTENSION)
   *
   * ------------------------------ */
  if ( SBase::readOtherXML(stream) )
    read = true;

  return read;
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
Delay::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);
}


/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Delay::readAttributes (const XMLAttributes& attributes,
                       const ExpectedAttributes& expectedAttributes)
{
  const unsigned int level   = getLevel  ();
  const unsigned int version = getVersion();

  SBase::readAttributes(attributes, expectedAttributes);

  switch (level)
  {
  case 1:
    logError(NotSchemaConformant, level, version,
	      "Delay is not a valid component for this level/version.");
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
Delay::readL2Attributes (const XMLAttributes& attributes)
{
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Delay::readL3Attributes (const XMLAttributes& attributes)
{
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write their XML attributes
 * to the XMLOutputStream.  Be sure to call your parents implementation
 * of this method as well.
 */
void
Delay::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  const unsigned int level = getLevel();

  /* invalid level/version */
  if (level < 2)
  {
    return;
  }

  //
  // sboTerm: SBOTerm { use="optional" }  (L2v3 ->)
  // is written in SBase::writeAttributes()
  //

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
Delay::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  if ( getLevel() > 1 && isSetMath() ) writeMathML(getMath(), stream, getSBMLNamespaces());

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}
/** @endcond */


/** @cond doxygen-c-only */


/**
 * Creates a new Delay_t structure using the given SBML @p level
 * and @p version values.
 *
 * @param level an unsigned int, the SBML Level to assign to this
 * Delay
 *
 * @param version an unsigned int, the SBML Version to assign to this
 * Delay
 *
 * @return a pointer to the newly created Delay_t structure.
 *
 * @note Once a Delay has been added to an SBMLDocument, the @p
 * level and @p version for the document @em override those used to create
 * the Delay.  Despite this, the ability to supply the values at
 * creation time is an important aid to creating valid SBML.  Knowledge of
 * the intended SBML Level and Version  determine whether it is valid to
 * assign a particular value to an attribute, or whether it is valid to add
 * an object to an existing SBMLDocument.
 */
LIBSBML_EXTERN
Delay_t *
Delay_create (unsigned int level, unsigned int version)
{
  try
  {
    Delay* obj = new Delay(level,version);
    return obj;
  }
  catch (SBMLConstructorException)
  {
    return NULL;
  }
}


/**
 * Creates a new Delay_t structure using the given
 * SBMLNamespaces_t structure.
 *
 * @param sbmlns SBMLNamespaces, a pointer to an SBMLNamespaces structure
 * to assign to this Delay
 *
 * @return a pointer to the newly created Delay_t structure.
 *
 * @note Once a Delay has been added to an SBMLDocument, the
 * @p sbmlns namespaces for the document @em override those used to create
 * the Delay.  Despite this, the ability to supply the values at creation time
 * is an important aid to creating valid SBML.  Knowledge of the intended SBML
 * Level and Version determine whether it is valid to assign a particular value
 * to an attribute, or whether it is valid to add an object to an existing
 * SBMLDocument.
 */
LIBSBML_EXTERN
Delay_t *
Delay_createWithNS (SBMLNamespaces_t* sbmlns)
{
  try
  {
    Delay* obj = new Delay(sbmlns);
    return obj;
  }
  catch (SBMLConstructorException)
  {
    return NULL;
  }
}


/**
 * Frees the given Delay_t structure.
 *
 * @param t the Delay_t structure to free.
 */
LIBSBML_EXTERN
void
Delay_free (Delay_t *t)
{
  if (t != NULL)
  delete t;
}


/**
 * Creates and returns a deep copy of the given Delay_t structure.
 *
 * @param t the Delay_t structure to copy. 
 *
 * @return a (deep) copy of the given Delay_t structure @p t.
 */
LIBSBML_EXTERN
Delay_t *
Delay_clone (const Delay_t *t)
{
  return (t != NULL) ? t->clone() : NULL;
}


/**
 * Returns a list of XMLNamespaces_t associated with this Delay_t
 * structure.
 *
 * @param d the Delay_t structure
 * 
 * @return pointer to the XMLNamespaces_t structure associated with 
 * this SBML object
 */
LIBSBML_EXTERN
const XMLNamespaces_t *
Delay_getNamespaces(Delay_t *d)
{
  return (d != NULL) ? d->getNamespaces() : NULL;
}

/**
 * Get the mathematical formula for a Delay_t structure and return it as
 * as an ASTNode structure.
 *
 * @param t the Delay_t structure to query.
 * 
 * @return an ASTNode_t structure representing the expression tree.
 */
LIBSBML_EXTERN
const ASTNode_t *
Delay_getMath (const Delay_t *t)
{
  return (t != NULL) ? t->getMath() : NULL;
}


/**
 * Predicate to test whether the formula for the given Delay_t structure
 * is set.
 *
 * @param t the Delay_t structure to query
 *
 * @return @c true if the formula (meaning the @c math subelement) of
 * this Delay is set, @c false otherwise.
 */
LIBSBML_EXTERN
int
Delay_isSetMath (const Delay_t *t)
{
  return (t != NULL) ? static_cast<int>( t->isSetMath() ) : 0;
}


/**
 * Sets the delay expression of the given Delay_t instance to a copy of the
 * given ASTNode_t structure.
 *
 * @param t the Delay_t structure to set.
 * @param math an ASTNode representing a formula tree.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
Delay_setMath (Delay_t *t, const ASTNode_t *math)
{
  if (t != NULL)
    return t->setMath(math);
  else
    return LIBSBML_INVALID_OBJECT;

}

/**
  * Calculates and returns a UnitDefinition_t that expresses the units
  * returned by the math expression of this Delay_t.
  *
  * @return a UnitDefinition_t that expresses the units of the math 
  * expression of this Delay_t.
  *
  * Note that the functionality that facilitates unit analysis depends 
  * on the model as a whole.  Thus, in cases where the object has not 
  * been added to a model or the model itself is incomplete,
  * unit analysis is not possible and this method will return @c NULL.
  *
  * @note The units are calculated by applying the mathematics 
  * from the expression to the units of the <ci> elements used 
  * within the expression. Where there are parameters/numbers
  * with undeclared units the UnitDefinition_t returned by this
  * function may not accurately represent the units of the expression.
  *
  * @see Delay_containsUndeclaredUnits()
  */
LIBSBML_EXTERN
UnitDefinition_t * 
Delay_getDerivedUnitDefinition(Delay_t *d)
{
  return (d != NULL) ? d->getDerivedUnitDefinition() : NULL;
}


/**
  * Predicate returning @c true or @c false depending on whether 
  * the math expression of this Delay_t contains
  * parameters/numbers with undeclared units.
  * 
  * @return @c true if the math expression of this Delay_t
  * includes parameters/numbers 
  * with undeclared units, @c false otherwise.
  *
  * @note a return value of @c true indicates that the UnitDefinition_t
  * returned by the getDerivedUnitDefinition function may not 
  * accurately represent the units of the expression.
  *
  * @see Delay_getDerivedUnitDefinition()
  */
LIBSBML_EXTERN
int 
Delay_containsUndeclaredUnits(Delay_t *d)
{
  return (d != NULL) ? static_cast<int>(d->containsUndeclaredUnits()) : 0;
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END
