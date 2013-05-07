/**
 * @file    CVTerm.cpp
 * @brief   CVTerm I/O
 * @author  Sarah Keating
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
 * the Free Software Foundation.  A copy of the license agreement is
 * provided in the file named "LICENSE.txt" included with this software
 * distribution.  It is also available online at
 * http://sbml.org/software/libsbml/license.html
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */


#include <sbml/common/common.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLErrorLog.h>

#include <sbml/SBase.h>

#include <sbml/SBMLErrorLog.h>

#include <sbml/util/util.h>
#include <sbml/util/List.h>

#include <sbml/annotation/CVTerm.h>


/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN


/*
 * create a new CVTerm
 */
CVTerm::CVTerm(QualifierType_t type) :
    mHasBeenModified (false)

{
  mResources = new XMLAttributes();

  mQualifier = UNKNOWN_QUALIFIER;
  mModelQualifier = BQM_UNKNOWN;
  mBiolQualifier = BQB_UNKNOWN;

  setQualifierType(type);

}


/*
 * create a new CVTerm from an XMLNode
 * this assumes that the XMLNode has a prefix 
 * that represents a CV term
 */
CVTerm::CVTerm(const XMLNode node) :
    mHasBeenModified (false)
{
  const string& name = node.getName();
  const string& prefix = node.getPrefix();
  XMLNode Bag = node.getChild(0);

  mResources = new XMLAttributes();

  mQualifier = UNKNOWN_QUALIFIER;
  mModelQualifier = BQM_UNKNOWN;
  mBiolQualifier = BQB_UNKNOWN;

  if (prefix == "bqbiol")
  {
    setQualifierType(BIOLOGICAL_QUALIFIER);
    setBiologicalQualifierType(name);
  }
  else if (prefix == "bqmodel")
  {
    setQualifierType(MODEL_QUALIFIER);
    setModelQualifierType(name);
  }


  for (unsigned int n = 0; n < Bag.getNumChildren(); n++)
  {
    for (int b = 0; b < Bag.getChild(n).getAttributes().getLength(); b++)
    {
      addResource(Bag.getChild(n).getAttributes().getValue(b));
    }
  }

}


/*
 * destructor
 */
CVTerm::~CVTerm()
{
  delete mResources;
}

/*
 * Copy constructor; creates a copy of a CVTerm.
 * 
 * @param orig the CVTerm instance to copy.
 */
CVTerm::CVTerm(const CVTerm& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mQualifier      = orig.mQualifier;
    mModelQualifier = orig.mModelQualifier;
    mBiolQualifier  = orig.mBiolQualifier;
    mResources      = new XMLAttributes(*orig.mResources);
    mHasBeenModified = orig.mHasBeenModified;
  }
}

/*
 * Assignment operator for CVTerm.
 */
CVTerm& 
CVTerm::operator=(const CVTerm& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    mQualifier       = rhs.mQualifier;
    mModelQualifier  = rhs.mModelQualifier;
    mBiolQualifier   = rhs.mBiolQualifier;

    delete mResources;
    mResources=new XMLAttributes(*rhs.mResources);

    mHasBeenModified = rhs.mHasBeenModified;
  }

  return *this;
}





/*
 * clones the CVTerm
 */  
CVTerm* CVTerm::clone() const
{
    CVTerm* term=new CVTerm(*this);
    return term;
}


/*
 * set the qualifier type
 */
int 
CVTerm::setQualifierType(QualifierType_t type)
{
  mQualifier = type;
  
  if (mQualifier == MODEL_QUALIFIER)
  {
    mBiolQualifier = BQB_UNKNOWN;
  }
  else
  {
    mModelQualifier = BQM_UNKNOWN;
  }

  mHasBeenModified = true;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * set the model qualifier type
 * this should be consistent with the mQualifier == MODEL_QUALIFIER
 */
int 
CVTerm::setModelQualifierType(ModelQualifierType_t type)
{
  if (mQualifier == MODEL_QUALIFIER)
  {
    mModelQualifier = type;
    mBiolQualifier = BQB_UNKNOWN;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    mModelQualifier = BQM_UNKNOWN;
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
}


/*
 * set the biological qualifier type
 * this should be consistent with the mQualifier == BIOLOGICAL_QUALIFIER
 */
int 
CVTerm::setBiologicalQualifierType(BiolQualifierType_t type)
{
  if (mQualifier == BIOLOGICAL_QUALIFIER)
  {
    mBiolQualifier = type;
    mModelQualifier = BQM_UNKNOWN;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    mBiolQualifier = BQB_UNKNOWN;
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
}

/*
 * set the model qualifier type
 * this should be consistent with the mQualifier == MODEL_QUALIFIER
 */
int 
CVTerm::setModelQualifierType(const std::string& qualifier)
{
  ModelQualifierType_t type;
  if (&qualifier == NULL)
    type = BQM_UNKNOWN;
  else
    type = ModelQualifierType_fromString(qualifier.c_str());
  
  return setModelQualifierType(type);  
}


/*
 * set the biological qualifier type
 * this should be consistent with the mQualifier == BIOLOGICAL_QUALIFIER
 */
int 
CVTerm::setBiologicalQualifierType(const std::string& qualifier)
{
  BiolQualifierType_t type;
  if (&qualifier == NULL)
    type = BQB_UNKNOWN;
  else
    type = BiolQualifierType_fromString(qualifier.c_str());
  return setBiologicalQualifierType(type);
}

/*
 * gets the Qualifier type
 */
QualifierType_t 
CVTerm::getQualifierType()
{
  return mQualifier;
}


/*
 * gets the Model Qualifier type
 */
ModelQualifierType_t 
CVTerm::getModelQualifierType()
{
  return mModelQualifier;
}


/*
 * gets the biological Qualifier type
 */
BiolQualifierType_t 
CVTerm::getBiologicalQualifierType()
{
  return mBiolQualifier;
}


/*
 * gets the resources
 */
XMLAttributes * 
CVTerm::getResources()
{
  return mResources;
}

/*
 * gets the resources
 */
const XMLAttributes * 
CVTerm::getResources() const
{
  return mResources;
}


/*
 * Returns the number of resources for this %CVTerm.
 */
unsigned int 
CVTerm::getNumResources()
{
  return mResources->getLength();
}

  
/*
 * Returns the value of the nth resource for this %CVTerm.
 */
std::string
CVTerm::getResourceURI(unsigned int n)
{
  return mResources->getValue(n);
}

  
/*
 * adds a resource to the term
 */
int 
CVTerm::addResource(const std::string& resource)
{
  if (&resource == NULL || resource.empty())
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else
  {
    mHasBeenModified = true;
    return mResources->addResource("rdf:resource", resource);
  }
}


/*
 * removes a resource to the term
 */
int 
CVTerm::removeResource(std::string resource)
{
  int result = LIBSBML_INVALID_ATTRIBUTE_VALUE;
  for (int n = 0; n < mResources->getLength(); n++)
  {
    if (resource == mResources->getValue(n))
    {
      mHasBeenModified = true;
      result = mResources->removeResource(n);
    }
  }

  if (mResources->getLength() == 0)
  {
    if (getQualifierType() == MODEL_QUALIFIER)
    {
      setModelQualifierType(BQM_UNKNOWN);
      setQualifierType(UNKNOWN_QUALIFIER);
    }
    else
    {
      setBiologicalQualifierType(BQB_UNKNOWN);
      setQualifierType(UNKNOWN_QUALIFIER);
    }
  }

  return result;
}

/** @cond doxygen-libsbml-internal */
bool 
CVTerm::hasRequiredAttributes()
{
  bool valid = true;

  if (getQualifierType() == UNKNOWN_QUALIFIER)
  {
    valid = false;
  }
  else if (getQualifierType() == MODEL_QUALIFIER)
  {
    if (getModelQualifierType() == BQM_UNKNOWN)
    {
      valid = false;
    }
  }
  else
  {
    if (getBiologicalQualifierType() == BQB_UNKNOWN)
    {
      valid = false;
    }
  }

  if (valid)
  {
    if (getResources()->isEmpty())
    {
      valid = false;
    }
  }

  return valid;
}

bool
CVTerm::hasBeenModified()
{
  return mHasBeenModified;
}

void
CVTerm::resetModifiedFlags()
{
  mHasBeenModified = false;
}




/** @endcond */





/** @cond doxygen-c-only */


/**
 * Creates a new CVTerm_t with the given #QualifierType_t value @p type and
 * returns a pointer to it.
 *
 * The possible QualifierTypes are MODEL_QUALIFIER and BIOLOGICAL_QUALIFIER.  
 *
 * @param type a #QualifierType_t
 *
 * @return a pointer to the newly created CVTerm_t structure.
 */
LIBSBML_EXTERN
CVTerm_t*
CVTerm_createWithQualifierType(QualifierType_t type)
{
  return new(nothrow) CVTerm(type);
}

/**
 * Create a new CVTerm_t from the given XMLNode_t and returns a 
 * pointer to it.
 *
 * RDFAnnotations within a model are stored as a List of CVTerms.  This allows
 * the user to interact with the %CVTerms directly.  When LibSBML reads in a 
 * model containing RDFAnnotations it parses them into a %List of CVTerms and
 * when writing a model it parses the CVTerms into the appropriate annotation
 * structure.  This function creates a %CVTerm from the %XMLNode supplied.
 *
 * @param node an XMLNode_t representing a CVTerm_t.
 *
 * @return a pointer to the newly created CVTerm_t structure.
 *
 * @note this method assumes that the %XMLNode_t is of the correct form
 */
LIBSBML_EXTERN
CVTerm_t*
CVTerm_createFromNode(const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return new(nothrow) CVTerm(*node);
}


/**
 * Frees the given CVTerm_t structure.
 *
 * @param term the CVTerm_t structure to be freed.
 */
LIBSBML_EXTERN
void
CVTerm_free(CVTerm_t * term)
{
  if (term == NULL ) return;
  delete static_cast<CVTerm*>(term);
}

/**
 * Creates a deep copy of the given CVTerm_t structure
 * 
 * @param p the CVTerm_t structure to be copied
 * 
 * @return a (deep) copy of the given CVTerm_t structure.
 */
LIBSBML_EXTERN
CVTerm_t *
CVTerm_clone (const CVTerm_t* c)
{
  if (c == NULL) return NULL;
  return static_cast<CVTerm*>( c->clone() );
}


/**
 * Takes a CVTerm_t structure and returns its #QualifierType_t type.
 *
 * @param term the CVTerm_t structure whose #QualifierType_t value is sought
 *
 * @return the #QualifierType_t value of this CVTerm_t or UNKNOWN_QUALIFIER
 * (default).
 */
LIBSBML_EXTERN
QualifierType_t 
CVTerm_getQualifierType(CVTerm_t * term)
{
  if (term == NULL) return UNKNOWN_QUALIFIER;
  return term->getQualifierType();
}

/**
 * Takes a CVTerm_t structure and returns the #ModelQualifierType_t type.
 *
 * @param term the CVTerm_t structure whose #ModelQualifierType_t is sought.
 *
 * @return the #ModelQualifierType_t value of this CVTerm_t or BQM_UNKNOWN
 * (default).
 */
LIBSBML_EXTERN
ModelQualifierType_t 
CVTerm_getModelQualifierType(CVTerm_t * term)
{
  if (term == NULL) return BQM_UNKNOWN;
  return term->getModelQualifierType();
}

/**
 * Takes a CVTerm_t structure and returns the #BiolQualifierType_t.
 *
 * @param term the CVTerm_t structure whose #BiolQualifierType_t value is
 * sought.
 *
 * @return the #BiolQualifierType_t value of this CVTerm_t or BQB_UNKNOWN
 * (default).
 */
LIBSBML_EXTERN
BiolQualifierType_t 
CVTerm_getBiologicalQualifierType(CVTerm_t * term)
{
  if (term == NULL) return BQB_UNKNOWN;
  return term->getBiologicalQualifierType();
}

/**
 * Takes a CVTerm_t structure and returns the resources.
 * 
 * @param term the CVTerm_t structure whose resources are sought.
 *
 * @return the XMLAttributes_t that store the resources of this CVTerm_t.
 */
LIBSBML_EXTERN
XMLAttributes_t * 
CVTerm_getResources(CVTerm_t * term)
{
  if (term == NULL) return NULL;
  return term->getResources();
}

/**
 * Returns the number of resources for this %CVTerm.
 *
 * @param term the CVTerm_t structure whose resources are sought.
 * 
 * @return the number of resources in the set of XMLAttributes
 * of this %CVTerm.
 */
LIBSBML_EXTERN
unsigned int
CVTerm_getNumResources(CVTerm_t* term)
{
  if (term == NULL) return SBML_INT_MAX;
  return term->getNumResources();
}


/**
 * Returns the value of the nth resource for this %CVTerm.
 *
 * @param term the CVTerm_t structure
 * @param n the index of the resource to query
 *
 * @return string representing the value of the nth resource
 * in the set of XMLAttributes of this %CVTerm.
 *
 * @note Since the values of the resource attributes in a CVTerm
 * are URIs this is a convenience function to facilitate
 * interaction with the CVTerm class.
 */
LIBSBML_EXTERN
char *
CVTerm_getResourceURI(CVTerm_t * cv, unsigned int n)
{
  if (cv == NULL) return NULL;
  return cv->getResourceURI(n).empty() ? NULL : safe_strdup(cv->getResourceURI(n).c_str());
}


/**
 * Sets the "QualifierType_t" of this %CVTerm_t.
 *
 * @param term the CVTerm_t structure to set.
 * @param type the QualifierType_t 
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
CVTerm_setQualifierType(CVTerm_t * term, QualifierType_t type)
{
  if (term == NULL) return LIBSBML_INVALID_OBJECT;
  return term->setQualifierType(type);
}


/**
 * Sets the "ModelQualifierType_t" of this %CVTerm.
 *
 * @param term the CVTerm_t structure to set.
 * @param type the ModelQualifierType_t
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note if the QualifierType for this object is not MODEL_QUALIFIER
 * then the ModelQualifierType will default to BQM_UNKNOWN.
 */
LIBSBML_EXTERN
int 
CVTerm_setModelQualifierType(CVTerm_t * term, ModelQualifierType_t type)
{
  if (term == NULL) return LIBSBML_INVALID_OBJECT;
  return term->setModelQualifierType(type);
}


/**
 * Sets the "BiolQualifierType_t" of this %CVTerm_t.
 *
 * @param term the CVTerm_t structure to set.
 * @param type the BiolQualifierType_t
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note if the QualifierType for this object is not BIOLOGICAL_QUALIFIER
 * then the BiolQualifierType_t will default to BQB_UNKNOWN.
 */
LIBSBML_EXTERN
int 
CVTerm_setBiologicalQualifierType(CVTerm_t * term, BiolQualifierType_t type)
{
  if (term == NULL) return LIBSBML_INVALID_OBJECT;
  return term->setBiologicalQualifierType(type);
}

/**
 * Sets the "ModelQualifierType_t" of this %CVTerm.
 *
 * @param term the CVTerm_t structure to set.
 * @param qualifier the string representing a model qualifier
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note if the QualifierType for this object is not MODEL_QUALIFIER
 * then the ModelQualifierType will default to BQM_UNKNOWN.
 */
LIBSBML_EXTERN
int 
CVTerm_setModelQualifierTypeByString(CVTerm_t * term, const char* qualifier)
{
  if (term == NULL) return LIBSBML_INVALID_OBJECT;
  if (qualifier == NULL)
    return term->setModelQualifierType(BQM_UNKNOWN);
  else 
    return term->setModelQualifierType(qualifier);
}


/**
 * Sets the "BiolQualifierType_t" of this %CVTerm_t.
 *
 * @param term the CVTerm_t structure to set.
 * @param qualifier the string representing a biol qualifier
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note if the QualifierType for this object is not BIOLOGICAL_QUALIFIER
 * then the BiolQualifierType_t will default to BQB_UNKNOWN.
 */
LIBSBML_EXTERN
int 
CVTerm_setBiologicalQualifierTypeByString(CVTerm_t * term, const char* qualifier)
{
  if (term == NULL) return LIBSBML_INVALID_OBJECT;
  if (qualifier == NULL)
    return term->setBiologicalQualifierType(BQB_UNKNOWN);
  else 
    return term->setBiologicalQualifierType(qualifier);
}

/**
 * Adds a resource to the CVTerm_t.
 *
 * @param term the CVTerm_t structure to set.
 * @param resource string representing the resource 
 * e.g. http://www.geneontology.org/#GO:0005892
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 *
 * @note this method adds the name "rdf:resource" to the attribute prior
 * to adding it to the resources in this CVTerm.
 */
LIBSBML_EXTERN
int 
CVTerm_addResource(CVTerm_t * term, const char * resource)
{
  if (term == NULL) return LIBSBML_OPERATION_FAILED;
  return term->addResource(resource);
}

/**
 * Removes a resource from the CVTerm_t.
 *
 * @param term the CVTerm_t structure.
 * @param resource string representing the resource 
 * e.g. http://www.geneontology.org/#GO:0005892
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
CVTerm_removeResource(CVTerm_t * term, const char * resource)
{
  if (term == NULL) return LIBSBML_INVALID_OBJECT;
  return term->removeResource(resource);
}


LIBSBML_EXTERN
int
CVTerm_hasRequiredAttributes(CVTerm_t *cvt)
{
  if (cvt == NULL) return (int)false;
  return static_cast<int> (cvt->hasRequiredAttributes());
}



static
const char* MODEL_QUALIFIER_STRINGS[] =
{
    "is"
  , "isDescribedBy"
  , "isDerivedFrom"
};

static
const char* BIOL_QUALIFIER_STRINGS[] =
{
    "is"
  , "hasPart"
  , "isPartOf"
  , "isVersionOf"
  , "hasVersion"
  , "isHomologTo"
  , "isDescribedBy"
  , "isEncodedBy"
  , "encodes"
  , "occursIn"
  , "hasProperty"
  , "isPropertyOf"    
};


/**
 * This method takes a model qualifier type code and returns a string 
 * representing the code.
 *
 * This method takes a model qualifier type as argument 
 * and returns a string name corresponding to that code.  For example, 
 * passing it the qualifier <code>BQM_IS_DESCRIBED_BY</code> will return 
 * the string "<code>isDescribedBy</code>". 
 *
 * @return a human readable qualifier name for the given type.
 *
 * @note The caller does not own the returned string and is therefore not
 * allowed to modify it.
 */
LIBSBML_EXTERN
const char* 
ModelQualifierType_toString(ModelQualifierType_t type)
{
  int max = BQM_UNKNOWN;

  if (type < BQM_IS || type >= max)
  {
      return NULL;
  }

  return MODEL_QUALIFIER_STRINGS[type];
}

/**
 * This method takes a biol qualifier type code and returns a string 
 * representing the code.
 *
 * This method takes a biol qualifier type as argument 
 * and returns a string name corresponding to that code.  For example, 
 * passing it the qualifier <code>BQB_HAS_VERSION</code> will return 
 * the string "<code>hasVersion</code>". 
 *
 * @return a human readable qualifier name for the given type.
 *
 * @note The caller does not own the returned string and is therefore not
 * allowed to modify it.
 */
LIBSBML_EXTERN
const char* 
BiolQualifierType_toString(BiolQualifierType_t type)
{
  int max = BQB_UNKNOWN;

  if (type < BQB_IS || type >= max)
  {
      return NULL;
  }

  return BIOL_QUALIFIER_STRINGS[type];
}

/**
 * This method takes a a string and returns a model qualifier
 * representing the string.
 *
 * This method takes a string as argument and returns a model qualifier type 
 * corresponding to that string.  For example, passing it the string 
 * "<code>isDescribedBy</code>" will return the qualifier 
 * <code>BQM_IS_DESCRIBED_BY</code>. 
 *
 * @return a qualifier for the given human readable qualifier name.
 *
 */
LIBSBML_EXTERN
ModelQualifierType_t 
ModelQualifierType_fromString(const char* s)
{
  if (s == NULL) return BQM_UNKNOWN;

  int max = BQM_UNKNOWN;
  for (int i = 0; i < max; i++)
  {
    if (strcmp(MODEL_QUALIFIER_STRINGS[i], s) == 0)
      return (ModelQualifierType_t)i;
  }
  return BQM_UNKNOWN;
}

/**
 * This method takes a a string and returns a biol qualifier
 * representing the string.
 *
 * This method takes a string as argument and returns a biol qualifier type 
 * corresponding to that string.  For example, passing it the string 
 * "<code>hasVersion</code>" will return the qualifier 
 * <code>BQB_HAS_VERSION</code>. 
 *
 * @return a qualifier for the given human readable qualifier name.
 *
 */
LIBSBML_EXTERN
BiolQualifierType_t 
BiolQualifierType_fromString(const char* s)
{  
  if (s == NULL) return BQB_UNKNOWN;

  int max = BQB_UNKNOWN;
  for (int i = 0; i < max; i++)
  {
    if (strcmp(BIOL_QUALIFIER_STRINGS[i], s) == 0)
      return (BiolQualifierType_t)i;
  }
  return BQB_UNKNOWN;
}




/** @endcond */

LIBSBML_CPP_NAMESPACE_END

