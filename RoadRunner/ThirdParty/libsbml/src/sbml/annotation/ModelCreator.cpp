/**
 * @file    ModelCreator.cpp
 * @brief   ModelCreator I/O
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


#include <sbml/annotation/ModelCreator.h>
#include <sbml/common/common.h>
#include <sbml/SBase.h>
#include <cstdio>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/*
 * Creates a new ModelCreator.
 */
ModelCreator::ModelCreator () :
    mAdditionalRDF(NULL)
  , mHasBeenModified (false)
{
}

/*
 * create a new ModelCreator from an XMLNode
 */
ModelCreator::ModelCreator(const XMLNode creator):
    mAdditionalRDF(NULL)
  , mHasBeenModified (false)
{
  // check that this is the right place in the RDF Annotation
  if (creator.getName() == "li")
  {
    int numChildren = static_cast<int>(creator.getNumChildren());
    int n;

    // we expect an N / EMAIL / ORG in that order 
    // find the positions of the first occurence of each
    int Npos = -1;
    int EMAILpos = -1;
    int ORGpos = -1;
    for (n = 0; n < numChildren; n++)
    {
      const string& name = creator.getChild(n).getName();
      if (name == "N" && Npos < 0)
        Npos = n;
      else if (name == "EMAIL" && EMAILpos < 0 && n > Npos)
        EMAILpos = n;
      else if (name == "ORG" && ORGpos < 0 && n > EMAILpos)
        ORGpos = n;
    }

    //get Names
    if (Npos >= 0)
    {
      setFamilyName(creator.getChild(Npos).getChild("Family").getChild(0).getCharacters());
      setGivenName(creator.getChild(Npos).getChild("Given").getChild(0).getCharacters());
    }

    // get EMAIL
    if (EMAILpos >= 0)
    {
      setEmail(creator.getChild(EMAILpos).getChild(0).getCharacters());
    }

    // get ORG
    if (ORGpos >= 0)
    {
      setOrganization(creator.getChild(ORGpos).getChild("Orgname")
                             .getChild(0).getCharacters());
    }
    // loop thru and save any other elements
    numChildren = static_cast<int>(creator.getNumChildren());
    for (n = 0; n < numChildren; n++)
    {
      if (n != Npos && n != EMAILpos && n!= ORGpos)
      {
        if (mAdditionalRDF == NULL)
        {
          mAdditionalRDF = new XMLNode();
        }
        mAdditionalRDF->addChild(creator.getChild(n));
      }
    }
  }
}


/*
 * destructor
 */
ModelCreator::~ModelCreator()
{
  delete mAdditionalRDF;
}


/*
 * Copy constructor.
 */
ModelCreator::ModelCreator(const ModelCreator& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mFamilyName   = orig.mFamilyName;
    mGivenName    = orig.mGivenName;
    mEmail        = orig.mEmail;
    mOrganization = orig.mOrganization;

    if (orig.mAdditionalRDF != NULL)
      this->mAdditionalRDF = orig.mAdditionalRDF->clone();
    else
      this->mAdditionalRDF = NULL;

    mHasBeenModified = orig.mHasBeenModified;

  }
}


/*
 * Assignment operator
 */
ModelCreator& ModelCreator::operator=(const ModelCreator& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    mFamilyName   = rhs.mFamilyName;
    mGivenName    = rhs.mGivenName;
    mEmail        = rhs.mEmail;
    mOrganization = rhs.mOrganization;

    delete this->mAdditionalRDF;
    if (rhs.mAdditionalRDF != NULL)
      this->mAdditionalRDF = rhs.mAdditionalRDF->clone();
    else
      this->mAdditionalRDF = NULL;

    mHasBeenModified = rhs.mHasBeenModified;
  }

  return *this;
}


/*
 * @return a (deep) copy of this ModelCreator.
 */
ModelCreator* ModelCreator::clone () const
{
  return new ModelCreator(*this);
}


bool 
ModelCreator::isSetFamilyName()
{
  return (mFamilyName.empty() == false);
}


bool 
ModelCreator::isSetGivenName()
{
  return (mGivenName.empty() == false);
}


bool 
ModelCreator::isSetEmail()
{
  return (mEmail.empty() == false);
}


bool 
ModelCreator::isSetOrganization()
{
  return (mOrganization.empty() == false);
}


bool 
ModelCreator::isSetOrganisation()
{
  return isSetOrganization();
}


/*
 * sets the family name
 */
int 
ModelCreator::setFamilyName(const std::string& name)
{
  if (&(name) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mFamilyName = name;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * sets the given name
 */
int 
ModelCreator::setGivenName(const std::string& name)
{
  if (&(name) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mGivenName = name;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * sets the email
 */
int 
ModelCreator::setEmail(const std::string& email)
{
  if (&(email) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mEmail = email;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


int 
ModelCreator::setOrganization(const std::string& organization)
{
  if (&(organization) == NULL)
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mOrganization = organization;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


int 
ModelCreator::setOrganisation(const std::string& organization)
{
  return setOrganization(organization);
}


int 
ModelCreator::unsetFamilyName()
{
  mFamilyName.erase();

  if (mFamilyName.empty()) 
  {
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


int 
ModelCreator::unsetGivenName()
{
  mGivenName.erase();

  if (mGivenName.empty()) 
  {
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


int 
ModelCreator::unsetEmail()
{
  mEmail.erase();

  if (mEmail.empty()) 
  {
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


int 
ModelCreator::unsetOrganization()
{
  mOrganization.erase();

  if (mOrganization.empty()) 
  {
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


int 
ModelCreator::unsetOrganisation()
{
  return unsetOrganization();
}

/** @cond doxygen-libsbml-internal */
XMLNode *
ModelCreator::getAdditionalRDF()
{
  return mAdditionalRDF;
}
/** @endcond */

bool
ModelCreator::hasRequiredAttributes()
{
  bool valid = true;

  if (!isSetFamilyName())
  {
    valid = false;
  }

  if (!isSetGivenName())
  {
    valid = false;
  }

  return valid;
}


/** @cond doxygen-libsbml-internal */
bool
ModelCreator::hasBeenModified()
{
  return mHasBeenModified;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
void
ModelCreator::resetModifiedFlags()
{
  mHasBeenModified = false;
}
/** @endcond */


/**
 * Creates a new ModelCreator_t structure and returns a pointer to it.
 *
 * @return pointer to newly created ModelCreator_t structure.
 */
LIBSBML_EXTERN
ModelCreator_t *
ModelCreator_create()
{
  return new(nothrow) ModelCreator();
}

/**
 * Creates a new ModelCreator_t structure from an XMLNode_t structure
 * and returns a pointer to it.
 *
 * @return pointer to newly created ModelCreator_t structure.
 */
LIBSBML_EXTERN
ModelCreator_t *
ModelCreator_createFromNode(const XMLNode_t * node)
{
  if (node == NULL) return NULL;
  return new(nothrow) ModelCreator(*node);
}


/**
 * Destroys this ModelCreator.
 *
 * @param mc ModelCreator_t structure to be freed.
 */
LIBSBML_EXTERN
void
ModelCreator_free(ModelCreator_t * mc)
{
  if (mc == NULL) return;
  delete static_cast<ModelCreator*>(mc);
}


/**
 * Creates a deep copy of the given ModelCreator_t structure
 * 
 * @param mc the ModelCreator_t structure to be copied
 * 
 * @return a (deep) copy of the given ModelCreator_t structure.
 */
LIBSBML_EXTERN
ModelCreator_t *
ModelCreator_clone (const ModelCreator_t* mc)
{
  if (mc == NULL) return NULL;
  return static_cast<ModelCreator*>( mc->clone() );
}


/**
 * Returns the familyName from the ModelCreator.
 * 
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return familyName from the ModelCreator.
 */
LIBSBML_EXTERN
const char * 
ModelCreator_getFamilyName(ModelCreator_t *mc)
{
  if (mc == NULL) return NULL;
  return mc->getFamilyName().c_str();
}


/**
 * Returns the givenName from the ModelCreator.
 * 
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return givenName from the ModelCreator.
 */
LIBSBML_EXTERN
const char * 
ModelCreator_getGivenName(ModelCreator_t *mc)
{
  if (mc == NULL) return NULL;
  return mc->getGivenName().c_str();
}


/**
 * Returns the email from the ModelCreator.
 * 
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return email from the ModelCreator.
 */
LIBSBML_EXTERN
const char * 
ModelCreator_getEmail(ModelCreator_t *mc)
{
  if (mc == NULL) return NULL;
  return mc->getEmail().c_str();
}


/**
 * Returns the organization from the ModelCreator.
 *
 * @note This function is an alias of ModelCreator_getOrganization().
 * 
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return organization from the ModelCreator.
 */
LIBSBML_EXTERN
const char * 
ModelCreator_getOrganisation(ModelCreator_t *mc)
{
  if (mc == NULL) return NULL;
  return mc->getOrganisation().c_str();
}


/**
 * Returns the organization from the ModelCreator.
 * 
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return organization from the ModelCreator.
 */
LIBSBML_EXTERN
const char * 
ModelCreator_getOrganization(ModelCreator_t *mc)
{
  return ModelCreator_getOrganisation(mc);
}


/**
 * Predicate indicating whether this
 * ModelCreator's familyName is set.
 *
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return true (non-zero) if the familyName of this 
 * ModelCreator_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int 
ModelCreator_isSetFamilyName(ModelCreator_t *mc)
{
  if (mc == NULL) return (int)false;
  return static_cast<int>(mc->isSetFamilyName());
}


/**
 * Predicate indicating whether this
 * ModelCreator's givenName is set.
 *
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return true (non-zero) if the givenName of this 
 * ModelCreator_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int 
ModelCreator_isSetGivenName(ModelCreator_t *mc)
{
  if (mc == NULL) return (int)false;
  return static_cast<int>(mc->isSetGivenName());
}


/**
 * Predicate indicating whether this
 * ModelCreator's email is set.
 *
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return true (non-zero) if the email of this 
 * ModelCreator_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int 
ModelCreator_isSetEmail(ModelCreator_t *mc)
{
  if (mc == NULL) return (int)false;
  return static_cast<int>(mc->isSetEmail());
}


/**
 * Predicate indicating whether this
 * ModelCreator's organization is set.
 *
 * @note This function is an alias of ModelCretor_isSetOrganization().
 *
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return true (non-zero) if the organization of this 
 * ModelCreator_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int 
ModelCreator_isSetOrganisation(ModelCreator_t *mc)
{
  if (mc == NULL) return (int)false;
  return static_cast<int>(mc->isSetOrganisation());
}


/**
 * Predicate indicating whether this
 * ModelCreator's organization is set.
 *
 * @param mc the ModelCreator_t structure to be queried
 *
 * @return true (non-zero) if the organization of this 
 * ModelCreator_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int 
ModelCreator_isSetOrganization(ModelCreator_t *mc)
{
  return ModelCreator_isSetOrganisation(mc);
}


/**
 * Sets the family name
 *  
 * @param mc the ModelCreator_t structure
 * @param name a string representing the familyName of the ModelCreator. 
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
ModelCreator_setFamilyName(ModelCreator_t *mc, char * name)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->setFamilyName(name);
}


/**
 * Sets the given name
 *  
 * @param mc the ModelCreator_t structure
 * @param name a string representing the givenName of the ModelCreator. 
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
ModelCreator_setGivenName(ModelCreator_t *mc, char * name)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->setGivenName(name);
}


/**
 * Sets the email
 *  
 * @param mc the ModelCreator_t structure
 * @param email a string representing the email of the ModelCreator. 
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
ModelCreator_setEmail(ModelCreator_t *mc, char * email)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->setEmail(email);
}


/**
 * Sets the organization
 *  
 * @param mc the ModelCreator_t structure
 * @param org a string representing the organisation of the ModelCreator. 
 *
 * @note This function is an alias of ModelCretor_setOrganization().
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
ModelCreator_setOrganisation(ModelCreator_t *mc, char * org)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->setOrganisation(org);
}


/**
 * Sets the organization
 *  
 * @param mc the ModelCreator_t structure
 * @param org a string representing the organisation of the ModelCreator. 
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
ModelCreator_setOrganization(ModelCreator_t *mc, char * org)
{
  return ModelCreator_setOrganisation(mc, org);
}


/**
 * Unsets the familyName of this ModelCreator.
 *
 * @param mc the ModelCreator_t structure.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
ModelCreator_unsetFamilyName(ModelCreator_t *mc)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->unsetFamilyName();
}


/**
 * Unsets the givenName of this ModelCreator.
 *
 * @param mc the ModelCreator_t structure.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
ModelCreator_unsetGivenName(ModelCreator_t *mc)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->unsetGivenName();
}


/**
 * Unsets the email of this ModelCreator.
 *
 * @param mc the ModelCreator_t structure.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
ModelCreator_unsetEmail(ModelCreator_t *mc)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->unsetEmail();
}


/**
 * Unsets the organization of this ModelCreator.
 *
 * @param mc the ModelCreator_t structure.
 *
 * @note This function is an alias of ModelCretor_unsetOrganization().
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
ModelCreator_unsetOrganisation(ModelCreator_t *mc)
{
  if (mc == NULL) return LIBSBML_INVALID_OBJECT;
  return mc->unsetOrganisation();
}


/**
 * Unsets the organization of this ModelCreator.
 *
 * @param mc the ModelCreator_t structure.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int 
ModelCreator_unsetOrganization(ModelCreator_t *mc)
{
  return ModelCreator_unsetOrganisation(mc);
}


LIBSBML_EXTERN
int
ModelCreator_hasRequiredAttributes(ModelCreator_t *mc)
{
  if (mc == NULL) return (int)false;
  return static_cast<int> (mc->hasRequiredAttributes());
}


LIBSBML_CPP_NAMESPACE_END

