/**
 * @file    ModelHistory.cpp
 * @brief   ModelHistory I/O
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


#include <sbml/annotation/ModelHistory.h>
#include <sbml/annotation/ModelCreator.h>
#include <sbml/annotation/Date.h>
#include <sbml/common/common.h>
#include <sbml/SBase.h>
#include <cstdio>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN


/*
 * Creates a new ModelHistory.
 */
 ModelHistory::ModelHistory ():
  mHasBeenModified (false)
{
  mCreatedDate = NULL;
//  mModifiedDate = NULL;
  mCreators = new List();
  mModifiedDates = new List();
}

/*
 * destructor
 */
ModelHistory::~ModelHistory()
{
  if (mCreators != NULL)
  {
    unsigned int size = mCreators->getSize();
    while (size--) delete static_cast<ModelCreator*>( mCreators->remove(0) );
    delete mCreators;
  }
  if (mCreatedDate != NULL) delete mCreatedDate;
//  if (mModifiedDate) delete mModifiedDate;
  if (mModifiedDates != NULL)
  {
    unsigned int size = mModifiedDates->getSize();
    while (size--) delete static_cast<Date*>
                                     ( mModifiedDates->remove(0) );
    delete mModifiedDates;
  }
}


/*
 * Copy constructor.
 */
ModelHistory::ModelHistory(const ModelHistory& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mCreators = new List();
    mModifiedDates = new List();
    unsigned int i;
    for (i = 0; i < orig.mCreators->getSize(); i++)
    {
      this->addCreator(static_cast<ModelCreator*>(orig.mCreators->get(i)));
    }
    for (i = 0; i < orig.mModifiedDates->getSize(); i++)
    {
      this->addModifiedDate(static_cast<Date*>(orig.mModifiedDates->get(i)));
    }
    if (orig.mCreatedDate != NULL) 
    {
      this->mCreatedDate = orig.mCreatedDate->clone();
    }
    else
    {
      mCreatedDate = NULL;
    }
    mHasBeenModified = orig.mHasBeenModified;
  }
}


/*
 * Assignment operator
 */
ModelHistory& 
ModelHistory::operator=(const ModelHistory& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    if (mCreators != NULL)
    {
      unsigned int size = mCreators->getSize();
      while (size--) delete static_cast<ModelCreator*>( mCreators->remove(0) );
    }
    else
    {
      mCreators = new List();
    }

    unsigned int i;
    for (i = 0; i < rhs.mCreators->getSize(); i++)
    {
      addCreator(static_cast<ModelCreator*>(rhs.mCreators->get(i)));
    }

    if (mModifiedDates != NULL)
    {
      unsigned int size = mModifiedDates->getSize();
      while (size--) delete static_cast<Date*>
                                       ( mModifiedDates->remove(0) );
    }
    else
    {
      mModifiedDates = new List();
    }

    for (i = 0; i < rhs.mModifiedDates->getSize(); i++)
    {
      addModifiedDate(static_cast<Date*>(rhs.mModifiedDates->get(i)));
    }

    delete mCreatedDate;
    if (rhs.mCreatedDate != NULL) 
      setCreatedDate(rhs.mCreatedDate);
    else
      mCreatedDate = NULL;

    mHasBeenModified = rhs.mHasBeenModified;
  }

  return *this;
}


/*
 * @return a (deep) copy of this ModelHistory.
 */
ModelHistory* 
ModelHistory::clone() const
{
  return new ModelHistory(*this);
}


/*
 * adds a creator to the model history
 */
int 
ModelHistory::addCreator(ModelCreator * creator)
{
  if (creator == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!creator->hasRequiredAttributes())
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else
  {
    mCreators->add((void *)(creator->clone()));
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * sets the created date
 */
int 
ModelHistory::setCreatedDate(Date* date)
{
  if (mCreatedDate == date)
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (date == NULL)
  {
    delete mCreatedDate;
    mCreatedDate = 0;
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (!date->representsValidDate())
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else
  {
    delete mCreatedDate;
    mCreatedDate = date->clone();
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/*
 * sets teh modiefied date
 */
int 
ModelHistory::setModifiedDate(Date* date)
{
  //mModifiedDate = date->clone();
  return addModifiedDate(date);
}

/*
 * adds a modifieddate to the model history
 */
int 
ModelHistory::addModifiedDate(Date * date)
{
  if (date == NULL)
  {
    //delete mModifiedDate;
    //mModifiedDate = 0;
    return LIBSBML_OPERATION_FAILED;
  }
  else if (!date->representsValidDate())
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else
  {
    mModifiedDates->add((void *)(date->clone()));
    mHasBeenModified = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
}

/*
 * return the List of creators
 */
List *
ModelHistory::getListCreators()
{
  return mCreators;
}


/*
 * return the List of modified dates
 */
List *
ModelHistory::getListModifiedDates()
{
  return mModifiedDates;
}


/*
 * return created date
 */
Date *
ModelHistory::getCreatedDate()
{
  return mCreatedDate;
}


/*
 * return modified date
 */
Date *
ModelHistory::getModifiedDate()
{
  return getModifiedDate(0);
}

/*
 * return modified date
 */
Date *
ModelHistory::getModifiedDate(unsigned int n)
{
  return (Date *) (mModifiedDates->get(n));
}


/*
 * @return number in List of Creator
 */
unsigned int 
ModelHistory::getNumCreators()
{
  return mCreators != NULL ? mCreators->getSize() : 0;
}


/*
 * @return number in List of modified dates
 */
unsigned int 
ModelHistory::getNumModifiedDates()
{
  return mModifiedDates->getSize();
}

/*
 * @return nth Creator
 */
ModelCreator* 
ModelHistory::getCreator(unsigned int n)
{
  return (ModelCreator *) (mCreators->get(n));
}


/*
 * @return true if the created Date has been set, false
 * otherwise.
 */
bool 
ModelHistory::isSetCreatedDate()
{
  return mCreatedDate != NULL;
}


/*
 * @return true if the modified Date has been set, false
 * otherwise.
 */
bool 
ModelHistory::isSetModifiedDate()
{
  return (getNumModifiedDates() != 0);
}

bool
ModelHistory::hasRequiredAttributes()
{
  bool valid = true;
  
  if ( getNumCreators() < 1 ||
      !isSetCreatedDate()  ||
      !isSetModifiedDate() )
  {
    valid = false;
  }

  unsigned int i = 0;
  while(valid && i < getNumCreators())
  {
    valid = static_cast<ModelCreator *>(getCreator(i))
      ->hasRequiredAttributes();
    i++;
  }

  if (!valid) 
  {
    return valid;
  }

  valid = getCreatedDate()->representsValidDate();

  if (!valid) 
  {
    return valid;
  }
  for (unsigned int i = 0; i < getNumModifiedDates(); i++)
  {
    valid = getModifiedDate(i)->representsValidDate();
  }

  return valid;
}


/** @cond doxygen-libsbml-internal */
bool
ModelHistory::hasBeenModified()
{
  unsigned int i = 0;
  
  // check whether individual creators have been modifed
  while (mHasBeenModified == false && i < getNumCreators())
  {
    mHasBeenModified = getCreator(i)->hasBeenModified();
    i++;
  }

  // check whether created date has been modified
  if (mHasBeenModified == false && isSetCreatedDate() == true)
  {
    mHasBeenModified = getCreatedDate()->hasBeenModified();
  }

  // check whether modified dates have been modified
  i = 0;
  while (mHasBeenModified == false && i < getNumModifiedDates())
  {
    mHasBeenModified = getModifiedDate(i)->hasBeenModified();
    i++;
  }

  return mHasBeenModified;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
void
ModelHistory::resetModifiedFlags()
{
  unsigned int i = 0;
  
  for (i = 0; i < getNumCreators(); i++)
  {
    getCreator(i)->resetModifiedFlags();
  }

  if (isSetCreatedDate() == true)
  {
    getCreatedDate()->resetModifiedFlags();
  }

  for (i = 0; i < getNumModifiedDates(); i++)
  {
    getModifiedDate(i)->resetModifiedFlags();
  }

  mHasBeenModified = false;
}
/** @endcond */


/**
 * Creates a new ModelHistory_t structure and returns a pointer to it.
 *
 * @return pointer to newly created ModelHistory_t structure.
 */
LIBSBML_EXTERN
ModelHistory_t * 
ModelHistory_create ()
{
  return new(nothrow) ModelHistory();
}


/**
 * Destroys this ModelHistory.
 *
 * @param mh ModelHistory_t structure to be freed.
 */
LIBSBML_EXTERN
void 
ModelHistory_free(ModelHistory_t * mh)
{
  delete static_cast<ModelHistory*>(mh);
}


/**
 * Creates a deep copy of the given ModelHistory_t structure
 * 
 * @param mh the ModelHistory_t structure to be copied
 * 
 * @return a (deep) copy of the given ModelHistory_t structure.
 */
LIBSBML_EXTERN
ModelHistory_t *
ModelHistory_clone (const ModelHistory_t* mh)
{
  if (mh == NULL) return NULL;
  return static_cast<ModelHistory*>( mh->clone() );
}


/**
 * Returns the createdDate from the ModelHistory.
 *
 * @param mh the ModelHistory_t structure
 * 
 * @return Date_t structure representing the createdDate
 * from the ModelHistory_t structure.
 */
LIBSBML_EXTERN
Date_t * ModelHistory_getCreatedDate(ModelHistory_t * mh)
{
  if (mh == NULL) return NULL;
  return mh->getCreatedDate();
}


/**
 * Returns the modifiedDate from the ModelHistory.
 *
 * @param mh the ModelHistory_t structure
 * 
 * @return Date_t structure representing the modifiedDate
 * from the ModelHistory_t structure.
 */
LIBSBML_EXTERN
Date_t * ModelHistory_getModifiedDate(ModelHistory_t * mh)
{
  if (mh == NULL) return NULL;
  return mh->getModifiedDate();
}


/**
 * Predicate indicating whether this
 * ModelHistory's createdDate is set.
 *
 * @param mh the ModelHistory_t structure to be queried
 *
 * @return true (non-zero) if the createdDate of this 
 * ModelHistory_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int ModelHistory_isSetCreatedDate(ModelHistory_t * mh)
{
  if (mh == NULL) return (int)false;
  return static_cast<int> (mh->isSetCreatedDate());
}


/**
 * Predicate indicating whether this
 * ModelHistory's modifiedDate is set.
 *
 * @param mh the ModelHistory_t structure to be queried
 *
 * @return true (non-zero) if the modifiedDate of this 
 * ModelHistory_t structure is set, false (0) otherwise.
 */
LIBSBML_EXTERN
int ModelHistory_isSetModifiedDate(ModelHistory_t * mh)
{
  if (mh == NULL) return (int)false;
  return static_cast<int> (mh->isSetModifiedDate());
}


/**
 * Sets the createdDate.
 *  
 * @param mh the ModelHistory_t structure
 * @param date the Date_t structure representing the date
 * the ModelHistory was created. 
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int ModelHistory_setCreatedDate(ModelHistory_t * mh, 
                                 Date_t * date)
{
  if (mh == NULL) return LIBSBML_INVALID_OBJECT;
  return mh->setCreatedDate(date);
}


/**
 * Sets the modifiedDate.
 *  
 * @param mh the ModelHistory_t structure
 * @param date the Date_t structure representing the date
 * the ModelHistory was modified. 
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
ModelHistory_setModifiedDate(ModelHistory_t * mh, 
                                  Date_t * date)
{
	if (mh == NULL) return LIBSBML_INVALID_OBJECT;
  return mh->setModifiedDate(date);
}


/**
 * Adds a copy of a ModelCreator_t structure to the 
 * ModelHistory_t structure.
 *
 * @param mh the ModelHistory_t structure
 * @param mc the ModelCreator_t structure to add.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int 
ModelHistory_addCreator(ModelHistory_t * mh, 
                             ModelCreator_t * mc)
{
  if (mh == NULL) return LIBSBML_INVALID_OBJECT;
  return mh->addCreator(mc);
}


/**
 * Get the List of ModelCreator objects in this 
 * ModelHistory.
 *
 * @param mh the ModelHistory_t structure
 * 
 * @return a pointer to the List_t structure of ModelCreators 
 * for this ModelHistory_t structure.
 */
LIBSBML_EXTERN
List_t * ModelHistory_getListCreators(ModelHistory_t * mh)
{
  if (mh == NULL) return NULL;
  return mh->getListCreators();
}


/**
 * Get the nth ModelCreator_t structure in this ModelHistory_t.
 * 
 * @param mh the ModelHistory_t structure
 * @param n an unsigned int indicating which ModelCreator
 *
 * @return the nth ModelCreator of this ModelHistory.
 */
LIBSBML_EXTERN
ModelCreator_t* ModelHistory_getCreator(ModelHistory_t * mh, unsigned int n)
{
  if (mh == NULL) return NULL;
  return mh->getCreator(n);
}


/**
 * Get the number of ModelCreator objects in this 
 * ModelHistory.
 * 
 * @param mh the ModelHistory_t structure
 * 
 * @return the number of ModelCreators in this 
 * ModelHistory.
 */
LIBSBML_EXTERN
unsigned int ModelHistory_getNumCreators(ModelHistory_t * mh)
{
  if (mh == NULL) return SBML_INT_MAX;
  return mh->getNumCreators();
}

/**
 * Adds a copy of a Date_t structure to the 
 * list of modifiedDates in the ModelHistory_t structure.
 *
 * @param mh the ModelHistory_t structure
 * @param date the Date_t structure to add.
 */
LIBSBML_EXTERN
int 
ModelHistory_addModifiedDate(ModelHistory_t * mh, Date_t * date)
{
  if (mh == NULL) return LIBSBML_INVALID_OBJECT;
  return mh->addModifiedDate(date);
}

/**
 * Get the List of Date objects in the list of ModifiedDates 
 * in this ModelHistory.
 *
 * @param mh the ModelHistory_t structure
 * 
 * @return a pointer to the List_t structure of Dates 
 * for this ModelHistory_t structure.
 */
LIBSBML_EXTERN
List_t * 
ModelHistory_getListModifiedDates(ModelHistory_t * mh)
{
  if (mh == NULL) return NULL;
  return mh->getListModifiedDates();
}

/**
 * Get the number of modified Date objects in the list of ModifiedDates 
 * in this ModelHistory.
 *
 * @param mh the ModelHistory_t structure
 * 
 * @return the number of Dates in the list of ModifiedDates in this 
 * ModelHistory.
 */
LIBSBML_EXTERN
unsigned int 
ModelHistory_getNumModifiedDates(ModelHistory_t * mh)
{
  if (mh == NULL) return SBML_INT_MAX;
  return mh->getNumModifiedDates();
}

/**
 * Get the nth Date_t structure in the list of ModifiedDates
 * in this ModelHistory_t.
 * 
 * @param mh the ModelHistory_t structure
 * @param n an unsigned int indicating which Date
 *
 * @return the nth Date in the list of ModifiedDates
 * of this ModelHistory.
 *
 * @note A bug in libSBML meant that originally a ModelHistory object
 * contained only one instance of a ModifiedDate.  In fact the MIRIAM
 * annotation expects zero or more modified dates and thus the
 * implementation was changed.  To avoid impacting on existing code
 * there is a ditinction between the function 
 * ModelHistory_getModifiedDate which requires no index value and
 * this function that indexes into a list.
 */
LIBSBML_EXTERN
Date_t* 
ModelHistory_getModifiedDateFromList(ModelHistory_t * mh, unsigned int n)
{
  if (mh == NULL) return NULL;
  return mh->getModifiedDate(n);
}

LIBSBML_EXTERN
int
ModelHistory_hasRequiredAttributes(ModelHistory_t *mh)
{
  if (mh == NULL) return (int)false;
  return static_cast<int> (mh->hasRequiredAttributes());
}


LIBSBML_CPP_NAMESPACE_END

