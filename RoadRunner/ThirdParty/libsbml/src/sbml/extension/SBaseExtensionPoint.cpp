/**
 * @file    SBaseExtensionPoint.h
 * @brief   Implementation of SBaseExtensionPoint
 * @author  Akiya Jouraku
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
 */

#include <sbml/common/common.h>
#include <sbml/common/operationReturnValues.h>
#include <sbml/extension/SBaseExtensionPoint.h>

#ifdef __cplusplus

LIBSBML_CPP_NAMESPACE_BEGIN

/*
 * constructor
 */
SBaseExtensionPoint::SBaseExtensionPoint(const std::string& pkgName, int typeCode) 
 : mPackageName(pkgName)
  ,mTypeCode(typeCode) 
{
}


/*
 * copy constructor
 */
SBaseExtensionPoint::SBaseExtensionPoint(const SBaseExtensionPoint& orig) 
 : mPackageName(orig.mPackageName)
  ,mTypeCode(orig.mTypeCode) 
{
}


/*
 * clone
 */
SBaseExtensionPoint* 
SBaseExtensionPoint::clone() const 
{ 
  return new SBaseExtensionPoint(*this); 
}


const std::string& 
SBaseExtensionPoint::getPackageName() const 
{ 
  return mPackageName; 
}


int 
SBaseExtensionPoint::getTypeCode() const 
{ 
  return mTypeCode; 
}


bool operator==(const SBaseExtensionPoint& lhs, const SBaseExtensionPoint& rhs) 
{
  if (&lhs == NULL || &rhs == NULL) return false;

  if (   (lhs.getTypeCode()    == rhs.getTypeCode()) 
      && (lhs.getPackageName() == rhs.getPackageName()) 
     )
  {
    return true;
  }
  return false;
}


bool operator<(const SBaseExtensionPoint& lhs, const SBaseExtensionPoint& rhs) 
{
  if (&lhs == NULL || &rhs == NULL) return false;

  if ( lhs.getPackageName() == rhs.getPackageName() )
  {
    if (lhs.getTypeCode()  < rhs.getTypeCode())
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  else if ( lhs.getPackageName() < rhs.getPackageName() )
  {
    return true;
  }

  return false;
}


/** @cond doxygen-c-only */

/**
 * Creates a new SBaseExtensionPoint_t structure with the given arguments
 * 
 * @param pkgName the package name for the new structure
 * @param typeCode the SBML Type code for the new structure
 * 
 * @return the newly created SBaseExtensionPoint_t structure or NULL in case 
 * the given pkgName is invalid (NULL).
 */
LIBSBML_EXTERN 
SBaseExtensionPoint_t *
SBaseExtensionPoint_create(const char* pkgName, int typeCode)
{
  if (pkgName == NULL) return NULL;
  return new SBaseExtensionPoint(pkgName, typeCode);
}

/**
 * Frees the given SBaseExtensionPoint_t structure
 * 
 * @param extPoint the SBaseExtensionPoint_t structure to be freed
 * 
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
 */
LIBSBML_EXTERN 
int
SBaseExtensionPoint_free(SBaseExtensionPoint_t *extPoint)
{
  if (extPoint == NULL) return LIBSBML_INVALID_OBJECT;
  delete extPoint;
  return LIBSBML_OPERATION_SUCCESS;
}


/**
 * Creates a deep copy of the given SBaseExtensionPoint_t structure
 * 
 * @param extPoint the SBaseExtensionPoint_t structure to be copied
 * 
 * @return a (deep) copy of the given SBaseExtensionPoint_t structure.
 */
LIBSBML_EXTERN 
SBaseExtensionPoint_t *
SBaseExtensionPoint_clone(const SBaseExtensionPoint_t *extPoint)
{
  if (extPoint == NULL) return NULL;
  return extPoint->clone();
}

/**
 * Returns the package name for the given SBaseExtensionPoint_t structure 
 * 
 * @param extPoint the SBaseExtensionPoint_t structure 
 * 
 * @return the package name for the given SBaseExtensionPoint_t structure or 
 * NULL. 
 */
LIBSBML_EXTERN 
char *
SBaseExtensionPoint_getPackageName(const SBaseExtensionPoint_t *extPoint)
{
  if (extPoint == NULL) return NULL;
  return safe_strdup(extPoint->getPackageName().c_str());
}

/**
 * Returns the type code for the given SBaseExtensionPoint_t structure 
 * 
 * @param extPoint the SBaseExtensionPoint_t structure 
 * 
 * @return the type code for the given SBaseExtensionPoint_t structure or 
 * LIBSBML_INVALID_OBJECT in case an invalid object is given. 
 */
LIBSBML_EXTERN 
int
SBaseExtensionPoint_getTypeCode(const SBaseExtensionPoint_t *extPoint)
{
  if (extPoint == NULL) return LIBSBML_INVALID_OBJECT;
  return extPoint->getTypeCode();
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


