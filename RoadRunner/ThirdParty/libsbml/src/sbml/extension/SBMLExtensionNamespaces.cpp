/**
 * @file    SBMLExtensionNamespaces.cpp
 * @brief   implementation of C-API for the SBMLExtensionNamespaces class 
 * @author  Frank Bergmann
 *
 * $Id $
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
 *
 */


#include <sbml/SBMLNamespaces.h>
#include <sbml/common/common.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/extension/SBMLExtensionException.h>
#include <sbml/extension/ISBMLExtensionNamespaces.h>
#include <sbml/extension/SBMLExtensionNamespaces.h>

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/** @cond doxygen-c-only */

/**
 * Creates a deep copy of the given SBMLExtensionNamespaces_t structure
 * 
 * @param extns the SBMLExtensionNamespaces_t structure to be copied
 * 
 * @return a (deep) copy of the given SBMLExtensionNamespaces_t structure.
 */
LIBSBML_EXTERN
SBMLExtensionNamespaces_t*
SBMLExtensionNamespaces_clone(SBMLExtensionNamespaces_t* extns)
{
  if (extns == NULL) return NULL;
  return (SBMLExtensionNamespaces_t*)extns->clone();
}

/**
 * Frees the given SBMLExtensionNamespaces_t structure
 * 
 * @param extns the SBMLExtensionNamespaces_t structure to be freed
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
SBMLExtensionNamespaces_free(SBMLExtensionNamespaces_t* extns)
{
  if (extns == NULL) return LIBSBML_INVALID_OBJECT;
  delete extns;
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Returns a copy of the string representing the Package XML namespace of the
 * given namespace structure.
 *
 * @param extns the SBMLExtensionNamespaces_t structure 
 *
 * @return a copy of the string representing the SBML namespace that reflects 
 * the SBML Level and Version of the namespace structure.
 */
LIBSBML_EXTERN
char*
SBMLExtensionNamespaces_getURI(SBMLExtensionNamespaces_t* extns)
{
  if (extns == NULL) return NULL;
  return safe_strdup(extns->getURI().c_str());
}

/**
 * Return the SBML Package Version of the SBMLExtensionNamespaces_t structure.
 *
 * @param extns the SBMLExtensionNamespaces_t structure 
 *
 * @return the SBML Package Version of the SBMLExtensionNamespaces_t structure.
 */
LIBSBML_EXTERN
unsigned int
SBMLExtensionNamespaces_getPackageVersion(SBMLExtensionNamespaces_t* extns)
{
  if (extns == NULL) return SBML_INT_MAX;
  return extns->getPackageVersion();
}

/**
 * Returns a copy of the string representing the Package name of the
 * given namespace structure.
 *
 * @param extns the SBMLExtensionNamespaces_t structure 
 *
 * @return a copy of the string representing the package name that of the 
 * namespace structure.
 */
LIBSBML_EXTERN
char*
SBMLExtensionNamespaces_getPackageName(SBMLExtensionNamespaces_t* extns)
{
  if (extns == NULL) return NULL;
  return safe_strdup(extns->getPackageName().c_str());
}

/*
 * Sets the package version of the namespace structure.
 *
 * @param extns the SBMLExtensionNamespaces_t structure 
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
SBMLExtensionNamespaces_setPackageVersion(SBMLExtensionNamespaces_t* extns,
    unsigned int pkgVersion)
{
  if (extns == NULL) return LIBSBML_INVALID_OBJECT;
  extns->setPackageVersion(pkgVersion);
  return LIBSBML_OPERATION_SUCCESS;
}


/** @endcond */

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END
