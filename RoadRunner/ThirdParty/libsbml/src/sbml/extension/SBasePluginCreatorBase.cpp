/**
 * @file    SBasePluginCreatorBase.cpp
 * @brief   Implementation of SBasePluginCreatorBase, the base class of 
 *          SBasePlugin creator classes.
 * @author  Akiya Jouraku
 *
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

#include <sbml/extension/SBasePluginCreatorBase.h>

#ifdef __cplusplus

#include <algorithm>
#include <string>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygen-libsbml-internal */
SBasePluginCreatorBase::SBasePluginCreatorBase (const SBaseExtensionPoint& extPoint,
                                                const std::vector<std::string>& packageURIs)
 : mSupportedPackageURI(packageURIs)
  ,mTargetExtensionPoint(extPoint)
{ 
#if 0
    for (int i=0; i < packageURIs.size(); i++)
    {
      std::cout << "[DEBUG] SBasePluginCreatorBase() : supported package "
                << mSupportedPackageURI[i] << std::endl;
                //<< packageURIs[i] << std::endl;
      std::cout << "[DEBUG] SBasePluginCreatorBase() : isSupported "
                << isSupported(mSupportedPackageURI[i]) << std::endl;
    }  
#endif
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 * Destructor
 */
SBasePluginCreatorBase::~SBasePluginCreatorBase()
{
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 * Copy Constructor
 */
SBasePluginCreatorBase::SBasePluginCreatorBase (const SBasePluginCreatorBase& orig)
:  mSupportedPackageURI(orig.mSupportedPackageURI)
  ,mTargetExtensionPoint(orig.mTargetExtensionPoint)
{
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
int
SBasePluginCreatorBase::getTargetSBMLTypeCode() const
{
  return mTargetExtensionPoint.getTypeCode();
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 * Get an SBMLTypeCode tied with this creator object.
 */
const std::string& 
SBasePluginCreatorBase::getTargetPackageName() const
{
  return mTargetExtensionPoint.getPackageName();
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 * Get an SBaseExtensionPoint tied with this creator object.
 */
const SBaseExtensionPoint& 
SBasePluginCreatorBase::getTargetExtensionPoint() const
{
  return mTargetExtensionPoint;
}
/** @endcond */



/** @cond doxygen-libsbml-internal */

/**
 *
 */
unsigned int 
SBasePluginCreatorBase::getNumOfSupportedPackageURI() const
{
  return (unsigned int)mSupportedPackageURI.size();
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 *
 */
std::string
SBasePluginCreatorBase::getSupportedPackageURI(unsigned int i) const
{
  return (i < mSupportedPackageURI.size()) ? mSupportedPackageURI[i] : std::string();
  return (i < mSupportedPackageURI.size()) ? mSupportedPackageURI[i] : std::string("");
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
 *
 */
bool 
SBasePluginCreatorBase::isSupported(const std::string& uri) const
{
  if (&uri == NULL) return false;
  return ( mSupportedPackageURI.end()
            !=
           find(mSupportedPackageURI.begin(), mSupportedPackageURI.end(), uri)
         );
}
/** @endcond */

/** @cond doxygen-c-only */

/**
 * Creates an SBasePlugin_t structure with the given uri and the prefix
 * of the target package extension.
 *
 * @param creator the SBasePluginCreatorBase_t structure  
 * @param uri the package extension uri
 * @param prefix the package extension prefix
 * @param xmlns the package extension namespaces
 *
 * @return an SBasePlugin_t structure with the given uri and the prefix
 * of the target package extension, or NULL in case an invalid creator, uri 
 * or prefix was given.
 */
LIBSBML_EXTERN
SBasePlugin_t*
SBasePluginCreator_createPlugin(SBasePluginCreatorBase_t* creator, 
  const char* uri, const char* prefix, const XMLNamespaces_t* xmlns)
{
  if (creator == NULL || uri == NULL || prefix == NULL) return NULL;
  string sUri(uri); string sPrefix(prefix);
  return creator->createPlugin(sUri, sPrefix, xmlns);
}

/**
 * Creates a deep copy of the given SBasePluginCreatorBase_t structure
 * 
 * @param creator the SBasePluginCreatorBase_t structure to be copied
 * 
 * @return a (deep) copy of the given SBasePluginCreatorBase_t structure.
 */
LIBSBML_EXTERN
SBasePluginCreatorBase_t*
SBasePluginCreator_clone(SBasePluginCreatorBase_t* creator)
{
  if (creator == NULL) return NULL;
  return creator->clone();
}

/**
 * Returns the number of supported packages by the given creator object.
 * 
 * @param creator the SBasePluginCreatorBase_t structure
 * 
 * @return the number of supported packages by the given creator object.
 */
LIBSBML_EXTERN
unsigned int
SBasePluginCreator_getNumOfSupportedPackageURI(SBasePluginCreatorBase_t* creator)
{
  if (creator == NULL) return 0;
  return creator->getNumOfSupportedPackageURI();
}

/**
 * Returns a copy of the package uri with the specified index. 
 * 
 * @param creator the SBasePluginCreatorBase_t structure
 * @param index the index of the package uri to return
 * 
 * @return a copy of the package uri with the specified index
 * (Has to be freed by the caller). If creator is invalid NULL will 
 * be returned.
 */
LIBSBML_EXTERN
char*
SBasePluginCreator_getSupportedPackageURI(SBasePluginCreatorBase_t* creator, 
    unsigned int index)
{
  if (creator == NULL) return NULL;
  return safe_strdup(creator->getSupportedPackageURI(index).c_str());
}

/**
 * Returns the SBMLTypeCode tied to the creator object.
 * 
 * @param creator the SBasePluginCreatorBase_t structure
 * 
 * @return the SBMLTypeCode tied with the creator object or 
 * LIBSBML_INVALID_OBJECT.
 */
LIBSBML_EXTERN
int
SBasePluginCreator_getTargetSBMLTypeCode(SBasePluginCreatorBase_t* creator)
{
  if (creator == NULL) return LIBSBML_INVALID_OBJECT;
  return creator->getTargetSBMLTypeCode();
}

/**
 * Returns the target package name of the creator object.
 * 
 * @param creator the SBasePluginCreatorBase_t structure
 * 
 * @return the target package name of the creator object, or NULL if 
 * creator is invalid.
 */
LIBSBML_EXTERN
const char*
SBasePluginCreator_getTargetPackageName(SBasePluginCreatorBase_t* creator)
{
  if (creator == NULL) return NULL;
  return creator->getTargetPackageName().c_str();
}

/**
 * Returns the SBaseExtensionPoint tied to this creator object.
 * 
 * @param creator the SBasePluginCreatorBase_t structure
 * 
 * @return the SBaseExtensionPoint of the creator object, or NULL if 
 * creator is invalid.
 */
LIBSBML_EXTERN
const SBaseExtensionPoint_t*
SBasePluginCreator_getTargetExtensionPoint(SBasePluginCreatorBase_t* creator)
{
  if (creator == NULL) return NULL;
  return &(creator->getTargetExtensionPoint());
}

/**
 * Returns true (1), if a package with the given namespace is supported. 
 * 
 * @param creator the SBasePluginCreatorBase_t structure
 * @param uri the package uri to test
 * 
 * @return true (1), if a package with the given namespace is supported.
 */
LIBSBML_EXTERN
int 
SBasePluginCreator_isSupported(SBasePluginCreatorBase_t* creator, const char* uri)
{
  if (creator == NULL) return (int)false;
  string sUri(uri);
  return creator->isSupported(sUri);
}


/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


