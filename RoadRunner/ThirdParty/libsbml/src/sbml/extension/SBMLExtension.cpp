/**
 * @file    SBMLExtension.cpp
 * @brief   Implementation of SBMLExtension, the base class of package extensions.
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

#include <sbml/extension/SBMLExtension.h>
#include <sbml/extension/SBMLExtensionRegistry.h>

#ifdef __cplusplus

#include <algorithm>
#include <string>

using namespace std;
LIBSBML_CPP_NAMESPACE_BEGIN

SBMLExtension::SBMLExtension ()
 : mIsEnabled(true)
{
}


/*
 * Copy constructor.
 */
SBMLExtension::SBMLExtension(const SBMLExtension& orig): 
 mIsEnabled(orig.mIsEnabled), 
 mSupportedPackageURI(orig.mSupportedPackageURI)
{
  for (size_t i=0; i < orig.mSBasePluginCreators.size(); i++)
    mSBasePluginCreators.push_back(orig.mSBasePluginCreators[i]->clone());
}


/*
 * Destroy this object.
 */
SBMLExtension::~SBMLExtension ()
{
  for (size_t i=0; i < mSBasePluginCreators.size(); i++)
    delete mSBasePluginCreators[i];
}


/*
 * Assignment operator for SBMLExtension.
 */
SBMLExtension& 
SBMLExtension::operator=(const SBMLExtension& orig)
{  
  mIsEnabled = orig.mIsEnabled; 
  mSupportedPackageURI = orig.mSupportedPackageURI; 

  for (size_t i=0; i < mSBasePluginCreators.size(); i++)
    delete mSBasePluginCreators[i];

  for (size_t i=0; i < orig.mSBasePluginCreators.size(); i++)
    mSBasePluginCreators.push_back(orig.mSBasePluginCreators[i]->clone());

  return *this;
}


/** @cond doxygen-libsbml-internal */
/*
 *
 */
int 
SBMLExtension::addSBasePluginCreator(const SBasePluginCreatorBase* sbaseExt)
{
  if (!sbaseExt)
  {
    return LIBSBML_INVALID_OBJECT;
  }

  //
  // (TODO) Checks the XMLNamespaces of the given SBaseAttributeExtension and
  //        that of this SBMLExtension object.
  //        Returns LIBSBML_INVALID_ATTRIBUTE_VALUE if the namespaces are mismatched.
  //

  if (sbaseExt->getNumOfSupportedPackageURI() == 0)
  {
    return LIBSBML_INVALID_OBJECT;
  }

  for (unsigned int i=0; i < sbaseExt->getNumOfSupportedPackageURI(); i++)
  {
    std::string uri = sbaseExt->getSupportedPackageURI(i);

#if 0
    std::cout << "[DEBUG] SBMLExtension::addSBasePluginCreator() : given package uri " 
              << uri << " typecode " << sbaseExt->getTargetSBMLTypeCode() << std::endl;
#endif

    if (! isSupported(uri) ) 
    {
      mSupportedPackageURI.push_back(uri);
    }
  }

  mSBasePluginCreators.push_back(sbaseExt->clone());

#if 0
    std::cout << "[DEBUG] SBMLExtension::addSBasePluginCreator() : supported package num " 
              <<  mSupportedPackageURI.size() << std::endl;

  for (int i=0; i < mSupportedPackageURI.size(); i++)
  {
      std::cout << "[DEBUG] SBMLExtension::addSBasePluginCreator() : supported package " 
                << mSupportedPackageURI[i] << std::endl;
  }
#endif

  return LIBSBML_OPERATION_SUCCESS;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
SBasePluginCreatorBase*
SBMLExtension::getSBasePluginCreator(const SBaseExtensionPoint& extPoint)
{
  if (&extPoint == NULL) return NULL;
  std::vector<SBasePluginCreatorBase*>::iterator it = mSBasePluginCreators.begin();
  while(it != mSBasePluginCreators.end())
  {
#if 0    
    static int i=0;
    std::cout << "[DEBUG] SBMLExtension::getSBasePluginCreator() : the given typeCode " 
              << extPoint.getTypeCode ()<< " (" << i << ") typecode " << (*it)->getTargetSBMLTypeCode() 
              << std::endl;
    i++;
#endif
    if ((*it)->getTargetExtensionPoint() == extPoint)
      return *it;  
    ++it;
  }

  return NULL;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
const SBasePluginCreatorBase*
SBMLExtension::getSBasePluginCreator(const SBaseExtensionPoint& extPoint) const
{
  return const_cast<SBMLExtension*>(this)->getSBasePluginCreator(extPoint);
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
SBasePluginCreatorBase*
SBMLExtension::getSBasePluginCreator(unsigned int n)
{
  return (n < mSBasePluginCreators.size()) ? mSBasePluginCreators[n] : NULL;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
const SBasePluginCreatorBase*
SBMLExtension::getSBasePluginCreator(unsigned int n) const
{
  return const_cast<SBMLExtension*>(this)->getSBasePluginCreator(n);
}
/** @endcond */


int 
SBMLExtension::getNumOfSBasePlugins() const
{
  return (int)mSBasePluginCreators.size();
}


/*
 *
 */
unsigned int 
SBMLExtension::getNumOfSupportedPackageURI() const
{
  return (unsigned int)mSupportedPackageURI.size();
}


/*
 *
 */
bool
SBMLExtension::isSupported(const std::string& uri) const
{
  if(&uri == NULL) return false;
  return ( mSupportedPackageURI.end() 
            != 
           find(mSupportedPackageURI.begin(),mSupportedPackageURI.end(), uri) );
}


const std::string&
SBMLExtension::getSupportedPackageURI(unsigned int i) const
{
  static std::string empty = "";
  return (i < mSupportedPackageURI.size()) ? mSupportedPackageURI[i] : empty;
}


/*
 * enable/disable this package.
 */
bool
SBMLExtension::setEnabled(bool isEnabled) 
{
  return SBMLExtensionRegistry::getInstance().setEnabled(getSupportedPackageURI(0), isEnabled);
}


/*
 * Check if this package is enabled (true) or disabled (false).
 */
bool 
SBMLExtension::isEnabled() const
{
  return SBMLExtensionRegistry::getInstance().isEnabled(getSupportedPackageURI(0));
}


/**
 * Removes the L2 Namespace
 *
 * This method should be overridden by all extensions that want to serialize
 * to an L2 annotation.
 */
void SBMLExtension::removeL2Namespaces(XMLNamespaces* xmlns)  const
{

}

/**
 * adds the L2 Namespace 
 *
 * This method should be overridden by all extensions that want to serialize
 * to an L2 annotation.
 */
void SBMLExtension::addL2Namespaces(XMLNamespaces* xmlns)  const
{

}

/**
 * Adds the L2 Namespace to the document and enables the extension.
 *
 * If the extension supports serialization to SBML L2 Annotations, this 
 * method should be overrridden, so it will be activated.
 */
void SBMLExtension::enableL2NamespaceForDocument(SBMLDocument* doc)  const
{

}


bool 
SBMLExtension::isInUse(SBMLDocument *doc) const
{
  return true;
}

/** @cond doxygen-c-only */

/**
 * Creates a deep copy of the given SBMLExtension_t structure
 * 
 * @param ext the SBMLExtension_t structure to be copied
 * 
 * @return a (deep) copy of the given SBMLExtension_t structure.
 */
LIBSBML_EXTERN
SBMLExtension_t*
SBMLExtension_clone(SBMLExtension_t* ext)
{
  if (ext == NULL) return NULL;
  return ext->clone();
}

/**
 * Frees the given SBMLExtension_t structure
 * 
 * @param ext the SBMLExtension_t structure to be freed
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
SBMLExtension_free(SBMLExtension_t* ext)
{
  if (ext == NULL) return LIBSBML_INVALID_OBJECT;
  delete ext;
  return LIBSBML_OPERATION_SUCCESS;

}

/**
 * Adds the given SBasePluginCreatorBase object to this package
 * extension.
 *
 * @param ext the SBMLExtension_t structure to be freed
 * @param sbaseExt the SBasePluginCreatorBase object bound to 
 * some SBML element and creates a corresponding SBasePlugin object 
 * of this package extension.
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
SBMLExtension_addSBasePluginCreator(SBMLExtension_t* ext, 
      SBasePluginCreatorBase_t *sbaseExt )
{
  if (ext == NULL) return LIBSBML_INVALID_OBJECT;
  return ext->addSBasePluginCreator(sbaseExt);
}

/*
 * Returns an SBasePluginCreatorBase structure of this package extension
 * bound to the given extension point.
 *
 * @param ext the SBMLExtension_t structure 
 * @param extPoint the SBaseExtensionPoint to which the returned 
 * SBasePluginCreatorBase object bound.
 *
 * @return an SBasePluginCreatorBase_t structure of this package extension 
 * bound to the given extension point, or NULL for invalid extension of 
 * extension point.
 */
LIBSBML_EXTERN
SBasePluginCreatorBase_t *
SBMLExtension_getSBasePluginCreator(SBMLExtension_t* ext, 
      SBaseExtensionPoint_t *extPoint )
{
  if (ext == NULL|| extPoint == NULL) return NULL;
  return ext->getSBasePluginCreator(*extPoint);
}

/**
 * Returns an SBasePluginCreatorBase_t structure of this package extension 
 * with the given index.
 *
 * @param ext the SBMLExtension_t structure 
 * @param i the index of the returned SBasePluginCreatorBase_t object for
 * this package extension.
 *
 * @return an SBasePluginCreatorBase structure of this package extension 
 * with the given index, or NULL for an invalid extension structure.
 */
LIBSBML_EXTERN
SBasePluginCreatorBase_t *
SBMLExtension_getSBasePluginCreatorByIndex(SBMLExtension_t* ext, 
      unsigned int index)
{
  if (ext == NULL) return NULL;
  return ext->getSBasePluginCreator(index);
}

/**
 * Returns the number of SBasePlugin_t structures stored in the structure.
 *
 * @param ext the SBMLExtension_t structure 
 *
 * @return the number of SBasePlugin_t structures stored in the structure, 
 * or LIBSBML_INVALID_OBJECT. 
 */
LIBSBML_EXTERN
int
SBMLExtension_getNumOfSBasePlugins(SBMLExtension_t* ext)
{
  if (ext == NULL) return LIBSBML_INVALID_OBJECT;
  return ext->getNumOfSBasePlugins();
}

/**
 * Returns the number of supported package namespaces (package versions) 
 * for this package extension.
 *
 * @param ext the SBMLExtension_t structure 
 *
 * @return the number of supported package namespaces (package versions) 
 * for this package extension or LIBSBML_INVALID_OBJECT. 
 */
LIBSBML_EXTERN
int
SBMLExtension_getNumOfSupportedPackageURI(SBMLExtension_t* ext)
{
  if (ext == NULL) return LIBSBML_INVALID_OBJECT;
  return ext->getNumOfSupportedPackageURI();
}

/**
 * Returns a flag indicating, whether the given URI (package version) is 
 * supported by this package extension.
 *
 * @param ext the SBMLExtension_t structure 
 * @param uri the package uri
 *
 * @return true (1) if the given URI (package version) is supported by this 
 * package extension, otherwise false (0) is returned.
 */
LIBSBML_EXTERN
int
SBMLExtension_isSupported(SBMLExtension_t* ext, const char* uri)
{
  if (ext == NULL || uri == NULL) return (int)false;
  string sUri(uri);
  return ext->isSupported(sUri);
}

/**
 * Returns the package URI (package version) for the given index.
 *
 * @param ext the SBMLExtension_t structure 
 * @param index the index of the supported package uri to return
 *
 * @return the package URI (package version) for the given index or NULL.
 */
LIBSBML_EXTERN
const char*
SBMLExtension_getSupportedPackageURI(SBMLExtension_t* ext, unsigned int index)
{
  if (ext == NULL) return NULL;
  return ext->getSupportedPackageURI(index).c_str();
}

/**
 * Returns the name of the package extension. (e.g. "layout", "multi").
 *
 * @param ext the SBMLExtension_t structure 
 *
 * @return the name of the package extension. (e.g. "layout", "multi").
 */
LIBSBML_EXTERN
const char*
SBMLExtension_getName(SBMLExtension_t* ext)
{
  if (ext == NULL) return NULL;
  return ext->getName().c_str();

}
/**
 * Returns the uri corresponding to the given SBML level, SBML version, 
 * and package version for this extension.
 *
 * @param ext the SBMLExtension structure
 * @param sbmlLevel the level of SBML
 * @param sbmlVersion the version of SBML
 * @param pkgVersion the version of package
 *
 * @return a string of the package URI
 */
LIBSBML_EXTERN
const char*
SBMLExtension_getURI(SBMLExtension_t* ext, unsigned int sbmlLevel, 
      unsigned int sbmlVersion, unsigned int pkgVersion)
{
  if (ext == NULL) return NULL;
  return ext->getURI(sbmlLevel, sbmlVersion, pkgVersion).c_str();
}

/**
 * Returns the SBML level associated with the given URI of this package.
 *
 * @param ext the SBMLExtension structure
 * @param uri the string of URI that represents a versions of the package
 *
 * @return the SBML level associated with the given URI of this package.
 */
LIBSBML_EXTERN
unsigned int
SBMLExtension_getLevel(SBMLExtension_t* ext, const char* uri)
{
  if (ext == NULL || uri == NULL) return SBML_INT_MAX;
  string sUri(uri);
  return ext->getLevel(sUri);
}

/**
 * Returns the SBML version associated with the given URI of this package.
 *
 * @param ext the SBMLExtension structure
 * @param uri the string of URI that represents a versions of the package
 *
 * @return the SBML version associated with the given URI of this package.
 */
LIBSBML_EXTERN
unsigned int
SBMLExtension_getVersion(SBMLExtension_t* ext, const char* uri)
{
  if (ext == NULL || uri == NULL) return SBML_INT_MAX;
  string sUri(uri);
  return ext->getVersion(sUri);

}

/**
 * Returns the package version associated with the given URI of this package.
 *
 * @param ext the SBMLExtension structure
 * @param uri the string of URI that represents a versions of the package
 *
 * @return the package version associated with the given URI of this package.
 */
LIBSBML_EXTERN
unsigned int
SBMLExtension_getPackageVersion(SBMLExtension_t* ext, const char* uri)
{
  if (ext == NULL || uri == NULL) return SBML_INT_MAX;
  string sUri(uri);
  return ext->getPackageVersion(sUri);
}

/**
 * This method takes a type code of this package and returns a string 
 * representing the code.
 * 
 * @param ext the SBMLExtension structure
 * @param typeCode the typeCode supported by the package
 * 
 * @return the string representing the given typecode, or NULL in case an 
 * invalid extension was provided. 
 */
LIBSBML_EXTERN
const char*
SBMLExtension_getStringFromTypeCode(SBMLExtension_t* ext, int typeCode)
{
  if (ext == NULL) return NULL;
  return ext->getStringFromTypeCode(typeCode);
    
}

/**
 * Returns an SBMLNamespaces_t structure corresponding to the given uri.
 * NULL will be returned if the given uri is not defined in the corresponding 
 * package.
 *
 * @param ext the SBMLExtension structure
 * @param uri the string of URI that represents one of versions of the package
 * 
 * @return an SBMLNamespaces_t structure corresponding to the uri. NULL
 *         will be returned if the given uri is not defined in the corresponding 
 *         package or an invalid extension structure was provided. 
 */
LIBSBML_EXTERN
SBMLNamespaces_t*
SBMLExtension_getSBMLExtensionNamespaces(SBMLExtension_t* ext, const char* uri)
{
  if (ext == NULL || uri == NULL) return NULL;
  string sUri(uri);
  return ext->getSBMLExtensionNamespaces(sUri);
}

/**
 * Enable/disable this package. 
 *
 * @param ext the SBMLExtension structure
 * @param isEnabled the value to set : true (1) (enabled) or false (0) (disabled)
 *
 * @return true (1) if this function call succeeded, otherwise false (0)is returned.
 * If the extension is invalid, LIBSBML_INVALID_OBJECT will be returned. 
 */
LIBSBML_EXTERN
int
SBMLExtension_setEnabled(SBMLExtension_t* ext, int isEnabled)
{
  if (ext == NULL) return LIBSBML_INVALID_OBJECT;
  return ext->setEnabled(isEnabled);
}

/**
 * Check if this package is enabled (true/1) or disabled (false/0).
 *
 * @param ext the SBMLExtension structure
 *
 * @return true if the package is enabled, otherwise false is returned.
 * If the extension is invalid, LIBSBML_INVALID_OBJECT will be returned. 
 */
LIBSBML_EXTERN
int
SBMLExtension_isEnabled(SBMLExtension_t* ext)
{
  if (ext == NULL) return LIBSBML_INVALID_OBJECT;
  return ext->isEnabled();
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


