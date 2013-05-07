/**
 * @file    SBMLExtensionRegistry.cpp
 * @brief   Implementation of SBMLExtensionRegistry, the registry class in which
 *          extension packages are registered.
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

#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/SBMLDocument.h>
#include <sbml/extension/SBasePlugin.h>
#include <algorithm>
#include <iostream>
#include <string>

#include "RegisterExtensions.h"


#ifdef __cplusplus

#include <sbml/validator/constraints/IdList.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_BEGIN

bool SBMLExtensionRegistry::registered = false;

/*
 *
 */
SBMLExtensionRegistry& 
SBMLExtensionRegistry::getInstance()
{
  static SBMLExtensionRegistry singletonObj;
  if (!registered)
  {
    registered = true;
    #include "RegisterExtensions.cxx"
  }
  return singletonObj;
}

SBMLExtensionRegistry::SBMLExtensionRegistry()
{
}


SBMLExtensionRegistry::SBMLExtensionRegistry(const SBMLExtensionRegistry& orig)
{
  if (&orig != NULL)
  {
    mSBMLExtensionMap =   orig.mSBMLExtensionMap;
    mSBasePluginMap   =   orig.mSBasePluginMap;
  }
}


/*
 * Add the given SBMLExtension to SBMLTypeCode_t element
 */
int 
SBMLExtensionRegistry::addExtension (const SBMLExtension* sbmlExt)
{
  //
  // null check
  //
  if (!sbmlExt)
  {
    //std::cout << "[DEBUG] SBMLExtensionRegistry::addExtension() : invalid attribute value " << std::endl;
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  
  //
  // duplication check
  //
  for (unsigned int i=0; i < sbmlExt->getNumOfSupportedPackageURI(); i++)
  {
	   SBMLExtensionMapIter it = mSBMLExtensionMap.find(sbmlExt->getSupportedPackageURI(i));
	   if (it != mSBMLExtensionMap.end())
		   return LIBSBML_PKG_CONFLICT;
  }

  
  SBMLExtension *sbmlExtClone = sbmlExt->clone();

  //
  // Register each (URI, SBMLExtension) pair and (pkgName, SBMLExtension) pair
  //
  for (unsigned int i=0; i < sbmlExt->getNumOfSupportedPackageURI(); i++)
  {    
    mSBMLExtensionMap.insert( SBMLExtensionPair(sbmlExt->getSupportedPackageURI(i), sbmlExtClone) );
  }
  mSBMLExtensionMap.insert( SBMLExtensionPair(sbmlExt->getName(), sbmlExtClone) );


  //
  // Register (SBMLTypeCode_t, SBasePluginCreatorBase) pair
  //
  for (int i=0; i < sbmlExtClone->getNumOfSBasePlugins(); i++)
  {
    const SBasePluginCreatorBase *sbPluginCreator = sbmlExtClone->getSBasePluginCreator(i);
#if 0
    std::cout << "[DEBUG] SBMLExtensionRegistry::addExtension() " << sbPluginCreator << std::endl;
#endif
    mSBasePluginMap.insert( SBasePluginPair(sbPluginCreator->getTargetExtensionPoint(), sbPluginCreator));
  }    

  return LIBSBML_OPERATION_SUCCESS;
}

SBMLExtension*
SBMLExtensionRegistry::getExtension(const std::string& uri)
{
	const SBMLExtension* extension = getExtensionInternal(uri);
	if (extension == NULL) return NULL;
	return extension->clone();
}

const SBMLExtension*
SBMLExtensionRegistry::getExtensionInternal(const std::string& uri)
{
  if(&uri == NULL) return NULL;
  
  SBMLExtensionMapIter it = mSBMLExtensionMap.find(uri);

#if 0
  if (it == mSBMLExtensionMap.end()) 
    std::cout << "[DEBUG] SBMLExtensionRegistry::getExtensionInternal() " << uri << " is NOT found." << std::endl;
  else
    std::cout << "[DEBUG] SBMLExtensionRegistry::getExtensionInternal() " << uri << " is FOUND." << std::endl;
#endif

  return (it != mSBMLExtensionMap.end()) ? mSBMLExtensionMap[uri] : NULL;  
}


/** @cond doxygen-libsbml-internal */
/*
 * Get the list of SBasePluginCreators with the given SBMLTypeCode_t element
 */
std::list<const SBasePluginCreatorBase*> 
SBMLExtensionRegistry::getSBasePluginCreators(const SBaseExtensionPoint& extPoint)
{
  std::list<const SBasePluginCreatorBase*> sbaseExtList;

  if (&extPoint != NULL)
  {
    SBasePluginMapIter it = mSBasePluginMap.find(extPoint);
    if (it != mSBasePluginMap.end())
    {    
      do 
      {
        sbaseExtList.push_back((*it).second);
        ++it;
      } while ( it != mSBasePluginMap.upper_bound(extPoint));
    }
  }  

  return sbaseExtList;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Get the list of SBasePluginCreators with the given URI (string)
 */
std::list<const SBasePluginCreatorBase*> 
SBMLExtensionRegistry::getSBasePluginCreators(const std::string& uri)
{
  std::list<const SBasePluginCreatorBase*> sbasePCList;

  if (&uri != NULL)
  {
  SBasePluginMapIter it = mSBasePluginMap.begin();
  if (it != mSBasePluginMap.end())
  {    
    do 
    {
     const SBasePluginCreatorBase* sbplug = (*it).second;

     if (sbplug->isSupported(uri))
     {
#if 0
        std::cout << "[DEBUG] SBMLExtensionRegistry::getPluginCreators() " 
                  << uri << " is found." << std::endl;
#endif
        sbasePCList.push_back((*it).second);
     }

      ++it;
    } while ( it != mSBasePluginMap.end() );
  }

#if 0
    if (sbasePluginList.size() == 0)
      std::cout << "[DEBUG] SBMLExtensionRegistry::getPluginCreators() " 
                << uri << " is NOT found." << std::endl;
#endif
  }

  return sbasePCList;  
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Get an SBasePluginCreator with the given extension point and URI pair
 */
const SBasePluginCreatorBase* 
SBMLExtensionRegistry::getSBasePluginCreator(const SBaseExtensionPoint& extPoint, const std::string &uri)
{
  if(&extPoint == NULL || &uri == NULL) return NULL;
  SBasePluginMapIter it = mSBasePluginMap.find(extPoint);
  if (it != mSBasePluginMap.end())
  {
    do
    {
      const SBasePluginCreatorBase* sbplugc = (*it).second;

      if (sbplugc->isSupported(uri))
      {
#if 0
          std::cout << "[DEBUG] SBMLExtensionRegistry::getSBasePluginCreators() " 
                    << uri << " is found." << std::endl;
#endif
        return sbplugc;
      }      
      ++it;
    } while ( it != mSBasePluginMap.end() );
  }      

#if 0
    std::cout << "[DEBUG] SBMLExtensionRegistry::getSBasePluginCreators() " 
              << uri << " is NOT found." << std::endl;
#endif

  return NULL;
}
/** @endcond */


unsigned int 
SBMLExtensionRegistry::getNumExtension(const SBaseExtensionPoint& extPoint)
{
  unsigned int numOfExtension = 0;
  if (&extPoint == NULL) return 0;
  SBasePluginMapIter it = mSBasePluginMap.find(extPoint);
  if (it != mSBasePluginMap.end())
  {    
    numOfExtension = (unsigned int)distance(it, mSBasePluginMap.upper_bound(extPoint));
  }    

  return numOfExtension;
}


/*
 * enable/disable the package with the given uri.
 * 
 * Returned value is the result of this function.
 */
bool 
SBMLExtensionRegistry::setEnabled(const std::string& uri, bool isEnabled)
{
  SBMLExtension *sbmlext = const_cast<SBMLExtension*>(getExtensionInternal(uri));  
  return (sbmlext) ? sbmlext->mIsEnabled = isEnabled : false;
}

void
SBMLExtensionRegistry::removeL2Namespaces(XMLNamespaces *xmlns)  const
{
  SBMLExtensionMap::const_iterator it = mSBMLExtensionMap.begin();
  while (it != mSBMLExtensionMap.end())
  {
    it->second->removeL2Namespaces(xmlns);
    it++;
  }
}

/**
 * adds all L2 Extension namespaces to the namespace list. This will call all 
 * overriden SBMLExtension::addL2Namespaces methods.
 */
void
SBMLExtensionRegistry::addL2Namespaces(XMLNamespaces *xmlns) const
{
  SBMLExtensionMap::const_iterator it = mSBMLExtensionMap.begin();
  while (it != mSBMLExtensionMap.end())
  {
    it->second->addL2Namespaces(xmlns);
    it++;
  }
}

/**
 * Enables all extensions that support serialization / deserialization with
 * SBML Annotations.
 */
void 
SBMLExtensionRegistry::enableL2NamespaceForDocument(SBMLDocument* doc)  const
{
  SBMLExtensionMap::const_iterator it = mSBMLExtensionMap.begin();
  while (it != mSBMLExtensionMap.end())
  {
    it->second->enableL2NamespaceForDocument(doc);
    it++;
  }
}

/*
 * Checks if the extension with the given URI is enabled (true) or disabled (false)
 */
bool 
SBMLExtensionRegistry::isEnabled(const std::string& uri)
{
  const SBMLExtension *sbmlext = getExtensionInternal(uri);  
  return (sbmlext) ? sbmlext->mIsEnabled : false;
}


/*
 * Checks if the extension with the given URI is registered (true) or not (false)
 */
bool 
SBMLExtensionRegistry::isRegistered(const std::string& uri)
{  
  return (getExtensionInternal(uri)) ? true : false;
}

List* 
SBMLExtensionRegistry::getRegisteredPackageNames()
{
  SBMLExtensionRegistry instance = getInstance();
  SBMLExtensionMap::const_iterator it = instance.mSBMLExtensionMap.begin();
  List* result = new List();
  IdList  * present = new IdList();
  while (it != instance.mSBMLExtensionMap.end())
  {    
    if (present->contains((*it).second->getName().c_str()) == false)
    {
      result->add(safe_strdup((*it).second->getName().c_str()));
      present->append(safe_strdup((*it).second->getName().c_str()));
    }
    it++;
  }
  delete present;
  return result;
}

unsigned int 
SBMLExtensionRegistry::getNumRegisteredPackages()
{
  return (unsigned int) getRegisteredPackageNames()->getSize();
  //SBMLExtensionRegistry instance = getInstance();
  //return (unsigned int)instance.mSBMLExtensionMap.size();
}


std::string
SBMLExtensionRegistry::getRegisteredPackageName(unsigned int index)
{
  char * name = (char *) (getRegisteredPackageNames()->get(index));
  return string(name);
}

void 
SBMLExtensionRegistry::disableUnusedPackages(SBMLDocument *doc)
{
  for (unsigned int i = doc->getNumPlugins(); i > 0; i--)
  {
    SBasePlugin *plugin = doc->getPlugin(i-1);
    if (plugin == NULL) continue;
    const SBMLExtension *ext = getExtensionInternal(plugin->getURI());
    if (!ext->isInUse(doc))
      doc->disablePackage(plugin->getURI(), plugin->getPrefix());
  }
}


/** @cond doxygen-c-only */


/**
 * Add the given SBMLExtension_t to the SBMLExtensionRegistry.
 *
 * @param SBMLExtension the SBMLExtension_t structure to be added.
 *   
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 * @li @link OperationReturnValues_t#LIBSBML_PKG_CONFLICT LIBSBML_PKG_CONFLICT @endlink
 * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
 */
LIBSBML_EXTERN
int 
SBMLExtensionRegistry_addExtension(const SBMLExtension_t* extension)
{
  if (extension == NULL) return LIBSBML_INVALID_OBJECT;
  return SBMLExtensionRegistry::getInstance().addExtension(extension);
}

/**
 * Returns an SBMLExtension_t structure with the given package URI or package name (string).
 *
 * @param package the URI or name of the package extension
 *
 * @return a clone of the SBMLExtension object with the given package URI or name. 
 * Or NULL in case of an invalid package name.
 * 
 * @note The returned extension is to be freed (i.e.: deleted) by the caller!
 */
LIBSBML_EXTERN
SBMLExtension_t* 
SBMLExtensionRegistry_getExtension(const char* package)
{
  if (package == NULL) return NULL;
  string sPackage(package);
  return SBMLExtensionRegistry::getInstance().getExtension(sPackage);
}

/**
 * Returns an SBasePluginCreator_t structure with the combination of the given 
 * extension point and URI of the package extension.
 *
 * @param extPoint the SBaseExtensionPoint
 * @param uri the URI of the target package extension.
 *
 * @return the SBasePluginCreator_t with the combination of the given 
 * SBMLTypeCode_t and the given URI of package extension, or NULL for 
 * invalid extensionPoint or uri.
 */
LIBSBML_EXTERN
const SBasePluginCreatorBase_t* 
SBMLExtensionRegistry_getSBasePluginCreator(const SBaseExtensionPoint_t* extPoint, const char* uri)
{
  if (extPoint == NULL || uri == NULL) return NULL;
  string sUri(uri);
  return SBMLExtensionRegistry::getInstance().getSBasePluginCreator(*extPoint, sUri);
}

/**
 * Returns a copied array of SBasePluginCreators with the given extension point.
 *
 * @param extPoint the SBaseExtensionPoint
 * @param length pointer to a variable holding the length of the array returned. 
 *
 * @return an array of SBasePluginCreators with the given typecode.
 */
LIBSBML_EXTERN
SBasePluginCreatorBase_t**
SBMLExtensionRegistry_getSBasePluginCreators(const SBaseExtensionPoint_t* extPoint, int* length)
{
  if (extPoint == NULL || length == NULL) return NULL;

  std::list<const SBasePluginCreatorBase*> list = 
    SBMLExtensionRegistry::getInstance().getSBasePluginCreators(*extPoint);

  *length = (int)list.size();
  SBasePluginCreatorBase_t** result = (SBasePluginCreatorBase_t**)malloc(sizeof(SBasePluginCreatorBase_t*)*(*length));
  
  std::list<const SBasePluginCreatorBase*>::iterator it;
  int count = 0;
  for (it = list.begin(); it != list.end(); it++)
  {
    result[count++] = (*it)->clone();
  }
  
  return result;
}

/**
 * Returns a copied array of SBasePluginCreators with the given URI
 * of package extension.
 *
 * @param uri the URI of the target package extension.
 * @param length pointer to a variable holding the length of the array returned. 
 *
 * @return an array of SBasePluginCreators with the given URI
 * of package extension to be freed by the caller.
 */
LIBSBML_EXTERN
SBasePluginCreatorBase_t**
SBMLExtensionRegistry_getSBasePluginCreatorsByURI(const char* uri, int* length)
{
   if (uri == NULL || length == NULL) return NULL;
   string sUri(uri);
   std::list<const SBasePluginCreatorBase*> list = 
     SBMLExtensionRegistry::getInstance().getSBasePluginCreators(sUri);
 
   *length = (int)list.size();
   SBasePluginCreatorBase_t** result = (SBasePluginCreatorBase_t**)malloc(sizeof(SBasePluginCreatorBase_t*)*(*length));
   
   std::list<const SBasePluginCreatorBase*>::iterator it;
   int count = 0;
   for (it = list.begin(); it != list.end(); it++)
   {
     result[count++] = (*it)->clone();
   }
  
  return result;
}


/**
 * Checks if the extension with the given URI is enabled (true) or 
 * disabled (false)
 *
 * @param uri the URI of the target package.
 *
 * @return false (0) will be returned if the given package is disabled 
 * or not registered, otherwise true (1) will be returned.
 */
LIBSBML_EXTERN
int
SBMLExtensionRegistry_isEnabled(const char* uri)
{
  if (uri == NULL) return 0;
  string sUri(uri);
  return SBMLExtensionRegistry::getInstance().isEnabled(sUri);
}

/**
 * Enable/disable the package with the given uri.
 *
 * @param uri the URI of the target package.
 * @param isEnabled the bool value corresponding to enabled (true/1) or 
 * disabled (false/0)
 *
 * @return false (0) will be returned if the given bool value is false 
 * or the given package is not registered, otherwise true (1) will be
 * returned.
 */
LIBSBML_EXTERN
int
SBMLExtensionRegistry_setEnabled(const char* uri, int isEnabled)
{
  if (uri == NULL) return 0;
  string sUri(uri);  
  return SBMLExtensionRegistry::getInstance().setEnabled(sUri, isEnabled);
}

/**
 * Checks if the extension with the given URI is registered (true/1) 
 * or not (false/0)
 *
 * @param uri the URI of the target package.
 *
 * @return true (1) will be returned if the package with the given URI
 * is registered, otherwise false (0) will be returned.
 */
LIBSBML_EXTERN
int
SBMLExtensionRegistry_isRegistered(const char* uri)
{
  if (uri == NULL) return 0;
  string sUri(uri);
  return (int)SBMLExtensionRegistry::getInstance().isRegistered(sUri);
}

/**
 * Returns the number of SBMLExtension_t structures for the given extension point.
 *
 * @param extPoint the SBaseExtensionPoint
 *
 * @return the number of SBMLExtension_t structures for the given extension point.
 */
LIBSBML_EXTERN
int 
SBMLExtensionRegistry_getNumExtensions(const SBaseExtensionPoint_t* extPoint)
{
  if (extPoint == NULL) return 0;
  return SBMLExtensionRegistry::getInstance().getNumExtension(*extPoint);
}


/** 
 * Returns a list of registered packages (such as 'layout', 'fbc' or 'comp')
 * the list contains char* strings and has to be freed by the caller. 
 * 
 * @return the names of the registered packages in a list
 */
LIBSBML_EXTERN
List_t*
SBMLExtensionRegistry_getRegisteredPackages()
{
  return (List_t*)SBMLExtensionRegistry::getRegisteredPackageNames();
}


/** 
 * Returns the number of registered packages.
 * 
 * @return the number of registered packages.
 */
LIBSBML_EXTERN
int
SBMLExtensionRegistry_getNumRegisteredPackages()
{
  return (int)SBMLExtensionRegistry::getNumRegisteredPackages();
}


/** 
 * Returns the registered package name at the given index
 * 
 * @param index zero based index of the package name to return
 * 
 * @return the package name with the given index or NULL
 */
LIBSBML_EXTERN
char*
SBMLExtensionRegistry_getRegisteredPackageName(int index)
{
  return safe_strdup(SBMLExtensionRegistry::getRegisteredPackageName(index).c_str());
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

