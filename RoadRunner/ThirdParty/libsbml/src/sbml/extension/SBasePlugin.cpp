/**
 * @file    SBasePlugin.cpp
 * @brief   Implementation of SBasePlugin, the base class of extension 
 *          entities plugged in SBase derived classes in the SBML Core package.
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

#include <sbml/extension/SBasePlugin.h>
#include <sbml/extension/SBMLExtensionRegistry.h>

#ifdef __cplusplus

#include <sstream>
#include <iostream>

LIBSBML_CPP_NAMESPACE_BEGIN


/** @cond doxygen-libsbml-internal */
/*
 * Constructor
 */
SBasePlugin::SBasePlugin (const std::string &uri, const std::string &prefix, 
                          SBMLNamespaces *sbmlns)
 : mSBMLExt(SBMLExtensionRegistry::getInstance().getExtensionInternal(uri))
  ,mSBML(NULL)
  ,mParent(NULL)
  ,mURI(uri)
  ,mSBMLNS(sbmlns == NULL ? NULL : sbmlns->clone())
  ,mPrefix(prefix)
{
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
* Copy constructor. Creates a copy of this SBasePlugin object.
*/
SBasePlugin::SBasePlugin(const SBasePlugin& orig)
  : mSBMLExt(orig.mSBMLExt)
   ,mSBML(NULL)   // (NOTE) NULL must be set to mSBML and mParent........ 
   ,mParent(NULL) // 
   ,mURI(orig.mURI)
   ,mSBMLNS(NULL)
   ,mPrefix(orig.mPrefix)
{
  if (orig.mSBMLNS) {
    mSBMLNS = orig.mSBMLNS->clone();
  }
}
/** @endcond */


/*
 * Destroy this object.
 */
SBasePlugin::~SBasePlugin ()
{
	if (mSBMLNS != NULL)
	delete mSBMLNS;
}


/*
 * Assignment operator for SBasePlugin.
 */
SBasePlugin& 
SBasePlugin::operator=(const SBasePlugin& orig)
{
  mSBMLExt = orig.mSBMLExt;
  mSBML    = orig.mSBML;    // (TODO)
  mParent  = orig.mParent;  // 0 should be set to mSBML and mParent?
  mURI     = orig.mURI;
  mPrefix  = orig.mPrefix;

  delete mSBMLNS;
  if (orig.mSBMLNS)
    mSBMLNS = orig.mSBMLNS->clone();
  else
    mSBMLNS = NULL;

  return *this;
}


/** @cond doxygen-libsbml-internal */
/*
 * Sets the given SBMLDocument as a parent document.
 */
void 
SBasePlugin::setSBMLDocument (SBMLDocument* d)
{
  mSBML = d;  
}
/** @endcond */


/*
 * Returns the parent SBMLDocument
 */
SBMLDocument*
SBasePlugin::getSBMLDocument ()
{
  return mSBML;
}


/*
 * Returns the parent SBMLDocument
 */
const SBMLDocument*
SBasePlugin::getSBMLDocument () const
{
  return mSBML;
}



SBase*
SBasePlugin::getElementBySId(std::string id)
{
  if (id.empty()) return NULL;
  return NULL;
}


SBase*
SBasePlugin::getElementByMetaId(std::string metaid)
{
  if (metaid.empty()) return NULL;
  return NULL;
}

List*
SBasePlugin::getAllElements()
{
  return NULL;
}

/** @cond doxygen-libsbml-internal */
/*
 * Sets the parent SBML object of this plugin object to
 * this object and child elements (if any).
 * (Creates a child-parent relationship by this plugin object)
 */
void
SBasePlugin::connectToParent (SBase* sbase)
{
  mParent = sbase;

  if (mParent)
  {
    setSBMLDocument(mParent->getSBMLDocument());
  }
  else
  {
	  setSBMLDocument(NULL);
  }
}
/** @endcond */



/**
 *
 * (Extension)
 *
 * Sets the XML namespace to which this element belogns to.
 * For example, all elements that belong to SBML Level 3 Version 1 Core
 * must set the namespace to "http://www.sbml.org/sbml/level3/version1/core";
 * all elements that belong to Layout Extension Version 1 for SBML Level 3
 * Version 1 Core must set the namespace to
 * "http://www.sbml.org/sbml/level3/version1/layout/version1/"
 *
 */
int
SBasePlugin::setElementNamespace(const std::string &uri)
{
//  cout << "[DEBUG] SBasePlugin::setElementNamespace() " << uri << endl;
  mURI = uri;

  return LIBSBML_OPERATION_SUCCESS;
}

/*
 * Returns the parent element.
 */
SBase*
SBasePlugin::getParentSBMLObject ()
{
  return mParent;
}


/*
 * Returns the parent element.
 */
const SBase*
SBasePlugin::getParentSBMLObject () const
{
  return mParent;
}


/** @cond doxygen-libsbml-internal */
/*
 * Enables/Disables the given package with child elements in this plugin
 * object (if any).
 */
void 
SBasePlugin::enablePackageInternal(const std::string& pkgURI,
                                   const std::string& pkgPrefix, bool flag)
{
 // do nothing.
}


bool 
SBasePlugin::stripPackage(const std::string& pkgPrefix, bool flag)
{
  return true;
}

/** @endcond */


/*
 * Returns the SBML level of this plugin object.
 *
 * @return the SBML level of this plugin object.
 */
unsigned int 
SBasePlugin::getLevel() const
{
  return mSBMLExt->getLevel(getURI());
}


/*
 * Returns the SBML version of this plugin object.
 *
 * @return the SBML version of this plugin object.
 */
unsigned int 
SBasePlugin::getVersion() const
{
  return mSBMLExt->getVersion(getURI());
}


/** @cond doxygen-libsbml-internal */
unsigned int 
SBasePlugin::getLine() const
{
  if (mParent == NULL) return 0;
  return mParent->getLine();
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
unsigned int 
SBasePlugin::getColumn() const
{
  if (mParent == NULL) return 0;
  return mParent->getColumn();
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
SBMLNamespaces *
SBasePlugin::getSBMLNamespaces() const
{
  if (mSBML != NULL)
    return mSBML->getSBMLNamespaces();
  else if (mParent != NULL)
    return mParent->getSBMLNamespaces();
  else if (mSBMLNS != NULL)
    return mSBMLNS;
  else
    return new SBMLNamespaces();
}
/** @endcond */


/*
 * Returns the package version of this plugin object.
 *
 * @return the package version of this plugin object.
 */
unsigned int 
SBasePlugin::getPackageVersion() const
{
  return mSBMLExt->getPackageVersion(getURI());
}


/** @cond doxygen-libsbml-internal */
void 
SBasePlugin::replaceSIDWithFunction(const std::string& id, const ASTNode* function)
{
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
void 
SBasePlugin::divideAssignmentsToSIdByFunction(const std::string& id, const ASTNode* function)
{
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
void 
SBasePlugin::multiplyAssignmentsToSIdByFunction(const std::string& id, const ASTNode* function)
{
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
bool SBasePlugin::hasIdentifierBeginningWith(const std::string& prefix)
{
  return false;
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
//Override and provide your own renaming scheme for the rest of the model if you do anything here.
int 
SBasePlugin::prependStringToAllIdentifiers(const std::string& prefix)
{
  return LIBSBML_OPERATION_SUCCESS;
}
  /** @endcond */
  
/*
 * Returns the namespace URI of this element.
 */
const std::string& 
SBasePlugin::getElementNamespace() const
{
  return mURI;  
}

std::string 
SBasePlugin::getURI() const
{
  if (mSBMLExt == NULL) 
    return getElementNamespace();
  
  const std::string &package = mSBMLExt->getName();
  const SBMLDocument* doc = getSBMLDocument();

  if (doc == NULL)
    return getElementNamespace();
  
  SBMLNamespaces* sbmlns = doc->getSBMLNamespaces();

  if (sbmlns == NULL)
    return getElementNamespace();

  if (package == "" || package == "core")
    return sbmlns->getURI();

  std::string packageURI = sbmlns->getNamespaces()->getURI(package);
  if (!packageURI.empty())
    return packageURI;

  return getElementNamespace();
}

/*
 * Returns the prefix bound to this element.
 */
const std::string& 
SBasePlugin::getPrefix() const
{
  
  return mPrefix;
}


/*
 * Returns the package name of this plugin object.
 */
const std::string& 
SBasePlugin::getPackageName() const
{
  return mSBMLExt->getName();
}

/** @cond doxygen-libsbml-internal */
/*
 * Intended to be overridden by package extensions of the Model object.
 */
int 
SBasePlugin::appendFrom(const Model* model)
{
  return LIBSBML_OPERATION_SUCCESS;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to create, store, and then
 * return an SBML object corresponding to the next XMLToken in the
 * XMLInputStream.
 *
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
SBasePlugin::createObject(XMLInputStream& stream)
{
  return NULL;
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
SBasePlugin::readOtherXML (SBase* parentObject, XMLInputStream& stream)
{
  return false;
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
/**
 * Synchronizes the annotation of this SBML object.
 *
 * Annotation element (XMLNode* mAnnotation) is synchronized with the 
 * current CVTerm objects (List* mCVTerm).
 * Currently, this method is called in getAnnotation, isSetAnnotation,
 * and writeElements methods.
 */
void 
SBasePlugin::syncAnnotation(SBase* parentObject, XMLNode *annotation)
{

}

/** 
 * Parse L2 annotation if supported
 *
 */
void 
SBasePlugin::parseAnnotation(SBase *parentObject, XMLNode *annotation)
{

}

/** @endcond */

/** @cond doxygen-libsbml-internal */
/* default for components that have no required elements */
bool
SBasePlugin::hasRequiredElements() const
{
  return true;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  
 */
void
SBasePlugin::writeElements (XMLOutputStream& stream) const
{
  // do nothing.  
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
SBasePlugin::addExpectedAttributes(ExpectedAttributes& attributes)
{
  // do nothing.
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.
 */
void
SBasePlugin::readAttributes (const XMLAttributes& attributes,
                                     const ExpectedAttributes& expectedAttributes)
{
  if (&attributes == NULL || &expectedAttributes == NULL ) return;

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();
  const unsigned int pkgVersion  = getPackageVersion();

   std::string element = (mParent) ? mParent->getElementName() : std::string();

  //
  // (NOTE)
  //
  // This function is just used to identify unexpected
  // attributes with the prefix of the package.
  //

#if 0
  std::cout << "[DEBUG] SBasePlugin::readAttributes() " << element << std::endl;
#endif

  //
  // check that all attributes of this plugin object are expected
  //
  for (int i = 0; i < attributes.getLength(); i++)
  {
    std::string name = attributes.getName(i);
    std::string uri  = attributes.getURI(i);

#if 0
    std::cout << "[DEBUG] SBasePlugin::readAttributes() name : " << name 
              << " uri " << uri << std::endl;
#endif
    
    if (uri != mURI) continue;

    if (!expectedAttributes.hasAttribute(name))
    {    
      logUnknownAttribute(name, sbmlLevel, sbmlVersion, pkgVersion, element);
    }      
  }
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write their XML attributes
 * to the XMLOutputStream. 
 */
void
SBasePlugin::writeAttributes (XMLOutputStream& stream) const
{
  // do nothing.  
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/* default for components that have no required attributes */
bool
SBasePlugin::hasRequiredAttributes() const
{
  return true;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Subclasses should override this method to write required xmlns attributes
 * to the XMLOutputStream. 
 * Tthe xmlns attribute will be written in the corresponding core element. 
 * For example, xmlns attribute written by this function will be
 * added to Model element if this plugin object connected to the element.
 */
void
SBasePlugin::writeXMLNS (XMLOutputStream& stream) const
{
  // do nothing.  
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * @return the SBMLErrorLog used to log errors during while reading and
 * validating SBML.
 */
SBMLErrorLog* 
SBasePlugin::getErrorLog ()
{
  return (mSBML) ? mSBML->getErrorLog() : 0;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Helper to log a common type of error.
 */
void 
SBasePlugin::logUnknownElement(const std::string &element,
			               const unsigned int sbmlLevel,
 			               const unsigned int sbmlVersion,
			               const unsigned int pkgVersion )
{
  if(&element == NULL) return;
  
  std::ostringstream msg;

  msg << "Element '"   << element << "' is not part of the definition of "
      << "SBML Level " << sbmlLevel << " Version " << sbmlVersion 
      << " Package \""   << mSBMLExt->getName() << "\" Version "
      << pkgVersion << ".";

  //
  // (TODO) Additional class such as SBMLExtensionError and SBMLExtensionErrorLog
  //        may need to be implemented
  //
  SBMLErrorLog* errlog = getErrorLog();
  if (errlog)
  {
    errlog->logError(UnrecognizedElement, sbmlLevel, sbmlVersion, msg.str());
  }
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Helper to log a common type of error.
 */
void 
SBasePlugin::logUnknownAttribute(const std::string &attribute,
                                         const unsigned int sbmlLevel,
 			                 const unsigned int sbmlVersion,
			                 const unsigned int pkgVersion,
					 const std::string& element)
{
  if (&attribute == NULL || &element == NULL) return;
  
  std::ostringstream msg;

  msg << "Attribute '" << attribute << "' is not part of the "
      << "definition of an SBML Level " << sbmlLevel
      << " Version " << sbmlVersion << " Package \"" 
      << mSBMLExt->getName() << "\" Version " << pkgVersion 
      << " on " << element << " element.";

  //
  // (TODO) Additional class such as SBMLExtensionError and SBMLExtensionErrorLog
  //        may need to be implemented
  //
  SBMLErrorLog* errlog = getErrorLog();
  if (errlog)
  {
    errlog->logError(NotSchemaConformant, sbmlLevel, sbmlVersion, msg.str());
  }
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Helper to log a common type of error.
 */
void 
SBasePlugin::logEmptyString(const std::string &attribute, 
                                    const unsigned int sbmlLevel,
                                    const unsigned int sbmlVersion,
			            const unsigned int pkgVersion,
			            const std::string& element)
{

  if (&attribute == NULL || &element == NULL) return;
  
  std::ostringstream msg;

  msg << "Attribute '" << attribute << "' on an "
      << element << " of package \"" << mSBMLExt->getName() 
      << "\" version " << pkgVersion << " must not be an empty string.";

  //
  // (TODO) Additional class such as SBMLExtensionError and SBMLExtensionErrorLog
  //        may need to be implemented
  //
  SBMLErrorLog* errlog = getErrorLog();
  if (errlog)
  {
    errlog->logError(NotSchemaConformant, sbmlLevel, sbmlVersion, msg.str());
  }
}
/** @endcond */


/** @cond doxygen-c-only */

/**
 * Returns the XML namespace (URI) of the package extension
 * of the given plugin structure.
 *
 * @param plugin the plugin structure
 * 
 * @return the URI of the package extension of this plugin object, or NULL
 * in case an invalid plugin structure is provided. 
 */
LIBSBML_EXTERN
const char* 
SBasePlugin_getURI(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return NULL;
  return plugin->getElementNamespace().c_str();
}

/**
 * Returns the prefix of the given plugin structure.
 *
 * @param plugin the plugin structure
 * 
 * @return the prefix of the given plugin structure, or NULL
 * in case an invalid plugin structure is provided. 
 */
LIBSBML_EXTERN
const char* 
SBasePlugin_getPrefix(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return NULL;
  return plugin->getPrefix().c_str();
}

/**
 * Returns the package name of the given plugin structure.
 *
 * @param plugin the plugin structure
 * 
 * @return the package name of the given plugin structure, or NULL
 * in case an invalid plugin structure is provided. 
 */
LIBSBML_EXTERN
const char* 
SBasePlugin_getPackageName(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return NULL;
  return plugin->getPackageName().c_str();
}

/**
 * Creates a deep copy of the given SBasePlugin_t structure
 * 
 * @param plugin the SBasePlugin_t structure to be copied
 * 
 * @return a (deep) copy of the given SBasePlugin_t structure.
 */
LIBSBML_EXTERN
SBasePlugin_t*
SBasePlugin_clone(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return NULL;
  return plugin->clone();
}

/**
 * Frees the given SBasePlugin_t structure
 * 
 * @param plugin the SBasePlugin_t structure to be freed
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
SBasePlugin_free(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return LIBSBML_INVALID_OBJECT;
  delete plugin;
  return LIBSBML_OPERATION_SUCCESS;
}


/**
 * Subclasses must override this method to create, store, and then
 * return an SBML object corresponding to the next XMLToken in the
 * XMLInputStream if they have their specific elements.
 *
 * @param plugin the SBasePlugin_t structure 
 * @param stream the XMLInputStream_t structure to read from
 *
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized or plugin or stream 
 * were NULL.
 */
LIBSBML_EXTERN
SBase_t*
SBasePlugin_createObject(SBasePlugin_t* plugin, XMLInputStream_t* stream)
{
  if (plugin == NULL || stream == NULL) return NULL;
  return plugin->createObject(*stream);
}

/**
 * Subclasses should override this method to read (and store) XHTML,
 * MathML, etc. directly from the XMLInputStream if the target elements
 * can't be parsed by SBase::readAnnotation() and/or SBase::readNotes() 
 * functions
 *
 * @param plugin the SBasePlugin_t structure 
 * @param parentObject the SBase_t structure that will store the annotation.
 * @param stream the XMLInputStream_t structure to read from
 *
 * @return true (1) if the subclass read from the stream, false (0) otherwise. 
 * If an invalid plugin or stream was provided LIBSBML_INVALID_OBJECT is returned.
 */
LIBSBML_EXTERN
int
SBasePlugin_readOtherXML(SBasePlugin_t* plugin, SBase_t* parentObject, XMLInputStream_t* stream)
{
  if (plugin == NULL || stream == NULL) return LIBSBML_INVALID_OBJECT;
  return plugin->readOtherXML(parentObject, *stream);
}

/**
 * Subclasses must override this method to write out their contained
 * SBML objects as XML elements if they have their specific elements.
 *
 * @param plugin the SBasePlugin_t structure  
 * @param stream the XMLOutputStream_t structure to write to
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
SBasePlugin_writeElements(SBasePlugin_t* plugin, XMLOutputStream_t* stream)
{
  if (plugin == NULL || stream == NULL) return LIBSBML_INVALID_OBJECT;
  plugin->writeElements(*stream);
  return LIBSBML_OPERATION_SUCCESS;
}

/** 
 * Checks if the plugin structure has all the required elements.
 *
 * Subclasses should override this function if they have their specific 
 * elements.
 *
 * @param plugin the SBasePlugin_t structure  
 * 
 * @return true (1) if this pugin object has all the required elements,
 * otherwise false (0) will be returned. If an invalid plugin 
 * was provided LIBSBML_INVALID_OBJECT is returned.
 */
LIBSBML_EXTERN
int
SBasePlugin_hasRequiredElements(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return LIBSBML_INVALID_OBJECT;
  return plugin->hasRequiredElements();
}

/**
 * Subclasses should override this method to get the list of
 * expected attributes if they have their specific attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 * 
 * @param plugin the SBasePlugin_t structure  
 * @param attributes the ExpectedAttributes_t structure  
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
SBasePlugin_addExpectedAttributes(SBasePlugin_t* plugin, ExpectedAttributes_t* attributes)
{
  if (plugin == NULL || attributes == NULL) return LIBSBML_INVALID_OBJECT;
  plugin->addExpectedAttributes(*attributes);
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Subclasses must override this method to read values from the given
 * XMLAttributes if they have their specific attributes.
 * 
 * @param plugin the SBasePlugin_t structure  
 * @param attributes the XMLAttributes_t structure  
 * @param expectedAttributes the ExpectedAttributes_t structure  
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
SBasePlugin_readAttributes(SBasePlugin_t* plugin, XMLAttributes_t* attributes, 
  ExpectedAttributes_t* expectedAttributes)
{
  if (plugin == NULL || attributes == NULL ||expectedAttributes == NULL ) 
    return LIBSBML_INVALID_OBJECT;
  plugin->readAttributes(*attributes, *expectedAttributes);
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Subclasses must override this method to write their XML attributes
 * to the XMLOutputStream if they have their specific attributes.
 * 
 * @param plugin the SBasePlugin_t structure  
 * @param stream the XMLOutputStream_t structure  
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
SBasePlugin_writeAttributes(SBasePlugin_t* plugin, XMLOutputStream_t* stream)
{
  if (plugin == NULL || stream == NULL) return LIBSBML_INVALID_OBJECT;
  plugin->writeAttributes(*stream);
  return LIBSBML_OPERATION_SUCCESS;
}

/** 
 * Checks if the plugin structure has all the required attributes.
 *
 * Subclasses should override this function if they have their specific 
 * attributes.
 *
 * @param plugin the SBasePlugin_t structure  
 * 
 * @return true (1) if this pugin object has all the required attributes,
 * otherwise false (0) will be returned. If an invalid plugin 
 * was provided LIBSBML_INVALID_OBJECT is returned.
 */
LIBSBML_EXTERN
int
SBasePlugin_hasRequiredAttributes(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return LIBSBML_INVALID_OBJECT;
  return plugin->hasRequiredAttributes();
}

/**
 * Subclasses should override this method to write required xmlns attributes
 * to the XMLOutputStream (if any). 
 * The xmlns attribute will be written in the element to which the object
 * is connected. For example, xmlns attributes written by this function will
 * be added to Model element if this plugin object connected to the Model 
 * element.
 * 
 * @param plugin the SBasePlugin_t structure  
 * @param stream the XMLOutputStream_t structure  
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
SBasePlugin_writeXMLNS(SBasePlugin_t* plugin, XMLOutputStream_t* stream)
{
  if (plugin == NULL || stream == NULL) return LIBSBML_INVALID_OBJECT;
  plugin->writeXMLNS(*stream);
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Sets the parent SBMLDocument of th plugin structure.
 *
 * Subclasses which contain one or more SBase derived elements must
 * override this function.
 *
 * @param plugin the SBasePlugin_t structure  
 * @param d the SBMLDocument object to use
 *
 * @see SBasePlugin_connectToParent
 * @see SBasePlugin_enablePackageInternal
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
SBasePlugin_setSBMLDocument(SBasePlugin_t* plugin, SBMLDocument_t* d)
{
  if (plugin == NULL) return LIBSBML_INVALID_OBJECT;
  plugin->setSBMLDocument(d);
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Sets the parent SBML object of this plugin structure to
 * this object and child elements (if any).
 * (Creates a child-parent relationship by this plugin object)
 *
 * This function is called when this object is created by
 * the parent element.
 * Subclasses must override this this function if they have one
 * or more child elements. Also, SBasePlugin::connectToParent()
 * must be called in the overridden function.
 *
 * @param plugin the SBasePlugin_t structure  
 * @param sbase the SBase_t structure to use
 *
 * @see SBasePlugin_setSBMLDocument
 * @see SBasePlugin_enablePackageInternal
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
SBasePlugin_connectToParent(SBasePlugin_t* plugin, SBase_t* sbase)
{
  if (plugin == NULL) return LIBSBML_INVALID_OBJECT;
  plugin->connectToParent(sbase);
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Enables/Disables the given package with child elements in this plugin 
 * object (if any).
 * (This is an internal implementation invoked from 
 *  SBase::enablePackageInternal() function)
 *
 * Subclasses which contain one or more SBase derived elements should 
 * override this function if elements defined in them can be extended by
 * some other package extension.
 *
 * @param plugin the SBasePlugin_t structure  
 * @param pkgURI the package uri
 * @param pkgPrefix the package prefix
 * @param flag indicating whether the package should be enabled (1) or disabled(0)
 *
 * @see SBasePlugin_setSBMLDocument
 * @see SBasePlugin_connectToParent
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
SBasePlugin_enablePackageInternal(SBasePlugin_t* plugin, 
    const char* pkgURI, const char* pkgPrefix, int flag)
{
  if (plugin == NULL || pkgURI == NULL || pkgPrefix == NULL)
    return LIBSBML_INVALID_OBJECT;
  plugin->enablePackageInternal(pkgURI, pkgPrefix, (bool)flag);
  return LIBSBML_OPERATION_SUCCESS;
}

/**
 * Returns the parent SBMLDocument of this plugin structure.
 *
 * @param plugin the SBasePlugin_t structure  
 *
 * @return the parent SBMLDocument object of this plugin object or NULL if 
 * no document is set, or the plugin structure is invalid. 
 */
LIBSBML_EXTERN
SBMLDocument_t*
SBasePlugin_getSBMLDocument(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return NULL;
  return plugin->getSBMLDocument();
}

/**
 * Returns the parent SBase structure to which this plugin structure is connected.
 *
 * @param plugin the SBasePlugin_t structure  
 *
 * @return the parent SBase structure to which this plugin structure is connected
 * or NULL if sbase structure is set, or the plugin structure is invalid. 
 */
LIBSBML_EXTERN
SBase_t*
SBasePlugin_getParentSBMLObject(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return NULL;
  return plugin->getParentSBMLObject();
}

/**
 * Returns the SBML level of the package extension of 
 * this plugin structure.
 *
 * @param plugin the SBasePlugin_t structure  
 *
 * @return the SBML level of the package extension of
 * this plugin object or SBML_INT_MAX if the structure is invalid.
 */
LIBSBML_EXTERN
unsigned int
SBasePlugin_getLevel(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return SBML_INT_MAX;
  return plugin->getLevel();
}

/**
 * Returns the SBML version of the package extension of 
 * this plugin structure.
 *
 * @param plugin the SBasePlugin_t structure  
 *
 * @return the SBML version of the package extension of
 * this plugin object or SBML_INT_MAX if the structure is invalid.
 */
LIBSBML_EXTERN
unsigned int
SBasePlugin_getVersion(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return SBML_INT_MAX;
  return plugin->getVersion();
}

/**
 * Returns the package version of the package extension of 
 * this plugin structure.
 *
 * @param plugin the SBasePlugin_t structure  
 *
 * @return the package version of the package extension of
 * this plugin object or SBML_INT_MAX if the structure is invalid.
 */
LIBSBML_EXTERN
unsigned int
SBasePlugin_getPackageVersion(SBasePlugin_t* plugin)
{
  if (plugin == NULL) return SBML_INT_MAX;
  return plugin->getPackageVersion();
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


