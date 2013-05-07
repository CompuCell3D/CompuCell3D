/**
 * @file    SBMLNamespaces.cpp
 * @brief   SBMLNamespaces class to store level/version and namespace 
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
 * ---------------------------------------------------------------------- -->
 */

#include <sbml/SBMLNamespaces.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/extension/SBMLExtensionException.h>
#include <sstream>
#include <sbml/common/common.h>
#include <iostream>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */


LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygen-libsbml-internal */
void 
SBMLNamespaces::initSBMLNamespace()
{
  mNamespaces = new XMLNamespaces();

  switch (mLevel)
  {
  case 1:
    switch (mVersion)
    {
    case 1:
    case 2:
      mNamespaces->add(SBML_XMLNS_L1);
      break;
    }
    break;
  case 2:
    switch (mVersion)
    {
    case 1:
      mNamespaces->add(SBML_XMLNS_L2V1);
      break;
    case 2:
      mNamespaces->add(SBML_XMLNS_L2V2);
      break;
    case 3:
      mNamespaces->add(SBML_XMLNS_L2V3);
      break;
    case 4:
      mNamespaces->add(SBML_XMLNS_L2V4);
      break;
    }
    break;
  case 3:
    switch (mVersion)
    {
    case 1:
      mNamespaces->add(SBML_XMLNS_L3V1);
      break;
    }
    break;
  }

  if (mNamespaces->getLength() == 0)
  {
    mLevel = SBML_INT_MAX;
    mVersion = SBML_INT_MAX;
    delete mNamespaces;
    mNamespaces = NULL;
  }
}
/** @endcond */


SBMLNamespaces::SBMLNamespaces(unsigned int level, unsigned int version)
 : mLevel(level)
  ,mVersion(version)
{
  initSBMLNamespace();
}


/**
 * (For Extension)
 *
 * Creates a new SBMLNamespaces object corresponding to the combination of
 * (1) the given SBML @p level and @p version, and (2) the given @p package
 * with the @p package @p version.
 *
 */
SBMLNamespaces::SBMLNamespaces(unsigned int level, unsigned int version, 
                               const std::string &pkgName, unsigned int pkgVersion, 
                               const std::string& pkgPrefix)
 : mLevel(level)
  ,mVersion(version)
{
  initSBMLNamespace();

  //
  // checks the URI of the given package
  //
  const SBMLExtension* sbmlext = SBMLExtensionRegistry::getInstance().getExtensionInternal(pkgName);
  if (sbmlext)
  {
    const std::string uri    = sbmlext->getURI(level, version, pkgVersion);
    const std::string prefix = (pkgPrefix.empty()) ? pkgName : pkgPrefix;

    if (!uri.empty())
    {
      mNamespaces->add(uri,prefix); 
    }
    else
    {
      std::ostringstream errMsg;

      errMsg << "Package \"" << pkgName << "\" SBML level " << level << " SBML version " 
             << version << " package version " << pkgVersion << " is not supported.";

      throw SBMLExtensionException(errMsg.str());
    }
  }
  else
  {
    std::ostringstream errMsg;

    errMsg << pkgName << " : No such package registered.";

    throw SBMLExtensionException(errMsg.str());
  }
}

SBMLNamespaces::~SBMLNamespaces()
{
  if (mNamespaces != NULL)
    delete mNamespaces;
}


/*
 * Copy constructor; creates a copy of a SBMLNamespaces.
 */
SBMLNamespaces::SBMLNamespaces(const SBMLNamespaces& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    mLevel   = orig.mLevel;
    mVersion = orig.mVersion;
 
    if(orig.mNamespaces != NULL)
      this->mNamespaces = 
            new XMLNamespaces(*const_cast<SBMLNamespaces&>(orig).mNamespaces);
    else
      this->mNamespaces = NULL;
  }
}


const List * 
SBMLNamespaces::getSupportedNamespaces()
{
  List *result = new List();
  result->add(new SBMLNamespaces(1,1));
  result->add(new SBMLNamespaces(1,2));
  result->add(new SBMLNamespaces(2,1));
  result->add(new SBMLNamespaces(2,2));
  result->add(new SBMLNamespaces(2,3));
  result->add(new SBMLNamespaces(2,4));
  result->add(new SBMLNamespaces(3,1));
  return result;
}

/*
 * Assignment operator for SBMLNamespaces.
 */
SBMLNamespaces&
SBMLNamespaces::operator=(const SBMLNamespaces& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if (&rhs != this)
  {
    mLevel   = rhs.mLevel;
    mVersion = rhs.mVersion;
    delete this->mNamespaces;
    if(rhs.mNamespaces != NULL)
      this->mNamespaces = 
            new XMLNamespaces(*const_cast<SBMLNamespaces&>(rhs).mNamespaces);
    else
      this->mNamespaces = NULL;
  }

  return *this;
}



/*
 * Creates and returns a deep copy of this SBMLNamespaces.
 */
SBMLNamespaces *
SBMLNamespaces::clone () const
{
  return new SBMLNamespaces(*this);
}


std::string 
SBMLNamespaces::getSBMLNamespaceURI(unsigned int level,
                                 unsigned int version)
{
  std::string uri = "";
  switch (level)
  {
  case 1:
    uri = SBML_XMLNS_L1;
    break;
  case 3:
    switch(version)
    {
    case 1:
    default:
      uri = SBML_XMLNS_L3V1;
      break;
    }
    break;
  case 2:
  default:
    switch (version)
    {
    case 1:
      uri = SBML_XMLNS_L2V1;
      break;
    case 2:
      uri = SBML_XMLNS_L2V2;
      break;
    case 3:
      uri = SBML_XMLNS_L2V3;
      break;
    case 4:
    default:
      uri = SBML_XMLNS_L2V4;
      break;
    }
    break;
  }
  return uri;
}


std::string
SBMLNamespaces::getURI() const
{
  return getSBMLNamespaceURI(mLevel,mVersion);
}


unsigned int 
SBMLNamespaces::getLevel()
{
  return mLevel;
}


unsigned int 
SBMLNamespaces::getLevel() const
{
  return mLevel;
}


unsigned int 
SBMLNamespaces::getVersion()
{
  return mVersion;
}


unsigned int 
SBMLNamespaces::getVersion() const
{
  return mVersion;
}


XMLNamespaces * 
SBMLNamespaces::getNamespaces()
{
  return mNamespaces;
}


const XMLNamespaces * 
SBMLNamespaces::getNamespaces() const
{
  return mNamespaces;
}


int
SBMLNamespaces::addNamespaces(const XMLNamespaces * xmlns)
{
  int success = LIBSBML_OPERATION_SUCCESS;

  if (xmlns == NULL)
    return LIBSBML_INVALID_OBJECT;

  if (!mNamespaces) 
  {
    initSBMLNamespace();
  }

  /* check whether the namespace already exists
   * add if it does not
   */
  for (int i = 0; i < xmlns->getLength(); i++)
  {
    if (!(mNamespaces->hasNS(xmlns->getURI(i), xmlns->getPrefix(i))))
    {
      success = mNamespaces->add(xmlns->getURI(i), xmlns->getPrefix(i));
    }
  }

  return success;
}

/*
 * (For Extension)
 *
 * Add an XML namespace (a pair of URI and prefix) of a package extension
 * to the set of namespaces within this SBMLNamespaces object.
 *
 */
int 
SBMLNamespaces::addPackageNamespace(const std::string &pkgName, unsigned int pkgVersion, 
                                const std::string &pkgPrefix)
{
  if (!mNamespaces) 
  {
    initSBMLNamespace();
  }

  //
  // checks the URI of the given package
  //
  const SBMLExtension* sbmlext = SBMLExtensionRegistry::getInstance().getExtensionInternal(pkgName);
  if (sbmlext)
  {
    const std::string uri    = sbmlext->getURI(mLevel, mVersion, pkgVersion);
    const std::string prefix = (pkgPrefix.empty()) ? pkgName : pkgPrefix;
    if (!uri.empty())
    {
      mNamespaces->add(uri,prefix);
    }
    else
    {
      return LIBSBML_INVALID_ATTRIBUTE_VALUE;
    }
  }
  else
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }

  return LIBSBML_OPERATION_SUCCESS;
}


/** @cond doxygen-libsbml-internal */
/*
 * (For Extension)
 *
 * Add an XML namespace (a pair of URI and prefix) of a package extension
 * to the set of namespaces within this SBMLNamespaces object.
 *
 */
int 
SBMLNamespaces::addPkgNamespace(const std::string &pkgName, unsigned int pkgVersion, 
                                const std::string &pkgPrefix)
{

  return addPackageNamespace(pkgName, pkgVersion, pkgPrefix);
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
/*
 * Add the XML namespaces of package extensions in the given
 * XMLNamespace object to the set of namespaces within this
 * SBMLNamespaces object (Non-package XML namespaces are not added
 * by this function).
 */
int
SBMLNamespaces::addPackageNamespaces (const XMLNamespaces *xmlns)
{
  if (!mNamespaces) 
  {
    initSBMLNamespace();
  }

  if (!xmlns) 
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }

  for (int i=0; i < xmlns->getLength(); i++)
  {
    const std::string uri = xmlns->getURI(i);

    if (SBMLExtensionRegistry::getInstance().isRegistered(uri))
    {
      mNamespaces->add(uri, xmlns->getPrefix(i));
    }
  }

  return LIBSBML_OPERATION_SUCCESS;
}
/** @endcond */

/** @cond doxygen-libsbml-internal */
int
SBMLNamespaces::addPkgNamespaces (const XMLNamespaces *xmlns)
{
  return addPackageNamespaces(xmlns);
}
/** @endcond */

int
SBMLNamespaces::addNamespace(const std::string &uri, const std::string &prefix)
{
  if (!mNamespaces) 
  {
    initSBMLNamespace();
  }

  return mNamespaces->add(uri, prefix);
}


int
SBMLNamespaces::removeNamespace(const std::string &uri)
{
  if (!mNamespaces) 
  {
    initSBMLNamespace();
  }

  return mNamespaces->remove(mNamespaces->getIndex(uri));
}


/*
 * Removes an XML namespace of a package extension from the set of namespaces 
 * within this SBMLNamespaces object.
 */
int
SBMLNamespaces::removePackageNamespace(unsigned int level, unsigned version, const std::string &pkgName,
                                   unsigned int pkgVersion)
{
  //
  // checks the URI of the given package
  //
  const SBMLExtension* sbmlext = SBMLExtensionRegistry::getInstance().getExtensionInternal(pkgName);
  if (sbmlext)
  {
    if (!mNamespaces) 
    {
      return LIBSBML_OPERATION_SUCCESS;
    }

    const std::string uri = sbmlext->getURI(level, version, pkgVersion);
    if (!uri.empty())
    {
      return mNamespaces->remove(mNamespaces->getIndex(uri));
    }
    else
    {
      return LIBSBML_INVALID_ATTRIBUTE_VALUE;
    }
  }
  else
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
}

/** @cond doxygen-libsbml-internal */
int
SBMLNamespaces::removePkgNamespace(unsigned int level, unsigned version, const std::string &pkgName,
                                   unsigned int pkgVersion)
{
  return removePackageNamespace(level, version, pkgName, pkgVersion);
}
/** @endcond */

/*
 * Predicate returning @c true if the given
 * URL is one of SBML XML namespaces.
 */
bool 
SBMLNamespaces::isSBMLNamespace(const std::string& uri)
{
  if (uri == SBML_XMLNS_L1)   return true;
  if (uri == SBML_XMLNS_L2V1) return true;
  if (uri == SBML_XMLNS_L2V2) return true;
  if (uri == SBML_XMLNS_L2V3) return true;
  if (uri == SBML_XMLNS_L2V4) return true;
  if (uri == SBML_XMLNS_L3V1) return true;

  return false;
}

bool 
SBMLNamespaces::isValidCombination()
{
  bool valid = true;
  bool sbmlDeclared = false;
  std::string declaredURI("");
  unsigned int version = getVersion();
  XMLNamespaces *xmlns = getNamespaces();

  if (xmlns != NULL)
  {
    // 
    // checks defined SBML XMLNamespace
    // returns false if different SBML XMLNamespaces 
    // (e.g. SBML_XMLNS_L2V1 and SBML_XMLNS_L2V3) are defined.
    //
    int numNS = 0;

    if (xmlns->hasURI(SBML_XMLNS_L3V1))
    {
      ++numNS;
      declaredURI.assign(SBML_XMLNS_L3V1);
    }

    if (xmlns->hasURI(SBML_XMLNS_L2V4))
    {
      if (numNS > 0) return false;
      ++numNS;
      declaredURI.assign(SBML_XMLNS_L2V4);
    }

    if (xmlns->hasURI(SBML_XMLNS_L2V3))
    {
      // checks different SBML XMLNamespaces
      if (numNS > 0) return false;
      ++numNS;
      declaredURI.assign(SBML_XMLNS_L2V3);
    }

    if (xmlns->hasURI(SBML_XMLNS_L2V2))
    {
      // checks different SBML XMLNamespaces
      if (numNS > 0) return false;
      ++numNS;
      declaredURI.assign(SBML_XMLNS_L2V2);
    }

    if (xmlns->hasURI(SBML_XMLNS_L2V1))
    {
      // checks different SBML XMLNamespaces
      if (numNS > 0) return false;
      ++numNS;
      declaredURI.assign(SBML_XMLNS_L2V1);
    }

    if (xmlns->hasURI(SBML_XMLNS_L1))
    {
      // checks different SBML XMLNamespaces
      if (numNS > 0) return false;
      ++numNS;
      declaredURI.assign(SBML_XMLNS_L1);
    }

    // checks if the SBML Namespace is explicitly defined.
    for (int i=0; i < xmlns->getLength(); i++)
    {
      if (!declaredURI.empty() && 
                      xmlns->getURI(i) == declaredURI)
      {
        sbmlDeclared = true;
        break;
      }
    }
  }


  switch (getLevel())
  {
    case 1:
     switch (version)
      {
        case 1:
        case 2:
          // the namespaces contains the sbml namespaces
          // check it is the correct ns for the level/version
          if (sbmlDeclared)
          {
            if (declaredURI != string(SBML_XMLNS_L1))
            {
              valid = false;
            }
          }
          break;
        default:
          valid = false;
          break;
        }
      break;
    case 2:
      switch (version)
      {
        case 1:
          // the namespaces contains the sbml namespaces
          // check it is the correct ns for the level/version
          if (sbmlDeclared)
          {
            if (declaredURI != string(SBML_XMLNS_L2V1))
            {
              valid = false;
            }
          }
          break;
        case 2:
          // the namespaces contains the sbml namespaces
          // check it is the correct ns for the level/version
          if (sbmlDeclared)
          {
            if (declaredURI != string(SBML_XMLNS_L2V2))
            {
              valid = false;
            }
          }
          break;
        case 3:
          // the namespaces contains the sbml namespaces
          // check it is the correct ns for the level/version
          if (sbmlDeclared)
          {
            if (declaredURI != string(SBML_XMLNS_L2V3))
            {
              valid = false;
            }
          }
          break;
        case 4:
          // the namespaces contains the sbml namespaces
          // check it is the correct ns for the level/version
          if (sbmlDeclared)
          {
            if (declaredURI != string(SBML_XMLNS_L2V4))
            {
              valid = false;
            }
          }
          break;
        default:
          valid = false;
          break;
        }
      break;
    case 3:
      switch (version)
      {
        case 1:
         // the namespaces contains the sbml namespaces
          // check it is the correct ns for the level/version
          if (sbmlDeclared)
          {
            if (declaredURI != string(SBML_XMLNS_L3V1))
            {
              valid = false;
            }
          }
          break;
        default:
          valid = false;
          break;
      }
      break;
    default:
      valid = false;
      break;
  }

  return valid;
}


/** @cond doxygen-libsbml-internal */
void 
SBMLNamespaces::setLevel(unsigned int level)
{
  mLevel = level;
}


void 
SBMLNamespaces::setVersion(unsigned int version)
{
  mVersion = version;
}

const std::string& 
SBMLNamespaces::getPackageName () const
{
	static const std::string pkgName = "core";
    return pkgName;
}

void 
SBMLNamespaces::setNamespaces(XMLNamespaces * xmlns)
{
  delete mNamespaces;
  if (xmlns != NULL)
    mNamespaces = xmlns->clone();
  else
    mNamespaces = NULL;
}
/** @endcond */

/** @cond doxygen-c-only */

/**
 * Creates a new SBMLNamespaces_t structure corresponding to the given SBML
 * @p level and @p version.
 *
 * SBMLNamespaces objects are used in libSBML to communicate SBML Level
 * and Version data between constructors and other methods.  The
 * SBMLNamespaces object class tracks 3-tuples (triples) consisting of
 * SBML Level, Version, and the corresponding SBML XML namespace.  Most
 * constructors for SBML objects in libSBML take a SBMLNamespaces object
 * as an argument, thereby allowing the constructor to produce the proper
 * combination of attributes and other internal data structures for the
 * given SBML Level and Version.
 *
 * The plural name "SBMLNamespaces" is not a mistake, because in SBML
 * Level&nbsp;3, objects may have extensions added by Level&nbsp;3
 * packages used by a given model; however, until the introduction of
 * SBML Level&nbsp;3, the SBMLNamespaces object only records one SBML
 * Level/Version/namespace combination at a time.
 *
 * @param level the SBML level
 * @param version the SBML version
 *
 * @return SBMLNamespaces_t structure created
 *
 * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
 */

LIBSBML_EXTERN
SBMLNamespaces_t *
SBMLNamespaces_create(unsigned int level, unsigned int version)
{
  return new SBMLNamespaces(level, version);
}


/**
 * Get the SBML Level of this SBMLNamespaces_t structure.
 *
 * @param sbmlns the SBMLNamespaces_t structure to query
 *
 * @return the SBML Level of this SBMLNamespaces_t structure.
 */
LIBSBML_EXTERN
unsigned int
SBMLNamespaces_getLevel(SBMLNamespaces_t *sbmlns)
{
  return (sbmlns != NULL) ? sbmlns->getLevel() : SBML_INT_MAX;
}


/**
 * Get the SBML Version of this SBMLNamespaces_t structure.
 *
 * @param sbmlns the SBMLNamespaces_t structure to query
 *
 * @return the SBML Version of this SBMLNamespaces_t structure.
 */
LIBSBML_EXTERN
unsigned int
SBMLNamespaces_getVersion(SBMLNamespaces_t *sbmlns)
{
  return (sbmlns != NULL) ? sbmlns->getVersion() : SBML_INT_MAX;
}


/**
 * Get the SBML Version of this SBMLNamespaces_t structure.
 *
 * @param sbmlns the SBMLNamespaces_t structure to query
 *
 * @return the XMLNamespaces_t structure of this SBMLNamespaces_t structure.
 */
LIBSBML_EXTERN
XMLNamespaces_t *
SBMLNamespaces_getNamespaces(SBMLNamespaces_t *sbmlns)
{
  return (sbmlns != NULL) ? sbmlns->getNamespaces() : NULL;
}


/**
 * Returns a string representing the SBML XML namespace for the 
 * given @p level and @p version of SBML.
 *
 * @param level the SBML level
 * @param version the SBML version
 *
 * @return a string representing the SBML namespace that reflects the
 * SBML Level and Version specified.
 */
LIBSBML_EXTERN
char *
SBMLNamespaces_getSBMLNamespaceURI(unsigned int level, unsigned int version)
{
  return safe_strdup(SBMLNamespaces::getSBMLNamespaceURI(level, version).c_str());
}


/**
 * Add the XML namespaces list to the set of namespaces
 * within this SBMLNamespaces_t structure.
 * 
 * @param sbmlns the SBMLNamespaces_t structure to add to
 * @param xmlns the XML namespaces to be added.
 */
LIBSBML_EXTERN
int
SBMLNamespaces_addNamespaces(SBMLNamespaces_t *sbmlns,
                             const XMLNamespaces_t * xmlns)
{
  if (sbmlns != NULL)
    return sbmlns->addNamespaces(xmlns);
  else
    return LIBSBML_INVALID_OBJECT;
}

/**
 * Returns an array of SBML Namespaces supported by this version of 
 * LibSBML. 
 *
 * @param length an integer holding the length of the array
 * @return an array of SBML namespaces, or NULL if length is NULL. The array 
 *         has to be freed by the caller.
 */
LIBSBML_EXTERN
SBMLNamespaces_t **
SBMLNamespaces_getSupportedNamespaces(int *length)
{
  if (length == NULL) return NULL;
   const List* supported = SBMLNamespaces::getSupportedNamespaces();
  
   *length = (int) supported->getSize();
  SBMLNamespaces_t ** result = (SBMLNamespaces_t**)malloc(sizeof(SBMLNamespaces_t*)*(*length));
  memset(result, 0, sizeof(SBMLNamespaces_t*)*(*length));
  for (int i = 0; i < *length; i++)
  {
    result[i] = ((SBMLNamespaces*)supported->get(i))->clone();
  }
  return result;
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

