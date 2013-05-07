/**
 * @file    SBasePluginCreatorBase.h
 * @brief   Definition of SBasePluginCreatorBase, the base class of 
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

#ifndef SBasePluginCreatorBase_h
#define SBasePluginCreatorBase_h


#include <sbml/SBMLDocument.h>
#include <sbml/SBMLNamespaces.h>
#include <sbml/extension/SBaseExtensionPoint.h>

#ifdef __cplusplus

LIBSBML_CPP_NAMESPACE_BEGIN

class SBasePlugin;

class LIBSBML_EXTERN SBasePluginCreatorBase
{
public:

  typedef std::vector<std::string>           SupportedPackageURIList;
  typedef std::vector<std::string>::iterator SupportedPackageURIListIter;

  /**
   * Destructor
   */
  virtual ~SBasePluginCreatorBase ();


  /**
   * Creates an SBasePlugin with the given uri and the prefix
   * of the target package extension.
   */
  virtual SBasePlugin* createPlugin(const std::string& uri, 
                                    const std::string& prefix,
                                    const XMLNamespaces *xmlns) const = 0;


  /**
   * clone
   */
  virtual SBasePluginCreatorBase* clone() const = 0;


  /**
   * Returns the number of supported packages by this creator object.
   */
  unsigned int getNumOfSupportedPackageURI() const;


  /**
   * Returns the supported package to the given index.
   */
  std::string getSupportedPackageURI(unsigned int) const;


  /**
   * Returns an SBMLTypeCode tied to this creator object.
   */
  int getTargetSBMLTypeCode() const;


  /**
   * Returns the target package name of this creator object.
   */
  const std::string& getTargetPackageName() const;


  /**
   * Returns an SBaseExtensionPoint tied to this creator object.
   */
  const SBaseExtensionPoint& getTargetExtensionPoint() const;


  /**
   * Returns true if a package with the given namespace is supported.
   */
  bool isSupported(const std::string& uri) const;

protected:

  /**
   * Constructor
   */
  SBasePluginCreatorBase (const SBaseExtensionPoint& extPoint,
                          const std::vector<std::string>&);


  /**
   * Copy Constructor
   */
  SBasePluginCreatorBase (const SBasePluginCreatorBase&);

  /** @cond doxygen-libsbml-internal */

  SupportedPackageURIList  mSupportedPackageURI;
  SBaseExtensionPoint       mTargetExtensionPoint;

  /** @endcond */


private:
  /** @cond doxygen-libsbml-internal */
  
  SBasePluginCreatorBase& operator=(const SBasePluginCreatorBase&);

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

  
#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

LIBSBML_EXTERN
SBasePlugin_t*
SBasePluginCreator_createPlugin(SBasePluginCreatorBase_t* creator, 
  const char* uri, const char* prefix, const XMLNamespaces_t* xmlns);

LIBSBML_EXTERN
SBasePluginCreatorBase_t*
SBasePluginCreator_clone(SBasePluginCreatorBase_t* creator);


LIBSBML_EXTERN
unsigned int
SBasePluginCreator_getNumOfSupportedPackageURI(SBasePluginCreatorBase_t* creator);

LIBSBML_EXTERN
char*
SBasePluginCreator_getSupportedPackageURI(SBasePluginCreatorBase_t* creator, 
    unsigned int index);

LIBSBML_EXTERN
int
SBasePluginCreator_getTargetSBMLTypeCode(SBasePluginCreatorBase_t* creator);

LIBSBML_EXTERN
const char*
SBasePluginCreator_getTargetPackageName(SBasePluginCreatorBase_t* creator);

LIBSBML_EXTERN
const SBaseExtensionPoint_t*
SBasePluginCreator_getTargetExtensionPoint(SBasePluginCreatorBase_t* creator);

LIBSBML_EXTERN
int 
SBasePluginCreator_isSupported(SBasePluginCreatorBase_t* creator, const char* uri);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */

#endif  /* SBasePluginCreatorBase_h */

