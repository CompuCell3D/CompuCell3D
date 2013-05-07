/**
 * @file    SBaseExtensionPoint.h
 * @brief   Definition of SBaseExtensionPoint
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
 *
 * @class SBaseExtensionPoint
 * @brief Representation of an extension point of SBML's package extension.
 *
 * SBaseExtensionPoint represents an element to be extended (extension point) and the
 * extension point is identified by a combination of a package name and a typecode of the 
 * element.
 * 
 * <p>
 * For example, an SBaseExtensionPoint object which represents an extension point of the model
 * element defined in the <em>core</em> package can be created as follows:
 *
@verbatim
      SBaseExtensionPoint  modelextp("core", SBML_MODEL);
@endverbatim
 * 
 * Similarly, an SBaseExtensionPoint object which represents an extension point of
 * the layout element defined in the layout extension can be created as follows:
 * 
@verbatim
      SBaseExtensionPoint  layoutextp("layout", SBML_LAYOUT_LAYOUT);
@endverbatim
 * 
 * SBaseExtensionPoint object is required as one of arguments of the constructor 
 * of SBasePluginCreator&lt;class SBasePluginType, class SBMLExtensionType&gt;
 * template class to identify an extension poitnt to which the plugin object created
 * by the creator class is plugged in.
 * For example, the SBasePluginCreator class which creates a LayoutModelPlugin object
 * of the layout extension which is plugged in to the model element of the <em>core</em>
 * package can be created with the corresponding SBaseExtensionPoint object as follows:
 *
@verbatim
  // std::vector object that contains a list of URI (package versions) supported 
  // by the plugin object.
  std::vector<std::string> packageURIs;
  packageURIs.push_back(getXmlnsL3V1V1());
  packageURIs.push_back(getXmlnsL2());  

  // creates an extension point (model element of the "core" package)
  SBaseExtensionPoint  modelExtPoint("core",SBML_MODEL);
   
  // creates an SBasePluginCreator object 
  SBasePluginCreator<LayoutModelPlugin, LayoutExtension>  modelPluginCreator(modelExtPoint,packageURIs);
@endverbatim
 *
 * This kind of code is implemented in init() function of each SBMLExtension derived classes.
 */

#ifndef SBaseExtensionPoint_h
#define SBaseExtensionPoint_h

#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>

#ifdef __cplusplus

#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN SBaseExtensionPoint
{
public:

  /**
   * constructor
   */
  SBaseExtensionPoint(const std::string& pkgName, int typeCode);  


  /**
   * copy constructor
   */
  SBaseExtensionPoint(const SBaseExtensionPoint& rhs);


  /**
   * clone 
   */
  SBaseExtensionPoint* clone() const;


  /**
   * Returns the package name of this extension point.
   */
  const std::string& getPackageName() const;


  /**
   * Returns the typecode of this extension point.
   */
  int getTypeCode() const;

private:
  std::string mPackageName;
  int         mTypeCode;
};


#ifndef SWIG

/**
 * Comparison (equal-to) operator for SBaseExtensionPoint
 */
bool operator==(const SBaseExtensionPoint& lhs, const SBaseExtensionPoint& rhs); 


/**
 * Comparison (less-than) operator for SBaseExtensionPoint
 */
bool operator<(const SBaseExtensionPoint& lhs, const SBaseExtensionPoint& rhs); 

#endif //SWIG


LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

LIBSBML_EXTERN 
SBaseExtensionPoint_t *
SBaseExtensionPoint_create(const char* pkgName, int typeCode);

LIBSBML_EXTERN 
int
SBaseExtensionPoint_free(SBaseExtensionPoint_t *extPoint);

LIBSBML_EXTERN 
SBaseExtensionPoint_t *
SBaseExtensionPoint_clone(const SBaseExtensionPoint_t *extPoint);

LIBSBML_EXTERN 
char *
SBaseExtensionPoint_getPackageName(const SBaseExtensionPoint_t *extPoint);

LIBSBML_EXTERN 
int
SBaseExtensionPoint_getTypeCode(const SBaseExtensionPoint_t *extPoint);



END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */

#endif  /* SBaseExtensionPoint_h */


