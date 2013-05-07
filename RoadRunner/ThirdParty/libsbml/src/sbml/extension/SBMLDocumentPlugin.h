/**
 * @file    SBMLDocumentPlugin.h
 * @brief   Definition of SBMLDocumentPlugin, the derived class of
 *          SBasePlugin.
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

#ifndef SBMLDocumentPlugin_h
#define SBMLDocumentPlugin_h

#include <sbml/common/sbmlfwd.h>
#include <sbml/SBMLTypeCodes.h>
#include <sbml/SBMLErrorLog.h>
#include <sbml/SBMLDocument.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>
#include <sbml/extension/SBasePlugin.h>

#ifdef __cplusplus

LIBSBML_CPP_NAMESPACE_BEGIN

//
// (NOTE) Plugin objects for the SBMLDocument element must be this class or 
//        a derived class of this class.
//        Package developers should use this class as-is if only "required" 
//        attribute is added in the SBMLDocument element by their packages, 
//        otherwise developers must implement a derived class of this class 
//        and use the class as the plugin object for the SBMLDocument element. 
//

class LIBSBML_EXTERN SBMLDocumentPlugin : public SBasePlugin
{
public:

  /**
   *  Constructor
   *
   * @param uri the URI of package 
   * @param prefix the prefix for the given package
   * @param sbmlns the SBMLNamespaces object for the package
   */
  SBMLDocumentPlugin (const std::string &uri, const std::string &prefix,
                      SBMLNamespaces *sbmlns);


  /**
   * Copy constructor. Creates a copy of this object.
   */
  SBMLDocumentPlugin(const SBMLDocumentPlugin& orig);


  /**
   * Destroy this object.
   */
  virtual ~SBMLDocumentPlugin ();

  /**
   * Assignment operator for SBMLDocumentPlugin.
   */
  SBMLDocumentPlugin& operator=(const SBMLDocumentPlugin& orig);


  /**
   * Creates and returns a deep copy of this SBMLDocumentPlugin object.
   * 
   * @return a (deep) copy of this object
   */
  virtual SBMLDocumentPlugin* clone () const;


  // ----------------------------------------------------------
  //
  // overridden virtual functions for reading/writing/checking
  // attributes
  //
  // ----------------------------------------------------------

#ifndef SWIG

  /** @cond doxygen-libsbml-internal */

  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);


  /**
   * Reads the attributes of corresponding package in SBMLDocument element.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);


  /**
   * Writes the attributes of corresponding package in SBMLDocument element.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;

  /** @endcond */

#endif //SWIG

  // -----------------------------------------------------------
  //
  // Additional public functions for manipulating attributes of 
  // corresponding package in SBMLDocument element.
  //
  // -----------------------------------------------------------


  /**
   *
   * Sets the bool value of "required" attribute of corresponding package
   * in SBMLDocument element.
   *
   * @param value the bool value of "required" attribute of corresponding 
   * package in SBMLDocument element.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   */
  virtual int setRequired(bool value);


  /**
   *
   * Returns the bool value of "required" attribute of corresponding 
   * package in SBMLDocument element.
   *
   * @return the bool value of "required" attribute of corresponding
   * package in SBMLDocument element.
   */
  virtual bool getRequired() const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * SBMLDocumentPlugin's "required" attribute has been set.
   *
   * @return @c true if the "required" attribute of this SBMLDocument has been
   * set, @c false otherwise.
   */
  virtual bool isSetRequired() const;


  /**
   * Unsets the value of the "required" attribute of this SBMLDocumentPlugin.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int unsetRequired();


  



protected:
  /** @cond doxygen-libsbml-internal */

  /*-- data members --*/

  bool mRequired;
  bool mIsSetRequired;

  /** @endcond */

};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

LIBSBML_EXTERN
SBMLDocumentPlugin_t*
SBMLDocumentPlugin_create(const char* uri, const char* prefix, 
      SBMLNamespaces_t* sbmlns);

LIBSBML_EXTERN
SBMLDocumentPlugin_t*
SBMLDocumentPlugin_clone(SBMLDocumentPlugin_t* plugin);

LIBSBML_EXTERN
int
SBMLDocumentPlugin_addExpectedAttributes(SBMLDocumentPlugin_t* plugin, 
      ExpectedAttributes_t* attributes);

LIBSBML_EXTERN
int
SBMLDocumentPlugin_readAttributes(SBMLDocumentPlugin_t* plugin, 
      const XMLAttributes_t* attributes, 
      const ExpectedAttributes_t* expectedAttributes);

LIBSBML_EXTERN
int
SBMLDocumentPlugin_writeAttributes(SBMLDocumentPlugin_t* plugin, 
      XMLOutputStream_t* stream);


LIBSBML_EXTERN
int
SBMLDocumentPlugin_getRequired(SBMLDocumentPlugin_t* plugin);


LIBSBML_EXTERN
int
SBMLDocumentPlugin_setRequired(SBMLDocumentPlugin_t* plugin, int required);


LIBSBML_EXTERN
int
SBMLDocumentPlugin_isSetRequired(SBMLDocumentPlugin_t* plugin);


LIBSBML_EXTERN
int
SBMLDocumentPlugin_unsetRequired(SBMLDocumentPlugin_t* plugin);

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */

#endif  /* SBMLDocumentPlugin_h */
