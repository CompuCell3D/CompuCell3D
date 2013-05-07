/**
 * @file    ModelCreator.h
 * @brief   ModelCreator I/O
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
 * ------------------------------------------------------------------------ -->
 * 
 * @class ModelCreator
 * @brief Representation of MIRIAM-compliant model creator data used
 * in ModelHistory. 
 *
 * @htmlinclude not-sbml-warning.html
 *
 * The SBML specification beginning with Level&nbsp;2 Version&nbsp;2
 * defines a standard approach to recording model history and model creator
 * information in a form that complies with MIRIAM ("Minimum Information
 * Requested in the Annotation of biochemical Models", <i>Nature
 * Biotechnology</i>, vol. 23, no. 12, Dec. 2005).  For the model creator,
 * this form involves the use of parts of the <a target="_blank"
 * href="http://en.wikipedia.org/wiki/VCard">vCard</a> representation.
 * LibSBML provides the ModelCreator class as a convenience high-level
 * interface for working with model creator data.  Objects of class
 * ModelCreator can be used to store and carry around creator data within a
 * program, and the various methods in this object class let callers
 * manipulate the different parts of the model creator representation.
 *
 * @section parts The different parts of a model creator definition
 *
 * The ModelCreator class mirrors the structure of the MIRIAM model creator
 * annotations in SBML.  The following template illustrates these different
 * fields when they are written in XML form:
 *
 <pre class="fragment">
 &lt;vCard:N rdf:parseType="Resource"&gt;
   &lt;vCard:Family&gt;<span style="background-color: #bbb">family name</span>&lt;/vCard:Family&gt;
   &lt;vCard:Given&gt;<span style="background-color: #bbb">given name</span>&lt;/vCard:Given&gt;
 &lt;/vCard:N&gt;
 ...
 &lt;vCard:EMAIL&gt;<span style="background-color: #bbb">email address</span>&lt;/vCard:EMAIL&gt;
 ...
 &lt;vCard:ORG rdf:parseType="Resource"&gt;
   &lt;vCard:Orgname&gt;<span style="background-color: #bbb">organization</span>&lt;/vCard:Orgname&gt;
 &lt;/vCard:ORG&gt;
 </pre>
 *
 * Each of the separate data values
 * <span class="code" style="background-color: #bbb">family name</span>,
 * <span class="code" style="background-color: #bbb">given name</span>,
 * <span class="code" style="background-color: #bbb">email address</span>, and
 * <span class="code" style="background-color: #bbb">organization</span> can
 * be set and retrieved via corresponding methods in the ModelCreator 
 * class.  These methods are documented in more detail below.
 *
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 


#ifndef ModelCreator_h
#define ModelCreator_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/common/operationReturnValues.h>
#include <sbml/util/List.h>

#include <sbml/xml/XMLNode.h>


#ifdef __cplusplus

#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN ModelCreator
{
public:

  /**
   * Creates a new ModelCreator object.
   */
  ModelCreator ();


  /**
   * Creates a new ModelCreator from an XMLNode.
   *
   * @param creator the XMLNode from which to create the ModelCreator.
   */
  ModelCreator(const XMLNode creator);


  /**
   * Destroys the ModelCreator.
   */
  ~ModelCreator();


  /**
   * Copy constructor; creates a copy of the ModelCreator.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  ModelCreator(const ModelCreator& orig);


  /**
   * Assignment operator.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  ModelCreator& operator=(const ModelCreator& rhs);


  /**
   * Creates and returns a copy of this ModelCreator.
   *
   * @return a (deep) copy of this ModelCreator.
   */
  ModelCreator* clone () const;


  /**
   * Returns the "family name" stored in this ModelCreator object.
   *
   * @return the "family name" portion of the ModelCreator object.
   */
  const std::string& getFamilyName()  const  {  return  mFamilyName;  }


  /**
   * Returns the "given name" stored in this ModelCreator object.
   *
   * @return the "given name" portion of the ModelCreator object.
   */
  const std::string& getGivenName() const    {  return  mGivenName;  }


  /**
   * Returns the "email" stored in this ModelCreator object.
   *
   * @return email from the ModelCreator.
   */
  const std::string& getEmail() const       {  return  mEmail;  }


  /**
   * Returns the "organization" stored in this ModelCreator object.
   *
   * @return organization from the ModelCreator.
   */
  const std::string& getOrganization() const{  return  mOrganization;  }


  /**
   * (Alternate spelling) Returns the "organization" stored in this
   * ModelCreator object.
   *
   * @note This function is an alias of getOrganization().
   *
   * @return organization from the ModelCreator.
   *
   * @see getOrganization()
   */
  const std::string& getOrganisation() const{  return  mOrganization;  }

 
  /**
   * Predicate returning @c true or @c false depending on whether this
   * ModelCreator's "family name" part is set.
   *
   * @return @c true if the familyName of this ModelCreator is set, @c false otherwise.
   */
  bool isSetFamilyName();


  /**
   * Predicate returning @c true or @c false depending on whether this
   * ModelCreator's "given name" part is set.
   *
   * @return @c true if the givenName of this ModelCreator is set, @c false otherwise.
   */
  bool isSetGivenName();


  /**
   * Predicate returning @c true or @c false depending on whether this
   * ModelCreator's "email" part is set.
   *
   * @return @c true if the email of this ModelCreator is set, @c false otherwise.
   */
  bool isSetEmail();


  /**
   * Predicate returning @c true or @c false depending on whether this
   * ModelCreator's "organization" part is set.
   *
   * @return @c true if the organization of this ModelCreator is set, @c false otherwise.
   */
  bool isSetOrganization();


  /**
   * (Alternate spelling) Predicate returning @c true or @c false depending
   * on whether this ModelCreator's "organization" part is set.
   *
   * @note This function is an alias of isSetOrganization().
   *
   * @return @c true if the organization of this ModelCreator is set, @c false otherwise.
   *
   * @see isSetOrganization()
   */
  bool isSetOrganisation();


  /**
   * Sets the "family name" portion of this ModelCreator object.
   *  
   * @param familyName a string representing the familyName of the ModelCreator. 
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int setFamilyName(const std::string& familyName);


  /**
   * Sets the "given name" portion of this ModelCreator object.
   *  
   * @param givenName a string representing the givenName of the ModelCreator. 
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int setGivenName(const std::string& givenName);


  /**
   * Sets the "email" portion of this ModelCreator object.
   *  
   * @param email a string representing the email of the ModelCreator. 
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int setEmail(const std::string& email);


  /**
   * Sets the "organization" portion of this ModelCreator object.
   *  
   * @param organization a string representing the organization of the 
   * ModelCreator. 
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int setOrganization(const std::string& organization);


  /**
   * (Alternate spelling) Sets the "organization" portion of this
   * ModelCreator object.
   *
   * @param organization a string representing the organization of the
   * ModelCreator.
   *
   * @note This function is an alias of setOrganization(const std::string& organization).
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   *
   * @see setOrganization(std::string organization)
   */
  int setOrganisation(const std::string& organization);


  /**
   * Unsets the "family name" portion of this ModelCreator object.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetFamilyName();


  /**
   * Unsets the "given name" portion of this ModelCreator object.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetGivenName();


  /**
   * Unsets the "email" portion of this ModelCreator object.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetEmail();


  /**
   * Unsets the "organization" portion of this ModelCreator object.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetOrganization();


  /**
   * (Alternate spelling) Unsets the "organization" portion of this ModelCreator object.
   *
   * @note This function is an alias of unsetOrganization().
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @see unsetOrganization()
   */
  int unsetOrganisation();


  /** @cond doxygen-libsbml-internal */
  XMLNode * getAdditionalRDF();
  /** @endcond */

  /**
   * Predicate returning @c true if all the required elements for this
   * ModelCreator object have been set.
   *
   * The only required elements for a ModelCreator object are the "family
   * name" and "given name".
   *
   * @return a boolean value indicating whether all the required
   * elements for this object have been defined.
   */ 
  bool hasRequiredAttributes();
  

  /** @cond doxygen-libsbml-internal */
  
  bool hasBeenModified();

  void resetModifiedFlags();
   
  
  /** @endcond */

protected:
  /** @cond doxygen-libsbml-internal */


  std::string mFamilyName;
  std::string mGivenName;
  std::string mEmail;
  std::string mOrganization;

  XMLNode * mAdditionalRDF;

  bool mHasBeenModified;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


LIBSBML_EXTERN
ModelCreator_t *
ModelCreator_create();

LIBSBML_EXTERN
ModelCreator_t *
ModelCreator_createFromNode(const XMLNode_t * node);

LIBSBML_EXTERN
void
ModelCreator_free(ModelCreator_t *);

LIBSBML_EXTERN
ModelCreator_t *
ModelCreator_clone (const ModelCreator_t* c);


LIBSBML_EXTERN
const char * 
ModelCreator_getFamilyName(ModelCreator_t *mc);

LIBSBML_EXTERN
const char * 
ModelCreator_getGivenName(ModelCreator_t *mc);

LIBSBML_EXTERN
const char * 
ModelCreator_getEmail(ModelCreator_t *mc);

LIBSBML_EXTERN
const char * 
ModelCreator_getOrganisation(ModelCreator_t *mc);

LIBSBML_EXTERN
const char * 
ModelCreator_getOrganization(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_isSetFamilyName(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_isSetGivenName(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_isSetEmail(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_isSetOrganisation(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_isSetOrganization(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_setFamilyName(ModelCreator_t *mc, char * name);

LIBSBML_EXTERN
int 
ModelCreator_setGivenName(ModelCreator_t *mc, char * name);

LIBSBML_EXTERN
int 
ModelCreator_setEmail(ModelCreator_t *mc, char * name);

LIBSBML_EXTERN
int 
ModelCreator_setOrganisation(ModelCreator_t *mc, char * org);

LIBSBML_EXTERN
int 
ModelCreator_setOrganization(ModelCreator_t *mc, char * name);

LIBSBML_EXTERN
int 
ModelCreator_unsetFamilyName(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_unsetGivenName(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_unsetEmail(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_unsetOrganisation(ModelCreator_t *mc);

LIBSBML_EXTERN
int 
ModelCreator_unsetOrganization(ModelCreator_t *mc);

LIBSBML_EXTERN
int
ModelCreator_hasRequiredAttributes(ModelCreator_t *mc);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /** ModelCreator_h **/

