/**
 * @file    XMLTriple.h
 * @brief   Stores an XML namespace triple.
 * @author  Ben Bornstein
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
 * @class XMLTriple.
 * @brief Representation of a qualified XML name.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * A "triple" in the libSBML XML layer encapsulates the notion of qualified
 * name, meaning an element name or an attribute name with an optional
 * namespace qualifier.  An XMLTriple instance carries up to three data items:
 * 
 * <ul>
 *
 * <li> The name of the attribute or element; that is, the attribute name
 * as it appears in an XML document or data stream;
 *
 * <li> The XML namespace prefix (if any) of the attribute.  For example,
 * in the following fragment of XML, the namespace prefix is the string
 * <code>mysim</code> and it appears on both the element
 * <code>someelement</code> and the attribute <code>attribA</code>.  When
 * both the element and the attribute are stored as XMLTriple objects,
 * their <i>prefix</i> is <code>mysim</code>.
 * @verbatim
<mysim:someelement mysim:attribA="value" />
@endverbatim
 *
 * <li> The XML namespace URI with which the prefix is associated.  In
 * XML, every namespace used must be declared and mapped to a URI.
 *
 * </ul>
 *
 * XMLTriple objects are the lowest-level data item in the XML layer
 * of libSBML.  Other objects such as XMLToken make use of XMLTriple
 * objects.
 */

#ifndef XMLTriple_h
#define XMLTriple_h

#include <sbml/xml/XMLExtern.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus

#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBLAX_EXTERN XMLTriple
{
public:

  /**
   * Creates a new, empty XMLTriple.
   */
  XMLTriple ();


  /**
   * Creates a new XMLTriple with the given @p name, @p uri and and @p
   * prefix.
   *
   * @param name a string, name for the XMLTriple.
   * @param uri a string, URI of the XMLTriple.
   * @param prefix a string, prefix for the URI of the XMLTriple,
   *
   * @throws @if python ValueError @else XMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  XMLTriple (  const std::string&  name
             , const std::string&  uri
             , const std::string&  prefix );


  /**
   * Creates a new XMLTriple by splitting the given @p triplet on the
   * separator character @p sepchar.
   *
   * Triplet may be in one of the following formats:
   * <ul>
   * <li> name
   * <li> URI sepchar name
   * <li> URI sepchar name sepchar prefix
   * </ul>
   * @param triplet a string representing the triplet as above
   * @param sepchar a character, the sepchar used in the triplet
   *
   * @throws @if python ValueError @else XMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  XMLTriple (const std::string& triplet, const char sepchar = ' ');

  
  /**
   * Copy constructor; creates a copy of this XMLTriple set.
   *
   * @param orig the XMLTriple object to copy.
   *
   * @throws @if python ValueError @else XMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  XMLTriple(const XMLTriple& orig);


  /**
   * Assignment operator for XMLTriple.
   *
   * @param rhs The XMLTriple object whose values are used as the basis
   * of the assignment.
   *
   * @throws @if python ValueError @else XMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  XMLTriple& operator=(const XMLTriple& rhs);


  /**
   * Creates and returns a deep copy of this XMLTriple set.
   * 
   * @return a (deep) copy of this XMLTriple set.
   */
  XMLTriple* clone () const;


  /**
   * Returns the @em name portion of this XMLTriple.
   *
   * @return a string, the name from this XMLTriple.
   */
  const std::string& getName () const;


  /**
   * Returns the @em prefix portion of this XMLTriple.
   *
   * @return a string, the @em prefix portion of this XMLTriple.
   */
  const std::string& getPrefix () const;


  /**
   * Returns the @em URI portion of this XMLTriple.
   *
   * @return URI a string, the @em prefix portion of this XMLTriple.
   */
  const std::string& getURI () const;


  /**
   * Returns the prefixed name from this XMLTriple.
   *
   * @return a string, the prefixed name from this XMLTriple.
   */
  const std::string getPrefixedName () const;


  /**
   * Predicate returning @c true or @c false depending on whether 
   * this XMLTriple is empty.
   * 
   * @return @c true if this XMLTriple is empty, @c false otherwise.
   */
  bool isEmpty () const;

private:
  /** @cond doxygen-libsbml-internal */

  std::string  mName;
  std::string  mURI;
  std::string  mPrefix;

  /** @endcond */
};


/**
 * Comparison (equal-to) operator for XMLTriple.
 *  
 * @param lhs XMLTriple object to be compared with rhs.
 * @param rhs XMLTriple object to be compared with lhs.
 *
 * return @c non-zero (true) if the combination of name, URI, and
 * prefix of lhs is equal to that of rhs @c zero (false) otherwise.
 */
bool operator==(const XMLTriple& lhs, const XMLTriple& rhs);


/**
 *  Comparison (not equal-to) operator for XMLTriple.
 *
 * @param lhs XMLTriple object to be compared with rhs.
 * @param rhs XMLTriple object to be compared with lhs.
 *
 * return @c non-zero (true) if the combination of name, URI, and
 * prefix of lhs is not equal to that of rhs @c zero (false) otherwise.
 */
bool operator!=(const XMLTriple& lhs, const XMLTriple& rhs);

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/* ----------------------------------------------------------------------------
 * See the .cpp file for the documentation of the following functions.
 * --------------------------------------------------------------------------*/


LIBLAX_EXTERN
XMLTriple_t *
XMLTriple_create (void);


LIBLAX_EXTERN
XMLTriple_t *
XMLTriple_createWith (const char *name, const char *uri, const char *prefix);


LIBLAX_EXTERN
void
XMLTriple_free (XMLTriple_t *triple);


LIBLAX_EXTERN
XMLTriple_t *
XMLTriple_clone (const XMLTriple_t* c);


LIBLAX_EXTERN
const char *
XMLTriple_getName (const XMLTriple_t *triple);


LIBLAX_EXTERN
const char *
XMLTriple_getPrefix (const XMLTriple_t *triple);


LIBLAX_EXTERN
const char *
XMLTriple_getURI (const XMLTriple_t *triple);


LIBLAX_EXTERN
const char *
XMLTriple_getPrefixedName (const XMLTriple_t *triple);


LIBLAX_EXTERN
int
XMLTriple_isEmpty(const XMLTriple_t *triple);


LIBLAX_EXTERN
int
XMLTriple_equalTo(const XMLTriple_t *lhs, const XMLTriple_t* rhs);


LIBLAX_EXTERN
int
XMLTriple_notEqualTo(const XMLTriple_t *lhs, const XMLTriple_t* rhs);



END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* XMLTriple_h */
