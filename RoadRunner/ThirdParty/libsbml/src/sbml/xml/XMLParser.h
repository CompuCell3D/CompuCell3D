/**
 * @file    XMLParser.h
 * @brief   XMLParser interface and factory
 * @author  Ben Bornstein
 * @author  Sarah Keating
 * @author  Michael Hucka
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->
 *
 * @class XMLParser
 * @brief Class providing a unified interface to different XML parsers.
 *
 * @if notclike @internal @endif@~
 */

#ifndef XMLParser_h
#define XMLParser_h

#ifdef __cplusplus

#include <string>
#include <sbml/xml/XMLExtern.h>
#include <sbml/common/operationReturnValues.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class XMLErrorLog;
class XMLHandler;


class LIBLAX_EXTERN XMLParser
{
public:

  /**
   * Creates a new XMLParser.  The parser will notify the given XMLHandler
   * of parse events and errors.
   *
   * The library parameter indicates the underlying XML library to use if
   * the XML compatibility layer has been linked against multiple XML
   * libraries.  It may be one of: "expat" (default), "libxml", or
   * "xerces".
   *
   * If the XML compatibility layer has been linked against only a single
   * XML library, the library parameter is ignored.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  static XMLParser* create (  XMLHandler&       handler
                            , const std::string library = "" );


  /**
   * Destroys this XMLParser.
   */
  virtual ~XMLParser ();


  /**
   * Parses XML content in one fell swoop.
   *
   * If isFile is true (default), content is treated as a filename from
   * which to read the XML content.  Otherwise, content is treated as a
   * null-terminated buffer containing XML data and is read directly.
   *
   * @return true if the parse was successful, false otherwise.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  virtual bool parse (const char* content, bool isFile = true) = 0;


  /**
   * Begins a progressive parse of XML content.  This parses the first
   * chunk of the XML content and returns.  Successive chunks are parsed by
   * calling parseNext().
   *
   * A chunk differs slightly depending on the underlying XML parser.  For
   * Xerces and libXML chunks correspond to XML elements.  For Expat, a
   * chunk is the size of its internal buffer.
   *
   * If isFile is true (default), content is treated as a filename from
   * which to read the XML content.  Otherwise, content is treated as a
   * null-terminated buffer containing XML data and is read directly.
   *
   * @return true if the first step of the progressive parse was
   * successful, false otherwise.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  virtual bool parseFirst (const char* content, bool isFile = true) = 0;


  /**
   * Parses the next chunk of XML content.
   *
   * @return true if the next step of the progressive parse was successful,
   * false otherwise or when at EOF.
   */
  virtual bool parseNext () = 0;


  /**
   * Resets the progressive parser.  Call between the last call to
   * parseNext() and the next call to parseFirst().
   */
  virtual void parseReset () = 0;


  /**
   * @return the current column position of the parser.
   */
  virtual unsigned int getColumn () const = 0;


  /**
   * @return the current line position of the parser.
   */
  virtual unsigned int getLine () const = 0;


  /**
   * @return an XMLErrorLog which can be used to log XML parse errors and
   * other validation errors (and messages).
   */
  XMLErrorLog* getErrorLog ();


  /**
   * Sets the XMLErrorLog this parser will use to log errors.
   *
   * @return integer value indicating success/failure of the
   * function. The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int setErrorLog (XMLErrorLog* log);


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * Creates a new XMLParser.  The parser will notify the given XMLHandler
   * of parse events and errors.
   *
   * Only subclasses may call this constructor directly.  Everyone else
   * should use XMLParser::create().
   */
  XMLParser ();


  XMLErrorLog* mErrorLog;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */
#endif  /* XMLParser_h */

