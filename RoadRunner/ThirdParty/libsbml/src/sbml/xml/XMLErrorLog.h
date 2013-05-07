/**
 * @file    XMLErrorLog.h
 * @brief   Stores errors (and messages) encountered while processing XML.
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 *
 * @class XMLErrorLog
 * @brief Log of errors and other events encountered while processing an XML
 * file or data stream.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * The error log is a list.  The XML layer of libSBML maintains an error
 * log associated with a given XML document or data stream.  When an
 * operation results in an error, or when there is something wrong with the
 * XML content, the problem is reported as an XMLError object stored in the
 * XMLErrorLog list.  Potential problems range from low-level issues (such
 * as the inability to open a file) to XML syntax errors (such as
 * mismatched tags or other problems).
 *
 * A typical approach for using this error log is to first use
 * @if java XMLErrorLog::getNumErrors()@else getNumErrors()@endif@~
 * to inquire how many XMLError object instances it contains, and then to
 * iterate over the list of objects one at a time using
 * getError(unsigned int n) const.  Indexing in the list begins at 0.
 *
 * In normal circumstances, programs using libSBML will actually obtain an
 * SBMLErrorLog rather than an XMLErrorLog.  The former is subclassed from
 * XMLErrorLog and simply wraps commands for working with SBMLError objects
 * rather than the low-level XMLError objects.  Classes such as
 * SBMLDocument use the higher-level SBMLErrorLog.
 */

#ifndef XMLErrorLog_h
#define XMLErrorLog_h

#include <sbml/xml/XMLExtern.h>
#include <sbml/xml/XMLError.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus

#include <iostream>
#include <string>
#include <vector>
#include <list>

LIBSBML_CPP_NAMESPACE_BEGIN

class XMLParser;


class LIBLAX_EXTERN XMLErrorLog
{
public:

  /**
   * Returns the number of errors that have been logged.
   *
   * To retrieve individual errors from the log, callers may use
   * @if clike getError() @else XMLErrorLog::getError(unsigned int n) @endif.
   *
   * @return the number of errors that have been logged.
   */
  unsigned int getNumErrors () const;


  /**
   * Returns the <i>n</i>th XMLError object in this log.
   *
   * Index @p n is counted from 0.  Callers should first inquire about the
   * number of items in the log by using the method
   * @if java XMLErrorLog::getNumErrors()@else getNumErrors()@endif.
   * Attempts to use an error index number that exceeds the actual number
   * of errors in the log will result in a @c NULL being returned.
   *
   * @param n the index number of the error to retrieve (with 0 being the
   * first error).
   *
   * @return the <i>n</i>th XMLError in this log, or @c NULL if @p n is
   * greater than or equal to
   * @if java XMLErrorLog::getNumErrors()@else getNumErrors()@endif.
   *
   * @see getNumErrors()
   */
  const XMLError* getError (unsigned int n) const;


  /**
   * Deletes all errors from this log.
   */
  void clearLog();


  /** @cond doxygen-libsbml-internal */

  /**
   * Creates a new empty XMLErrorLog.
   */
  XMLErrorLog ();


  /**
   * Destroys this XMLErrorLog.
   */
  virtual ~XMLErrorLog ();


  /**
   * Logs the given XMLError.
   *
   * @param error XMLError, the error to be logged.
   */
  void add (const XMLError& error);


  /**
   * Logs (copies) the XMLErrors in the given XMLError list to this
   * XMLErrorLog.
   *
   * @param errors list, a list of XMLError to be added to the log.
   */
  void add (const std::list<XMLError>& errors);


  /**
   * Sets the XMLParser associated with this XMLErrorLog.
   *
   * The XMLParser will be used to obtain the current line and column
   * number for XMLError objects that lack line and column numbers when
   * they are logged.  This method is used by libSBML's internal XML
   * parsing code and probably has no useful reason to be called from
   * application programs.
   *
   * @param p XMLParser, the parser to use
   *
   * @return integer value indicating success/failure of the
   * function.   The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int setParser (const XMLParser* p);

  /** @endcond */


  /** 
   * Writes all errors contained in this log to a string and returns it.
   *
   * This method uses printErrors() to format the diagnostic messages.
   * Please consult that method for information about the organization
   * of the messages in the string returned by this method.
   *
   * @return a string containing all logged errors and warnings.
   *
   * @see printErrors()
   */
  std::string toString() const;


  /**
   * Prints all the errors or warnings stored in this error log
   *
   * It prints the text to the stream given by the optional parameter @p
   * stream.  If no parameter is given, it prints the output to the
   * standard error stream.
   *
   * If no errors have occurred, i.e., <code>getNumErrors() == 0</code>, no
   * output will be sent to the stream.
   *
   * The format of the output is:
   * @verbatim
   N error(s):
     line NNN: (id) message
 @endverbatim
   *
   * @param stream the ostream or ostringstream object indicating where
   * the output should be printed.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  void printErrors (std::ostream& stream = std::cerr) const;

protected:
  /** @cond doxygen-libsbml-internal */

  std::vector<XMLError*> mErrors;
  const XMLParser*       mParser;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/* ----------------------------------------------------------------------------
 * See the .cpp file for the documentation of the following functions.
 * --------------------------------------------------------------------------*/


LIBLAX_EXTERN
XMLErrorLog_t *
XMLErrorLog_create (void);


LIBLAX_EXTERN
void
XMLErrorLog_free (XMLErrorLog_t *log);


LIBLAX_EXTERN
void
XMLErrorLog_add (XMLErrorLog_t *log, const XMLError_t *error);


LIBLAX_EXTERN
const XMLError_t *
XMLErrorLog_getError (const XMLErrorLog_t *log, unsigned int n);


LIBLAX_EXTERN
unsigned int
XMLErrorLog_getNumErrors (const XMLErrorLog_t *log);


LIBLAX_EXTERN
void
XMLErrorLog_clearLog (XMLErrorLog_t *log);

LIBLAX_EXTERN
char*
XMLErrorLog_toString (XMLErrorLog_t *log);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* XMLErrorLog_h */
