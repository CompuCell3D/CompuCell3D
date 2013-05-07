/**
 * @file    XMLTokenizer.cpp
 * @brief   Uses an XMLHandler to deliver an XML stream as a series of tokens
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
 * ---------------------------------------------------------------------- -->*/

#include <sstream>

#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLTokenizer.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygen-libsbml-internal */

/*
 * Creates a new XMLTokenizer.
 */
XMLTokenizer::XMLTokenizer () :
   mInChars( false )
 , mInStart( false )
 , mEOFSeen( false )
{
}


/*
 * Destroys this XMLTokenizer.
 */
XMLTokenizer::~XMLTokenizer ()
{
}


/*
 * @return the encoding of the underlying XML document.
 */
const string&
XMLTokenizer::getEncoding ()
{
  return mEncoding;
}


/*
 * @return the version of the underlying XML document.
 */
const string&
XMLTokenizer::getVersion ()
{
  return mVersion;
}


/*
 * @return true if this XMLTokenizer has at least one XMLToken ready to
 * deliver, false otherwise.
 *
 * Note that hasNext() == false does not imply isEOF() == true.  The
 * XMLTokenizer may simply be waiting for the XMLParser to parse more of
 * the document.
 */
bool
XMLTokenizer::hasNext () const
{
  return (mTokens.size() > 0);
}


/*
 * @return true if the end of the XML file (document) has been reached
 * and there are no more tokens to consume, false otherwise.
 */
bool
XMLTokenizer::isEOF () const
{
  return mEOFSeen && !hasNext();
}


/*
 * Consume the next XMLToken and return it.
 *
 * @return the next XMLToken.
 */
XMLToken
XMLTokenizer::next ()
{
  XMLToken token( peek() );
  mTokens.pop_front();

  return token;
}


/*
 * Returns the next XMLToken without consuming it.  A subsequent call to
 * either peek() or next() will return the same token.
 *
 * @return the next XMLToken.
 */
const XMLToken&
XMLTokenizer::peek ()
{
  return mTokens.front();
}


/*
 * Prints a string representation of the underlying token stream, for
 * debugging purposes.
 */
string
XMLTokenizer::toString ()
{
  ostringstream stream;

  for (unsigned int n = 0; n < mTokens.size(); ++n)
  {
    stream << '[' << mTokens[n].toString() << ']' << endl;
  }

  return stream.str();
}


/*
 * Receive notification of the XML declaration, i.e.
 * <?xml version="1.0" encoding="UTF-8"?>
 */
void
XMLTokenizer::XML (const string& version, const string& encoding)
{
  mVersion = version;
  mEncoding = encoding;
}


/*
 * Receive notification of the start of an element.
 */
void
XMLTokenizer::startElement (const XMLToken& element)
{
  if (&element == NULL) return; 

  if (mInChars || mInStart)
  {
    mInChars = false;
    mTokens.push_back( mCurrent );
  }

  //
  // We delay pushing element onto mTokens until we see either an end
  // elment (in which case we can collapse start and end elements into a
  // single token) or the beginning of character data.
  //
  mInStart = true;
  mCurrent = element;
}


/*
 * Receive notification of the end of the document.
 */
void
XMLTokenizer::endDocument ()
{
  mEOFSeen = true;
}


/*
 * Receive notification of the end of an element.
 */
void
XMLTokenizer::endElement (const XMLToken& element)
{
  if (mInChars)
  {
    mInChars = false;
    mTokens.push_back( mCurrent );
  }

  if (mInStart)
  {
    mInStart = false;
    mCurrent.setEnd();
    mTokens.push_back( mCurrent );
  }
  else if (&element != NULL) 
  {
    mTokens.push_back(element);
  }
}


/*
 * Receive notification of character data inside an element.
 */
void
XMLTokenizer::characters (const XMLToken& data)
{
  if (&data == NULL) return; 

  if (mInStart)
  {
    mInStart = false;
    mTokens.push_back( mCurrent );
  }

  if (mInChars)
  {
    mCurrent.append( data.getCharacters() );
  }
  else
  {
    mInChars = true;
    mCurrent = data;
  }
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END

