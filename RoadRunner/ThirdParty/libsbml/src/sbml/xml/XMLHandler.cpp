/**
 * @file    XMLHandler.cpp
 * @brief   XMLHandler interface
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

#include <sbml/xml/XMLHandler.h>
#include <sbml/xml/XMLToken.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygen-libsbml-internal */

/*
 * Creates a new XMLHandler.
 */
XMLHandler::XMLHandler ()
{
}


/*
 * Destroys this XMLHandler.
 */
XMLHandler::~XMLHandler ()
{
}


/*
 * Receive notification of the beginning of the document.
 *
 * By default, do nothing. Application writers may override this method
 * in a subclass to take specific actions at the start of the document.
 */
void
XMLHandler::startDocument ()
{
}


/*
 * Receive notification of the XML declaration,
 * i.e.  <?xml version="1.0" encoding="UTF-8"?>
 *
 * By default, do nothing. Application writers may override this method in
 * a subclass to take specific actions at the declaration.
 */
void
XMLHandler::XML (const string& version, const string& encoding)
{
}


/*
 * Receive notification of the start of an element.
 *
 * By default, do nothing. Application writers may override this method
 * in a subclass to take specific actions at the start of each element.
 */
void
XMLHandler::startElement (const XMLToken& element)
{
}


/*
 * Receive notification of the end of the document.
 *
 * By default, do nothing. Application writers may override this method
 * in a subclass to take specific actions at the end of the document.
 */
void
XMLHandler::endDocument ()
{
}


/*
 * Receive notification of the end of an element.
 *
 * By default, do nothing. Application writers may override this method
 * in a subclass to take specific actions at the end of each element.
 */
void
XMLHandler::endElement (const XMLToken& element)
{
}


/*
 * Receive notification of character data inside an element.
 *
 * By default, do nothing. Application writers may override this method
 * to take specific actions for each chunk of character data.
 */
void
XMLHandler::characters (const XMLToken& data)
{
}


/** @endcond */

LIBSBML_CPP_NAMESPACE_END

