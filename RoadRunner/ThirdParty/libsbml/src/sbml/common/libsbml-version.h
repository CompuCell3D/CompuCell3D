/**
 * @file    libsbml-version.h
 * @brief   Define libSBML version numbers for access from client software.
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
 *------------------------------------------------------------------------- -->
 */

#ifndef LIBSBML_VERSION_H
#define LIBSBML_VERSION_H 

#include <sbml/common/extern.h>


/**
 * LIBSBML_DOTTED_VERSION:
 *
 * A version string of the form "1.2.3".
 */
#define LIBSBML_DOTTED_VERSION	"5.6.0"


/**
 * LIBSBML_VERSION:
 *
 * The version as an integer: version 1.2.3 becomes 10203.  Since the major
 * number comes first, the overall number will always increase when a new
 * libSBML is released, making it easy to use less-than and greater-than
 * comparisons when testing versions numbers.
 */
#define LIBSBML_VERSION		50600


/**
 * LIBSBML_VERSION_STRING:
 *
 * The numeric version as a string: version 1.2.3 becomes "10203".
 */
#define LIBSBML_VERSION_STRING	"50600"


LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/**
 * Returns the version number of this copy of libSBML as an integer.
 *
 * @return the libSBML version as an integer; version 1.2.3 becomes 10203.
 */
LIBSBML_EXTERN
int 
getLibSBMLVersion () ;


/**
 * Returns the version number of this copy of libSBML as a string.
 *
 * @return the libSBML version as a string; version 1.2.3 becomes
 * "1.2.3".
 *
 * @see getLibSBMLVersionString()
 */
LIBSBML_EXTERN
const char* 
getLibSBMLDottedVersion ();


/**
 * Returns the version number of this copy of libSBML as a string without
 * periods.
 *
 * @return the libSBML version as a string: version 1.2.3 becomes "10203".
 *
 * @see getLibSBMLDottedVersion()
 */
LIBSBML_EXTERN
const char* 
getLibSBMLVersionString ();


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* LIBSBML_VERSION_H */

