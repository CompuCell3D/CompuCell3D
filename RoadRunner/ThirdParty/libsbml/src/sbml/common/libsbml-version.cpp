/**
 * @file    libsbml-version.cpp
 * @brief   Define libSBML version numbers for access from client software.
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 */

#include "libsbml-version.h"

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * Returns the libSBML version as an integer: version 1.2.3 becomes 10203.
 *
 * @return the libSBML version as an integer: version 1.2.3 becomes 10203.
 */
LIBSBML_EXTERN
int 
getLibSBMLVersion () 
{ 
  return LIBSBML_VERSION; 
}


/**
 * Returns the libSBML version as a string of the form "1.2.3".
 *
 * @return the libSBML version as a string of the form "1.2.3".
 */
LIBSBML_EXTERN
const char* 
getLibSBMLDottedVersion () 
{ 
  return LIBSBML_DOTTED_VERSION;
}


/**
 * Returns the libSBML version as a string: version 1.2.3 becomes "10203".
 *
 * @return the libSBML version as a string: version 1.2.3 becomes "10203".
 */
LIBSBML_EXTERN
const char* 
getLibSBMLVersionString () 
{ 
  return LIBSBML_VERSION_STRING;
}

LIBSBML_CPP_NAMESPACE_END


