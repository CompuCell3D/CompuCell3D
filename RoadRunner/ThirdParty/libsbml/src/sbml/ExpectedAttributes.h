/**
 * @file    ExpectedAttributes.h
 * @brief   Definition of ExpectedAttributes, the class allowing the specification
 *          of attributes to expect.
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
 */

#ifndef EXPECTED_ATTRIBUTES_H
#define EXPECTED_ATTRIBUTES_H

#include <sbml/common/extern.h>


#ifdef __cplusplus

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

LIBSBML_CPP_NAMESPACE_BEGIN
/** @cond doxygen-libsbml-internal */
  #ifndef SWIG
class LIBSBML_EXTERN ExpectedAttributes
{
public:

  ExpectedAttributes() 
  {}

  ExpectedAttributes(const ExpectedAttributes& orig) 
    : mAttributes(orig.mAttributes) 
  {}
    
  void add(const std::string& attribute) { mAttributes.push_back(attribute); }

  std::string get(unsigned int i) const
  {
    return (mAttributes.size() < i) ? mAttributes[i] : std::string(); 
  }

  bool hasAttribute(const std::string& attribute) const
  {
    return ( std::find(mAttributes.begin(), mAttributes.end(), attribute)
             != mAttributes.end() );
  }

private:
  std::vector<std::string> mAttributes;
};


#endif //SWIG
/** @endcond */


LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


LIBSBML_EXTERN 
ExpectedAttributes_t *
ExpectedAttributes_create();

LIBSBML_EXTERN 
ExpectedAttributes_t *
ExpectedAttributes_clone(ExpectedAttributes_t *attr);

LIBSBML_EXTERN 
int
ExpectedAttributes_add(ExpectedAttributes_t *attr, const char* attribute);

LIBSBML_EXTERN 
char*
ExpectedAttributes_get(ExpectedAttributes_t *attr, unsigned int index);

LIBSBML_EXTERN 
int
ExpectedAttributes_hasAttribute(ExpectedAttributes_t *attr, const char* attribute);

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG   */
#endif  /* EXPECTED_ATTRIBUTES_H */
