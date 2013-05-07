/**
 * @file    RateRule.cpp
 * @brief   Implementations of RateRule.
 * @author  Ben Bornstein
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
 * ---------------------------------------------------------------------- -->*/

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>
#include <sbml/xml/XMLNamespaces.h>

#include <sbml/math/FormulaFormatter.h>
#include <sbml/math/FormulaParser.h>
#include <sbml/math/MathML.h>
#include <sbml/math/ASTNode.h>

#include <sbml/SBO.h>
#include <sbml/SBMLTypeCodes.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/SBMLError.h>
#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/RateRule.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

RateRule::RateRule (unsigned int level, unsigned int version) :
  Rule(SBML_RATE_RULE, level, version)
{
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();
}

RateRule::RateRule (SBMLNamespaces *sbmlns) :
  Rule(SBML_RATE_RULE, sbmlns)
{
  if (!hasValidLevelVersionNamespaceCombination())
  {
    throw SBMLConstructorException(getElementName(), sbmlns);
  }

  loadPlugins(sbmlns);
}


/*
 * Destroys this RateRule.
 */
RateRule::~RateRule ()
{
}

/*
 * @return a (deep) copy of this Rule.
 */
RateRule*
RateRule::clone () const
{
  return new RateRule(*this);
}


bool 
RateRule::hasRequiredAttributes() const
{
  bool allPresent = Rule::hasRequiredAttributes();

  /* required attributes for rateRule: variable (comp/species/name in L1) */

  if (!isSetVariable())
    allPresent = false;

  return allPresent;
}


/*
 * Accepts the given SBMLVisitor.
 *
 * @return the result of calling <code>v.visit()</code>, which indicates
 * whether or not the Visitor would like to visit the Model's next Rule
 * (if available).
 */
bool
RateRule::accept (SBMLVisitor& v) const
{
  return v.visit(*this);
}


void
RateRule::renameSIdRefs(std::string oldid, std::string newid)
{
  Rule::renameSIdRefs(oldid, newid);
  if (isSetVariable()) {
    if (getVariable()==oldid) {
      setVariable(newid);
    }
  }
}

LIBSBML_CPP_NAMESPACE_END
