/**
 * @file    AlgebraicRule.cpp
 * @brief   Implementations of AlgebraicRule.
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
#include <sbml/AlgebraicRule.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN


AlgebraicRule::AlgebraicRule (unsigned int level, unsigned int version) :
  Rule(SBML_ALGEBRAIC_RULE, level, version)
{
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();

  mInternalIdOnly = false;
}


AlgebraicRule::AlgebraicRule (SBMLNamespaces * sbmlns) :
  Rule(SBML_ALGEBRAIC_RULE, sbmlns)
{
  if (!hasValidLevelVersionNamespaceCombination())
  {
    throw SBMLConstructorException(getElementName(), sbmlns);
  }

  mInternalIdOnly = false;

  loadPlugins(sbmlns);
}


/*
 * Destroys this AlgebraicRule.
 */
AlgebraicRule::~AlgebraicRule ()
{
}

/*
 * @return a (deep) copy of this Rule.
 */
AlgebraicRule*
AlgebraicRule::clone () const
{
  return new AlgebraicRule(*this);
}


/*
 * Accepts the given SBMLVisitor.
 *
 * @return the result of calling <code>v.visit()</code>, which indicates
 * whether or not the Visitor would like to visit the Model's next Rule
 * (if available).
 */
bool
AlgebraicRule::accept (SBMLVisitor& v) const
{
  return v.visit(*this);
}

bool 
AlgebraicRule::hasRequiredAttributes() const
{
  bool allPresent = Rule::hasRequiredAttributes();

  return allPresent;
}



/** @cond doxygen-libsbml-internal */

/*
 * sets the mInternalIdOnly flag
 */
void 
AlgebraicRule::setInternalIdOnly()
{
  mInternalIdOnly = true;
}

/*
 * gets the mInternalIdOnly flag
 */
bool 
AlgebraicRule::getInternalIdOnly() const
{
  return mInternalIdOnly;
}

/** @endcond */

LIBSBML_CPP_NAMESPACE_END
