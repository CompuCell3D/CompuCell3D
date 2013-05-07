/**
 * @file    ModifierSpeciesReference.cpp
 * @brief   Implementation of ModifierSpeciesReference. 
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

#include <sbml/math/FormulaParser.h>
#include <sbml/math/MathML.h>
#include <sbml/math/ASTNode.h>

#include <sbml/SBO.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/SBMLError.h>
#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/ModifierSpeciesReference.h>
#include <sbml/extension/SBasePlugin.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN


ModifierSpeciesReference::ModifierSpeciesReference (unsigned int level, 
                          unsigned int version) :
  SimpleSpeciesReference(level, version)
{
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();
}


ModifierSpeciesReference::ModifierSpeciesReference (SBMLNamespaces *sbmlns) :
  SimpleSpeciesReference(sbmlns)
{
  if (!hasValidLevelVersionNamespaceCombination())
    throw SBMLConstructorException();

  loadPlugins(sbmlns);
}


/*
 * Destroys this ModifierSpeciesReference.
 */
ModifierSpeciesReference::~ModifierSpeciesReference ()
{
}


/*
 * Accepts the given SBMLVisitor.
 *
 * @return the result of calling <code>v.visit()</code>, which indicates
 * whether or not the Visitor would like to visit the Reaction's next
 * ModifierSpeciesReference (if available).
 */
bool
ModifierSpeciesReference::accept (SBMLVisitor& v) const
{
  return v.visit(*this);
}


/*
 * @return a (deep) copy of this ModifierSpeciesReference.
 */
ModifierSpeciesReference*
ModifierSpeciesReference::clone () const
{
  return new ModifierSpeciesReference(*this);
}


/*
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * (default).
 *
 * @see getElementName()
 */
int
ModifierSpeciesReference::getTypeCode () const
{
  return SBML_MODIFIER_SPECIES_REFERENCE;
}


/*
 * @return the name of this element ie "modifierSpeciesReference".
 
 */
const string&
ModifierSpeciesReference::getElementName () const
{
  static const string name = "modifierSpeciesReference";
  return name;
}


bool 
ModifierSpeciesReference::hasRequiredAttributes() const
{
  bool allPresent = SimpleSpeciesReference::hasRequiredAttributes();

  return allPresent;
}


LIBSBML_CPP_NAMESPACE_END

