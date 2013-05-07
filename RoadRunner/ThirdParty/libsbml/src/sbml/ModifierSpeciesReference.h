/**
 * @file    ModifierSpeciesReference.h
 * @brief   Definitions of ModifierSpeciesReference. 
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
 * ------------------------------------------------------------------------ -->
 *
 * @class ModifierSpeciesReference
 * @brief LibSBML implementation of %SBML's %ModifierSpeciesReference construct.
 *
 * Sometimes a species appears in the kinetic rate formula of a reaction
 * but is itself neither created nor destroyed in that reaction (for
 * example, because it acts as a catalyst or inhibitor).  In SBML, all such
 * species are simply called @em modifiers without regard to the detailed
 * role of those species in the model.  The Reaction structure provides a
 * way to express which species act as modifiers in a given reaction.  This
 * is the purpose of the list of modifiers available in Reaction.  The list
 * contains instances of ModifierSpeciesReference structures.
 *
 * The ModifierSpeciesReference structure inherits the mandatory attribute
 * "species" and optional attributes "id" and "name" from the parent class
 * SimpleSpeciesReference.  See the description of SimpleSpeciesReference
 * for more information about these.
 *
 * The value of the "species" attribute must be the identifier of a species
 * defined in the enclosing Model; this species is designated as a modifier
 * for the current reaction.  A reaction may have any number of modifiers.
 * It is permissible for a modifier species to appear simultaneously in the
 * list of reactants and products of the same reaction where it is
 * designated as a modifier, as well as to appear in the list of reactants,
 * products and modifiers of other reactions in the model.
 *
 * 
 */


#ifndef ModifierSpeciesReference_h
#define ModifierSpeciesReference_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SimpleSpeciesReference.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/ListOf.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLNamespaces;

class LIBSBML_EXTERN ModifierSpeciesReference : public SimpleSpeciesReference
{
public:

  /**
   * Creates a new ModifierSpeciesReference using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this ModifierSpeciesReference
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * ModifierSpeciesReference
   * 
   * @note Upon the addition of a ModifierSpeciesReference object to an
   * SBMLDocument (e.g., using Reaction::addModifier(@if java ModifierSpeciesReference msr@endif)), the
   * SBML Level, SBML Version and XML namespace of the document @em
   * override the values used when creating the ModifierSpeciesReference
   * object via this constructor.  This is necessary to ensure that an SBML
   * document is a consistent structure.  Nevertheless, the ability to
   * supply the values at the time of creation of a
   * ModifierSpeciesReference is an important aid to producing valid SBML.
   * Knowledge of the intented SBML Level and Version determine whether it
   * is valid to assign a particular value to an attribute, or whether it
   * is valid to add an object to an existing SBMLDocument.
   */
  ModifierSpeciesReference (unsigned int level, unsigned int version);


  /**
   * Creates a new ModifierSpeciesReference using the given SBMLNamespaces object
   * @p sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object.
   *
   * @note Upon the addition of a ModifierSpeciesReference object to an
   * SBMLDocument (e.g., using Reaction::addModifier(@if java ModifierSpeciesReference msr@endif)), the
   * SBML XML namespace of the document @em overrides the value used when
   * creating the ModifierSpeciesReference object via this constructor.
   * This is necessary to ensure that an SBML document is a consistent
   * structure.  Nevertheless, the ability to supply the values at the time
   * of creation of a ModifierSpeciesReference is an important aid to
   * producing valid SBML.  Knowledge of the intented SBML Level and
   * Version determine whether it is valid to assign a particular value to
   * an attribute, or whether it is valid to add an object to an existing
   * SBMLDocument.
   */
  ModifierSpeciesReference (SBMLNamespaces* sbmlns);


  /**
   * Destroys this ModifierSpeciesReference.
   */
  virtual ~ModifierSpeciesReference();


  /**
   * Accepts the given SBMLVisitor.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this ModifierSpeciesReference
   * instance.
   *
   * @return a (deep) copy of this ModifierSpeciesReference.
   */
  virtual ModifierSpeciesReference* clone () const;


  /**
   * Returns the libSBML type code for this %SBML object.
   * 
   * @if clike LibSBML attaches an identifying code to every kind of SBML
   * object.  These are known as <em>SBML type codes</em>.  The set of
   * possible type codes is defined in the enumeration #SBMLTypeCode_t.
   * The names of the type codes all begin with the characters @c
   * SBML_. @endif@if java LibSBML attaches an identifying code to every
   * kind of SBML object.  These are known as <em>SBML type codes</em>.  In
   * other languages, the set of type codes is stored in an enumeration; in
   * the Java language interface for libSBML, the type codes are defined as
   * static integer constants in the interface class {@link
   * libsbmlConstants}.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if python LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the Python language interface for libSBML, the type
   * codes are defined as static integer constants in the interface class
   * @link libsbml@endlink.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if csharp LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the C# language interface for libSBML, the type codes
   * are defined as static integer constants in the interface class @link
   * libsbmlcs.libsbml@endlink.  The names of the type codes all begin with
   * the characters @c SBML_. @endif@~
   *
   * @return the SBML type code for this object, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Returns the XML element name of this object, which for Species, is
   * always @c "modifierSpeciesReference".
   * 
   * @return the name of this element, i.e., @c "modifierSpeciesReference".
   */
  virtual const std::string& getElementName () const;


  /**
   * Predicate returning @c true if
   * all the required attributes for this ModifierSpeciesReference object
   * have been set.
   *
   * @note The required attributes for a ModifierSpeciesReference object are:
   * species
   */
  virtual bool hasRequiredAttributes() const ;


protected:
  /** @cond doxygen-libsbml-internal */

  /* the validator classes need to be friends to access the 
   * protected constructor that takes no arguments
   */
  friend class Validator;
  friend class ConsistencyValidator;
  friend class IdentifierConsistencyValidator;
  friend class InternalConsistencyValidator;
  friend class L1CompatibilityValidator;
  friend class L2v1CompatibilityValidator;
  friend class L2v2CompatibilityValidator;
  friend class L2v3CompatibilityValidator;
  friend class L2v4CompatibilityValidator;
  friend class MathMLConsistencyValidator;
  friend class ModelingPracticeValidator;
  friend class OverdeterminedValidator;
  friend class SBOConsistencyValidator;
  friend class UnitConsistencyValidator;

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


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* ModifierSpeciesReference_h */
