/**
 * @file    SBMLTypeCodes.h
 * @brief   Enumeration to identify SBML objects at runtime
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
 * ---------------------------------------------------------------------- -->*/

#ifndef SBMLTypeCodes_h
#define SBMLTypeCodes_h


#include <sbml/common/libsbml-config.h>
#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * An enumeration of SBML types to help identify SBML objects at runtime.
 * Abstract types do not have a typecode since they cannot be instantiated.
 *
 * (NOTES)
 *
 *  - Each typecode is used as a return value (int) of the following functions
 *
 *     - virtual int SBase::getTypeCode() const;
 *     - virtual int ListOf::getItemTypeCode() const;
 *
 *    (In libSBML 5, the type of return values in these functions have been changed
 *     from typecode (int) to int for extensibility.)
 *
 *  - Each pacakge extension must define similar enum type for each SBase subclass
 *    (e.g. SBMLLayoutTypeCode_t for the layout extension, SBMLGroupTypeCode_t for
 *          group extension).
 *
 *  - The value of each typecode can be duplicated between those of different 
 *    packages.
 *
 *  - To distinguish the typecodes of different packages, not only the return
 *    value of getTypeCode() but also that of getPackageName() must be checked
 *    as follows:
 *
 *       void example (const SBase *sb)
 *       {
 *         cons std::string pkgName = sb->getPackageName();
 *         if (pkgName == "core")
 *         {
 *           switch (sb->getTypeCode())
 *           {
 *             case SBML_MODEL:
 *                ....
 *                break;
 *             case SBML_REACTION:
 *                ....
 *           }
 *         } 
 *         else if (pkgName == "layout")
 *         {
 *           switch (sb->getTypeCode())
 *           {
 *             case SBML_LAYOUT_LAYOUT:
 *                ....
 *                break;
 *             case SBML_LAYOUT_REACTIONGLYPH:
 *                ....
 *           }
 *         } 
 *         ...
 *       } 
 *      
 */
typedef enum
{
      SBML_UNKNOWN                    =  0
    , SBML_COMPARTMENT                =  1
    , SBML_COMPARTMENT_TYPE           =  2
    , SBML_CONSTRAINT                 =  3
    , SBML_DOCUMENT                   =  4
    , SBML_EVENT                      =  5
    , SBML_EVENT_ASSIGNMENT           =  6
    , SBML_FUNCTION_DEFINITION        =  7
    , SBML_INITIAL_ASSIGNMENT         =  8
    , SBML_KINETIC_LAW                =  9
    , SBML_LIST_OF                    = 10
    , SBML_MODEL                      = 11
    , SBML_PARAMETER                  = 12
    , SBML_REACTION                   = 13
    , SBML_RULE                       = 14
    , SBML_SPECIES                    = 15
    , SBML_SPECIES_REFERENCE          = 16
    , SBML_SPECIES_TYPE               = 17
    , SBML_MODIFIER_SPECIES_REFERENCE = 18
    , SBML_UNIT_DEFINITION            = 19
    , SBML_UNIT                       = 20
    , SBML_ALGEBRAIC_RULE             = 21
    , SBML_ASSIGNMENT_RULE            = 22
    , SBML_RATE_RULE                  = 23
    , SBML_SPECIES_CONCENTRATION_RULE = 24
    , SBML_COMPARTMENT_VOLUME_RULE    = 25
    , SBML_PARAMETER_RULE             = 26
    , SBML_TRIGGER                    = 27
    , SBML_DELAY                      = 28
    , SBML_STOICHIOMETRY_MATH         = 29
    , SBML_LOCAL_PARAMETER            = 30
    , SBML_PRIORITY                   = 31
} SBMLTypeCode_t;



/**
 * This method takes an SBML type code and returns a string representing
 * the code.
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
 * libsbml@endlink.  The names of the type codes all begin with
 * the characters @c SBML_. @endif@~
 *
 * @return a human readable name for the given
 * @if clike #SBMLTypeCode_t value@else SBML type code@endif.
 *
 * @note The caller does not own the returned string and is therefore not
 * allowed to modify it.
 */
LIBSBML_EXTERN
const char *
SBMLTypeCode_toString (int tc, const char* pkgName);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* SBMLTypeCodes_h */
