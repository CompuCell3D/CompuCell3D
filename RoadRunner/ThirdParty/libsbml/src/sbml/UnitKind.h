/**
 * @file    UnitKind.h
 * @brief   Definition of SBML's UnitKind enumeration
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
 *
 * @var typedef enum UnitKind_t
 * @brief Enumeration of predefined SBML base units
 *
 * For more information, please refer to the class documentation for Unit.
 * 
 * @see UnitDefinition_t
 * @see Unit_t
 */

#ifndef UnitKind_h
#define UnitKind_h


#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/**
 * @var typedef enum UnitKind_t
 */
typedef enum
{
    UNIT_KIND_AMPERE
  , UNIT_KIND_AVOGADRO
  , UNIT_KIND_BECQUEREL
  , UNIT_KIND_CANDELA
  , UNIT_KIND_CELSIUS
  , UNIT_KIND_COULOMB
  , UNIT_KIND_DIMENSIONLESS
  , UNIT_KIND_FARAD
  , UNIT_KIND_GRAM
  , UNIT_KIND_GRAY
  , UNIT_KIND_HENRY
  , UNIT_KIND_HERTZ
  , UNIT_KIND_ITEM
  , UNIT_KIND_JOULE
  , UNIT_KIND_KATAL
  , UNIT_KIND_KELVIN
  , UNIT_KIND_KILOGRAM
  , UNIT_KIND_LITER
  , UNIT_KIND_LITRE
  , UNIT_KIND_LUMEN
  , UNIT_KIND_LUX
  , UNIT_KIND_METER
  , UNIT_KIND_METRE
  , UNIT_KIND_MOLE
  , UNIT_KIND_NEWTON
  , UNIT_KIND_OHM
  , UNIT_KIND_PASCAL
  , UNIT_KIND_RADIAN
  , UNIT_KIND_SECOND
  , UNIT_KIND_SIEMENS
  , UNIT_KIND_SIEVERT
  , UNIT_KIND_STERADIAN
  , UNIT_KIND_TESLA
  , UNIT_KIND_VOLT
  , UNIT_KIND_WATT
  , UNIT_KIND_WEBER
  , UNIT_KIND_INVALID
} UnitKind_t;


/**
 * Tests for logical equality between two given <code>UNIT_KIND_</code>
 * code values.
 *
 * This function behaves exactly like C's <code>==</code> operator, except
 * for the following two cases:
 * <ul>
 * <li>@link UnitKind_t#UNIT_KIND_LITER UNIT_KIND_LITER@endlink <code>==</code> @link UnitKind_t#UNIT_KIND_LITRE UNIT_KIND_LITRE@endlink
 * <li>@link UnitKind_t#UNIT_KIND_METER UNIT_KIND_METER@endlink <code>==</code> @link UnitKind_t#UNIT_KIND_METRE UNIT_KIND_METRE@endlink
 * </ul>
 *
 * In the two cases above, C equality comparison would yield @c false
 * (because each of the above is a distinct enumeration value), but
 * this function returns @c true.
 *
 * @param uk1 a <code>UNIT_KIND_</code> value 
 * @param uk2 a second <code>UNIT_KIND_</code> value to compare to @p uk1
 *
 * @return nonzero (for @c true) if @p uk1 is logically equivalent to @p
 * uk2, zero (for @c false) otherwise.
 *
 * @note For more information about the libSBML unit codes, please refer to
 * the class documentation for Unit.
 */
LIBSBML_EXTERN
int
UnitKind_equals (UnitKind_t uk1, UnitKind_t uk2);


/**
 * Converts a text string naming a kind of unit to its corresponding
 * libSBML <code>UNIT_KIND_</code> constant/enumeration value.
 *
 * @param name a string, the name of a predefined base unit in SBML
 * 
 * @return @if clike a value from UnitKind_t corresponding to the given
 * string @p name (determined in a case-insensitive manner).
 * @endif@if python a value the set of <code>UNIT_KIND_</code> codes
 * defined in class @link libsbml libsbml@endlink, corresponding to the
 * string @p name (determined in a case-insensitive
 * manner).@endif@if java a value the set of <code>UNIT_KIND_</code> codes
 * defined in class {@link libsbmlConstants}, corresponding to the string
 * @p name (determined in a case-insensitive manner).@endif@~
 *
 * @note For more information about the libSBML unit codes, please refer to
 * the class documentation for Unit.
 */
LIBSBML_EXTERN
UnitKind_t
UnitKind_forName (const char *name);


/**
 * Converts a unit code to a text string equivalent.
 *
 * @param uk @if clike a value from the UnitKind_t enumeration
 * @endif@if python a value from the set of <code>UNIT_KIND_</code> codes
 * defined in the class @link libsbml libsbml@endlink
 * @endif@if java a value from the set of <code>UNIT_KIND_</code> codes
 * defined in the class {@link libsbmlConstants}
 * @endif@~
 *
 * @return the name corresponding to the given unit code.
 *
 * @note For more information about the libSBML unit codes, please refer to
 * the class documentation for Unit.
 * 
 * @warning The string returned is a static data value.  The caller does not
 * own the returned string and is therefore not allowed to modify it.
 */
LIBSBML_EXTERN
const char *
UnitKind_toString (UnitKind_t uk);


/**
 * Predicate for testing whether a given string corresponds to a
 * predefined libSBML unit code.
 *
 * @param str a text string naming a base unit defined by SBML
 * @param level the Level of SBML
 * @param version the Version within the Level of SBML
 *
 * @return nonzero (for @c true) if string is the name of a valid
 * <code>UNIT_KIND_</code> value, zero (for @c false) otherwise.
 *
 * @note For more information about the libSBML unit codes, please refer to
 * the class documentation for Unit.
 */
LIBSBML_EXTERN
int
UnitKind_isValidUnitKindString (const char *str, unsigned int level, unsigned int version);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /** UnitKind_h **/

