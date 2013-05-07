/**
 * @file    SBMLInitialAssignmentConverter.h
 * @brief   Definition of SBMLInitialAssignmentConverter, a converter inlining function definitions
 * @author  Frank Bergmann 
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
 * @class SBMLInitialAssignmentConverter
 * @brief SBML converter for replacing initial assignments.
 *
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * This is an SBML converter for replacing InitialAssignment objects 
 * (when possible) by setting the initial value attributes on the model
 * objects being assigned.  In other words, for every object that is
 * the target of an initial assignment in the model, it evaluates the
 * mathematical expression of the assignment to get a numerical value,
 * and then sets the corresponding attribute of the object to the
 * value.  The effects for different kinds of SBML components are
 * as follows:
 * 
 * <center>
 * <table border="0" class="text-table width80 normal-font alt-row-colors">
 *  <tr style="background: lightgray; font-size: 14px;">
 *      <th align="left" width="200">Component</th>
 *      <th align="left">Effect</th>
 *  </tr>
 *  <tr>
 *  <td>Compartment</td>
 *  <td>Sets the value of the <code>size</code> attribute.</td>
 *  </tr>
 *  <tr>
 *  <td>Species</td>
 *  <td>Sets the value of either the <code>initialAmount</code>
 *  or the <code>initialConcentration</code> attributes, depending
 *  on the value of the Species object's
 *  <code>hasOnlySubstanceUnits</code> attribute.</td>
 *  </tr>
 *  <tr>
 *  <td>Parameter</td>
 *  <td>Sets the value of the <code>value</code> attribute.</td>
 *  </tr>
 *  <tr>
 *  <td>SpeciesReference</td>
 *  <td>Sets the value of the <code>stoichiometry</code> attribute
 *  in the Reaction object where the SpeciesReference object appears.</td>
 *  </tr>
 * </table>
 * </center>
 *
 * @see SBMLFunctionDefinitionConverter
 * @see SBMLLevelVersionConverter
 * @see SBMLRuleConverter
 * @see SBMLStripPackageConverter
 * @see SBMLUnitsConverter
 */

#ifndef SBMLInitialAssignmentConverter_h
#define SBMLInitialAssignmentConverter_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/conversion/SBMLConverter.h>
#include <sbml/conversion/SBMLConverterRegister.h>


#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SBMLInitialAssignmentConverter : public SBMLConverter
{
public:

  /** @cond doxygen-libsbml-internal */
  
  /* register with the ConversionRegistry */
  static void init();  

  /** @endcond */


  /**
   * Creates a new SBMLInitialAssignmentConverter object.
   */
  SBMLInitialAssignmentConverter();


  /**
   * Copy constructor; creates a copy of an SBMLInitialAssignmentConverter
   * object.
   *
   * @param obj the SBMLInitialAssignmentConverter object to copy.
   */
  SBMLInitialAssignmentConverter(const SBMLInitialAssignmentConverter& obj);


  /**
   * Creates and returns a deep copy of this SBMLInitialAssignmentConverter
   * object.
   * 
   * @return a (deep) copy of this converter.
   */
  virtual SBMLConverter* clone() const;


  /**
   * Returns @c true if this converter object's properties match the given
   * properties.
   *
   * A typical use of this method involves creating a ConversionProperties
   * object, setting the options desired, and then calling this method on
   * an SBMLInitialAssignmentConverter object to find out if the object's
   * property values match the given ones.  This method is also used by
   * SBMLConverterRegistry::getConverterFor(@if java const ConversionProperties& props@endif)
   * to search across all registered converters for one matching particular
   * properties.
   * 
   * @param props the properties to match.
   * 
   * @return @c true if this converter's properties match, @c false
   * otherwise.
   */
  virtual bool matchesProperties(const ConversionProperties &props) const;

  
  /** 
   * Perform the conversion.
   *
   * This method causes the converter to do the actual conversion work,
   * that is, to convert the SBMLDocument object set by
   * SBMLConverter::setDocument(@if java const SBMLDocument* doc@endif) and
   * with the configuration options set by
   * SBMLConverter::setProperties(@if java const ConversionProperties *props@endif).
   * 
   * @return  integer value indicating the success/failure of the operation.
   * @if clike The value is drawn from the enumeration
   * #OperationReturnValues_t. @endif@~ The possible values are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   */
  virtual int convert();


  /**
   * Returns the default properties of this converter.
   * 
   * A given converter exposes one or more properties that can be adjusted
   * in order to influence the behavior of the converter.  This method
   * returns the @em default property settings for this converter.  It is
   * meant to be called in order to discover all the settings for the
   * converter object.
   *
   * @return the ConversionProperties object describing the default properties
   * for this converter.
   */
  virtual ConversionProperties getDefaultProperties() const;

};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

  
#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* SBMLInitialAssignmentConverter_h */

