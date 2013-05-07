/**
 * @file    SBMLFunctionDefinitionConverter.h
 * @brief   Definition of SBMLFunctionDefinitionConverter, a converter replacing function definitions
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
 * @class SBMLFunctionDefinitionConverter
 * @brief SBML converter for replacing function definitions.
 * 
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * This is an SBML converter for manipulating user-defined functions in an
 * SBML file.  When invoked on the current model, it performs the following
 * operation:
 * <ol>
 * <li>Read the list of user-defined functions in the model (i.e., the
 * list of FunctionDefinition objects);
 * <li>Look for invocations of the function in mathematical expressions
 * throughout the model; and
 * <li>For each invocation found, replaces the invocation with a
 * in-line copy of the function's body, similar to how macro expansions
 * might be performed in scripting and programming languages.
 * </ol>
 *
 * For example, suppose the model contains a function definition
 * representing the function <i>f(x, y) = x * y</i>.  Further
 * suppose this functions invoked somewhere else in the model, in
 * a mathematical formula, as <i>f(s, p)</i>.  The outcome of running
 * SBMLFunctionDefinitionConverter on the model will be to replace
 * the call to <i>f</i> with the expression <i>s * p</i>.
 *
 * @see SBMLInitialAssignmentConverter
 * @see SBMLLevelVersionConverter
 * @see SBMLRuleConverter
 * @see SBMLStripPackageConverter
 * @see SBMLUnitsConverter
 */

#ifndef SBMLFunctionDefinitionConverter_h
#define SBMLFunctionDefinitionConverter_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/conversion/SBMLConverter.h>
#include <sbml/conversion/SBMLConverterRegister.h>


#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SBMLFunctionDefinitionConverter : public SBMLConverter
{
public:

  /** @cond doxygen-libsbml-internal */
  
  /* register with the ConversionRegistry */
  static void init();  

  /** @endcond */


  /**
   * Creates a new SBMLFunctionDefinitionConverter object.
   */
  SBMLFunctionDefinitionConverter();


  /**
   * Copy constructor; creates a copy of an SBMLFunctionDefinitionConverter
   * object.
   *
   * @param obj the SBMLFunctionDefinitionConverter object to copy.
   */
  SBMLFunctionDefinitionConverter(const SBMLFunctionDefinitionConverter& obj);


  /**
   * Creates and returns a deep copy of this SBMLFunctionDefinitionConverter
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
   * an SBMLFunctionDefinitionConverter object to find out if the object's
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
   * Replaces invocations of each user-defined function with an in-line
   * copy, similar to macro expansion.
   *
   * @return  integer value indicating the success/failure of the operation.
   * @if clike The value is drawn from the enumeration
   * #OperationReturnValues_t. @endif@~ The possible values are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   * @li @link OperationReturnValues_t#LIBSBML_CONV_INVALID_SRC_DOCUMENT LIBSBML_CONV_INVALID_SRC_DOCUMENT @endlink
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


private:
  /** @cond doxygen-libsbml-internal */

  bool expandFD_errors(unsigned int errors);

  /** @endcond */


};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

  
#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* SBMLFunctionDefinitionConverter_h */

