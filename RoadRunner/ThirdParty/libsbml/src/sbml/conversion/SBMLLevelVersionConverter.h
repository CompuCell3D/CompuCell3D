/**
 * @file    SBMLLevelVersionConverter.h
 * @brief   Definition of SBMLLevelVersionConverter, the base class for SBML conversion.
 * @author  Sarah Keating
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
 * @class SBMLLevelVersionConverter
 * @brief SBML converter for transforming documents from one Level+Version to another.
 *
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * This SBML converter takes an SBML document of one SBML Level+Version
 * combination and attempts to convert it to another Level+Version combination.
 * The target Level+Version is set using an SBMLNamespace object in the
 * ConversionProperties object that controls this converter.
 *
 * This class is the basis for
 * SBMLDocument::setLevelAndVersion(@if java long lev, long ver, boolean strict@endif).
 * 
 * @see SBMLFunctionDefinitionConverter
 * @see SBMLInitialAssignmentConverter
 * @see SBMLRuleConverter
 * @see SBMLStripPackageConverter
 * @see SBMLUnitsConverter
 */

#ifndef SBMLLevelVersionConverter_h
#define SBMLLevelVersionConverter_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/conversion/SBMLConverter.h>
#include <sbml/conversion/SBMLConverterRegister.h>

#include <sbml/validator/ConsistencyValidator.h>
#include <sbml/validator/IdentifierConsistencyValidator.h>
#include <sbml/validator/MathMLConsistencyValidator.h>
#include <sbml/validator/SBOConsistencyValidator.h>
#include <sbml/validator/UnitConsistencyValidator.h>
#include <sbml/validator/OverdeterminedValidator.h>
#include <sbml/validator/ModelingPracticeValidator.h>
#include <sbml/validator/L1CompatibilityValidator.h>
#include <sbml/validator/L2v1CompatibilityValidator.h>
#include <sbml/validator/L2v2CompatibilityValidator.h>
#include <sbml/validator/L2v3CompatibilityValidator.h>
#include <sbml/validator/L2v4CompatibilityValidator.h>
#include <sbml/validator/L3v1CompatibilityValidator.h>
#include <sbml/validator/InternalConsistencyValidator.h>


#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN  SBMLLevelVersionConverter : public SBMLConverter
{
public:

  /** @cond doxygen-libsbml-internal */
  
  /* register with the ConversionRegistry */
  static void init();  

  /** @endcond */


  /**
   * Creates a new SBMLLevelVersionConverter object.
   */
  SBMLLevelVersionConverter ();


  /**
   * Copy constructor; creates a copy of an SBMLLevelVersionConverter
   * object.
   *
   * @param obj the SBMLLevelVersionConverter object to copy.
   */
  SBMLLevelVersionConverter(const SBMLLevelVersionConverter& obj);

  
  /**
   * Destroys this object.
   */
  virtual ~SBMLLevelVersionConverter ();


  /**
   * Assignment operator for SBMLLevelVersionConverter.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  SBMLLevelVersionConverter& operator=(const SBMLLevelVersionConverter& rhs);


  /**
   * Creates and returns a deep copy of this SBMLLevelVersionConverter
   * object.
   * 
   * @return a (deep) copy of this converter.
   */
  virtual SBMLLevelVersionConverter* clone() const;


  /**
   * Returns @c true if this converter object's properties match the given
   * properties.
   *
   * A typical use of this method involves creating a ConversionProperties
   * object, setting the options desired, and then calling this method on
   * an SBMLLevelVersionConverter object to find out if the object's
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


  /* Convenience functions for this converter */

  /**
   * Returns the target SBML Level for the conversion.
   *
   * @return an integer indicating the SBML Level.
   */
  unsigned int getTargetLevel();


  /**
   * Returns the target SBML Version for the conversion.
   *
   * @return an integer indicating the Version within the SBML Level.
   */
  unsigned int getTargetVersion();

 
  /**
   * Returns the flag indicating whether the conversion has been set to "strict".
   *
   * @return @c true if strict validity has been requested, @c false
   * otherwise.
   */
  bool getValidityFlag();


#ifndef SWIG

#endif // SWIG


private:
  /** @cond doxygen-libsbml-internal */

  bool conversion_errors(unsigned int errors, bool strictUnits = true);

  /*
   * Predicate returning true if model has strict unit consistency.
   */
  bool hasStrictUnits();

  /*
   * Predicate returning true if model has strict sbo consistency.
   */
  bool hasStrictSBO();

  /*
   * do actual conversion
   */  
  bool performConversion(bool strict, bool strictUnits, bool duplicateAnn);  

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
#endif  /* SBMLLevelVersionConverter_h */

