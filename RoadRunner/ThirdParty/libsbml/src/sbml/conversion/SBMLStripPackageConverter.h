/**
 * @file    SBMLStripPackageConverter.h
 * @brief   Definition of SBMLStripPackageConverter, the base class for SBML conversion.
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
 * @class SBMLStripPackageConverter
 * @brief SBML converter for removing packages.
 * 
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * This SBML converter takes an SBML document and removes (strips) a package
 * from it.  No conversion is performed; the package constructs are simply
 * removed from the SBML document.  The package to be stripped is determined
 * by the value of the option "package" on the conversion properties.
 *
 * @see SBMLFunctionDefinitionConverter
 * @see SBMLLevelVersionConverter
 * @see SBMLRuleConverter
 * @see SBMLLevelVersionConverter
 * @see SBMLUnitsConverter
 */

#ifndef SBMLStripPackageConverter_h
#define SBMLStripPackageConverter_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/conversion/SBMLConverter.h>
#include <sbml/conversion/SBMLConverterRegister.h>

#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SBMLStripPackageConverter : public SBMLConverter
{
public:

  /** @cond doxygen-libsbml-internal */
  
  /* register with the ConversionRegistry */
  static void init();  

  /** @endcond */


  /**
   * Creates a new SBMLStripPackageConverter object.
   */
  SBMLStripPackageConverter ();


  /**
   * Copy constructor; creates a copy of an SBMLStripPackageConverter
   * object.
   *
   * @param obj the SBMLStripPackageConverter object to copy.
   */
  SBMLStripPackageConverter(const SBMLStripPackageConverter& obj);

  
  /**
   * Destroys this object.
   */
  virtual ~SBMLStripPackageConverter ();


  /**
   * Assignment operator for SBMLStripPackageConverter.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  SBMLStripPackageConverter& operator=(const SBMLStripPackageConverter& rhs);


  /**
   * Creates and returns a deep copy of this SBMLStripPackageConverter
   * object.
   * 
   * @return a (deep) copy of this converter.
   */
  virtual SBMLStripPackageConverter* clone() const;


  /**
   * Returns @c true if this converter object's properties match the given
   * properties.
   *
   * A typical use of this method involves creating a ConversionProperties
   * object, setting the options desired, and then calling this method on
   * an SBMLStripPackageConverter object to find out if the object's
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
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
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



#ifndef SWIG

#endif // SWIG



private:
  /** @cond doxygen-libsbml-internal */

  std::string getPackageToStrip();


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
#endif  /* SBMLStripPackageConverter_h */

