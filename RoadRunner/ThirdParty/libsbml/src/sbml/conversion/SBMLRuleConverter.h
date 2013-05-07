/**
 * @file    SBMLRuleConverter.h
 * @brief   Definition of SBMLRuleConverter, a converter sorting rules
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
 * @class SBMLRuleConverter
 * @brief SBML converter for reordering rules and assignments in a model.
 * 
 * @htmlinclude libsbml-facility-only-warning.html
 *
 * This converter reorders assignments in a model.  Specifically, it sorts
 * the list of assignment rules (i.e., the AssignmentRule objects contained
 * in the ListOfAssignmentRules within the Model object) and the initial
 * assignments (i.e., the InitialAssignment objects contained in the
 * ListOfInitialAssignments) such that, within each set, assignments that
 * depend on prior values are placed after the values are set.  For
 * example, if there is an assignment rule stating <i>a = b + 1</i>, and
 * another rule stating <i>b = 3</i>, the list of rules is sorted and the
 * rules are arranged so that the rule for <i>b = 3</i> appears @em before
 * the rule for <i>a = b + 1</i>.  Similarly, if dependencies of this
 * sort exist in the list of initial assignments in the model, the initial
 * assignments are sorted as well.
 *
 * Beginning with SBML Level 2, assignment rules have no ordering
 * required&mdash;the order in which the rules appear in an SBML file has
 * no significance.  Software tools, however, may need to reorder
 * assignments for purposes of evaluating them.  For example, for
 * simulators that use time integration methods, it would be a good idea to
 * reorder assignment rules such as the following,
 *
 * <i>b = a + 10 seconds</i><br>
 * <i>a = time</i>
 *
 * so that the evaluation of the rules is independent of integrator
 * step sizes. (This is due to the fact that, in this case, the order in
 * which the rules are evaluated changes the result.)  This converter
 * can be used to reorder the SBML objects regardless of whether the
 * input file contained them in the desired order.  Here is a code
 * fragment to illustrate how to do that:
 * @verbatim
ConversionProperties props;
props.addOption("sortRules", true, "sort rules");

SBMLConverter converter;
converter.setProperties(&props);
converter.setDocument(&doc);
converter.convert(); 
@endverbatim
 * 
 * @note The two sets of assignments (list of assignment rules on the one
 * hand, and list of initial assignments on the other hand) are handled @em
 * independently.  In an SBML model, these entities are treated differently
 * and no amount of sorting can deal with inter-dependencies between
 * assignments of the two kinds.
 *
 * @see SBMLFunctionDefinitionConverter
 * @see SBMLInitialAssignmentConverter
 * @see SBMLLevelVersionConverter
 * @see SBMLStripPackageConverter
 * @see SBMLUnitsConverter
 */


#ifndef SBMLRuleConverter_h
#define SBMLRuleConverter_h

#include <sbml/SBMLNamespaces.h>
#include <sbml/conversion/SBMLConverter.h>
#include <sbml/conversion/SBMLConverterRegister.h>

#ifdef __cplusplus


LIBSBML_CPP_NAMESPACE_BEGIN


class LIBSBML_EXTERN SBMLRuleConverter : public SBMLConverter
{
public:

  /** @cond doxygen-libsbml-internal */
  
  /* register with the ConversionRegistry */
  static void init();  

  /** @endcond */


  /**
   * Creates a new SBMLLevelVersionConverter object.
   */
  SBMLRuleConverter();


  /**
   * Copy constructor; creates a copy of an SBMLLevelVersionConverter
   * object.
   *
   * @param obj the SBMLLevelVersionConverter object to copy.
   */
  SBMLRuleConverter(const SBMLRuleConverter& obj);


  /**
   * Creates and returns a deep copy of this SBMLLevelVersionConverter
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
   * 
   * @return  integer value indicating the success/failure of the operation.
   * @if clike The value is drawn from the enumeration
   * #OperationReturnValues_t. @endif@~ The possible values are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
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
#endif  /* SBMLRuleConverter_h */

