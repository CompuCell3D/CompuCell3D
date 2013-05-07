/**
 * @file    Compartment.h
 * @brief   Definitions of Compartment and ListOfCompartments
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
 * @class Compartment
 * @brief  LibSBML implementation of SBML's %Compartment construct.
 *
 * A compartment in SBML represents a bounded space in which species are
 * located.  Compartments do not necessarily have to correspond to actual
 * structures inside or outside of a biological cell.
 * 
 * It is important to note that although compartments are optional in the
 * overall definition of Model, every species in an SBML model must be
 * located in a compartment.  This in turn means that if a model defines
 * any species, the model must also define at least one compartment.  The
 * reason is simply that species represent physical things, and therefore
 * must exist @em somewhere.  Compartments represent the @em somewhere.
 *
 * Compartment has one required attribute, "id", to give the compartment a
 * unique identifier by which other parts of an SBML model definition can
 * refer to it.  A compartment can also have an optional "name" attribute
 * of type @c string.  Identifiers and names must be used according to the
 * guidelines described in the SBML specifications.
 * 
 * Compartment also has an optional attribute "spatialDimensions" that is
 * used to indicate the number of spatial dimensions possessed by the
 * compartment.  Most modeling scenarios involve compartments with integer
 * values of "spatialDimensions" of @c 3 (i.e., a three-dimensional
 * compartment, which is to say, a volume), or 2 (a two-dimensional
 * compartment, a surface), or @c 1 (a one-dimensional compartment, a
 * line).  In SBML Level&nbsp;3, the type of this attribute is @c double,
 * there are no restrictions on the permitted values of the
 * "spatialDimensions" attribute, and there are no default values.  In SBML
 * Level&nbsp;2, the value must be a positive @c integer, and the default
 * value is @c 3; the permissible values in SBML Level&nbsp;2 are @c 3, @c
 * 2, @c 1, and @c 0 (for a point).
 *
 * Another optional attribute on Compartment is "size", representing the
 * @em initial total size of that compartment in the model.  The "size"
 * attribute must be a floating-point value and may represent a volume (if
 * the compartment is a three-dimensional one), or an area (if the
 * compartment is two-dimensional), or a length (if the compartment is
 * one-dimensional).  There is no default value of compartment size in SBML
 * Level&nbsp;2 or Level&nbsp;3.  In particular, a missing "size" value
 * <em>does not imply that the compartment size is 1</em>.  (This is unlike
 * the definition of compartment "volume" in SBML Level&nbsp;1.)  When the
 * compartment's "spatialDimensions" attribute does not have a value of @c
 * 0, a missing value of "size" for a given compartment signifies that the
 * value either is unknown, or to be obtained from an external source, or
 * determined by an InitialAssignment, AssignmentRule, AlgebraicRule or
 * RateRule elsewhere in the model.  In SBML Level&nbsp;2, there are
 * additional special requirements on the values of "size"; we discuss them
 * in a <a href="#comp-l2">separate section below</a>.
 *
 * The units associated with a compartment's "size" attribute value may be
 * set using the optional attribute "units".  The rules for setting and
 * using compartment size units differ between SBML Level&nbsp;2 and
 * Level&nbsp;3, and are discussed separately below.
 * 
 * Finally, the optional Compartment attribute named "constant" is used to
 * indicate whether the compartment's size stays constant after simulation
 * begins.  A value of @c true indicates the compartment's "size" cannot be
 * changed by any other construct except InitialAssignment; a value of @c
 * false indicates the compartment's "size" can be changed by other
 * constructs in SBML.  In SBML Level&nbsp;2, there is an additional
 * explicit restriction that if "spatialDimensions"=@c "0", the value
 * cannot be changed by InitialAssignment either.  Further, in
 * Level&nbsp;2, "constant" has a default value of @c true.  In SBML
 * Level&nbsp;3, there is no default value for the "constant" attribute.
 *
 * 
 * @section comp-l2 Additional considerations in SBML Level&nbsp;2
 * 
 * In SBML Level&nbsp;2, the default units of compartment size, and the
 * kinds of units allowed as values of the attribute "units", interact with
 * the number of spatial dimensions of the compartment.  The value of the
 * "units" attribute of a Compartment object must be one of the base units
 * (see Unit), or the predefined unit identifiers @c volume, @c area, @c
 * length or @c dimensionless, or a new unit defined by a UnitDefinition
 * object in the enclosing Model, subject to the restrictions detailed in
 * the following table:
 *
 * @htmlinclude compartment-size-restrictions.html 
 *
 * In SBML Level&nbsp;2, the units of the compartment size, as defined by the
 * "units" attribute or (if "units" is not set) the default value listed in
 * the table above, are used in the following ways when the compartment has
 * a "spatialDimensions" value greater than @c 0:
 * <ul>
 * <li> The value of the "units" attribute is used as the units of the
 * compartment identifier when the identifier appears as a numerical
 * quantity in a mathematical formula expressed in MathML.
 * 
 * <li> The @c math element of an AssignmentRule or InitialAssignment
 * referring to this compartment must have identical units.
 *
 * <li> In RateRule objects that set the rate of change of the compartment's
 * size, the units of the rule's @c math element must be identical to the
 * compartment's "units" attribute divided by the default @em time units.
 * (In other words, the units for the rate of change of compartment size
 * are <em>compartment size</em>/<em>time</em> units.
 *
 * <li> When a Species is to be treated in terms of concentrations or
 * density, the units of the spatial size portion of the concentration
 * value (i.e., the denominator in the units formula @em substance/@em
 * size) are those indicated by the value of the "units" attribute on the
 * compartment in which the species is located.
 * </ul>
 *
 * Compartments with "spatialDimensions"=@c 0 require special treatment in
 * this framework.  As implied above, the "size" attribute must not have a
 * value on an SBML Level&nbsp;2 Compartment object if the
 * "spatialDimensions" attribute has a value of @c 0.  An additional
 * related restriction is that the "constant" attribute must default to or
 * be set to @c true if the value of the "spatialDimensions" attribute is
 * @c 0, because a zero-dimensional compartment cannot ever have a size.
 *
 * If a compartment has no size or dimensional units, how should such a
 * compartment's identifier be interpreted when it appears in mathematical
 * formulas?  The answer is that such a compartment's identifier should not
 * appear in mathematical formulas in the first place&mdash;it has no
 * value, and its value cannot change.  Note also that a zero-dimensional
 * compartment is a point, and species located at points can only be
 * described in terms of amounts, not spatially-dependent measures such as
 * concentration.  Since SBML KineticLaw formulas are already in terms of
 * @em substance/@em time and not (say) @em concentration/@em time, volume
 * or other factors in principle are not needed for species located in
 * zero-dimensional compartments.
 *
 * Finally, in SBML Level&nbsp;2 Versions 2&ndash;4, each compartment in a
 * model may optionally be designated as belonging to a particular
 * compartment @em type.  The optional attribute "compartmentType" is used
 * identify the compartment type represented by the Compartment structure.
 * The "compartmentType" attribute's value must be the identifier of a
 * CompartmentType instance defined in the model.  If the "compartmentType"
 * attribute is not present on a particular compartment definition, a
 * unique virtual compartment type is assumed for that compartment, and no
 * other compartment can belong to that compartment type.  The values of
 * "compartmentType" attributes on compartments have no effect on the
 * numerical interpretation of a model.  Simulators and other numerical
 * analysis software may ignore "compartmentType" attributes.  The
 * "compartmentType" attribute and the CompartmentType class of objects are
 * not present in SBML Level&nbsp;3 Core nor in SBML Level&nbsp;1.
 * 
 * 
 * @section comp-l3 Additional considerations in SBML Level&nbsp;3
 *
 * One difference between SBML Level&nbsp;3 and lower Levels of SBML is
 * that there are no restrictions on the permissible values of the
 * "spatialDimensions" attribute, and there is no default value defined for
 * the attribute.  The value of "spatialDimensions" does not have to be an
 * integer, either; this is to allow for the possibility of representing
 * structures with fractal dimensions.
 *
 * The number of spatial dimensions possessed by a compartment cannot enter
 * into mathematical formulas, and therefore cannot directly alter the
 * numerical interpretation of a model.  However, the value of
 * "spatialDimensions" @em does affect the interpretation of the units
 * associated with a compartment's size.  Specifically, the value of
 * "spatialDimensions" is used to select among the Model attributes
 * "volumeUnits", "areaUnits" and "lengthUnits" when a Compartment object
 * does not define a value for its "units" attribute.
 *
 * The "units" attribute may be left unspecified for a given compartment in
 * a model; in that case, the compartment inherits the unit of measurement
 * specified by one of the attributes on the enclosing Model object
 * instance.  The applicable attribute on Model depends on the value of the
 * compartment's "spatialDimensions" attribute; the relationship is shown
 * in the table below.  If the Model object does not define the relevant
 * attribute ("volumeUnits", "areaUnits" or "lengthUnits") for a given
 * "spatialDimensions" value, the unit associated with that Compartment
 * object's size is undefined.  If @em both "spatialDimensions" and "units"
 * are left unset on a given Compartment object instance, then no unit can
 * be chosen from among the Model's "volumeUnits", "areaUnits" or
 * "lengthUnits" attributes (even if the Model instance provides values for
 * those attributes), because there is no basis to select between them and
 * there is no default value of "spatialDimensions".  Leaving the units of
 * compartments' sizes undefined in an SBML model does not render the model
 * invalid; however, as a matter of best practice, we strongly recommend
 * that all models specify the units of measurement for all compartment
 * sizes.
 *
 * @htmlinclude compartment-size-recommendations.html
 *
 * The unit of measurement associated with a compartment's size, as defined
 * by the "units" attribute or (if "units" is not set) the inherited value
 * from Model according to the table above, is used in the following ways:
 *
 * <ul>
 * 
 * <li> When the identifier of the compartment appears as a numerical
 * quantity in a mathematical formula expressed in MathML, it represents
 * the size of the compartment, and the unit associated with the size is
 * the value of the "units" attribute.
 * 
 * <li> When a Species is to be treated in terms of concentrations or
 * density, the unit associated with the spatial size portion of the
 * concentration value (i.e., the denominator in the formula
 * <em>amount</em>/<em>size</em>) is specified by the value of the "units"
 * attribute on the compartment in which the species is located.
 * 
 * <li> The "math" elements of AssignmentRule, InitialAssignment and
 * EventAssignment objects setting the value of the compartment size
 * should all have the same units as the unit associated with the
 * compartment's size.
 * 
 * <li> In a RateRule object that defines a rate of change for a
 * compartment's size, the unit of the rule's "math" element should be
 * identical to the compartment's "units" attribute divided by the
 * model-wide unit of <em>time</em>.  (In other words, {<em>unit of
 * compartment size</em>}/{<em>unit of time</em>}.)
 * 
 * </ul>
 * 
 *
 * @section comp-other Other aspects of Compartment
 *
 * In SBML Level&nbsp;1 and Level&nbsp;2, Compartment has an optional
 * attribute named "outside", whose value can be the identifier of another
 * Compartment object defined in the enclosing Model object.  Doing so
 * means that the other compartment contains it or is outside of it.  This
 * enables the representation of simple topological relationships between
 * compartments, for those simulation systems that can make use of the
 * information (e.g., for drawing simple diagrams of compartments).  It is
 * worth noting that in SBML, there is no relationship between compartment
 * sizes when compartment positioning is expressed using the "outside"
 * attribute.  The size of a given compartment does not in any sense
 * include the sizes of other compartments having it as the value of their
 * "outside" attributes.  In other words, if a compartment @em B has the
 * identifier of compartment @em A as its "outside" attribute value, the
 * size of @em A does not include the size of @em B.  The compartment sizes
 * are separate.
 *
 * In Level&nbsp;2, there are two restrictions on the "outside" attribute.
 * First, because a compartment with "spatialDimensions" of @c 0 has no
 * size, such a compartment cannot act as the container of any other
 * compartment @em except compartments that @em also have
 * "spatialDimensions" values of @c 0.  Second, the directed graph formed
 * by representing Compartment structures as vertexes and the "outside"
 * attribute values as edges must be acyclic.  The latter condition is
 * imposed to prevent a compartment from being contained inside itself.  In
 * the absence of a value for "outside", compartment definitions in SBML
 * Level&nbsp;2 do not have any implied spatial relationships between each
 * other.
 * 
 * 
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 
/**
 * @class ListOfCompartments
 * @brief LibSBML implementation of SBML Level&nbsp;2's %ListOfCompartments construct.
 * 
 * The various ListOf___ classes in SBML are merely containers used for
 * organizing the main components of an SBML model.  All are derived from
 * the abstract class SBase, and inherit the various attributes and
 * subelements of SBase, such as "metaid" as and "annotation".  The
 * ListOf___ classes do not add any attributes of their own.
 *
 * The relationship between the lists and the rest of an SBML model is
 * illustrated by the following (for SBML Level&nbsp;2 Version&nbsp;4):
 *
 * @image html listof-illustration.jpg "ListOf___ elements in an SBML Model"
 * @image latex listof-illustration.jpg "ListOf___ elements in an SBML Model"
 *
 * Readers may wonder about the motivations for using the ListOf___
 * containers.  A simpler approach in XML might be to place the components
 * all directly at the top level of the model definition.  The choice made
 * in SBML is to group them within XML elements named after
 * ListOf<em>Classname</em>, in part because it helps organize the
 * components.  More importantly, the fact that the container classes are
 * derived from SBase means that software tools can add information @em about
 * the lists themselves into each list container's "annotation".
 *
 * @see ListOfFunctionDefinitions
 * @see ListOfUnitDefinitions
 * @see ListOfCompartmentTypes
 * @see ListOfSpeciesTypes
 * @see ListOfCompartments
 * @see ListOfSpecies
 * @see ListOfParameters
 * @see ListOfInitialAssignments
 * @see ListOfRules
 * @see ListOfConstraints
 * @see ListOfReactions
 * @see ListOfEvents
 */

#ifndef Compartment_h
#define Compartment_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#ifndef LIBSBML_USE_STRICT_INCLUDES
#include <sbml/annotation/RDFAnnotation.h>
#endif
#include <sbml/common/operationReturnValues.h>

#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLVisitor;


class LIBSBML_EXTERN Compartment : public SBase
{
public:

  /**
   * Creates a new Compartment using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this Compartment
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * Compartment
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of a Compartment object to an SBMLDocument
   * (e.g., using Model::addCompartment(@if java Compartment c@endif)), the SBML Level, SBML Version
   * and XML namespace of the document @em override the values used
   * when creating the Compartment object via this constructor.  This is
   * necessary to ensure that an SBML document is a consistent structure.
   * Nevertheless, the ability to supply the values at the time of creation
   * of a Compartment is an important aid to producing valid SBML.
   * Knowledge of the intented SBML Level and Version determine whether it
   * is valid to assign a particular value to an attribute, or whether it
   * is valid to add an object to an existing SBMLDocument.
   */
  Compartment (unsigned int level, unsigned int version);
  
  
  /**
   * Creates a new Compartment using the given SBMLNamespaces object 
   * @p sbmlns.
   *
   * The SBMLNamespaces object encapsulates SBML Level/Version/namespaces
   * information.  It is used to communicate the SBML Level, Version, and
   * (in Level&nbsp;3) packages used in addition to SBML Level&nbsp;3 Core.
   * A common approach to using this class constructor is to create an
   * SBMLNamespaces object somewhere in a program, once, then pass it to
   * object constructors such as this one when needed.
   *
   * It is worth emphasizing that although this constructor does not take
   * an identifier argument, in SBML Level&nbsp;2 and beyond, the "id"
   * (identifier) attribute of a Compartment is required to have a value.
   * Thus, callers are cautioned to assign a value after calling this
   * constructor.  Setting the identifier can be accomplished using the
   * method @if java Compartment::setId(String id)@else setId()@endif.
   *
   * @param sbmlns an SBMLNamespaces object.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   *
   * @note Upon the addition of a Compartment object to an SBMLDocument
   * (e.g., using Model::addCompartment(@if java Compartment c@endif)), the SBML XML namespace of the
   * document @em overrides the value used when creating the Compartment
   * object via this constructor.  This is necessary to ensure that an SBML
   * document is a consistent structure.  Nevertheless, the ability to
   * supply the values at the time of creation of a Compartment is an
   * important aid to producing valid SBML.  Knowledge of the intented SBML
   * Level and Version determine whether it is valid to assign a particular
   * value to an attribute, or whether it is valid to add an object to an
   * existing SBMLDocument.
   */
  Compartment (SBMLNamespaces* sbmlns);


  /**
   * Destroys this Compartment.
   */
  virtual ~Compartment ();


  /**
   * Copy constructor; creates a copy of a Compartment.
   * 
   * @param orig the Compartment instance to copy.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  Compartment(const Compartment& orig);


  /**
   * Assignment operator for Compartment.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  Compartment& operator=(const Compartment& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of Compartment.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next Compartment in the
   * list of compartments within which this Compartment is embedded (i.e.,
   * the ListOfCompartments in the parent Model).
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this Compartment object.
   * 
   * @return a (deep) copy of this Compartment.
   */
  virtual Compartment* clone () const;


  /**
   * Initializes the fields of this Compartment object to "typical" default
   * values.
   *
   * The SBML Compartment component has slightly different aspects and
   * default attribute values in different SBML Levels and Versions.
   * This method sets the values to certain common defaults, based
   * mostly on what they are in SBML Level&nbsp;2.  Specifically:
   * <ul>
   * <li> Sets attribute "spatialDimensions" to @c 3
   * <li> Sets attribute "constant" to @c true
   * <li> (Applies to Level&nbsp;1 models only) Sets attribute "volume" to @c 1.0
   * <li> (Applies to Level&nbsp;3 models only) Sets attribute "units" to @c litre
   * </ul>
   */
  void initDefaults ();


  /**
   * Returns the value of the "id" attribute of this Compartment object.
   * 
   * @return the id of this Compartment.
   */
  virtual const std::string& getId () const;


  /**
   * Returns the value of the "name" attribute of this Compartment object.
   * 
   * @return the name of this Compartment.
   */
  virtual const std::string& getName () const;


  /**
   * Get the value of the "compartmentType" attribute of this Compartment
   * object.
   * 
   * @return the value of the "compartmentType" attribute of this
   * Compartment as a string.
   *
   * @note The "compartmentType" attribute is only available in SBML
   * Level&nbsp;2 Versions&nbsp;2&ndash;4.
   */
  const std::string& getCompartmentType () const;


  /**
   * Get the number of spatial dimensions of this Compartment object.
   *
   * @note In SBML Level&nbsp;3, the data type of the "spatialDimensions"
   * attribute is @c double, whereas in Level&nbsp;2, it is @c integer.
   * LibSBML provides a separate method for obtaining the value as a double,
   * for models where it is relevant.
   *
   * @return the value of the "spatialDimensions" attribute of this
   * Compartment as an unsigned integer
   *
   * @see getSpatialDimensionsAsDouble()
   */
  unsigned int getSpatialDimensions () const;


  /**
   * Get the number of spatial dimensions of this Compartment object
   * as a double.
   *
   * @note In SBML Level&nbsp;3, the data type of the "spatialDimensions"
   * attribute is @c double, whereas in Level&nbsp;2, it is @c integer.  To
   * avoid backward compatibility issues, libSBML provides a separate
   * method for obtaining the value as a double, for models where it is
   * relevant.
   *
   * @return the value of the "spatialDimensions" attribute of this
   * Compartment as a double, or @c NaN if this model is not in SBML
   * Level&nbsp;3 format.
   *
   * @see getSpatialDimensions()
   */
  double getSpatialDimensionsAsDouble () const;


  /**
   * Get the size of this Compartment.
   *
   * This method is identical to
   * @if java Compartment::getVolume()@else getVolume()@endif.
   * In SBML Level&nbsp;1, compartments are always three-dimensional
   * constructs and only have volumes, whereas in SBML Level&nbsp;2,
   * compartments may be other than three-dimensional and therefore the
   * "volume" attribute is named "size" in Level&nbsp;2.  LibSBML provides
   * both
   * @if java Compartment::getSize()@else getSize()@endif@~ and
   * @if java Compartment::getVolume()@else getVolume()@endif@~ for
   * easier compatibility between SBML Levels.
   *
   * @return the value of the "size" attribute ("volume" in Level&nbsp;1) of
   * this Compartment as a float-point number.
   *
   * @see isSetSize()
   * @see getVolume()
   */
  double getSize () const;


  /**
   * Get the volume of this Compartment.
   * 
   * This method is identical to
   * @if java Compartment::getSize()@else getSize()@endif.  In
   * SBML Level&nbsp;1, compartments are always three-dimensional
   * constructs and only have volumes, whereas in SBML Level&nbsp;2,
   * compartments may be other than three-dimensional and therefore the
   * "volume" attribute is named "size" in Level&nbsp;2.  LibSBML provides
   * both
   * @if java Compartment::getSize()@else getSize()@endif@~ and
   * @if java Compartment::getVolume()@else getVolume()@endif@~
   * for easier compatibility between SBML Levels.
   *
   * @return the value of the "volume" attribute ("size" in Level&nbsp;2) of
   * this Compartment, as a floating-point number.
   *
   * @note The attribute "volume" only exists by that name in SBML
   * Level&nbsp;1.  In Level&nbsp;2 and above, the equivalent attribute is
   * named "size".
   * 
   * @see isSetVolume()
   * @see getSize()
   */
  double getVolume () const;


  /**
   * Get the units of this compartment's size.
   * 
   * The value of an SBML compartment's "units" attribute establishes the
   * unit of measurement associated with the compartment's size.
   *
   * @return the value of the "units" attribute of this Compartment, as a
   * string.  An empty string indicates that no units have been assigned to
   * the value of the size.
   *
   * @note @htmlinclude unassigned-units-are-not-a-default.html
   *
   * @see isSetUnits()
   * @see @if java Compartment::setUnits(String sid)@else setUnits()@endif@~
   * @see getSize()
   */
  const std::string& getUnits () const;


  /**
   * Get the identifier, if any, of the compartment that is designated
   * as being outside of this one.
   * 
   * @return the value of the "outside" attribute of this Compartment.
   *
   * @note The "outside" attribute is defined in SBML Level&nbsp;1 and
   * Level&nbsp;2, but does not exist in SBML Level&nbsp;3 Version&nbsp;1
   * Core.
   */
  const std::string& getOutside () const;


  /**
   * Get the value of the "constant" attribute of this Compartment.
   *
   * @return @c true if this Compartment's size is flagged as being
   * constant, @c false otherwise.
   */
  bool getConstant () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "id" attribute is set.
   *
   * @return @c true if the "id" attribute of this Compartment is 
   * set, @c false otherwise.
   */
  virtual bool isSetId () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "name" attribute is set.
   *
   * @return @c true if the "name" attribute of this Compartment is
   * set, @c false otherwise.
   */
  virtual bool isSetName () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "compartmentType" attribute is set.
   *
   * @return @c true if the "compartmentType" attribute of this Compartment
   * is set, @c false otherwise.
   *
   * @note The "compartmentType" attribute is only available in SBML
   * Level&nbsp;2 Versions&nbsp;2&ndash;4.
   */
  bool isSetCompartmentType () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "size" attribute is set.
   *
   * This method is similar but not identical to
   * @if java Compartment::isSetVolume()@else isSetVolume()@endif.  The latter
   * should be used in the context of SBML Level&nbsp;1 models instead of
   * @if java Compartment::isSetSize()@else isSetSize()@endif@~
   * because @if java Compartment::isSetVolume()@else isSetVolume()@endif@~
   * performs extra processing to take into account the difference in
   * default values between SBML Levels 1 and 2.
   * 
   * @return @c true if the "size" attribute ("volume" in Level&nbsp;2) of
   * this Compartment is set, @c false otherwise.
   *
   * @see isSetVolume()
   * @see setSize(double value)
   */
  bool isSetSize () const;


  /**
   * Predicate returning @c true if this Compartment's
   * "volume" attribute is set.
   * 
   * This method is similar but not identical to
   * @if java Compartment::isSetSize()@else isSetSize()@endif.  The latter
   * should not be used in the context of SBML Level&nbsp;1 models because this
   * method performs extra processing to take into account
   * the difference in default values between SBML Levels 1 and 2.
   * 
   * @return @c true if the "volume" attribute ("size" in Level&nbsp;2 and
   * above) of this Compartment is set, @c false otherwise.
   *
   * @note The attribute "volume" only exists by that name in SBML
   * Level&nbsp;1.  In Level&nbsp;2 and above, the equivalent attribute is
   * named "size".  In SBML Level&nbsp;1, a compartment's volume has a
   * default value (@c 1.0) and therefore this method will always return @c
   * true.  In Level 2, a compartment's size (the equivalent of SBML
   * Level&nbsp;1's "volume") is optional and has no default value, and
   * therefore may or may not be set.
   *
   * @see isSetSize()
   * @see @if java Compartment::setVolume(double value)@else setVolume()@endif@~
   */
  bool isSetVolume () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "units" attribute is set.
   * 
   * @return @c true if the "units" attribute of this Compartment is
   * set, @c false otherwise.
   *
   * @note @htmlinclude unassigned-units-are-not-a-default.html
   */
  bool isSetUnits () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "outside" attribute is set.
   * 
   * @return @c true if the "outside" attribute of this Compartment is
   * set, @c false otherwise.
   * 
   * @note The "outside" attribute is defined in SBML Level&nbsp;1 and
   * Level&nbsp;2, but does not exist in SBML Level&nbsp;3 Version&nbsp;1
   * Core.
   */
  bool isSetOutside () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "spatialDimensions" attribute is set.
   * 
   * @return @c true if the "spatialDimensions" attribute of this
   * Compartment is set, @c false otherwise.
   */
  bool isSetSpatialDimensions () const;


  /**
   * Predicate returning @c true if this
   * Compartment's "constant" attribute is set.
   * 
   * @return @c true if the "constant" attribute of this Compartment is
   * set, @c false otherwise.
   */
  bool isSetConstant () const;


  /**
   * Sets the value of the "id" attribute of this Compartment.
   *
   * The string @p sid is copied.  Note that SBML has strict requirements
   * for the syntax of identifiers.  @htmlinclude id-syntax.html
   *
   * @param sid the string to use as the identifier of this Compartment
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setId (const std::string& sid);


  /**
   * Sets the value of the "name" attribute of this Compartment.
   *
   * The string in @p name is copied.
   *
   * @param name the new name for the Compartment
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setName (const std::string& name);


  /**
   * Sets the "compartmentType" attribute of this Compartment.
   *
   * @param sid the identifier of a CompartmentType object defined
   * elsewhere in this Model.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * 
   * @note The "compartmentType" attribute is only available in SBML
   * Level&nbsp;2 Versions&nbsp;2&ndash;4.
   */
  int setCompartmentType (const std::string& sid);


  /**
   * Sets the "spatialDimensions" attribute of this Compartment.
   *
   * If @p value is not one of @c 0, @c 1, @c 2, or @c 3, this method will
   * have no effect (i.e., the "spatialDimensions" attribute will not be
   * set).
   * 
   * @param value an unsigned integer indicating the number of dimensions
   * of this compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   */
  int setSpatialDimensions (unsigned int value);


  /**
   * Sets the "spatialDimensions" attribute of this Compartment as a double.
   *
   * @param value a double indicating the number of dimensions
   * of this compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   */
  int setSpatialDimensions (double value);


  /**
   * Sets the "size" attribute (or "volume" in SBML Level&nbsp;1) of this
   * Compartment.
   *
   * This method is identical to
   * @if java Compartment::setVolume(double value)@else setVolume()@endif@~
   * and is provided for compatibility between
   * SBML Level&nbsp;1 and Level&nbsp;2.
   *
   * @param value a @c double representing the size of this compartment
   * instance in whatever units are in effect for the compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   *
   * @note The attribute "volume" only exists by that name in SBML
   * Level&nbsp;1.  In Level&nbsp;2 and above, the equivalent attribute is
   * named "size".
   */
  int setSize (double value);


  /**
   * Sets the "volume" attribute (or "size" in SBML Level&nbsp;2) of this
   * Compartment.
   *
   * This method is identical to
   * @if java Compartment::setVolume(double value)@else setVolume()@endif@~
   * and is provided for compatibility between SBML Level&nbsp;1 and
   * Level&nbsp;2.
   * 
   * @param value a @c double representing the volume of this compartment
   * instance in whatever units are in effect for the compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   *
   * @note The attribute "volume" only exists by that name in SBML
   * Level&nbsp;1.  In Level&nbsp;2 and above, the equivalent attribute is
   * named "size".
   */
  int setVolume (double value);


  /**
   * Sets the "units" attribute of this Compartment.
   *
   * @param sid the identifier of the defined units to use.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  int setUnits (const std::string& sid);


  /**
   * Sets the "outside" attribute of this Compartment.
   *
   * @param sid the identifier of a compartment that encloses this one.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   *
   * @note The "outside" attribute is defined in SBML Level&nbsp;1 and
   * Level&nbsp;2, but does not exist in SBML Level&nbsp;3 Version&nbsp;1
   * Core.
   */
  int setOutside (const std::string& sid);


  /**
   * Sets the value of the "constant" attribute of this Compartment.
   *
   * @param value a boolean indicating whether the size/volume of this
   * compartment should be considered constant (@c true) or variable
   * (@c false)
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   */
  int setConstant (bool value);


  /**
   * Renames all the SIdRef attributes on this element, including any found in MathML
   */
  virtual void renameSIdRefs(std::string oldid, std::string newid);


  /**
   * Renames all the UnitSIdRef attributes on this element
   */
  virtual void renameUnitSIdRefs(std::string oldid, std::string newid);


  /**
   * Unsets the value of the "name" attribute of this Compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int unsetName ();


  /**
   * Unsets the value of the "compartmentType"
   * attribute of this Compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * 
   * @note The "compartmentType" attribute is only available in SBML
   * Level&nbsp;2 Versions&nbsp;2&ndash;4.
   *
   * @see setCompartmentType(const std::string& sid)
   * @see isSetCompartmentType()
   */
  int unsetCompartmentType ();


  /**
   * Unsets the value of the "size" attribute of this Compartment.
   * 
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note The attribute "volume" only exists by that name in SBML
   * Level&nbsp;1.  In Level&nbsp;2 and above, the equivalent attribute is
   * named "size".
   */
  int unsetSize ();


  /**
   * Unsets the value of the "volume" attribute of this
   * Compartment.
   * 
   * In SBML Level&nbsp;1, a Compartment volume has a default value (@c 1.0) and
   * therefore <em>should always be set</em>.  In Level&nbsp;2, "size" is
   * optional with no default value and as such may or may not be set.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note The attribute "volume" only exists by that name in SBML
   * Level&nbsp;1.  In Level&nbsp;2 and above, the equivalent attribute is
   * named "size".
   */
  int unsetVolume ();


  /**
   * Unsets the value of the "units" attribute of this Compartment.
   * 
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetUnits ();


  /**
   * Unsets the value of the "outside" attribute of this Compartment.
   * 
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note The "outside" attribute is defined in SBML Level&nbsp;1 and
   * Level&nbsp;2, but does not exist in SBML Level&nbsp;3 Version&nbsp;1
   * Core.
   */
  int unsetOutside ();


  /**
   * Unsets the value of the "spatialDimensions" attribute of this Compartment.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   *
   * @note This function is only valid for SBML Level&nbsp;3.
   */
  int unsetSpatialDimensions ();


  /**
   * Constructs and returns a UnitDefinition that corresponds to the units
   * of this Compartment's designated size.
   *
   * Compartments in SBML have an attribute ("units") for declaring the
   * units of measurement intended for the value of the compartment's size.
   * In the absence of a value given for this attribute, the units are
   * inherited from values either defined on the enclosing Model (in SBML
   * Level&nbsp;3) or in defaults (in SBML Level&nbsp;2).  This method
   * returns a UnitDefinition object based on how this compartment's units
   * are interpreted according to the relevant SBML guidelines, or it
   * returns @c NULL if no units have been declared and no defaults are
   * defined by the relevant SBML specification.
   *
   * Note that unit declarations for Compartment objects are specified in
   * terms of the @em identifier of a unit (e.g., using
   * @if java Compartment::setUnits(String sid)@else setUnits()@endif), but
   * @em this method returns a UnitDefinition object, not a unit
   * identifier.  It does this by constructing an appropriate
   * UnitDefinition.  For SBML Level&nbsp;2 models, it will do this even
   * when the value of the "units" attribute is one of the special SBML
   * Level&nbsp;2 unit identifiers @c "substance", @c "volume", @c "area",
   * @c "length" or @c "time".  Callers may find this useful in conjunction
   * with the helper methods provided by the UnitDefinition class for
   * comparing different UnitDefinition objects.
   * 
   * @return a UnitDefinition that expresses the units of this 
   * Compartment, or @c NULL if one cannot be constructed.
   *
   * @note The libSBML system for unit analysis depends on the model as a
   * whole.  In cases where the Compartment object has not yet been added
   * to a model, or the model itself is incomplete, unit analysis is not
   * possible, and consequently this method will return @c NULL.
   *
   * @see isSetUnits()
   * @see getUnits()
   */
  UnitDefinition * getDerivedUnitDefinition();


  /**
   * Constructs and returns a UnitDefinition that corresponds to the units
   * of this Compartment's designated size.
   *
   * Compartments in SBML have an attribute ("units") for declaring the
   * units of measurement intended for the value of the compartment's size.
   * In the absence of a value given for this attribute, the units are
   * inherited from values either defined on the enclosing Model (in SBML
   * Level&nbsp;3) or in defaults (in SBML Level&nbsp;2).  This method
   * returns a UnitDefinition object based on how this compartment's units
   * are interpreted according to the relevant SBML guidelines, or it
   * returns @c NULL if no units have been declared and no defaults are
   * defined by the relevant SBML specification.
   *
   * Note that unit declarations for Compartment objects are specified in
   * terms of the @em identifier of a unit (e.g., using setUnits(@if java String sid@endif)), but
   * @em this method returns a UnitDefinition object, not a unit
   * identifier.  It does this by constructing an appropriate
   * UnitDefinition.  For SBML Level&nbsp;2 models, it will do this even
   * when the value of the "units" attribute is one of the special SBML
   * Level&nbsp;2 unit identifiers @c "substance", @c "volume", @c "area",
   * @c "length" or @c "time".  Callers may find this useful in conjunction
   * with the helper methods provided by the UnitDefinition class for
   * comparing different UnitDefinition objects.
   * 
   * @return a UnitDefinition that expresses the units of this 
   * Compartment, or @c NULL if one cannot be constructed.
   *
   * @note The libSBML system for unit analysis depends on the model as a
   * whole.  In cases where the Compartment object has not yet been added
   * to a model, or the model itself is incomplete, unit analysis is not
   * possible, and consequently this method will return @c NULL.
   *
   * @see isSetUnits()
   * @see getUnits()
   */
  const UnitDefinition * getDerivedUnitDefinition() const;


  /**
   * Returns the libSBML type code for this SBML object.
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
   * @return the SBML type code for this object, or
   * @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Returns the XML element name of this object, which for Compartment, is
   * always @c "compartment".
   * 
   * @return the name of this element, i.e., @c "compartment".
   */
  virtual const std::string& getElementName () const;


  /** @cond doxygen-libsbml-internal */

  /**
   * Subclasses should override this method to write out their contained
   * SBML objects as XML elements.  Be sure to call your parents
   * implementation of this method as well.
   */
  virtual void writeElements (XMLOutputStream& stream) const;

  /** @endcond */


  /**
   * Predicate returning @c true if
   * all the required attributes for this Compartment object
   * have been set.
   *
   * @note The required attributes for a Compartment object are:
   * @li "id" (or "name" in SBML Level&nbsp;1)
   * @li "constant" (in SBML Level&nbsp;3 only)
   *
   * @return a boolean value indicating whether all the required
   * attributes for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const;


protected:
  /** @cond doxygen-libsbml-internal */



  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);


  /**
   * Subclasses should override this method to read values from the given
   * XMLAttributes set into their specific fields.  Be sure to call your
   * parents implementation of this method as well.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);

  void readL1Attributes (const XMLAttributes& attributes);

  void readL2Attributes (const XMLAttributes& attributes);
  
  void readL3Attributes (const XMLAttributes& attributes);


  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;

  bool isExplicitlySetSpatialDimensions() const { 
    return mExplicitlySetSpatialDimensions; };

  bool isExplicitlySetConstant() const { return mExplicitlySetConstant; } ;


  std::string   mId;
  std::string   mName;
  std::string   mCompartmentType;
  unsigned int  mSpatialDimensions;
  double        mSpatialDimensionsDouble;
  double        mSize;
  std::string   mUnits;
  std::string   mOutside;
  bool          mConstant;

  bool  mIsSetSize;
  bool  mIsSetSpatialDimensions;
  bool  mIsSetConstant;
  bool  mExplicitlySetSpatialDimensions;
  bool  mExplicitlySetConstant;

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
  friend class L3v1CompatibilityValidator;
  friend class MathMLConsistencyValidator;
  friend class ModelingPracticeValidator;
  friend class OverdeterminedValidator;
  friend class SBOConsistencyValidator;
  friend class UnitConsistencyValidator;

  /** @endcond */
};


class LIBSBML_EXTERN ListOfCompartments : public ListOf
{
public:

  /**
   * Creates a new ListOfCompartments object.
   *
   * The object is constructed such that it is valid for the given SBML
   * Level and Version combination.
   *
   * @param level the SBML Level
   * 
   * @param version the Version within the SBML Level
   */
  ListOfCompartments (unsigned int level, unsigned int version);


  /**
   * Creates a new ListOfCompartments object.
   *
   * The object is constructed such that it is valid for the SBML Level and
   * Version combination determined by the SBMLNamespaces object in @p
   * sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object that is used to determine the
   * characteristics of the ListOfCompartments object to be created.
   */
  ListOfCompartments (SBMLNamespaces* sbmlns);


  /**
   * Creates and returns a deep copy of this ListOfCompartments instance.
   *
   * @return a (deep) copy of this ListOfCompartments.
   */
  virtual ListOfCompartments* clone () const;


  /**
   * Returns the libSBML type code for this SBML object.
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
  virtual int getTypeCode () const { return SBML_LIST_OF; };


  /**
   * Returns the libSBML type code for the objects contained in this ListOf
   * (i.e., Compartment objects, if the list is non-empty).
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
   * @return the SBML type code for the objects contained in this ListOf
   * instance, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getItemTypeCode () const;


  /**
   * Returns the XML element name of this object.
   *
   * For ListOfCompartments, the XML element name is @c "listOfCompartments".
   * 
   * @return the name of this element, i.e., @c "listOfCompartments".
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a Compartment from the ListOfCompartments.
   *
   * @param n the index number of the Compartment to get.
   * 
   * @return the nth Compartment in this ListOfCompartments.
   *
   * @see size()
   */
  virtual Compartment * get(unsigned int n); 


  /**
   * Get a Compartment from the ListOfCompartments.
   *
   * @param n the index number of the Compartment to get.
   * 
   * @return the nth Compartment in this ListOfCompartments.
   *
   * @see size()
   */
  virtual const Compartment * get(unsigned int n) const; 


  /**
   * Get a Compartment from the ListOfCompartments
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Compartment to get.
   * 
   * @return Compartment in this ListOfCompartments
   * with the given id or @c NULL if no such
   * Compartment exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual Compartment* get (const std::string& sid);


  /**
   * Get a Compartment from the ListOfCompartments
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Compartment to get.
   * 
   * @return Compartment in this ListOfCompartments
   * with the given id or @c NULL if no such
   * Compartment exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const Compartment* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfCompartments items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual Compartment* remove (unsigned int n);


  /**
   * Removes item in this ListOfCompartments items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then
   * @c NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual Compartment* remove (const std::string& sid);


  /** @cond doxygen-libsbml-internal */

  /**
   * Get the ordinal position of this element in the containing object
   * (which in this case is the Model object).
   *
   * The ordering of elements in the XML form of SBML is generally fixed
   * for most components in SBML.  So, for example, the ListOfCompartments
   * in a model is (in SBML Level&nbsp;2 Version&nbsp;4) the fifth
   * ListOf___.  (However, it differs for different Levels and Versions of
   * SBML.)
   *
   * @return the ordinal position of the element with respect to its
   * siblings, or @c -1 (default) to indicate the position is not significant.
   */
  virtual int getElementPosition () const;

  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or @c NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/* ----------------------------------------------------------------------------
 * See the .cpp file for the documentation of the following functions.
 * --------------------------------------------------------------------------*/


/*
LIBSBML_EXTERN
Compartment_t *
Compartment_createWithLevelVersionAndNamespaces (unsigned int level,
              unsigned int version, XMLNamespaces_t *xmlns);
*/

LIBSBML_EXTERN
Compartment_t *
Compartment_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Compartment_t *
Compartment_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
Compartment_free (Compartment_t *c);


LIBSBML_EXTERN
Compartment_t *
Compartment_clone (const Compartment_t* c);


LIBSBML_EXTERN
void
Compartment_initDefaults (Compartment_t *c);


LIBSBML_EXTERN
const XMLNamespaces_t *
Compartment_getNamespaces(Compartment_t *c);


LIBSBML_EXTERN
const char *
Compartment_getId (const Compartment_t *c);


LIBSBML_EXTERN
const char *
Compartment_getName (const Compartment_t *c);


LIBSBML_EXTERN
const char *
Compartment_getCompartmentType (const Compartment_t *c);


LIBSBML_EXTERN
unsigned int
Compartment_getSpatialDimensions (const Compartment_t *c);


LIBSBML_EXTERN
double
Compartment_getSpatialDimensionsAsDouble (const Compartment_t *c);


LIBSBML_EXTERN
double
Compartment_getSize (const Compartment_t *c);


LIBSBML_EXTERN
double
Compartment_getVolume (const Compartment_t *c);


LIBSBML_EXTERN
const char *
Compartment_getUnits (const Compartment_t *c);


LIBSBML_EXTERN
const char *
Compartment_getOutside (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_getConstant (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetId (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetName (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetCompartmentType (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetSize (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetVolume (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetUnits (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetOutside (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetSpatialDimensions (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_isSetConstant (const Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_setId (Compartment_t *c, const char *sid);


LIBSBML_EXTERN
int
Compartment_setName (Compartment_t *c, const char *string);


LIBSBML_EXTERN
int
Compartment_setCompartmentType (Compartment_t *c, const char *sid);


LIBSBML_EXTERN
int
Compartment_setSpatialDimensions (Compartment_t *c, unsigned int value);


LIBSBML_EXTERN
int
Compartment_setSpatialDimensionsAsDouble (Compartment_t *c, double value);


LIBSBML_EXTERN
int
Compartment_setSize (Compartment_t *c, double value);


LIBSBML_EXTERN
int
Compartment_setVolume (Compartment_t *c, double value);


LIBSBML_EXTERN
int
Compartment_setUnits (Compartment_t *c, const char *sid);


LIBSBML_EXTERN
int
Compartment_setOutside (Compartment_t *c, const char *sid);


LIBSBML_EXTERN
int
Compartment_setConstant (Compartment_t *c, int value);


LIBSBML_EXTERN
int
Compartment_unsetName (Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_unsetCompartmentType (Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_unsetSize (Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_unsetVolume (Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_unsetUnits (Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_unsetOutside (Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_unsetSpatialDimensions (Compartment_t *c);


LIBSBML_EXTERN
UnitDefinition_t * 
Compartment_getDerivedUnitDefinition(Compartment_t *c);


LIBSBML_EXTERN
int
Compartment_hasRequiredAttributes (Compartment_t *c);


LIBSBML_EXTERN
Compartment_t *
ListOfCompartments_getById (ListOf_t *lo, const char *sid);


LIBSBML_EXTERN
Compartment_t *
ListOfCompartments_removeById (ListOf_t *lo, const char *sid);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* Compartment_h */

