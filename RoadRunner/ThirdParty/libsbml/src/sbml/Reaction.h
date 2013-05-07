/**
 * @file    Reaction.h
 * @brief   Definitions of Reaction and ListOfReactions.
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
 * @class Reaction
 * @brief LibSBML implementation of SBML's %Reaction construct.
 *
 * A @em reaction represents any transformation, transport or binding
 * process, typically a chemical reaction, that can change the quantity of
 * one or more species.  In SBML, a reaction is defined primarily in terms
 * of the participating reactants and products (and their corresponding
 * stoichiometries), along with optional modifier species, an optional rate
 * at which the reaction takes place, and optional parameters.  
 * 
 * As with other major object in SBML, Reaction has a mandatory attribute,
 * "id", used to give the compartment type an identifier.  The identifier
 * must be a text string conforming to the identifer syntax permitted in
 * SBML.  In SBML Level&nbsp;2 and Level&nbsp;3, the reaction "id"
 * identifier can be used in mathematical formulas elsewhere in an SBML
 * model to represent the rate of that reaction; this usage is explained
 * below.  Reaction also has an optional "name" attribute, of type @c
 * string.  The "id" and "name" must be used according to the guidelines
 * described in the SBML specification.
 *
 * The species participating as reactants, products, and/or modifiers in a
 * reaction are declared using lists of SpeciesReference and/or
 * ModifierSpeciesReference instances stored in subelements
 * "listOfReactants", "listOfProducts" and "listOfModifiers".  Certain
 * restrictions are placed on the appearance of species in reaction
 * definitions:
 * <ul>
 * <li> The ability of a species to appear as a reactant or product of any
 * reaction in a model is governed by certain flags in that species'
 * definition; see the definition of Species for more information.
 *
 * <li> Any species appearing in the mathematical formula of the subelement
 * "kineticLaw" (described below) of a Reaction must be declared in at
 * least one of that Reaction's lists of reactants, products, and/or
 * modifiers.  Put another way, it is an error for a reaction's kinetic law
 * formula to refer to species that have not been declared for that
 * reaction.
 *
 * <li> A reaction definition can contain an empty list of reactants
 * <em>or</em> an empty list of products, but it must have at least one
 * reactant or product; in other words, a reaction without any reactant or
 * product species is not permitted.  (This restriction does not apply to
 * modifier species, which remain optional in all cases.)
 * </ul>
 * 
 * A reaction can contain up to one KineticLaw object in a subelement named
 * "kineticLaw".  It defines the speed at which the process defined by the
 * reaction takes place.  The description of KineticLaw provides more
 * details about its use.  Note that although the inclusion of a KineticLaw
 * object in an instance of a Reaction component is optional, there is no
 * useful default that can be substituted in place of a missing rate
 * expression in a reaction.  Moreover, a reaction's rate cannot be defined
 * in any other way in SBML&mdash;InitialAssignment, AssignmentRule,
 * RateRule, AlgebraicRule, Event, and other constructs in SBML cannot be
 * used to set the reaction rate separately.  Nevertheless, for some
 * modeling applications, reactions without any defined rate can be
 * perfectly acceptable.
 *
 * Reaction also has a boolean attribute named "reversible" for indicating
 * whether the reaction is reversible.  This attribute is optional in SBML
 * Level&nbsp;2, with a default of @c true; it is mandatory in SBML
 * Level&nbsp;3 (with no default value).  To say that a reaction is @em
 * reversible is to say it can proceed in either the forward or the reverse
 * direction.  Although the reversibility of a reaction can sometimes be
 * deduced by inspecting its rate expression, this is not always the case,
 * especially for complicated expressions.  Moreover, the need in SBML to
 * allow rate expressions (i.e., KineticLaw) to be optional leads to the
 * need for a separate flag indicating reversibility.  Note that labeling a
 * reaction as irreversible is an assertion that the reaction always
 * proceeds in the given forward direction.  (Why else would it be flagged
 * as irreversible?)  This implies the rate expression in the KineticLaw
 * always has a non-negative value during simulations.  Software tools
 * could provide a means of optionally testing that this condition holds.
 * The presence of reversibility information in two places (i.e., the rate
 * expression and the "reversible" attribute on Reaction) leaves open the
 * possibility that a model could contain contradictory information, but
 * the creation of such a model would be an error on the part of the
 * software generating it.
 *
 * The Reaction object class has another boolean attribute called "fast".
 * This attribute is optional in SBML Level&nbsp;2, with a default of @c
 * false; it is mandatory in SBML Level&nbsp;3 (with no default value).  It
 * is used to indicate that a reaction occurs on a vastly faster time scale
 * than others in a system.  Readers are directed to the SBML Level&nbsp;2
 * Version&nbsp;4 specification, which provides more detail about the
 * conditions under which a reaction can be considered to be fast in this
 * sense.  The attribute's default value is @c false.  SBML Level&nbsp;1
 * and Level&nbsp;2 Version&nbsp;1 incorrectly claimed that software tools
 * could ignore this attribute if they did not implement support for the
 * corresponding concept; however, further research in SBML has revealed
 * that this is not true, and "fast" <em>cannot be ignored</em> if it is
 * set to @c true.  SBML Level&nbsp;2 Versions&nbsp;2&ndash;4 therefore
 * stipulate that if a model has any reactions with "fast" set to @c true,
 * a software tool must be able to respect the attribute or else indicate
 * to the user that it does not have the capacity to do so.  Analysis
 * software cannot ignore the value of the "fast" attribute because doing
 * so may lead to different results as compared to a software system that
 * <em>does</em> make use of "fast".
 *
 * In SBML Level&nbsp;3 Version&nbsp;1, the Reaction object has an
 * additional optional attribute named "compartment", whose value must be
 * the identifier of a compartment defined in the enclosing Model object.
 * The "compartment" attribute can be used to indicate the compartment in
 * which the reaction is assumed to take place.  If the attribute is
 * present, its value must be the identifier of a Compartment object
 * defined in the enclosing Model object.  Similar to the "reversible"
 * attribute, the value of the "compartment" attribute has no direct impact
 * on the construction of mathematical equations for the SBML model.  When
 * a kinetic law is given for a reaction, the compartment location may
 * already be implicit in the kinetic law (although this cannot always be
 * guaranteed).  Nevertheless, software tools may find the "compartment"
 * attribute value useful for such purposes as analyzing the structure of
 * the model, guiding the modeler in constructing correct rate formulas,
 * and visualization purposes.
 *
 * Readers are urged to read the SBML specification for more details about
 * the proper use of Reaction.
 * 
 * 
 * @class ListOfReactions.
 * @brief LibSBML implementation of SBML's %ListOfReactions construct.
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


#ifndef Reaction_h
#define Reaction_h


#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>
#include <sbml/SpeciesReference.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class KineticLaw;
class SBMLVisitor;


class LIBSBML_EXTERN Reaction : public SBase
{
public:

  /**
   * Creates a new Reaction using the given SBML @p level and @p version
   * values.
   *
   * @param level an unsigned int, the SBML Level to assign to this Reaction
   *
   * @param version an unsigned int, the SBML Version to assign to this
   * Reaction
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   * 
   * @note Upon the addition of a Reaction object to an SBMLDocument (e.g.,
   * using Model::addReaction(@if java Reaction r@endif)), the SBML Level, SBML Version and
   * XML namespace of the document @em override the values used when
   * creating the Reaction object via this constructor.  This is necessary
   * to ensure that an SBML document is a consistent structure.
   * Nevertheless, the ability to supply the values at the time of creation
   * of a Reaction is an important aid to producing valid SBML.  Knowledge
   * of the intented SBML Level and Version determine whether it is valid
   * to assign a particular value to an attribute, or whether it is valid
   * to add an object to an existing SBMLDocument.
   */
  Reaction (unsigned int level, unsigned int version);


  /**
   * Creates a new Reaction using the given SBMLNamespaces object
   * @p  sbmlns.
   *
   * The SBMLNamespaces object encapsulates SBML Level/Version/namespaces
   * information.  It is used to communicate the SBML Level, Version, and
   * (in Level&nbsp;3) packages used in addition to SBML Level&nbsp;3 Core.
   * A common approach to using this class constructor is to create an
   * SBMLNamespaces object somewhere in a program, once, then pass it to
   * object constructors such as this one when needed.
   *
   * @param sbmlns an SBMLNamespaces object.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the given @p level and @p version combination, or this kind
   * of SBML object, are either invalid or mismatched with respect to the
   * parent SBMLDocument object.
   *
   * @note Upon the addition of a Reaction object to an SBMLDocument (e.g.,
   * using Model::addReaction(@if java Reaction r@endif)), the SBML XML namespace of the document
   * @em overrides the value used when creating the Reaction object via
   * this constructor.  This is necessary to ensure that an SBML document
   * is a consistent structure.  Nevertheless, the ability to supply the
   * values at the time of creation of a Reaction is an important aid to
   * producing valid SBML.  Knowledge of the intented SBML Level and
   * Version determine whether it is valid to assign a particular value to
   * an attribute, or whether it is valid to add an object to an existing
   * SBMLDocument.
   */
  Reaction (SBMLNamespaces* sbmlns);


  /**
   * Destroys this Reaction.
   */
  virtual ~Reaction ();


  /**
   * Copy constructor; creates a copy of this Reaction.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  Reaction (const Reaction& orig);


  /**
   * Assignment operator for Reaction.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  Reaction& operator=(const Reaction& rhs);


  /**
   * Accepts the given SBMLVisitor for this instance of Reaction.
   *
   * @param v the SBMLVisitor instance to be used.
   *
   * @return the result of calling <code>v.visit()</code>.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this Reaction.
   * 
   * @return a (deep) copy of this Reaction.
   */
  virtual Reaction* clone () const;


   /**
   * Returns the first child element found that has the given id in the model-wide SId namespace, or NULL if no such object is found.
   *
   * @param id string representing the id of objects to find.
   *
   * @return pointer to the first element found with the given id.
   */
  virtual SBase* getElementBySId(std::string id);
  
  
  /**
   * Returns the first child element it can find with the given metaid, or NULL if no such object is found.
   *
   * @param metaid string representing the metaid of objects to find
   *
   * @return pointer to the first element found with the given metaid.
   */
  virtual SBase* getElementByMetaId(std::string metaid);
  
  
  /**
   * Returns a List of all child SBase* objects, including those nested to an arbitrary depth
   *
   * @return a List* of pointers to all children objects.
   */
  virtual List* getAllElements();
  
  
  /**
   * Renames all the SIdRef attributes on this element, including any found in MathML
   */
  virtual void renameSIdRefs(std::string oldid, std::string newid);


/**
   * Initializes the fields of this Reaction object to "typical" default
   * values.
   *
   * The SBML Reaction component has slightly different aspects and
   * default attribute values in different SBML Levels and Versions.
   * This method sets the values to certain common defaults, based
   * mostly on what they are in SBML Level&nbsp;2.  Specifically:
   * <ul>
   * <li> Sets the "reversible" attribute to @c true
   * <li> Sets the "fast" attribute to @c false
   * <li> Marks the "fast" attribute as @em not having been set
   * </ul>
   * 
   * @warning The "fast" attribute must be used with care.  SBML
   * definitions before SBML Level&nbsp;2 Version&nbsp;2 incorrectly
   * indicated that software tools could ignore this attribute if they did
   * not implement support for the corresponding concept; however, further
   * research in SBML has revealed that this is not true, and "fast"
   * <em>cannot be ignored</em> if it is set to @c true.  Beginning with
   * SBML Level&nbsp;2 Versions&nbsp;2, the SBML specifications therefore
   * stipulate that if a model has any reactions with "fast" set to @c
   * true, a software tool must be able to respect the attribute or else
   * indicate to the user that it does not have the capacity to do so.
   * Readers are directed to the SBML specifications, which provides more
   * detail about the conditions under which a reaction can be considered
   * to be fast in this sense.
   */
  void initDefaults ();


  /**
   * Returns the value of the "id" attribute of this Reaction.
   * 
   * @return the id of this Reaction.
   */
  virtual const std::string& getId () const;


  /**
   * Returns the value of the "name" attribute of this Reaction.
   * 
   * @return the name of this Reaction.
   */
  virtual const std::string& getName () const;


  /**
   * Returns the KineticLaw object contained in this Reaction.
   * 
   * @return the KineticLaw instance.
   */
  const KineticLaw* getKineticLaw () const;


  /**
   * Returns the KineticLaw object contained in this Reaction.
   * 
   * @return the KineticLaw instance.
   */
  KineticLaw* getKineticLaw ();


  /**
   * Returns the value of the "reversible" attribute on the Reaction as a
   * boolean value.
   * 
   * @return the reversibility status of this Reaction.
   */
  bool getReversible () const;


  /**
   * Returns the value of the "fast" attribute of this Reaction.
   * 
   * @return the "fast" status of this Reaction.
   *
   * @warning SBML definitions before SBML Level&nbsp;2 Version&nbsp;2 incorrectly
   * indicated that software tools could ignore this attribute if they did
   * not implement support for the corresponding concept; however, further
   * research in SBML has revealed that this is not true, and "fast"
   * <em>cannot be ignored</em> if it is set to @c true.  SBML Level&nbsp;2
   * Versions 2, 3 and 4 therefore stipulate that if a model has any reactions
   * with "fast" set to @c true, a software tool must be able to respect
   * the attribute or else indicate to the user that it does not have the
   * capacity to do so.  Readers are directed to the SBML Level&nbsp;2 Version
   * 4 specification, which provides more detail about the conditions under
   * which a reaction can be considered to be fast in this sense.
   */
  bool getFast () const;


  /**
   * (SBML Level&nbsp;3 only) Returns the value of the "compartment"
   * attribute on the Reaction.
   * 
   * @return the compartment of this Reaction.
   *
   * @note The "compartment" attribute is available in SBML Level&nbsp;3
   * Version&nbsp;1 Core, but is not present on Reaction in lower Levels of
   * SBML.
   */
  const std::string& getCompartment () const;


  /**
   * Predicate returning @c true if this
   * Reaction's "id" attribute is set.
   *
   * @return @c true if the "id" attribute of this Reaction is
   * set, @c false otherwise.
   */
  virtual bool isSetId () const;


  /**
   * Predicate returning @c true if this
   * Reaction's "name" attribute is set.
   *
   * @return @c true if the "name" attribute of this Reaction is
   * set, @c false otherwise.
   */
  virtual bool isSetName () const;


  /**
   * Predicate returning @c true if this
   * Reaction contains a kinetic law object.
   *
   * @return @c true if a KineticLaw is present in this Reaction,, @c false
   * otherwise.
   */
  bool isSetKineticLaw () const;


  /**
   * Predicate returning @c true if the value of
   * the "fast" attribute on this Reaction.
   *
   * @return @c true if the "fast" attribute is true, @c false otherwise.
   *
   * @warning SBML definitions before SBML Level&nbsp;2 Version&nbsp;2 incorrectly
   * indicated that software tools could ignore this attribute if they did
   * not implement support for the corresponding concept; however, further
   * research in SBML has revealed that this is not true, and "fast"
   * <em>cannot be ignored</em> if it is set to @c true.  SBML Level&nbsp;2
   * Versions 2, 3 and 4 therefore stipulate that if a model has any reactions
   * with "fast" set to @c true, a software tool must be able to respect
   * the attribute or else indicate to the user that it does not have the
   * capacity to do so.  Readers are directed to the SBML Level&nbsp;2 Version
   * 4 specification, which provides more detail about the conditions under
   * which a reaction can be considered to be fast in this sense.  Note
   * also that in SBML Level&nbsp;1, "fast" is defined as optional with a
   * default of @c false, which means it is effectively always set.
   */
  bool isSetFast () const;


  /**
   * Predicate returning @c true if this
   * Reaction's "compartment" attribute is set.
   *
   * @return @c true if the "compartment" attribute of this Reaction is
   * set, @c false otherwise.
   *
   * @note The "compartment" attribute is available in SBML
   * Level&nbsp;3 Version&nbsp;1 Core, but is not present on Reaction in
   * lower Levels of SBML.
   */
  bool isSetCompartment () const;


  /**
   * Predicate returning @c true if this
   * Reaction's "reversible" attribute is set.
   *
   * @return @c true if the "reversible" attribute of this Reaction is
   * set, @c false otherwise.
   */
  bool isSetReversible () const;


  /**
   * Sets the value of the "id" attribute of this Reaction.
   *
   * The string @p sid is copied.  Note that SBML has strict requirements
   * for the syntax of identifiers.  @htmlinclude id-syntax.html
   *
   * @param sid the string to use as the identifier of this Reaction
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setId (const std::string& sid);


  /**
   * Sets the value of the "name" attribute of this Reaction.
   *
   * The string in @p name is copied.
   *
   * @param name the new name for the Reaction
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  virtual int setName (const std::string& name);


  /**
   * Sets the "kineticLaw" subelement of this Reaction to a copy of the
   * given KineticLaw object.
   *
   * @param kl the KineticLaw object to use.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_LEVEL_MISMATCH LIBSBML_LEVEL_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_VERSION_MISMATCH LIBSBML_VERSION_MISMATCH @endlink
  */
  int setKineticLaw (const KineticLaw* kl);


  /**
   * Sets the value of the "reversible" attribute of this Reaction.
   *
   * @param value the value of the "reversible" attribute.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int setReversible (bool value);


  /**
   * Sets the value of the "fast" attribute of this Reaction.
   *
   * @param value the value of the "fast" attribute.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * 
   * @warning SBML definitions before SBML Level&nbsp;2 Version&nbsp;2 incorrectly
   * indicated that software tools could ignore this attribute if they did
   * not implement support for the corresponding concept; however, further
   * research in SBML has revealed that this is not true, and "fast"
   * <em>cannot be ignored</em> if it is set to @c true.  SBML Level&nbsp;2
   * Versions 2, 3 and 4 therefore stipulate that if a model has any reactions
   * with "fast" set to @c true, a software tool must be able to respect
   * the attribute or else indicate to the user that it does not have the
   * capacity to do so.  Readers are directed to the SBML Level&nbsp;2 Version
   * 4 specification, which provides more detail about the conditions under
   * which a reaction can be considered to be fast in this sense.
   */
  int setFast (bool value);


  /**
   * Sets the value of the "compartment" attribute of this Reaction.
   *
   * The string @p sid is copied.  
   *
   * @param sid the string to use as the compartment of this Reaction
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   *
   * @note The "compartment" attribute is available in SBML
   * Level&nbsp;3 Version&nbsp;1 Core, but is not present on Reaction in
   * lower Levels of SBML.
   */
  int setCompartment (const std::string& sid);


  /**
   * Unsets the value of the "name" attribute of this Reaction.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int unsetName ();


  /**
   * Unsets the "kineticLaw" subelement of this Reaction.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int unsetKineticLaw ();


  /**
   * Unsets the value of the "fast" attribute of this Reaction.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @warning In SBML Level&nbsp;1, "fast" is optional with a default of @c
   * false, which means it is effectively always set (and reset to @c false
   * if this method is called).  Further, SBML definitions before SBML
   * Level&nbsp;2 Version&nbsp;2 incorrectly indicated that software tools could
   * ignore this attribute if they did not implement support for the
   * corresponding concept; however, further research in SBML has revealed
   * that this is not true, and "fast" <em>cannot be ignored</em> if it is
   * set to @c true.  SBML Level&nbsp;2 Versions 2, 3 and 4 therefore stipulate
   * that if a model has any reactions with "fast" set to @c true, a
   * software tool must be able to respect the attribute or else indicate
   * to the user that it does not have the capacity to do so.  Readers are
   * directed to the SBML Level&nbsp;2 Version&nbsp;4 specification, which provides
   * more detail about the conditions under which a reaction can be
   * considered to be fast in this sense.
   */
  int unsetFast ();


  /**
   * Unsets the value of the "compartment" attribute of this Reaction.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note The "compartment" attribute is available in SBML
   * Level&nbsp;3 Version&nbsp;1 Core, but is not present on Reaction in
   * lower Levels of SBML.
   */
  int unsetCompartment ();


  /**
   * Adds a given SpeciesReference object as a reactant in this Reaction.
   *
   * The SpeciesReference instance in @p sr is copied.
   *
   * @param sr a SpeciesReference object referring to a Species in the
   * enclosing Model
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_LEVEL_MISMATCH LIBSBML_LEVEL_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_VERSION_MISMATCH LIBSBML_VERSION_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_DUPLICATE_OBJECT_ID LIBSBML_DUPLICATE_OBJECT_ID @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note This method should be used with some caution.  The fact that
   * this method @em copies the object passed to it means that the caller
   * will be left holding a physically different object instance than the
   * one contained in this Reaction.  Changes made to the original object
   * instance (such as resetting attribute values) will <em>not affect the
   * instance in the Reaction</em>.  In addition, the caller should make
   * sure to free the original object if it is no longer being used, or
   * else a memory leak will result.  Please see Reaction::createReactant()
   * for a method that does not lead to these issues.
   *
   * @see createReactant()
   */
  int addReactant (const SpeciesReference* sr);


  /**
   * Adds a given SpeciesReference object as a product in this Reaction.
   *
   * The SpeciesReference instance in @p sr is copied.
   *
   * @param sr a SpeciesReference object referring to a Species in the
   * enclosing Model
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_LEVEL_MISMATCH LIBSBML_LEVEL_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_VERSION_MISMATCH LIBSBML_VERSION_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note This method should be used with some caution.  The fact that
   * this method @em copies the object passed to it means that the caller
   * will be left holding a physically different object instance than the
   * one contained in this Reaction.  Changes made to the original object
   * instance (such as resetting attribute values) will <em>not affect the
   * instance in the Reaction</em>.  In addition, the caller should make
   * sure to free the original object if it is no longer being used, or
   * else a memory leak will result.  Please see Reaction::createProduct()
   * for a method that does not lead to these issues.
   *
   * @see createProduct()
   */
  int addProduct (const SpeciesReference* sr);


  /**
   * Adds a given ModifierSpeciesReference object as a product in this
   * Reaction.
   *
   * The ModifierSpeciesReference instance in @p msr is copied.
   *
   * @param msr a ModifierSpeciesReference object referring to a Species in
   * the enclosing Model
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_LEVEL_MISMATCH LIBSBML_LEVEL_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_DUPLICATE_OBJECT_ID LIBSBML_DUPLICATE_OBJECT_ID @endlink
   * @li @link OperationReturnValues_t#LIBSBML_VERSION_MISMATCH LIBSBML_VERSION_MISMATCH @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note This method should be used with some caution.  The fact that
   * this method @em copies the object passed to it means that the caller
   * will be left holding a physically different object instance than the
   * one contained in this Reaction.  Changes made to the original object
   * instance (such as resetting attribute values) will <em>not affect the
   * instance in the Reaction</em>.  In addition, the caller should make
   * sure to free the original object if it is no longer being used, or
   * else a memory leak will result.  Please see Reaction::createModifier()
   * for a method that does not lead to these issues.
   *
   * @see createModifier()
   */
  int addModifier (const ModifierSpeciesReference* msr);


  /**
   * Creates a new SpeciesReference, adds it to this Reaction's list of
   * reactants, and returns it.
   *
   * @return a new SpeciesReference object.
   */
  SpeciesReference* createReactant ();


  /**
   * Creates a new SpeciesReference, adds it to this Reaction's list of
   * products, and returns it.
   *
   * @return a new SpeciesReference object.
   */
  SpeciesReference* createProduct ();


  /**
   * Creates a new ModifierSpeciesReference, adds it to this Reaction's
   * list of modifiers and returns it.
   *
   * @return a new ModifierSpeciesReference object.
   */
  ModifierSpeciesReference* createModifier ();


  /**
   * Creates a new KineticLaw object, installs it as this Reaction's
   * "kineticLaw" subelement, and returns it.
   *
   * If this Reaction had a previous KineticLaw, it will be destroyed.
   *
   * @return the new KineticLaw object
   */
  KineticLaw* createKineticLaw ();


  /**
   * Returns the list of reactants in this Reaction object.
   * 
   * @return the ListOfSpeciesReferences containing the references to the
   * species acting as reactants in this reaction
   */
  const ListOfSpeciesReferences* getListOfReactants () const;


  /**
   * Returns the list of reactants in this Reaction object.
   * 
   * @return the ListOfSpeciesReferences containing the references to the
   * species acting as reactants in this reaction
   */
  ListOfSpeciesReferences* getListOfReactants ();


  /**
   * Returns the list of products in this Reaction object.
   * 
   * @return the ListOfSpeciesReferences containing the references to the
   * species acting as products in this reaction
   */
  const ListOfSpeciesReferences* getListOfProducts () const;


  /**
   * Returns the list of products in this Reaction object.
   * 
   * @return the ListOfSpeciesReferences containing the references to the
   * species acting as products in this reaction
   */
  ListOfSpeciesReferences* getListOfProducts ();


  /**
   * Returns the list of modifiers in this Reaction object.
   * 
   * @return the ListOfSpeciesReferences containing the references to the
   * species acting as modifiers in this reaction
   */
  const ListOfSpeciesReferences* getListOfModifiers () const;


  /**
   * Returns the list of modifiers in this Reaction object.
   * 
   * @return the ListOfSpeciesReferences containing the references to the
   * species acting as modifiers in this reaction
   */
  ListOfSpeciesReferences* getListOfModifiers ();


  /**
   * Returns the nth reactant species (as a SpeciesReference object) in 
   * the list of reactants in this Reaction.
   *
   * Callers should first call getNumReactants() to find out how many
   * reactants there are, to avoid using an invalid index number.
   *
   * @param n the index of the reactant sought.
   * 
   * @return the nth reactant (as a SpeciesReference object) of this
   * Reaction.
   */
  const SpeciesReference* getReactant (unsigned int n) const;


  /**
   * Returns the nth reactant species (as a SpeciesReference object) 
   * in the list of reactants in this Reaction.
   *
   * Callers should first call getNumReactants() to find out how many
   * reactants there are, to avoid using an invalid index number.
   *
   * @param n the index of the reactant sought.
   * 
   * @return the nth reactant (as a SpeciesReference object) of this
   * Reaction.
   */
  SpeciesReference* getReactant (unsigned int n);


  /**
   * Returns the reactant species (as a SpeciesReference object) having 
   * a specific identifier in this Reaction.
   *
   * @param species the identifier of the reactant Species ("species" 
   * attribute of the reactant SpeciesReference object)
   *
   * @return a SpeciesReference object, or @c NULL if no species with the
   * given identifier @p species appears as a reactant in this Reaction.
   */
  const SpeciesReference* getReactant (const std::string& species) const;


  /**
   * Returns the reactant species (as a SpeciesReference object) having 
   * a specific identifier in this Reaction.
   *
   * @param species the identifier of the reactant Species ("species" 
   * attribute of the reactant SpeciesReference object)
   *
   * @return a SpeciesReference object, or @c NULL if no species with the
   * given identifier @p species appears as a reactant in this Reaction.
   */
  SpeciesReference* getReactant (const std::string& species);


  /**
   * Returns the nth product species (as a SpeciesReference object) in 
   * the list of products in this Reaction.
   *
   * Callers should first call getNumProducts() to find out how many
   * products there are, to avoid using an invalid index number.
   *
   * @param n the index of the product sought.
   * 
   * @return the nth product (as a SpeciesReference object) of this
   * Reaction.
   */
  const SpeciesReference* getProduct (unsigned int n) const;


  /**
   * Returns the nth product species (as a SpeciesReference object) 
   * in the list of products in this Reaction.
   *
   * Callers should first call getNumProducts() to find out how many
   * products there are, to avoid using an invalid index number.
   *
   * @param n the index of the product sought.
   * 
   * @return the nth product (as a SpeciesReference object) of this
   * Reaction.
   */
  SpeciesReference* getProduct (unsigned int n);


  /**
   * Returns the product species (as a SpeciesReference object) having 
   * a specific identifier in this Reaction.
   *
   * @param species the identifier of the product Species ("species"
   * attribute of the product SpeciesReference object)
   *
   * @return a SpeciesReference object, or @c NULL if no species with the
   * given identifier @p species appears as a product in this Reaction.
   */
  const SpeciesReference* getProduct (const std::string& species) const;


  /**
   * Returns the product species (as a SpeciesReference object) having 
   * a specific identifier in this Reaction.
   *
   * @param species the identifier of the product Species ("species"
   * attribute of the product SpeciesReference object)
   *
   * @return a SpeciesReference object, or @c NULL if no species with the
   * given identifier @p species appears as a product in this Reaction.
   */
  SpeciesReference* getProduct (const std::string& species);


  /**
   * Returns the nth modifier species (as a ModifierSpeciesReference object) 
   * in the list of modifiers of this Reaction.
   *
   * Callers should first call getNumModifiers() to find out how many
   * modifiers there are, to avoid using an invalid index number.
   *
   * @param n the index of the modifier species sought
   * 
   * @return the nth modifier (as a ModifierSpeciesReference object) of
   * this Reaction.
   */
  const ModifierSpeciesReference* getModifier (unsigned int n) const;


  /**
   * Returns the nth modifier species (as a ModifierSpeciesReference object) 
   * in the list of modifiers of this Reaction.
   *
   * Callers should first call getNumModifiers() to find out how many
   * modifiers there are, to avoid using an invalid index number.
   *
   * @param n the index of the modifier species sought
   * 
   * @return the nth modifier (as a ModifierSpeciesReference object) of
   * this Reaction.
   */
  ModifierSpeciesReference* getModifier (unsigned int n);


  /**
   * Returns the modifier species (as a ModifierSpeciesReference object) 
   * having a specific identifier in this Reaction.
   *
   * @param species the identifier of the modifier Species ("species" 
   * attribute of the ModifierSpeciesReference object)
   *
   * @return a ModifierSpeciesReference object, or @c NULL if no species with
   * the given identifier @p species appears as a modifier in this
   * Reaction.
   */
  const ModifierSpeciesReference*
  getModifier (const std::string& species) const;


  /**
   * Returns the modifier species (as a ModifierSpeciesReference object) 
   * having a specific identifier in this Reaction.
   *
   * @param species the identifier of the modifier Species ("species" 
   * attribute of the ModifierSpeciesReference object)
   *
   * @return a ModifierSpeciesReference object, or @c NULL if no species with
   * the given identifier @p species appears as a modifier in this
   * Reaction.
   */
  ModifierSpeciesReference* getModifier (const std::string& species);


  /**
   * Returns the number of reactant species in this Reaction.
   * 
   * @return the number of reactants in this Reaction.
   */
  unsigned int getNumReactants () const;


  /**
   * Returns the number of product species in this Reaction.
   * 
   * @return the number of products in this Reaction.
   */
  unsigned int getNumProducts () const;


  /**
   * Returns the number of modifier species in this Reaction.
   * 
   * @return the number of modifiers in this Reaction.
   */
  unsigned int getNumModifiers () const;


  /**
   * Removes the nth reactant species (SpeciesReference object) in the list of 
   * reactants in this Reaction and returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   * The caller should first call getNumReactants() to find out how many
   * reactants there are, to avoid using an invalid index number.
   *
   * @param n the index of the reactant SpeciesReference object to remove
   *
   * @return the removed reactant SpeciesReference object, or @c NULL if the 
   * given index is out of range.
   */
  SpeciesReference* removeReactant (unsigned int n);


  /**
   * Removes the reactant species (SpeciesReference object) having the given  
   * "species" attribute in this Reaction and returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   *
   * @param species the "species" attribute of the reactant SpeciesReference 
   * object
   *
   * @return the removed reactant SpeciesReference object, or @c NULL if no 
   * reactant SpeciesReference object with the given "species" attribute 
   * @p species exists in this Reaction.
   */
  SpeciesReference* removeReactant (const std::string& species);


  /**
   * Removes the nth product species (SpeciesReference object) in the list of 
   * products in this Reaction and returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   * The caller should first call getNumProducts() to find out how many
   * products there are, to avoid using an invalid index number.
   *
   * @param n the index of the product SpeciesReference object to remove
   *
   * @return the removed product SpeciesReference object, or @c NULL if the 
   * given index is out of range.
   */
  SpeciesReference* removeProduct (unsigned int n);


  /**
   * Removes the product species (SpeciesReference object) having the given  
   * "species" attribute in this Reaction and returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   *
   * @param species the "species" attribute of the product SpeciesReference 
   * object
   *
   * @return the removed product SpeciesReference object, or @c NULL if no 
   * product SpeciesReference object with the given "species" attribute 
   * @p species exists in this Reaction.
   */
  SpeciesReference* removeProduct (const std::string& species);


  /**
   * Removes the nth modifier species (ModifierSpeciesReference object) in 
   * the list of  modifiers in this Reaction and returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   * The caller should first call getNumModifiers() to find out how many
   * modifiers there are, to avoid using an invalid index number.
   *
   * @param n the index of the ModifierSpeciesReference object to remove
   *
   * @return the removed ModifierSpeciesReference object, or @c NULL if the 
   * given index is out of range.
   */
  ModifierSpeciesReference* removeModifier (unsigned int n);


  /**
   * Removes the modifier species (ModifierSpeciesReference object) having 
   * the given "species" attribute in this Reaction and returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   *
   * @param species the "species" attribute of the ModifierSpeciesReference 
   * object
   *
   * @return the removed ModifierSpeciesReference object, or @c NULL if no 
   * ModifierSpeciesReference object with the given "species" attribute @p 
   * species exists in this Reaction.
   */
  ModifierSpeciesReference* removeModifier (const std::string& species);


  /** @cond doxygen-libsbml-internal */

  /**
   * Sets the parent SBMLDocument of this SBML object.
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets this SBML object to child SBML objects (if any).
   * (Creates a child-parent relationship by the parent)
   *
   * Subclasses must override this function if they define
   * one ore more child elements.
   * Basically, this function needs to be called in
   * constructor, copy constructor and assignment operator.
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToChild ();

  /**
   * Enables/Disables the given package with this element and child
   * elements (if any).
   * (This is an internal implementation for enablePackage function)
   *
   * @note Subclasses of the SBML Core package in which one or more child
   * elements are defined must override this function.
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);

  /** @endcond */

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
  virtual int getTypeCode () const;


  /**
   * Returns the XML element name of this object, which for Reaction, is
   * always @c "reaction".
   * 
   * @return the name of this element, i.e., @c "reaction".
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
   * Predicate returning @c true if all the required attributes for this
   * Reaction object have been set.
   *
   * @note The required attributes for a Reaction object are:
   * @li "id" (or "name" in SBML Level&nbsp;1)
   * @li "fast" (in Level&nbsp;3 only, where it is defined as a required attribute)
   * @li "reversible" (in Level&nbsp;3 only, where it is defined as a required attribute)
   *
   * @return a boolean value indicating whether all the required
   * attributes for this object have been defined.
   */
  virtual bool hasRequiredAttributes() const ;


protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or @c NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);


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

  bool isExplicitlySetReversible() const { return mExplicitlySetReversible; };
  bool isExplicitlySetFast() const { return mExplicitlySetFast; };

  std::string mId;
  std::string mName;
 
  ListOfSpeciesReferences  mReactants;
  ListOfSpeciesReferences  mProducts;
  ListOfSpeciesReferences  mModifiers;

  KineticLaw* mKineticLaw;
  bool        mReversible;
  bool        mFast;

  bool mIsSetFast;
  std::string mCompartment;
  bool mIsSetReversible;

  bool mExplicitlySetReversible;
  bool mExplicitlySetFast;

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



class LIBSBML_EXTERN ListOfReactions : public ListOf
{
public:

  /**
   * Creates a new ListOfReactions object.
   *
   * The object is constructed such that it is valid for the given SBML
   * Level and Version combination.
   *
   * @param level the SBML Level
   * 
   * @param version the Version within the SBML Level
   */
  ListOfReactions (unsigned int level, unsigned int version);
          

  /**
   * Creates a new ListOfReactions object.
   *
   * The object is constructed such that it is valid for the SBML Level and
   * Version combination determined by the SBMLNamespaces object in @p
   * sbmlns.
   *
   * @param sbmlns an SBMLNamespaces object that is used to determine the
   * characteristics of the ListOfReactions object to be created.
   */
  ListOfReactions (SBMLNamespaces* sbmlns);


  /**
   * Creates and returns a deep copy of this ListOfReactions instance.
   *
   * @return a (deep) copy of this ListOfReactions.
   */
  virtual ListOfReactions* clone () const;


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
   * @return the SBML type code for this object, or @link
   * SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const { return SBML_LIST_OF; };


  /**
   * Returns the libSBML type code for the objects contained in this ListOf
   * (i.e., Reaction objects, if the list is non-empty).
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
   * Returns the XML element name of this object
   *
   * For ListOfReactions, the XML element name is @c "listOfReactions".
   * 
   * @return the name of this element, i.e., @c "listOfReactions".
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a Reaction from the ListOfReactions.
   *
   * @param n the index number of the Reaction to get.
   * 
   * @return the nth Reaction in this ListOfReactions.
   *
   * @see size()
   */
  virtual Reaction * get(unsigned int n); 


  /**
   * Get a Reaction from the ListOfReactions.
   *
   * @param n the index number of the Reaction to get.
   * 
   * @return the nth Reaction in this ListOfReactions.
   *
   * @see size()
   */
  virtual const Reaction * get(unsigned int n) const; 

  /**
   * Get a Reaction from the ListOfReactions
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Reaction to get.
   * 
   * @return Reaction in this ListOfReactions
   * with the given id or @c NULL if no such
   * Reaction exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual Reaction* get (const std::string& sid);


  /**
   * Get a Reaction from the ListOfReactions
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the Reaction to get.
   * 
   * @return Reaction in this ListOfReactions
   * with the given id or @c NULL if no such
   * Reaction exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const Reaction* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfReactions items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual Reaction* remove (unsigned int n);


  /**
   * Removes item in this ListOfReactions items with the given identifier.
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
  virtual Reaction* remove (const std::string& sid);


  /** @cond doxygen-libsbml-internal */

  /**
   * @return the ordinal position of the element with respect to its
   * siblings or -1 (default) to indicate the position is not significant.
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

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/* ----------------------------------------------------------------------------
 * See the .cpp file for the documentation of the following functions.
 * --------------------------------------------------------------------------*/


LIBSBML_EXTERN
Reaction_t *
Reaction_create (unsigned int level, unsigned int version);


LIBSBML_EXTERN
Reaction_t *
Reaction_createWithNS (SBMLNamespaces_t *sbmlns);


LIBSBML_EXTERN
void
Reaction_free (Reaction_t *r);


LIBSBML_EXTERN
Reaction_t *
Reaction_clone (const Reaction_t *r);


LIBSBML_EXTERN
void
Reaction_initDefaults (Reaction_t *r);


LIBSBML_EXTERN
const XMLNamespaces_t *
Reaction_getNamespaces(Reaction_t *c);


LIBSBML_EXTERN
const char *
Reaction_getId (const Reaction_t *r);


LIBSBML_EXTERN
const char *
Reaction_getName (const Reaction_t *r);


LIBSBML_EXTERN
KineticLaw_t *
Reaction_getKineticLaw (Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_getReversible (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_getFast (const Reaction_t *r);


LIBSBML_EXTERN
const char *
Reaction_getCompartment (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_isSetId (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_isSetName (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_isSetKineticLaw (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_isSetCompartment (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_isSetReversible (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_isSetFast (const Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_setId (Reaction_t *r, const char *sid);


LIBSBML_EXTERN
int
Reaction_setName (Reaction_t *r, const char *name);


LIBSBML_EXTERN
int
Reaction_setKineticLaw (Reaction_t *r, const KineticLaw_t *kl);


LIBSBML_EXTERN
int
Reaction_setReversible (Reaction_t *r, int value);


LIBSBML_EXTERN
int
Reaction_setFast (Reaction_t *r, int value);


LIBSBML_EXTERN
int
Reaction_setCompartment (Reaction_t *r, const char *compartment);


LIBSBML_EXTERN
int
Reaction_unsetName (Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_unsetKineticLaw (Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_unsetFast (Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_unsetCompartment (Reaction_t *r);


LIBSBML_EXTERN
int
Reaction_hasRequiredAttributes (Reaction_t *p);


LIBSBML_EXTERN
int
Reaction_addReactant (Reaction_t *r, const SpeciesReference_t *sr);


LIBSBML_EXTERN
int
Reaction_addProduct (Reaction_t *r, const SpeciesReference_t *sr);


LIBSBML_EXTERN
int
Reaction_addModifier (Reaction_t *r, const SpeciesReference_t *msr);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_createReactant (Reaction_t *r);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_createProduct (Reaction_t *r);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_createModifier (Reaction_t *r);


LIBSBML_EXTERN
KineticLaw_t *
Reaction_createKineticLaw (Reaction_t *r);


LIBSBML_EXTERN
ListOf_t *
Reaction_getListOfReactants (Reaction_t *r);


LIBSBML_EXTERN
ListOf_t *
Reaction_getListOfProducts (Reaction_t *r);


LIBSBML_EXTERN
ListOf_t *
Reaction_getListOfModifiers (Reaction_t *r);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_getReactant (Reaction_t *r, unsigned int n);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_getReactantBySpecies (Reaction_t *r, const char *species);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_getProduct (Reaction_t *r, unsigned int n);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_getProductBySpecies (Reaction_t *r, const char *species);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_getModifier (Reaction_t *r, unsigned int n);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_getModifierBySpecies (Reaction_t *r, const char *species);


LIBSBML_EXTERN
unsigned int
Reaction_getNumReactants (const Reaction_t *r);


LIBSBML_EXTERN
unsigned int
Reaction_getNumProducts (const Reaction_t *r);


LIBSBML_EXTERN
unsigned int
Reaction_getNumModifiers (const Reaction_t *r);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_removeReactant (Reaction_t *r, unsigned int n);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_removeReactantBySpecies (Reaction_t *r, const char *species);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_removeProduct (Reaction_t *r, unsigned int n);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_removeProductBySpecies (Reaction_t *r, const char *species);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_removeModifier (Reaction_t *r, unsigned int n);


LIBSBML_EXTERN
SpeciesReference_t *
Reaction_removeModifierBySpecies (Reaction_t *r, const char *species);


LIBSBML_EXTERN
Reaction_t *
ListOfReactions_getById (ListOf_t *lo, const char *sid);


LIBSBML_EXTERN
Reaction_t *
ListOfReactions_removeById (ListOf_t *lo, const char *sid);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* Reaction_h */

