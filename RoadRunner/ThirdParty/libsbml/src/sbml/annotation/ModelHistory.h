/**
 * @file    ModelHistory.h
 * @brief   ModelHistory I/O
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
 * @class ModelHistory
 * @brief Representation of MIRIAM-compliant model history data.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * The SBML specification beginning with Level&nbsp;2 Version&nbsp;2
 * defines a standard approach to recording optional model history and
 * model creator information in a form that complies with MIRIAM ("Minimum
 * Information Requested in the Annotation of biochemical Models",
 * <i>Nature Biotechnology</i>, vol. 23, no. 12, Dec. 2005).  LibSBML
 * provides the ModelHistory class as a convenient high-level interface
 * for working with model history data.
 *
 * Model histories in SBML consist of one or more <em>model creators</em>,
 * a single date of @em creation, and one or more @em modification dates.
 * The overall XML form of this data takes the following form:
 * 
 <pre class="fragment">
 &lt;dc:creator&gt;
   &lt;rdf:Bag&gt;
     &lt;rdf:li rdf:parseType="Resource"&gt;
       <span style="background-color: #d0eed0">+++</span>
       &lt;vCard:N rdf:parseType="Resource"&gt;
         &lt;vCard:Family&gt;<span style="background-color: #bbb">family name</span>&lt;/vCard:Family&gt;
         &lt;vCard:Given&gt;<span style="background-color: #bbb">given name</span>&lt;/vCard:Given&gt;
       &lt;/vCard:N&gt;
       <span style="background-color: #d0eed0">+++</span>
       <span style="border-bottom: 2px dotted #888">&lt;vCard:EMAIL&gt;<span style="background-color: #bbb">email address</span>&lt;/vCard:EMAIL&gt;</span>
       <span style="background-color: #d0eed0">+++</span>
       <span style="border-bottom: 2px dotted #888">&lt;vCard:ORG rdf:parseType="Resource"&gt;</span>
        <span style="border-bottom: 2px dotted #888">&lt;vCard:Orgname&gt;<span style="background-color: #bbb">organization name</span>&lt;/vCard:Orgname&gt;</span>
       <span style="border-bottom: 2px dotted #888">&lt;/vCard:ORG&gt;</span>
       <span style="background-color: #d0eed0">+++</span>
     &lt;/rdf:li&gt;
     <span style="background-color: #edd">...</span>
   &lt;/rdf:Bag&gt;
 &lt;/dc:creator&gt;
 &lt;dcterms:created rdf:parseType="Resource"&gt;
   &lt;dcterms:W3CDTF&gt;<span style="background-color: #bbb">creation date</span>&lt;/dcterms:W3CDTF&gt;
 &lt;/dcterms:created&gt;
 &lt;dcterms:modified rdf:parseType="Resource"&gt;
   &lt;dcterms:W3CDTF&gt;<span style="background-color: #bbb">modification date</span>&lt;/dcterms:W3CDTF&gt;
 &lt;/dcterms:modified&gt;
 <span style="background-color: #edd">...</span>
 </pre>
 *
 * In the template above, the <span style="border-bottom: 2px dotted #888">underlined</span>
 * portions are optional, the symbol
 * <span class="code" style="background-color: #d0eed0">+++</span> is a placeholder
 * for either no content or valid XML content that is not defined by
 * the annotation scheme, and the ellipses
 * <span class="code" style="background-color: #edd">...</span>
 * are placeholders for zero or more elements of the same form as the
 * immediately preceding element.  The various placeholders for content, namely
 * <span class="code" style="background-color: #bbb">family name</span>,
 * <span class="code" style="background-color: #bbb">given name</span>,
 * <span class="code" style="background-color: #bbb">email address</span>,
 * <span class="code" style="background-color: #bbb">organization</span>,
 * <span class="code" style="background-color: #bbb">creation date</span>, and
 * <span class="code" style="background-color: #bbb">modification date</span>
 * are data that can be filled in using the various methods on
 * the ModelHistory class described below.
 *
 *
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 
/**
 * @class ModelCreator
 * @brief Representation of MIRIAM-compliant model creator data used
 * in ModelHistory. 
 *
 * @htmlinclude not-sbml-warning.html
 *
 * The SBML specification beginning with Level&nbsp;2 Version&nbsp;2
 * defines a standard approach to recording model history and model creator
 * information in a form that complies with MIRIAM ("Minimum Information
 * Requested in the Annotation of biochemical Models", <i>Nature
 * Biotechnology</i>, vol. 23, no. 12, Dec. 2005).  For the model creator,
 * this form involves the use of parts of the <a target="_blank"
 * href="http://en.wikipedia.org/wiki/VCard">vCard</a> representation.
 * LibSBML provides the ModelCreator class as a convenience high-level
 * interface for working with model creator data.  Objects of class
 * ModelCreator can be used to store and carry around creator data within a
 * program, and the various methods in this object class let callers
 * manipulate the different parts of the model creator representation.
 *
 * @section parts The different parts of a model creator definition
 *
 * The ModelCreator class mirrors the structure of the MIRIAM model creator
 * annotations in SBML.  The following template illustrates these different
 * fields when they are written in XML form:
 *
 <pre class="fragment">
 &lt;vCard:N rdf:parseType="Resource"&gt;
   &lt;vCard:Family&gt;<span style="background-color: #bbb">family name</span>&lt;/vCard:Family&gt;
   &lt;vCard:Given&gt;<span style="background-color: #bbb">given name</span>&lt;/vCard:Given&gt;
 &lt;/vCard:N&gt;
 ...
 &lt;vCard:EMAIL&gt;<span style="background-color: #bbb">email address</span>&lt;/vCard:EMAIL&gt;
 ...
 &lt;vCard:ORG rdf:parseType="Resource"&gt;
   &lt;vCard:Orgname&gt;<span style="background-color: #bbb">organization</span>&lt;/vCard:Orgname&gt;
 &lt;/vCard:ORG&gt;
 </pre>
 *
 * Each of the separate data values
 * <span class="code" style="background-color: #bbb">family name</span>,
 * <span class="code" style="background-color: #bbb">given name</span>,
 * <span class="code" style="background-color: #bbb">email address</span>, and
 * <span class="code" style="background-color: #bbb">organization</span> can
 * be set and retrieved via corresponding methods in the ModelCreator 
 * class.  These methods are documented in more detail below.
 *
 * <!-- leave this next break as-is to work around some doxygen bug -->
 */ 
/**
 * @class Date
 * @brief Representation of MIRIAM-compliant dates used in ModelHistory.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * A Date object stores a reasonably complete representation of date and
 * time.  Its purpose is to serve as a way to store dates to be read and
 * written in the <a target="_blank"
 * href="http://www.w3.org/TR/NOTE-datetime">W3C date format</a> used in
 * RDF Dublin Core annotations within SBML.  The W3C date format is a
 * restricted form of <a target="_blank"
 * href="http://en.wikipedia.org/wiki/ISO_8601">ISO 8601</a>, the
 * international standard for the representation of dates and times.  A
 * time and date value in this W3C format takes the form
 * YYYY-MM-DDThh:mm:ssXHH:ZZ (e.g., <code>1997-07-16T19:20:30+01:00</code>)
 * where XHH:ZZ is the time zone offset.  The libSBML Date object contains
 * the following fields to represent these values:
 * <ul>
 * 
 * <li> @em year: an unsigned int representing the year.  This should be a
 * four-digit number such as @c 2011.
 * 
 * <li> @em month: an unsigned int representing the month, with a range of
 * values of 1&ndash;12.  The value @c 1 represents January, and so on.
 *
 * <li> @em day: an unsigned int representing the day of the month, with a
 * range of values of 1&ndash;31.
 * 
 * <li> @em hour: an unsigned int representing the hour on a 24-hour clock,
 * with a range of values of 0&ndash;23.
 * 
 * <li> @em minute: an unsigned int representing the minute, with a range
 * of 0&ndash;59.
 * 
 * <li> @em second: an unsigned int representing the second, with a range
 * of 0&ndash;59.
 * 
 * <li> @em sign: an unsigned int representing the sign of the offset (@c 0
 * signifying @c + and @c 1 signifying @c -).  See the paragraph below for
 * further explanations.
 * 
 * <li> @em hours offset: an unsigned int representing the time zone's hour
 * offset from GMT.
 * 
 * <li> @em minute offset: an unsigned int representing the time zone's
 * minute offset from GMT.
 * 
 * </ul>
 *
 * To illustrate the time zone offset, a value of <code>-05:00</code> would
 * correspond to USA Eastern Standard Time.  In the Date object, this would
 * require a value of @c 1 for the sign field, @c 5 for the hour offset and
 * @c 0 for the minutes offset.
 *
 * In the restricted RDF annotations used in SBML, described in
 * Section&nbsp;6 of the SBML Level&nbsp;2 and Level&nbsp;3 specification
 * documents, date/time stamps can be used to indicate the time of
 * creation and modification of a model.  The following SBML model fragment
 * illustrates this:
@verbatim
<model metaid="_180340" id="GMO" name="Goldbeter1991_MinMitOscil">
    <annotation>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:dc="http://purl.org/dc/elements/1.1/"
                 xmlns:dcterms="http://purl.org/dc/terms/"
                 xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#" >
            <rdf:Description rdf:about="#_180340">
                <dc:creator>
                    <rdf:Bag>
                        <rdf:li rdf:parseType="Resource">
                            <vCard:N rdf:parseType="Resource">
                                <vCard:Family>Shapiro</vCard:Family>
                                <vCard:Given>Bruce</vCard:Given>
                            </vCard:N>
                            <vCard:EMAIL>bshapiro@jpl.nasa.gov</vCard:EMAIL>
                            <vCard:ORG rdf:parseType="Resource">
                                <vCard:Orgname>NASA Jet Propulsion Laboratory</vCard:Orgname>
                            </vCard:ORG>
                        </rdf:li>
                    </rdf:Bag>
                </dc:creator>
                <dcterms:created rdf:parseType="Resource">
                    <dcterms:W3CDTF>2005-02-06T23:39:40+00:00</dcterms:W3CDTF>
                </dcterms:created>
                <dcterms:modified rdf:parseType="Resource">
                    <dcterms:W3CDTF>2005-09-13T13:24:56+00:00</dcterms:W3CDTF>
                </dcterms:modified>
            </rdf:Description>
        </rdf:RDF>
    </annotation>
</model>@endverbatim
 */

#ifndef ModelHistory_h
#define ModelHistory_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/common/operationReturnValues.h>
#include <sbml/util/List.h>

#include <sbml/xml/XMLNode.h>

#ifndef LIBSBML_USE_STRICT_INCLUDES
#include <sbml/annotation/Date.h>
#include <sbml/annotation/ModelCreator.h>
#endif

#ifdef __cplusplus

#include <string>

LIBSBML_CPP_NAMESPACE_BEGIN

#ifdef LIBSBML_USE_STRICT_INCLUDES
  class Date;
  class ModelCreator;
#endif

class LIBSBML_EXTERN ModelHistory
{
public:

  /**
   * Creates a new ModelHistory object.
   */
  ModelHistory ();


  /**
   * Destroys this ModelHistory object.
   */
  ~ModelHistory();


  /**
   * Copy constructor; creates a copy of this ModelHistory object.
   *
   * @param orig the object to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  ModelHistory(const ModelHistory& orig);


  /**
   * Assignment operator for ModelHistory.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  ModelHistory& operator=(const ModelHistory& rhs);


  /**
   * Creates and returns a copy of this ModelHistory object
   *
   * @return a (deep) copy of this ModelHistory object.
   */
  ModelHistory* clone () const;


  /**
   * Returns the "creation date" portion of this ModelHistory object.
   *
   * @return a Date object representing the creation date stored in
   * this ModelHistory object.
   */
  Date * getCreatedDate();

  
  /**
   * Returns the "modified date" portion of this ModelHistory object.
   * 
   * Note that in the MIRIAM format for annotations, there can be multiple
   * modification dates.  The libSBML ModelHistory class supports this by
   * storing a list of "modified date" values.  If this ModelHistory object
   * contains more than one "modified date" value in the list, this method
   * will return the first one in the list.
   *
   * @return a Date object representing the date of modification
   * stored in this ModelHistory object.
   */
  Date * getModifiedDate();

  
  /**
   * Predicate returning @c true or @c false depending on whether this
   * ModelHistory's "creation date" is set.
   *
   * @return @c true if the creation date value of this ModelHistory is
   * set, @c false otherwise.
   */
  bool isSetCreatedDate();

  
  /**
   * Predicate returning @c true or @c false depending on whether this
   * ModelHistory's "modified date" is set.
   *
   * @return @c true if the modification date value of this ModelHistory
   * object is set, @c false otherwise.
   */
  bool isSetModifiedDate();

  
  /**
   * Sets the creation date of this ModelHistory object.
   *  
   * @param date a Date object representing the date to which the "created
   * date" portion of this ModelHistory should be set.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   */
  int setCreatedDate(Date* date);

  
  /**
   * Sets the modification date of this ModelHistory object.
   *  
   * @param date a Date object representing the date to which the "modified
   * date" portion of this ModelHistory should be set.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   */
  int setModifiedDate(Date* date);

  
  /**
   * Adds a copy of a Date object to the list of "modified date" values
   * stored in this ModelHistory object.
   *
   * In the MIRIAM format for annotations, there can be multiple
   * modification dates.  The libSBML ModelHistory class supports this by
   * storing a list of "modified date" values.
   *  
   * @param date a Date object representing the "modified date" that should
   * be added to this ModelHistory object.
   * 
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   */
  int addModifiedDate(Date* date);

  
  /**
   * Returns the list of "modified date" values (as Date objects) stored in
   * this ModelHistory object.
   * 
   * In the MIRIAM format for annotations, there can be multiple
   * modification dates.  The libSBML ModelHistory class supports this by
   * storing a list of "modified date" values.
   * 
   * @return the list of modification dates for this ModelHistory object.
   */
  List * getListModifiedDates();

  
  /**
   * Get the nth Date object in the list of "modified date" values stored
   * in this ModelHistory object.
   * 
   * In the MIRIAM format for annotations, there can be multiple
   * modification dates.  The libSBML ModelHistory class supports this by
   * storing a list of "modified date" values.
   * 
   * @return the nth Date in the list of ModifiedDates of this
   * ModelHistory.
   */
  Date* getModifiedDate(unsigned int n);

  
  /**
   * Get the number of Date objects in this ModelHistory object's list of
   * "modified dates".
   * 
   * In the MIRIAM format for annotations, there can be multiple
   * modification dates.  The libSBML ModelHistory class supports this by
   * storing a list of "modified date" values.
   * 
   * @return the number of ModifiedDates in this ModelHistory.
   */
  unsigned int getNumModifiedDates();

  
  /**
   * Adds a copy of a ModelCreator object to the list of "model creator"
   * values stored in this ModelHistory object.
   *
   * In the MIRIAM format for annotations, there can be multiple model
   * creators.  The libSBML ModelHistory class supports this by storing a
   * list of "model creator" values.
   * 
   * @param mc the ModelCreator to add
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  int addCreator(ModelCreator * mc);

  
  /**
   * Returns the list of ModelCreator objects stored in this ModelHistory
   * object.
   *
   * In the MIRIAM format for annotations, there can be multiple model
   * creators.  The libSBML ModelHistory class supports this by storing a
   * list of "model creator" values.
   * 
   * @return the list of ModelCreator objects.
   */
  List * getListCreators();

  
  /**
   * Get the nth ModelCreator object stored in this ModelHistory object.
   *
   * In the MIRIAM format for annotations, there can be multiple model
   * creators.  The libSBML ModelHistory class supports this by storing a
   * list of "model creator" values.
   * 
   * @return the nth ModelCreator object.
   */
  ModelCreator* getCreator(unsigned int n);

  
  /**
   * Get the number of ModelCreator objects stored in this ModelHistory
   * object.
   *
   * In the MIRIAM format for annotations, there can be multiple model
   * creators.  The libSBML ModelHistory class supports this by storing a
   * list of "model creator" values.
   * 
   * @return the number of ModelCreators objects.
   */
  unsigned int getNumCreators();


  /**
   * Predicate returning @c true if all the required elements for this
   * ModelHistory object have been set.
   *
   * The required elements for a ModelHistory object are "created
   * name", "modified date", and at least one "model creator".
   *
   * @return a boolean value indicating whether all the required
   * elements for this object have been defined.
   */ 
  bool hasRequiredAttributes();

    
  /** @cond doxygen-libsbml-internal */
   
  bool hasBeenModified();

  void resetModifiedFlags();
   
  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /* Can have more than one creator. */

  List * mCreators;

  Date* mCreatedDate;

  /*
   * there can be more than one modified date
   * this is a bug and so as to not break code 
   * I'll hack the old code to interact with a list.
   */
  
  List * mModifiedDates;

  bool mHasBeenModified;


  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


LIBSBML_EXTERN
ModelHistory_t * ModelHistory_create ();

LIBSBML_EXTERN
void ModelHistory_free(ModelHistory_t *);

LIBSBML_EXTERN
ModelHistory_t *
ModelHistory_clone (const ModelHistory_t* mh);


LIBSBML_EXTERN
int ModelHistory_addCreator(ModelHistory_t * mh, 
                             ModelCreator_t * mc);

LIBSBML_EXTERN
int ModelHistory_setCreatedDate(ModelHistory_t * mh, 
                                 Date_t * date);

LIBSBML_EXTERN
int ModelHistory_setModifiedDate(ModelHistory_t * mh, 
                                  Date_t * date);

LIBSBML_EXTERN
List_t * ModelHistory_getListCreators(ModelHistory_t * mh);

LIBSBML_EXTERN
Date_t * ModelHistory_getCreatedDate(ModelHistory_t * mh);

LIBSBML_EXTERN
Date_t * ModelHistory_getModifiedDate(ModelHistory_t * mh);

LIBSBML_EXTERN
unsigned int ModelHistory_getNumCreators(ModelHistory_t * mh);

LIBSBML_EXTERN
ModelCreator_t* ModelHistory_getCreator(ModelHistory_t * mh, unsigned int n);

LIBSBML_EXTERN
int ModelHistory_isSetCreatedDate(ModelHistory_t * mh);

LIBSBML_EXTERN
int ModelHistory_isSetModifiedDate(ModelHistory_t * mh);

LIBSBML_EXTERN
int 
ModelHistory_addModifiedDate(ModelHistory_t * mh, Date_t * date);

LIBSBML_EXTERN
List_t * 
ModelHistory_getListModifiedDates(ModelHistory_t * mh);

LIBSBML_EXTERN
unsigned int 
ModelHistory_getNumModifiedDates(ModelHistory_t * mh);

LIBSBML_EXTERN
Date_t* 
ModelHistory_getModifiedDateFromList(ModelHistory_t * mh, unsigned int n);


LIBSBML_EXTERN
int
ModelHistory_hasRequiredAttributes(ModelHistory_t *mh);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /** ModelHistory_h **/

