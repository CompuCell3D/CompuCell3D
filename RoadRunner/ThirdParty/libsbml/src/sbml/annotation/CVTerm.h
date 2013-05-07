/**
 * @file    CVTerm.h
 * @brief   Definition of a CVTerm class for adding annotations to a Model.
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
 * @class CVTerm.
 * @brief Representation of MIRIAM-compliant controlled vocabulary annotation.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * The SBML Level&nbsp;2 and Level&nbsp;3 specifications define a simple
 * format for annotating models when (a) referring to controlled vocabulary
 * terms and database identifiers that define and describe biological and
 * biochemical entities, and (b) describing the creator of a model and the
 * model's modification history.  This SBML format is a concrete syntax that
 * conforms to the guidelines of MIRIAM ("Minimum Information Requested in
 * the Annotation of biochemical Models", <i>Nature Biotechnology</i>,
 * vol. 23, no. 12, Dec. 2005).  The format uses a subset of W3C RDF (<a
 * target="_blank" href="http://www.w3.org/RDF/">Resource Description
 * Format</a>).  In order to help application developers work with
 * annotations in this format, libSBML provides several helper classes that
 * provide higher-level interfaces to the data elements; these classes
 * include CVTerm, ModelCreator, ModelHistory, RDFAnnotationParser, and
 * Date.
 *
 * @section annotation-parts Components of an SBML annotation
 *
 * The SBML annotation format consists of RDF-based content placed inside
 * an <code>&lt;annotation&gt;</code> element attached to an SBML component
 * such as Species, Compartment, etc.  The following template illustrates
 * the different parts of SBML annotations in XML form:
 * 
 <pre class="fragment">
 &lt;<span style="background-color: #bbb">SBML_ELEMENT</span> <span style="background-color: #d0eed0">+++</span> metaid=&quot;<span style="border-bottom: 1px solid black">meta id</span>&quot; <span style="background-color: #d0eed0">+++</span>&gt;
   <span style="background-color: #d0eed0">+++</span>
   &lt;annotation&gt;
     <span style="background-color: #d0eed0">+++</span>
     &lt;rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
              xmlns:dc="http://purl.org/dc/elements/1.1/"
              xmlns:dcterm="http://purl.org/dc/terms/"
              xmlns:vcard="http://www.w3.org/2001/vcard-rdf/3.0#"
              xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
              xmlns:bqmodel="http://biomodels.net/model-qualifiers/" &gt;
       &lt;rdf:Description rdf:about=&quot;#<span style="border-bottom: 1px solid black">meta id</span>&quot;&gt;
         <span style="background-color: #e0e0e0; border-bottom: 2px dotted #888">HISTORY</span>
         &lt;<span style="background-color: #bbb">RELATION_ELEMENT</span>&gt;
           &lt;rdf:Bag&gt;
             &lt;rdf:li rdf:resource=&quot;<span style="background-color: #d0d0ee">URI</span>&quot; /&gt;
             <span style="background-color: #edd">...</span>
           &lt;/rdf:Bag&gt;
         &lt;/<span style="background-color: #bbb">RELATION_ELEMENT</span>&gt;
         <span style="background-color: #edd">...</span>
       &lt;/rdf:Description&gt;
       <span style="background-color: #d0eed0">+++</span>
     &lt;/rdf:RDF&gt;
     <span style="background-color: #d0eed0">+++</span>
   &lt;/annotation&gt;
   <span style="background-color: #d0eed0">+++</span>
 &lt;/<span style="background-color: #bbb">SBML_ELEMENT</span>&gt;
 </pre>
 * 
 * In the template above, the placeholder
 * <span class="code" style="background-color: #bbb">SBML_ELEMENT</span> stands for
 * the XML tag name of an SBML model component (e.g., <code>model</code>,
 * <code>reaction</code>, etc.) and the placeholder 
 * <span class="code" style="border-bottom: 1px solid black">meta id</span>
 * stands for the element's meta identifier, which is a field available
 * on all SBML components derived from the SBase base object class.
 * The <span style="border-bottom: 2px dotted #888">dotted</span>
 * portions are optional, the symbol
 * <span class="code" style="background-color: #d0eed0">+++</span> is a placeholder
 * for either no content or valid XML content that is not defined by
 * this annotation scheme, and the ellipses
 * <span class="code" style="background-color: #edd">...</span>
 * are placeholders for zero or more elements of the same form as the
 * immediately preceding element.  The optional content
 * <span class="code" style="background-color: #e0e0e0; border-bottom: 2px dotted #888">HISTORY</span>
 * is a creation and modification history; in libSBML, this is stored
 * using ModelHistory objects.
 *
 * The placeholder <span class="code" style="background-color: #bbb">RELATION_ELEMENT</span>
 * refers to a BioModels.net qualifier element name.  This is an element in
 * either the XML namespace
 * <code>"http://biomodels.net/model-qualifiers"</code> (for model
 * qualifiers) or <code>"http://biomodels.net/biology-qualifiers"</code>
 * (for biological qualifier).  The <span class="code" style="background-color: #d0d0ee">URI</span>
 * is a required data value that uniquely identifies a resource and
 * data within that resource to which the annotation refers.  (Since
 * a URI is only a label, not an address, applications will often
 * want a means of looking up the resource to which the URI refers.
 * Providing the facilities for doing this is the purpose of
 * <a target="_blank" href="http://biomodels.net/miriam">MIRIAM Resources</a>.)
 *
 * The relation-resource pairs above are the "controlled vocabulary" terms
 * that which CVTerm is designed to store and manipulate.  The next section
 * describes these parts in more detail.  For more information about
 * SBML annotations in general, please refer to Section&nbsp;6 in the
 * SBML Level&nbsp;2 (Versions 2&ndash;4) or Level&nbsp;3 specification
 * documents.
 * 
 *
 * @section cvterm-parts The parts of a CVTerm
 * 
 * Annotations that refer to controlled vocabularies are managed in libSBML
 * using CVTerm objects.  A set of RDF-based annotations attached to a
 * given SBML <code>&lt;annotation&gt;</code> element are read by
 * RDFAnnotationParser and converted into a list of these CVTerm objects.
 * Each CVTerm object instance stores the following components of an
 * annotation:
 * 
 * <ul>
 *
 * <li>The @em qualifier, which can be a BioModels.net "biological
 * qualifier", a BioModels.net "model qualifier", or an unknown qualifier
 * (as far as the CVTerm class is concerned).  Qualifiers are used in
 * MIRIAM to indicate the nature of the relationship between the object
 * being annotated and the resource.  In CVTerm, the qualifiers can be
 * manipulated using the methods CVTerm::getQualifierType(),
 * CVTerm::setQualifierType(@if java int type@endif), and related methods.
 * 
 * <li>The @em resource, represented by a URI (which, we must remind
 * developers, is not the same as a URL).  In the CVTerm class, the
 * resource component can be manipulated using the methods
 * CVTerm::addResource(@if java String resource@endif) and
 * CVTerm::removeResource(@if java String resource@endif).
 *
 * </ul>
 *
 * Note that a CVTerm contains a single qualifier, but possibly more than
 * one resource.  This corresponds to the possibility of an annotation that
 * points to multiple resources, all of which are qualified by the same
 * BioModels.net qualifier.  The CVTerm object class supports this by
 * supporting a list of resources.
 *
 * Detailed explanations of the qualifiers defined by BioModels.net can be
 * found at <a target="_blank"
 * href="http://biomodels.net/qualifiers">http://biomodels.net/qualifiers</a>.
 */

#ifndef CVTerm_h
#define CVTerm_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/common/operationReturnValues.h>

#include <sbml/xml/XMLAttributes.h>

LIBSBML_CPP_NAMESPACE_BEGIN

typedef enum
{
    MODEL_QUALIFIER
  , BIOLOGICAL_QUALIFIER
  , UNKNOWN_QUALIFIER
} QualifierType_t;

typedef enum
{
    BQM_IS
  , BQM_IS_DESCRIBED_BY
  , BQM_IS_DERIVED_FROM
  , BQM_UNKNOWN
} ModelQualifierType_t;

typedef enum
{
    BQB_IS
  , BQB_HAS_PART
  , BQB_IS_PART_OF
  , BQB_IS_VERSION_OF
  , BQB_HAS_VERSION
  , BQB_IS_HOMOLOG_TO
  , BQB_IS_DESCRIBED_BY
  , BQB_IS_ENCODED_BY
  , BQB_ENCODES
  , BQB_OCCURS_IN
  , BQB_HAS_PROPERTY
  , BQB_IS_PROPERTY_OF
  , BQB_UNKNOWN
} BiolQualifierType_t;

LIBSBML_CPP_NAMESPACE_END

#ifdef __cplusplus


#include <limits>
#include <iomanip>
#include <string>
#include <sstream>

#include <cstdlib>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN CVTerm
{
public:

  /**
   * Creates an empty CVTerm, optionally with the given @if clike #QualifierType_t value@else qualifier@endif@~ @p type.
   *
   * The SBML Level&nbsp;2 and Level&nbsp;3 specifications define a simple
   * format for annotating models when (a) referring to controlled
   * vocabulary terms and database identifiers that define and describe
   * biological and other entities, and (b) describing the creator of a
   * model and the model's modification history.  The annotation content is
   * stored in <code>&lt;annotation&gt;</code> elements attached to
   * individual SBML elements.  The format for storing the content inside
   * SBML <code>&lt;annotation&gt;</code> elements is a subset of W3C RDF
   * (<a target="_blank" href="http://www.w3.org/RDF/">Resource Description
   * Format</a>) expressed in XML.  The CVTerm class provides a programming
   * interface for working directly with controlled vocabulary term ("CV
   * term") objects without having to deal directly with the XML form.
   * When libSBML reads in an SBML model containing RDF annotations, it
   * parses those annotations into a list of CVTerm objects, and when
   * writing a model, it parses the CVTerm objects back into the
   * appropriate SBML <code>&lt;annotation&gt;</code> structure.
   *
   * This method creates an empty CVTerm object.  The possible qualifier
   * types usable as values of @p type are @link
   * QualifierType_t#MODEL_QUALIFIER MODEL_QUALIFIER@endlink and @link
   * QualifierType_t#BIOLOGICAL_QUALIFIER BIOLOGICAL_QUALIFIER@endlink.  If
   * an explicit value for @p type is not given, this method defaults to
   * using @link QualifierType_t#UNKNOWN_QUALIFIER
   * UNKNOWN_QUALIFIER@endlink.  The @if clike #QualifierType_t value@else qualifier type@endif@~ 
   * can be set later using the
   * CVTerm::setQualifierType(@if java int type@endif) method.
   *
   * Different BioModels.net qualifier elements encode different types of
   * relationships.  Please refer to the SBML specification or the <a
   * target="_blank" href="http://biomodels.net/qualifiers/">BioModels.net
   * qualifiers web page</a> for an explanation of the meaning of these
   * different qualifiers.
   *
   * @param type a @if clike #QualifierType_t value@else qualifier type@endif@~
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  CVTerm(QualifierType_t type = UNKNOWN_QUALIFIER);


  /**
   * Creates a new CVTerm from the given XMLNode.
   *
   * The SBML Level&nbsp;2 and Level&nbsp;3 specifications define a simple
   * format for annotating models when (a) referring to controlled
   * vocabulary terms and database identifiers that define and describe
   * biological and other entities, and (b) describing the creator of a
   * model and the model's modification history.  The annotation content is
   * stored in <code>&lt;annotation&gt;</code> elements attached to
   * individual SBML elements.  The format for storing the content inside
   * SBML <code>&lt;annotation&gt;</code> elements is a subset of W3C RDF
   * (<a target="_blank" href="http://www.w3.org/RDF/">Resource Description
   * Format</a>) expressed in XML.  The CVTerm class provides a programming
   * interface for working directly with controlled vocabulary term ("CV
   * term") objects without having to deal directly with the XML form.
   * When libSBML reads in an SBML model containing RDF annotations, it
   * parses those annotations into a list of CVTerm objects, and when
   * writing a model, it parses the CVTerm objects back into the
   * appropriate SBML <code>&lt;annotation&gt;</code> structure.
   * 
   * This method creates a CVTerm object from the XMLNode object @p node.
   * Recall that XMLNode is a node in an XML tree of elements, and each
   * such element can be placed in a namespace.  This constructor looks for
   * the element to be in the XML namespaces
   * <code>"http://biomodels.net/model-qualifiers"</code> (for
   * model qualifiers) and
   * <code>"http://biomodels.net/biology-qualifiers"</code> (for
   * biological qualifier), and if they are, creates CVTerm objects for
   * the result.
   *
   * @param node an %XMLNode representing a CVTerm.
   *
   * @note This method assumes that the given XMLNode object @p node is of
   * the correct structural form.
   */
  CVTerm(const XMLNode node);


  /**
   * Destroys this CVTerm.
   */
  ~CVTerm();


  /**
   * Copy constructor; creates a copy of a CVTerm object.
   * 
   * @param orig the CVTerm instance to copy.
   * 
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  CVTerm(const CVTerm& orig);


  /**
   * Assignment operator for CVTerm.
   *
   * @param rhs The object whose values are used as the basis of the
   * assignment.
   *
   * @throws @if python ValueError @else SBMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  CVTerm& operator=(const CVTerm& rhs);


  /**
   * Creates and returns a deep copy of this CVTerm object.
   * 
   * @return a (deep) copy of this CVTerm.
   */  
  CVTerm* clone() const; 


  /**
   * Returns the qualifier type of this CVTerm object.
   *
   * @htmlinclude cvterm-common-description-text.html
   *
   * The placeholder <span class="code" style="background-color: #bbb">
   * RELATION_ELEMENT</span> refers to a BioModels.net qualifier
   * element name.  This is an element in either the XML namespace
   * <code>"http://biomodels.net/model-qualifiers"</code> (for model
   * qualifiers) or <code>"http://biomodels.net/biology-qualifiers"</code>
   * (for biological qualifier).  The present method returns a code
   * identifying which one of these two relationship namespaces is being
   * used; any other qualifier in libSBML is considered unknown (as far as
   * the CVTerm class is concerned).  Consequently, this method will return
   * one of the following values:
   * 
   * @li @link QualifierType_t#MODEL_QUALIFIER MODEL_QUALIFIER@endlink
   * @li @link QualifierType_t#BIOLOGICAL_QUALIFIER BIOLOGICAL_QUALIFIER@endlink
   * @li @link QualifierType_t#UNKNOWN_QUALIFIER UNKNOWN_QUALIFIER@endlink
   *
   * The specific relationship of this CVTerm to the enclosing SBML object
   * can be determined using the CVTerm methods such as
   * getModelQualifierType() and getBiologicalQualifierType().  Callers
   * will typically want to use the present method to find out which one of
   * the @em other two methods to call to find out the specific
   * relationship.
   *
   * @return the @if clike #QualifierType_t value@else qualifier type@endif@~
   * of this object or @link QualifierType_t#UNKNOWN_QUALIFIER UNKNOWN_QUALIFIER@endlink
   * (the default).
   *
   * @see getResources()
   * @see getModelQualifierType()
   * @see getBiologicalQualifierType()
   */
  QualifierType_t getQualifierType();


  /**
   * Returns the model qualifier type of this CVTerm object.
   * 
   * @htmlinclude cvterm-common-description-text.html
   *
   * The placeholder <span class="code" style="background-color: #bbb">
   * RELATION_ELEMENT</span> refers to a BioModels.net qualifier
   * element name.  This is an element in either the XML namespace
   * <code>"http://biomodels.net/model-qualifiers"</code> (for model
   * qualifiers) or <code>"http://biomodels.net/biology-qualifiers"</code>
   * (for biological qualifier).  Callers will typically use
   * getQualifierType() to find out the type of qualifier relevant to this
   * particular CVTerm object, then if it is a @em model qualifier, use the
   * present method to determine the specific qualifier.  The set of known
   * model qualifiers is, at the time of this libSBML release, the
   * following:
   *
   * @li @link ModelQualifierType_t#BQM_IS BQM_IS@endlink
   * @li @link ModelQualifierType_t#BQM_IS_DESCRIBED_BY BQM_IS_DESCRIBED_BY@endlink
   * @li @link ModelQualifierType_t#BQM_IS_DERIVED_FROM BQM_IS_DERIVED_FROM@endlink
   *
   * Any other BioModels.net qualifier found in the model is considered
   * unknown by libSBML and reported as
   * @link ModelQualifierType_t#BQM_UNKNOWN BQM_UNKNOWN@endlink.
   *
   * @return the @if clike #ModelQualifierType_t value@else model qualifier type@endif@~
   * of this object or @link ModelQualifierType_t#BQM_UNKNOWN BQM_UNKNOWN@endlink
   * (the default).
   */
  ModelQualifierType_t getModelQualifierType();


  /**
   * Returns the biological qualifier type of this CVTerm object.
   * 
   * @htmlinclude cvterm-common-description-text.html
   *
   * The placeholder <span class="code" style="background-color: #bbb">
   * RELATION_ELEMENT</span> refers to a BioModels.net qualifier
   * element name.  This is an element in either the XML namespace
   * <code>"http://biomodels.net/model-qualifiers"</code> (for model
   * qualifiers) or <code>"http://biomodels.net/biology-qualifiers"</code>
   * (for biological qualifier).  Callers will typically use
   * getQualifierType() to find out the type of qualifier relevant to this
   * particular CVTerm object, then if it is a @em biological qualifier,
   * use the present method to determine the specific qualifier.  The set
   * of known biological qualifiers is, at the time of this libSBML
   * release, the following:
   *
   * @li @link BiolQualifierType_t#BQB_IS BQB_IS@endlink
   * @li @link BiolQualifierType_t#BQB_HAS_PART BQB_HAS_PART@endlink
   * @li @link BiolQualifierType_t#BQB_IS_PART_OF BQB_IS_PART_OF@endlink
   * @li @link BiolQualifierType_t#BQB_IS_VERSION_OF BQB_IS_VERSION_OF@endlink
   * @li @link BiolQualifierType_t#BQB_HAS_VERSION BQB_HAS_VERSION@endlink
   * @li @link BiolQualifierType_t#BQB_IS_HOMOLOG_TO BQB_IS_HOMOLOG_TO@endlink
   * @li @link BiolQualifierType_t#BQB_IS_DESCRIBED_BY BQB_IS_DESCRIBED_BY@endlink
   * @li @link BiolQualifierType_t#BQB_IS_ENCODED_BY BQB_IS_ENCODED_BY@endlink
   * @li @link BiolQualifierType_t#BQB_ENCODES BQB_ENCODES@endlink
   * @li @link BiolQualifierType_t#BQB_OCCURS_IN BQB_OCCURS_IN@endlink
   * @li @link BiolQualifierType_t#BQB_HAS_PROPERTY BQB_HAS_PROPERTY@endlink
   * @li @link BiolQualifierType_t#BQB_IS_PROPERTY_OF BQB_IS_PROPERTY_OF@endlink
   *
   * Any other BioModels.net qualifier found in the model is considered
   * unknown by libSBML and reported as
   * @link BiolQualifierType_t#BQB_UNKNOWN BQB_UNKNOWN@endlink.
   *
   * @return the @if clike #BiolQualifierType_t value@else biology qualifier type@endif@~
   * of this object or @link BiolQualifierType_t#BQB_UNKNOWN BQB_UNKNOWN@endlink
   * (the default).
   */
  BiolQualifierType_t getBiologicalQualifierType();


  /**
   * Returns the resource references for this CVTerm object.
   *
   * @htmlinclude cvterm-common-description-text.html
   *
   * The <span class="code" style="background-color: #d0d0ee">resource
   * URI</span> values shown in the template above are stored internally in
   * CVTerm objects using an XMLAttributes object.  Each attribute stored
   * inside the XMLAttributes will have the same name (specifically,
   * &quot;<code>rdf:resource</code>&quot;) but a different value, and the
   * value will be a <span class="code" style="background-color: #d0d0ee">
   * resource URI</span> shown in the XML template above.
   *
   * A valid CVTerm entity must always have at least one resource and
   * a value for the relationship qualifier.
   * 
   * @return the XMLAttributes that store the resources of this CVTerm.
   *
   * @see getQualifierType()
   * @see addResource(const std::string& resource)
   * @see getResourceURI(unsigned int n)
   */
  XMLAttributes * getResources(); 

  
  /**
   * Returns the resources for this CVTerm object.
   * 
   * @htmlinclude cvterm-common-description-text.html
   *
   * The <span class="code" style="background-color: #d0d0ee">resource
   * URI</span> values shown in the template above are stored internally in
   * CVTerm objects using an XMLAttributes object.  Each attribute stored
   * inside the XMLAttributes will have the same name (specifically,
   * &quot;<code>rdf:resource</code>&quot;) but a different value, and the
   * value will be a <span class="code" style="background-color: #d0d0ee">
   * resource URI</span> shown in the XML template above.
   *
   * A valid CVTerm entity must always have at least one resource and
   * a value for the relationship qualifier.
   * 
   * @return the XMLAttributes that store the resources of this CVTerm.
   *
   * @see getQualifierType()
   * @see addResource(const std::string& resource)
   * @see getResourceURI(unsigned int n)
   */
  const XMLAttributes * getResources() const; 

  
  /**
   * Returns the number of resources for this CVTerm object.
   * 
   * @htmlinclude cvterm-common-description-text.html
   *
   * The fragment above illustrates that there can be more than one
   * resource referenced by a given relationship annotation (i.e., the
   * <span class="code" style="background-color: #d0d0ee">resource
   * URI</span> values associated with a particular <span class="code"
   * style="background-color: #bbb">RELATION_ELEMENT</span>).  The present
   * method returns a count of the resources stored in this CVTerm object.
   *
   * @return the number of resources in the set of XMLAttributes
   * of this CVTerm.
   *
   * @see getResources()
   * @see getResourceURI(unsigned int n)
   */
  unsigned int getNumResources(); 

  
  /**
   * Returns the value of the <em>n</em>th resource for this CVTerm object.
   *
   * @htmlinclude cvterm-common-description-text.html
   *
   * The fragment above illustrates that there can be more than one
   * resource referenced by a given relationship annotation (i.e., the
   * <span class="code" style="background-color: #d0d0ee">resource
   * URI</span> values associated with a particular <span class="code"
   * style="background-color: #bbb">RELATION_ELEMENT</span>).  LibSBML
   * stores all resource URIs in a single CVTerm object for a given
   * relationship.  Callers can use getNumResources() to find out how many
   * resources are stored in this CVTerm object, then call this method to
   * retrieve the <em>n</em>th resource URI.
   * 
   * @param n the index of the resource to query
   *
   * @return string representing the value of the nth resource
   * in the set of XMLAttributes of this CVTerm.
   *
   * @see getNumResources()
   * @see getQualifierType()
   */
  std::string getResourceURI(unsigned int n); 

  
  /**
   * Sets the @if clike #QualifierType_t@else qualifier code@endif@~ of this
   * CVTerm object.
   *
   * @param type the @if clike #QualifierType_t value@else qualifier type@endif.
   * The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   *
   * @see getQualifierType()
   */
  int setQualifierType(QualifierType_t type);


  /**
   * Sets the @if clike #ModelQualifierType_t value@else model qualifier type@endif@~
   * of this CVTerm object.
   *
   * @param type the @if clike #ModelQualifierType_t value@else model qualifier type@endif@~
   *
   * @return integer value indicating success/failure of the
   * function. The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE@endlink
   *
   * @note If the Qualifier Type of this object is not
   * @link QualifierType_t#MODEL_QUALIFIER MODEL_QUALIFIER@endlink, 
   * then the ModelQualifierType_t value will default to
   * @link QualifierType_t#BQM_UNKNOWN BQM_UNKNOWN@endlink.
   *
   * @see getQualifierType()
   * @see setQualifierType(@if java int type@endif)
   */
  int setModelQualifierType(ModelQualifierType_t type);


  /**
   * Sets the @if clike #BiolQualifierType_t value@else biology qualifier type@endif@~
   * of this CVTerm object.
   *
   * @param type the @if clike #BiolQualifierType_t value@else biology qualifier type@endif.
   *
   * @return integer value indicating success/failure of the
   * function. The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE@endlink
   *
   * @note If the Qualifier Type of this object is not
   * @link QualifierType_t#BIOLOGICAL_QUALIFIER BIOLOGICAL_QUALIFIER@endlink,
   * then the @if clike #BiolQualifierType_t value@else biology qualifier type@endif@~ will default
   * to @link BiolQualifierType_t#BQB_UNKNOWN BQB_UNKNOWN@endlink.
   *
   * @see getQualifierType()
   * @see setQualifierType(@if java int type@endif)
   */
  int setBiologicalQualifierType(BiolQualifierType_t type);


  /**
   * Sets the @if clike #ModelQualifierType_t@endif@if java model qualifier type code@endif@~ value of this CVTerm object.
   *
   * @param qualifier the string representing a model qualifier
   *
   * @return integer value indicating success/failure of the
   * function. The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE@endlink
   *
   * @note If the Qualifier Type of this object is not
   * @link QualifierType_t#MODEL_QUALIFIER MODEL_QUALIFIER@endlink, 
   * then the ModelQualifierType_t value will default to
   * @link QualifierType_t#BQM_UNKNOWN BQM_UNKNOWN@endlink.
   *
   * @see getQualifierType()
   * @see setQualifierType(@if java int type@endif)
   */
  int setModelQualifierType(const std::string& qualifier);


  /**
   * Sets the @if clike #BiolQualifierType_t@endif@if java biology qualifier type code@endif@~ of this CVTerm object.
   *
   * @param qualifier the string representing a biology qualifier
   *
   * @return integer value indicating success/failure of the
   * function. The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE@endlink
   *
   * @note If the Qualifier Type of this object is not
   * @link QualifierType_t#BIOLOGICAL_QUALIFIER BIOLOGICAL_QUALIFIER@endlink,
   * then the @if clike #BiolQualifierType_t@endif@if java biology qualifier type code@endif@~ value will default
   * to @link BiolQualifierType_t#BQB_UNKNOWN BQB_UNKNOWN@endlink.
   *
   * @see getQualifierType()
   * @see setQualifierType(@if java int type@endif)
   */
  int setBiologicalQualifierType(const std::string& qualifier);


  /**
   * Adds a resource reference to this CVTerm object.
   *
   * The SBML Level&nbsp;2 and Level&nbsp;3 specifications define a simple
   * standardized format for annotating models with references to
   * controlled vocabulary terms and database identifiers that define and
   * describe biological or other entities.  This annotation format
   * consists of RDF-based content placed inside an
   * <code>&lt;annotation&gt;</code> element attached to an SBML component
   * such as Species, Compartment, etc.
   *
   * The specific RDF element used in this SBML format for referring to
   * external entities is <code>&lt;rdf:Description&gt;</code>, with a
   * <code>&lt;rdf:Bag&gt;</code> element containing one or more
   * <code>&lt;rdf:li&gt;</code> elements.  Each such element refers to a
   * data item in an external resource; the resource and data item are
   * together identified uniquely using a URI.  The following template
   * illustrates the structure:
   *
   <pre class="fragment">
   &lt;rdf:Description rdf:about=&quot;#<span style="border-bottom: 1px solid black">meta id</span>&quot;&gt;
     <span style="background-color: #e0e0e0; border-bottom: 2px dotted #888">HISTORY</span>
     &lt;<span style="background-color: #bbb">RELATION_ELEMENT</span>&gt;
       &lt;rdf:Bag&gt;
         &lt;rdf:li rdf:resource=&quot;<span style="background-color: #d0d0ee">resource URI</span>&quot; /&gt;
         <span style="background-color: #edd">...</span>
       &lt;/rdf:Bag&gt;
     &lt;/<span style="background-color: #bbb">RELATION_ELEMENT</span>&gt;
     <span style="background-color: #edd">...</span>
   &lt;/rdf:Description&gt;
   </pre>
   *
   * In the template above, the placeholder <span class="code"
   * style="border-bottom: 1px solid black">meta id</span> stands for the
   * element's meta identifier, which is a field available on all SBML
   * components derived from the SBase base object class.  The <span
   * style="border-bottom: 2px dotted #888">dotted</span> portions are
   * optional, and the ellipses <span class="code"
   * style="background-color: #edd">...</span> are placeholders for zero or
   * more elements of the same form as the immediately preceding element.
   * The placeholder <span class="code" style="background-color: #bbb">
   * RELATION_ELEMENT</span> refers to a BioModels.net qualifier element
   * name.  This is an element in either the XML namespace
   * <code>"http://biomodels.net/model-qualifiers"</code> (for model
   * qualifiers) or <code>"http://biomodels.net/biology-qualifiers"</code>
   * (for biological qualifier).
   *
   * The <span class="code" style="background-color: #d0d0ee">resource
   * URI</span> is a required data value that uniquely identifies a
   * resource and data within that resource to which the annotation refers.
   * The present method allows callers to add a reference to a resource URI
   * with the same relationship to the enclosing SBML object.  (In other
   * words, the argument to this method is a <span class="code"
   * style="background-color: #d0d0ee">resource URI</span> as shown in the
   * XML fragment above.)  Resources are stored in this CVTerm object
   * within an XMLAttributes object.
   * 
   * The relationship of this CVTerm to the enclosing SBML object can be
   * determined using the CVTerm methods such as getModelQualifierType()
   * and getBiologicalQualifierType().
   *
   * @param resource a string representing the URI of the resource and data
   * item being referenced; e.g.,
   * <code>"http://www.geneontology.org/#GO:0005892"</code>.
   *
   * @return integer value indicating success/failure of the call. The
   * possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED@endlink
   *
   * @see getResources()
   * @see removeResource(std::string resource)
   * @see getQualifierType()
   * @see getModelQualifierType()
   * @see getBiologicalQualifierType()
   */
  int addResource(const std::string& resource);


  /**
   * Removes a resource URI from the set of resources stored in this CVTerm
   * object.
   *
   * @param resource a string representing the resource URI to remove;
   * e.g., <code>"http://www.geneontology.org/#GO:0005892"</code>.
   *
   * @return integer value indicating success/failure of the
   * function. The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS@endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE@endlink
   *
   * @see addResource(const std::string& resource)
   */
  int removeResource(std::string resource);
  

  /**
   * Predicate returning @c true if all the required elements for this
   * CVTerm object have been set.
   *
   * @note The required attributes for a CVTerm are:
   * @li a <em>qualifier type</em>, which can be either a model qualifier or a biological qualifier
   * @li at least one resource
   */ 
  bool hasRequiredAttributes();

  /** @cond doxygen-libsbml-internal */
  
  bool hasBeenModified();

  void resetModifiedFlags();
   
  
  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  XMLAttributes * mResources;

  QualifierType_t       mQualifier;
  ModelQualifierType_t  mModelQualifier;
  BiolQualifierType_t   mBiolQualifier;

  bool mHasBeenModified;

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
CVTerm_t*
CVTerm_createWithQualifierType(QualifierType_t type);


LIBSBML_EXTERN
CVTerm_t*
CVTerm_createFromNode(const XMLNode_t *);


LIBSBML_EXTERN
void
CVTerm_free(CVTerm_t *);


LIBSBML_EXTERN
CVTerm_t *
CVTerm_clone (const CVTerm_t* c);


LIBSBML_EXTERN
QualifierType_t 
CVTerm_getQualifierType(CVTerm_t *);


LIBSBML_EXTERN
ModelQualifierType_t 
CVTerm_getModelQualifierType(CVTerm_t *);


LIBSBML_EXTERN
BiolQualifierType_t 
CVTerm_getBiologicalQualifierType(CVTerm_t *);


LIBSBML_EXTERN
XMLAttributes_t * 
CVTerm_getResources(CVTerm_t *); 


LIBSBML_EXTERN
unsigned int
CVTerm_getNumResources(CVTerm_t*);


LIBSBML_EXTERN
char *
CVTerm_getResourceURI(CVTerm_t * cv, unsigned int n);


LIBSBML_EXTERN
int 
CVTerm_setQualifierType(CVTerm_t * CVT, QualifierType_t type);


LIBSBML_EXTERN
int 
CVTerm_setModelQualifierType(CVTerm_t * CVT, ModelQualifierType_t type);


LIBSBML_EXTERN
int 
CVTerm_setBiologicalQualifierType(CVTerm_t * CVT, BiolQualifierType_t type);


LIBSBML_EXTERN
int 
CVTerm_setModelQualifierTypeByString(CVTerm_t * CVT, const char* qualifier);


LIBSBML_EXTERN
int 
CVTerm_setBiologicalQualifierTypeByString(CVTerm_t * CVT, const char* qualifier);

LIBSBML_EXTERN
int 
CVTerm_addResource(CVTerm_t * CVT, const char * resource);


LIBSBML_EXTERN
int 
CVTerm_removeResource(CVTerm_t * CVT, const char * resource);


LIBSBML_EXTERN
int
CVTerm_hasRequiredAttributes(CVTerm_t *cvt);


LIBSBML_EXTERN
const char* 
ModelQualifierType_toString(ModelQualifierType_t type);

LIBSBML_EXTERN
const char* 
BiolQualifierType_toString(BiolQualifierType_t type);

LIBSBML_EXTERN
ModelQualifierType_t 
ModelQualifierType_fromString(const char* s);

LIBSBML_EXTERN
BiolQualifierType_t 
BiolQualifierType_fromString(const char* s);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /** CVTerm_h **/
