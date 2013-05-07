/**
 * @file    SBasePlugin.h
 * @brief   Definition of SBasePlugin, the base class of extension entities
 *          plugged in SBase derived classes in the SBML Core package.
 * @author  Akiya Jouraku
 *
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
 * @class SBasePlugin
 * @brief Representation of a plug-in object of SBML's package extension.
 * 
 * Additional attributes and/or elements of a package extension which are directly 
 * contained by some pre-defined element are contained/accessed by <a href="#SBasePlugin"> 
 * SBasePlugin </a> class which is extended by package developers for each extension point.
 * The extension point, which represents an element to be extended, is identified by a 
 * combination of a Package name and a typecode of the element, and is represented by
 * SBaseExtensionPoint class.
 * </p>
 *
 * <p>
 * For example, the layout extension defines <em>&lt;listOfLayouts&gt;</em> element which is 
 * directly contained in <em>&lt;model&gt;</em> element of the core package. 
 * In the layout package (provided as one of example packages in libSBML-5), the additional 
 * element for the model element is implemented as ListOfLayouts class (an SBase derived class) and 
 * the object is contained/accessed by a LayoutModelPlugin class (an SBasePlugin derived class). 
 * </p>
 *
 * <p>
 * SBasePlugin class defines basic virtual functions for reading/writing/checking 
 * additional attributes and/or top-level elements which should or must be overridden by 
 * subclasses like SBase class and its derived classes.
 * </p>
 *
 * <p>
 *  Package developers must implement an SBasePlugin exntended class for 
 *  each element to be extended (e.g. SBMLDocument, Model, ...) in which additional 
 *  attributes and/or top-level elements of the package extension are directly contained.
 *</p>
 *
 *  To implement reading/writing functions for attributes and/or top-level 
 *  elements of the SBsaePlugin extended class, package developers should or must
 *  override the corresponding virtual functions below provided in the SBasePlugin class:
 *
 *   <ul>
 *     <li> <p>reading elements : </p>
 *       <ol>
 *         <li> <code>virtual SBase* createObject (XMLInputStream& stream) </code>
 *         <p>This function must be overridden if one or more additional elements are defined.</p>
 *         </li>
 *         <li> <code>virtual bool readOtherXML (SBase* parentObject, XMLInputStream& stream)</code>
 *         <p>This function should be overridden if elements of annotation, notes, MathML, etc. need 
 *            to be directly parsed from the given XMLInputStream object instead of the
 *            SBase::readAnnotation(XMLInputStream& stream) 
 *            and/or SBase::readNotes(XMLInputStream& stream) functions.
 *         </p> 
 *         </li>
 *       </ol>
 *     </li>
 *     <li> <p>reading attributes (must be overridden if additional attributes are defined) :</p>
 *       <ol>
 *         <li><code>virtual void addExpectedAttributes(ExpectedAttributes& attributes) </code></li>
 *         <li><code>virtual void readAttributes (const XMLAttributes& attributes, const ExpectedAttributes& expectedAttributes)</code></li>
 *       </ol>
 *     </li>
 *     <li> <p>writing elements (must be overridden if additional elements are defined) :</p>
 *       <ol>
 *         <li><code>virtual void writeElements (XMLOutputStream& stream) const </code></li>
 *       </ol>
 *     </li>
 *     <li> <p>writing attributes : </p>
 *       <ol>
 *        <li><code>virtual void writeAttributes (XMLOutputStream& stream) const </code>
 *         <p>This function must be overridden if one or more additional attributes are defined.</p>
 *        </li>
 *        <li><code>virtual void writeXMLNS (XMLOutputStream& stream) const </code>
 *         <p>This function must be overridden if one or more additional xmlns attributes are defined.</p>
 *        </li>
 *       </ol>
 *     </li>
 *
 *     <li> <p>checking elements (should be overridden) :</p>
 *       <ol>
 *         <li><code>virtual bool hasRequiredElements() const </code></li>
 *       </ol>
 *     </li>
 *
 *     <li> <p>checking attributes (should be overridden) :</p>
 *       <ol>
 *         <li><code>virtual bool hasRequiredAttributes() const </code></li>
 *       </ol>
 *     </li>
 *   </ul>
 *
 *<p>
 *   To implement package-specific creating/getting/manipulating functions of the
 *   SBasePlugin derived class (e.g., getListOfLayouts(), createLyout(), getLayout(),
 *   and etc are implemented in LayoutModelPlugin class of the layout package), package
 *   developers must newly implement such functions (as they like) in the derived class.
 *</p>
 *
 *<p>
 *   SBasePlugin class defines other virtual functions of internal implementations
 *   such as:
 *
 *   <ul>
 *    <li><code> virtual void setSBMLDocument(SBMLDocument* d) </code>
 *    <li><code> virtual void connectToParent(SBase *sbase) </code>
 *    <li><code> virtual void enablePackageInternal(const std::string& pkgURI, const std::string& pkgPrefix, bool flag) </code>
 *   </ul>
 *
 *   These functions must be overridden by subclasses in which one or more top-level elements are defined.
 *</p>
 *
 *<p>
 *   For example, the following three SBasePlugin extended classes are implemented in
 *   the layout extension:
 *</p>
 *
 *<ol>
 *
 *  <li> <p><a href="class_s_b_m_l_document_plugin.html"> SBMLDocumentPlugin </a> class for SBMLDocument element</p>
 *
 *    <ul>
 *         <li> <em> required </em> attribute is added to SBMLDocument object.
 *         </li>
 *    </ul>
 *
 *<p>
 *(<a href="class_s_b_m_l_document_plugin.html"> SBMLDocumentPlugin </a> class is a common SBasePlugin 
 *extended class for SBMLDocument class. Package developers can use this class as-is if no additional 
 *elements/attributes (except for <em> required </em> attribute) is needed for the SBMLDocument class 
 *in their packages, otherwise package developers must implement a new SBMLDocumentPlugin derived class.)
 *</p>
 *
 *  <li> <p>LayoutModelPlugin class for Model element</p>
 *    <ul>
 *       <li> &lt;listOfLayouts&gt; element is added to Model object.
 *       </li>
 *
 *       <li> <p>
 *            The following virtual functions for reading/writing/checking
 *            are overridden: (type of arguments and return values are omitted)
 *            </p>
 *           <ul>
 *              <li> <code> createObject() </code> : (read elements)
 *              </li>
 *              <li> <code> readOtherXML() </code> : (read elements in annotation of SBML L2)
 *              </li>
 *              <li> <code> writeElements() </code> : (write elements)
 *              </li>
 *           </ul>
 *       </li>
 *
 *        <li> <p>
 *             The following virtual functions of internal implementations
 *             are overridden: (type of arguments and return values are omitted)
 *            </p>  
 *            <ul>
 *              <li> <code> setSBMLDocument() </code> 
 *              </li>
 *              <li> <code> connectToParent() </code>
 *              </li>
 *              <li> <code> enablePackageInternal() </code>
 *              </li>
 *            </ul>
 *        </li>
 *
 *
 *        <li> <p>
 *             The following creating/getting/manipulating functions are newly 
 *             implemented: (type of arguments and return values are omitted)
 *            </p>
 *            <ul>
 *              <li> <code> getListOfLayouts() </code>
 *              </li>
 *              <li> <code> getLayout ()  </code>
 *              </li>
 *              <li> <code> addLayout() </code>
 *              </li>
 *              <li> <code> createLayout() </code>
 *              </li>
 *              <li> <code> removeLayout() </code>
 *              </li>	   
 *              <li> <code> getNumLayouts() </code>
 *              </li>
 *           </ul>
 *        </li>
 *
 *    </ul>
 *  </li>
 *
 *  <li> <p>LayoutSpeciesReferencePlugin class for SpeciesReference element (used only for SBML L2V1) </p>
 *
 *      <ul>
 *        <li>
 *         <em> id </em> attribute is internally added to SpeciesReference object
 *          only for SBML L2V1 
 *        </li>
 *
 *        <li>
 *         The following virtual functions for reading/writing/checking
 *          are overridden: (type of arguments and return values are omitted)
 *        </li>
 *
 *         <ul>
 *          <li>
 *          <code> readOtherXML() </code>
 *          </li>
 *          <li>
 *          <code> writeAttributes() </code>
 *          </li>
 *        </ul>
 *      </ul>
 *    </li>
 *
 * </ol>
 */

#ifndef SBasePlugin_h
#define SBasePlugin_h


#include <sbml/common/sbmlfwd.h>
#include <sbml/SBMLTypeCodes.h>
#include <sbml/SBMLErrorLog.h>
#include <sbml/SBase.h>
#include <sbml/SBMLDocument.h>

#include <sbml/extension/SBMLExtension.h>


#ifdef __cplusplus

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN SBasePlugin
{
public:

  /**
   * Destroy this object.
   */
  virtual ~SBasePlugin ();


  /**
   * Assignment operator for SBasePlugin.
   */
  SBasePlugin& operator=(const SBasePlugin& orig);


  /**
   * Returns the XML namespace (URI) of the package extension
   * of this plugin object.
   *
   * @return the URI of the package extension of this plugin object.
   */
  const std::string& getElementNamespace() const;


  /**
   * Returns the prefix of the package extension of this plugin object.
   *
   * @return the prefix of the package extension of this plugin object.
   */
  const std::string& getPrefix() const;


  /**
   * Returns the package name of this plugin object.
   *
   * @return the package name of this plugin object.
   */
  const std::string& getPackageName() const;


  /**
   * Creates and returns a deep copy of this SBasePlugin object.
   * 
   * @return a (deep) copy of this SBase object
   */
  virtual SBasePlugin* clone () const = 0;


  /**
   * Returns the first child element found that has the given id in the model-wide SId namespace, or NULL if no such object is found.
   *
   * @param id string representing the id of objects to find
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
  
  
  // --------------------------------------------------------
  //
  // virtual functions for reading/writing/checking elements
  //
  // --------------------------------------------------------

#ifndef SWIG

  /** @cond doxygen-libsbml-internal */

  /**
   * Takes the contents of the passed-in Model, makes copies of everything, and appends those copies to the appropriate places in this Model.  Only called from Model::appendFrom, and is intended to be extended for packages that add new things to the Model object.
   *
   * @param The Model to merge with this one.
   *
   */
  virtual int appendFrom(const Model* model);

  /**
   * Subclasses must override this method to create, store, and then
   * return an SBML object corresponding to the next XMLToken in the
   * XMLInputStream if they have their specific elements.
   *
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);


  /**
   * Subclasses should override this method to read (and store) XHTML,
   * MathML, etc. directly from the XMLInputStream if the target elements
   * can't be parsed by SBase::readAnnotation(XMLInputStream& stream)
   * and/or SBase::readNotes(XMLInputStream& stream) functions.
   *
   * @return true if the subclass read from the stream, false otherwise.
   */
  virtual bool readOtherXML (SBase* parentObject, XMLInputStream& stream);

  
  /**
   * Synchronizes the annotation of this SBML object.
   *
   * Annotation element (XMLNode* mAnnotation) is synchronized with the 
   * current CVTerm objects (List* mCVTerm).
   * Currently, this method is called in getAnnotation, isSetAnnotation,
   * and writeElements methods.
   */
  virtual void syncAnnotation(SBase* parentObject, XMLNode *annotation);


  /** 
   * Parse L2 annotation if supported
   *
   */
  virtual void parseAnnotation(SBase *parentObject, XMLNode *annotation);

  /**
   * Subclasses must override this method to write out their contained
   * SBML objects as XML elements if they have their specific elements.
   */
  virtual void writeElements (XMLOutputStream& stream) const;


  /** 
   * Checks if this plugin object has all the required elements.
   *
   * Subclasses should override this function if they have their specific 
   * elements.
   *
   * @return true if this pugin object has all the required elements,
   * otherwise false will be returned.
   */
  virtual bool hasRequiredElements() const ;

  /** @endcond */


  // ----------------------------------------------------------
  //
  // virtual functions for reading/writing/checking attributes
  //
  // ----------------------------------------------------------


  /** @cond doxygen-libsbml-internal */

  /**
   * Subclasses should override this method to get the list of
   * expected attributes if they have their specific attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);


  /**
   * Subclasses must override this method to read values from the given
   * XMLAttributes if they have their specific attributes.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);


  /**
   * Subclasses must override this method to write their XML attributes
   * to the XMLOutputStream if they have their specific attributes.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;


  /* 
   * Checks if this plugin object has all the required attributes .
   *
   * Subclasses should override this function if if they have their specific 
   * attributes.
   *
   * @return true if this pugin object has all the required attributes,
   * otherwise false will be returned.
   */
  virtual bool hasRequiredAttributes() const ;


  /**
   * Subclasses should override this method to write required xmlns attributes
   * to the XMLOutputStream (if any). 
   * The xmlns attribute will be written in the element to which the object
   * is connected. For example, xmlns attributes written by this function will
   * be added to Model element if this plugin object connected to the Model 
   * element.
   */
  virtual void writeXMLNS (XMLOutputStream& stream) const;


  /** @endcond */

#endif // SWIG


  /** @cond doxygen-libsbml-internal */

  // ---------------------------------------------------------
  //
  // virtual functions (internal implementation) which should 
  // be overridden by subclasses.
  //
  // ---------------------------------------------------------

  /**
   * Sets the parent SBMLDocument of this plugin object.
   *
   * Subclasses which contain one or more SBase derived elements must
   * override this function.
   *
   * @param d the SBMLDocument object to use
   *
   * @see connectToParent
   * @see enablePackageInternal
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets the parent SBML object of this plugin object to
   * this object and child elements (if any).
   * (Creates a child-parent relationship by this plugin object)
   *
   * This function is called when this object is created by
   * the parent element.
   * Subclasses must override this this function if they have one
   * or more child elements. Also, SBasePlugin::connectToParent(@if java SBase *sbase@endif)
   * must be called in the overridden function.
   *
   * @param sbase the SBase object to use
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToParent (SBase *sbase);


  /**
   * Enables/Disables the given package with child elements in this plugin 
   * object (if any).
   * (This is an internal implementation invoked from 
   *  SBase::enablePackageInternal() function)
   *
   * Subclasses which contain one or more SBase derived elements should 
   * override this function if elements defined in them can be extended by
   * some other package extension.
   *
   * @see setSBMLDocument
   * @see connectToParent
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);


  virtual bool stripPackage(const std::string& pkgPrefix, bool flag);


  /** @endcond */

  // ----------------------------------------------------------


  /**
   * Returns the parent SBMLDocument of this plugin object.
   *
   * @return the parent SBMLDocument object of this plugin object.
   */
  SBMLDocument* getSBMLDocument ();


  /**
   * Returns the parent SBMLDocument of this plugin object.
   *
   * @return the parent SBMLDocument object of this plugin object.
   */
  const SBMLDocument* getSBMLDocument () const;

  /**
   * Gets the URI to which this element belongs to.
   * For example, all elements that belong to SBML Level 3 Version 1 Core
   * must would have the URI "http://www.sbml.org/sbml/level3/version1/core"; 
   * all elements that belong to Layout Extension Version 1 for SBML Level 3
   * Version 1 Core must would have the URI
   * "http://www.sbml.org/sbml/level3/version1/layout/version1/"
   *
   * Unlike getElementNamespace, this function first returns the URI for this 
   * element by looking into the SBMLNamespaces object of the document with 
   * the its package name. if not found it will return the result of 
   * getElementNamespace
   *
   * @return the URI this elements  
   *
   * @see getPackageName
   * @see getElementNamespace
   * @see SBMLDocument::getSBMLNamespaces
   * @see getSBMLDocument
   */
  std::string getURI() const;

  /**
   * Returns the parent SBase object to which this plugin 
   * object connected.
   *
   * @return the parent SBase object to which this plugin 
   * object connected.
   */
  SBase* getParentSBMLObject ();


  /**
   * Returns the parent SBase object to which this plugin 
   * object connected.
   *
   * @return the parent SBase object to which this plugin 
   * object connected.
   */
  const SBase* getParentSBMLObject () const;

  
  /**
   * Sets the XML namespace to which this element belongs to.
   * For example, all elements that belong to SBML Level 3 Version 1 Core
   * must set the namespace to "http://www.sbml.org/sbml/level3/version1/core"; 
   * all elements that belong to Layout Extension Version 1 for SBML Level 3
   * Version 1 Core must set the namespace to 
   * "http://www.sbml.org/sbml/level3/version1/layout/version1/"
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  int setElementNamespace(const std::string &uri);

  /**
   * Returns the SBML level of the package extension of 
   * this plugin object.
   *
   * @return the SBML level of the package extension of
   * this plugin object.
   */
  unsigned int getLevel() const;


  /**
   * Returns the SBML version of the package extension of
   * this plugin object.
   *
   * @return the SBML version of the package extension of
   * this plugin object.
   */
  unsigned int getVersion() const;


  /**
   * Returns the package version of the package extension of
   * this plugin object.
   *
   * @return the package version of the package extension of
   * this plugin object.
   */
  unsigned int getPackageVersion() const;


  /** @cond doxygen-libsbml-internal */
  /**
   * If this object has a child 'math' object (or anything with ASTNodes in general), replace all nodes with the name 'id' with the provided function. 
   *
   * @note This function does nothing itself--subclasses with ASTNode subelements must override this function.
   */
  virtual void replaceSIDWithFunction(const std::string& id, const ASTNode* function);
  /** @endcond */


  /** @cond doxygen-libsbml-internal */
  /**
   * If the function of this object is to assign a value has a child 'math' object (or anything with ASTNodes in general), replace  the 'math' object with the function (existing/function).  
   *
   * @note This function does nothing itself--subclasses with ASTNode subelements must override this function.
   */
  virtual void divideAssignmentsToSIdByFunction(const std::string& id, const ASTNode* function);
  /** @endcond */

  /** @cond doxygen-libsbml-internal */
  /**
   * If this assignment assigns a value to the 'id' element, replace the 'math' object with the function (existing*function). 
   */
  virtual void multiplyAssignmentsToSIdByFunction(const std::string& id, const ASTNode* function);
  /** @endcond */

  /** @cond doxygen-libsbml-internal */
  /**
   * Check to see if the given prefix is used by any of the IDs defined by extension elements.  A package that defines its own 'id' attribute for a core element would check that attribute here.
   */
  virtual bool hasIdentifierBeginningWith(const std::string& prefix);
  /** @endcond */

  /** @cond doxygen-libsbml-internal */
  /**
   * Add the given string to all identifiers in the object.  If the string is added to anything other than an id or a metaid, this code is responsible for tracking down and renaming all *idRefs in the package extention that identifier comes from.
   */
  virtual int prependStringToAllIdentifiers(const std::string& prefix);
  /** @endcond */
  
  /** @cond doxygen-libsbml-internal */
  /**
   * Returns the line number on which this object first appears in the XML
   * representation of the SBML document.
   * 
   * @return the line number of the underlying SBML object.
   *
   * @note The line number for each construct in an SBML model is set upon
   * reading the model.  The accuracy of the line number depends on the
   * correctness of the XML representation of the model, and on the
   * particular XML parser library being used.  The former limitation
   * relates to the following problem: if the model is actually invalid
   * XML, then the parser may not be able to interpret the data correctly
   * and consequently may not be able to establish the real line number.
   * The latter limitation is simply that different parsers seem to have
   * their own accuracy limitations, and out of all the parsers supported
   * by libSBML, none have been 100% accurate in all situations. (At this
   * time, libSBML supports the use of <a target="_blank"
   * href="http://xmlsoft.org">libxml2</a>, <a target="_blank"
   * href="http://expat.sourceforge.net/">Expat</a> and <a target="_blank"
   * href="http://xerces.apache.org/xerces-c/">Xerces</a>.)
   *
   * @see getColumn()
   */
  unsigned int getLine() const;
  /** @endcond */

  /** @cond doxygen-libsbml-internal */
  /**
   * Returns the column number on which this object first appears in the XML
   * representation of the SBML document.
   * 
   * @return the column number of the underlying SBML object.
   * 
   * @note The column number for each construct in an SBML model is set
   * upon reading the model.  The accuracy of the column number depends on
   * the correctness of the XML representation of the model, and on the
   * particular XML parser library being used.  The former limitation
   * relates to the following problem: if the model is actually invalid
   * XML, then the parser may not be able to interpret the data correctly
   * and consequently may not be able to establish the real column number.
   * The latter limitation is simply that different parsers seem to have
   * their own accuracy limitations, and out of all the parsers supported
   * by libSBML, none have been 100% accurate in all situations. (At this
   * time, libSBML supports the use of <a target="_blank"
   * href="http://xmlsoft.org">libxml2</a>, <a target="_blank"
   * href="http://expat.sourceforge.net/">Expat</a> and <a target="_blank"
   * href="http://xerces.apache.org/xerces-c/">Xerces</a>.)
   * 
   * @see getLine()
   */
  unsigned int getColumn() const;
  /** @endcond */

  /** @cond doxygen-libsbml-internal */
  /* gets the SBMLnamespaces - internal use only*/
  virtual SBMLNamespaces * getSBMLNamespaces() const;
  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */
  /**
   * Constructor. Creates an SBasePlugin object with the URI and 
   * prefix of an package extension.
   */
  SBasePlugin (const std::string &uri, const std::string &prefix,
               SBMLNamespaces *sbmlns);


  /**
   * Copy constructor. Creates a copy of this SBase object.
   */
  SBasePlugin(const SBasePlugin& orig);


  /**
   * @return the SBMLErrorLog used to log errors during while reading and
   * validating SBML.
   */
  SBMLErrorLog* getErrorLog ();


  // -----------------------------------------------
  //
  // virtual functions for elements
  //
  // ------------------------------------------------

  /**
   * Helper to log a common type of error for elements.
   */
  virtual void logUnknownElement(const std::string &element, 
                                 const unsigned int sbmlLevel,
 			         const unsigned int sbmlVersion,
			         const unsigned int pkgVersion );

  // -----------------------------------------------
  //
  // virtual functions for attributes
  //
  // ------------------------------------------------

  /**
   * Helper to log a common type of error.
   */
  virtual void logUnknownAttribute(const std::string &attribute, 
                                   const unsigned int sbmlLevel,
 			           const unsigned int sbmlVersion,
			           const unsigned int pkgVersion,
                                   const std::string& element);


  /**
   * Helper to log a common type of error.
   */
  virtual void logEmptyString(const std::string &attribute,
                              const unsigned int sbmlLevel,
                              const unsigned int sbmlVersion,
                              const unsigned int pkgVersion,
                              const std::string& element);


  /*-- data members --*/

  //
  // An SBMLExtension derived object of corresponding package extension
  // The owner of this object is SBMLExtensionRegistry class.
  //
  const SBMLExtension  *mSBMLExt;

  //
  // Parent SBMLDocument object of this plugin object.
  //
  SBMLDocument         *mSBML;

  //
  // Parent SBase derived object to which this plugin object
  // connected.
  //
  SBase                *mParent;

  //
  // XML namespace of corresponding package extension
  //
  std::string          mURI;

  //
  // SBMLNamespaces derived object of this plugin object.
  //
  SBMLNamespaces      *mSBMLNS;

  //
  // Prefix of corresponding package extension
  //
  std::string          mPrefix;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */

#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

LIBSBML_EXTERN
const char* 
SBasePlugin_getURI(SBasePlugin_t* plugin);

LIBSBML_EXTERN
const char* 
SBasePlugin_getPrefix(SBasePlugin_t* plugin);

LIBSBML_EXTERN
const char* 
SBasePlugin_getPackageName(SBasePlugin_t* plugin);

LIBSBML_EXTERN
SBasePlugin_t*
SBasePlugin_clone(SBasePlugin_t* plugin);

LIBSBML_EXTERN
int
SBasePlugin_free(SBasePlugin_t* plugin);

LIBSBML_EXTERN
SBase_t*
SBasePlugin_createObject(SBasePlugin_t* plugin, XMLInputStream_t* stream);

LIBSBML_EXTERN
int
SBasePlugin_readOtherXML(SBasePlugin_t* plugin, SBase_t* parentObject, XMLInputStream_t* stream);

LIBSBML_EXTERN
int
SBasePlugin_writeElements(SBasePlugin_t* plugin, XMLInputStream_t* stream);

LIBSBML_EXTERN
int
SBasePlugin_hasRequiredElements(SBasePlugin_t* plugin);

LIBSBML_EXTERN
int
SBasePlugin_addExpectedAttributes(SBasePlugin_t* plugin, 
        ExpectedAttributes_t* attributes);

LIBSBML_EXTERN
int
SBasePlugin_readAttributes(SBasePlugin_t* plugin,
        XMLAttributes_t* attributes, 
        ExpectedAttributes_t* expectedAttributes);

LIBSBML_EXTERN
int
SBasePlugin_writeAttributes(SBasePlugin_t* plugin,
        XMLOutputStream_t* stream);

LIBSBML_EXTERN
int
SBasePlugin_hasRequiredAttributes(SBasePlugin_t* plugin);

LIBSBML_EXTERN
int
SBasePlugin_writeXMLNS(SBasePlugin_t* plugin, XMLOutputStream_t* stream);

LIBSBML_EXTERN
int
SBasePlugin_setSBMLDocument(SBasePlugin_t* plugin, SBMLDocument_t* d);

LIBSBML_EXTERN
int
SBasePlugin_connectToParent(SBasePlugin_t* plugin, SBase_t* sbase);

LIBSBML_EXTERN
int
SBasePlugin_enablePackageInternal(SBasePlugin_t* plugin, 
        const char* pkgURI, const char* pkgPrefix, int flag);

LIBSBML_EXTERN
SBMLDocument_t*
SBasePlugin_getSBMLDocument(SBasePlugin_t* plugin);

LIBSBML_EXTERN
SBase_t*
SBasePlugin_getParentSBMLObject(SBasePlugin_t* plugin);

LIBSBML_EXTERN
unsigned int
SBasePlugin_getLevel(SBasePlugin_t* plugin);

LIBSBML_EXTERN
unsigned int
SBasePlugin_getVersion(SBasePlugin_t* plugin);

LIBSBML_EXTERN
unsigned int
SBasePlugin_getPackageVersion(SBasePlugin_t* plugin);



END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */

#endif  /* SBasePlugin_h */
