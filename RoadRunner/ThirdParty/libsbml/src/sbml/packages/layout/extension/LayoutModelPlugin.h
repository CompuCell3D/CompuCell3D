/**
 * @file    LayoutModelPlugin.h
 * @brief   Definition of LayoutModelPlugin, the plugin class of 
 *          layout package for Model element.
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
 */

#ifndef LayoutModelPlugin_h
#define LayoutModelPlugin_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>

#ifdef __cplusplus

#include <sbml/SBMLTypeCodes.h>
#include <sbml/SBMLErrorLog.h>
#include <sbml/Model.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/extension/SBasePlugin.h>
#include <sbml/packages/layout/sbml/Layout.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN LayoutModelPlugin : public SBasePlugin
{
public:

  /**
   * Constructor
   */
  LayoutModelPlugin (const std::string &uri, const std::string &prefix,
                     LayoutPkgNamespaces* layoutns);


  /**
   * Copy constructor. Creates a copy of this SBase object.
   */
  LayoutModelPlugin(const LayoutModelPlugin& orig);


  /**
   * Destroy this object.
   */
  virtual ~LayoutModelPlugin ();

  /**
   * Assignment operator for LayoutModelPlugin.
   */
  LayoutModelPlugin& operator=(const LayoutModelPlugin& orig);


  /**
   * Creates and returns a deep copy of this LayoutModelPlugin object.
   * 
   * @return a (deep) copy of this SBase object
   */
  virtual LayoutModelPlugin* clone () const;

#ifndef SWIG

  // --------------------------------------------------------
  //
  // overridden virtual functions for reading/writing/checking 
  // elements
  //
  // --------------------------------------------------------

  /** @cond doxygen-libsbml-internal */

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
   * Parses Layout Extension of SBML Level 2
   */
  virtual bool readOtherXML (SBase* parentObject, XMLInputStream& stream);


  /**
   * Subclasses must override this method to write out their contained
   * SBML objects as XML elements if they have their specific elements.
   */
  virtual void writeElements (XMLOutputStream& stream) const;


  /**
   * This function is a bit tricky.
   * This function is used only for setting annotation element of layout
   * extension for SBML Level2 because annotation element needs to be
   * set before invoking the above writeElements function.
   * Thus, no attribute is written by this function.
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;


  /* function returns true if component has all the required
   * elements
   * needs to be overloaded for each component
   */
  virtual bool hasRequiredElements() const ;

  /** @endcond */

#endif

  /** ------------------------------------------------------------------
   *
   *  Additional public functions
   *
   * ------------------------------------------------------------------
   */
  
  /**
   * Returns the ListOfLayouts object for this Model.
   *
   * @return the ListOfLayouts object for this Model.
   */
  const ListOfLayouts* getListOfLayouts () const;


  /**
   * Returns the ListOfLayouts object for this Model.
   *
   * @return the ListOfLayouts object for this Model.
   */
  ListOfLayouts* getListOfLayouts ();


  /**
   * Returns the layout object that belongs to the given index. If the
   * index is invalid, NULL is returned.
   *
   * @param index the index of list of layout objects.
   *
   * @return the Layout object that belongs to the given index. NULL
   * is returned if the index is invalid. 
   */
  Layout* getLayout (unsigned int index);


  /**
   * Returns the layout object that belongs to the given index. If the
   * index is invalid, NULL is returned.
   *
   * @param index the index of list of layout objects.
   *
   * @return the Layout object that belongs to the given index. NULL
   * is returned if the index is invalid. 
   */
  const Layout* getLayout (unsigned int index) const;


  /**
   * Returns the layout object with the given id attribute. If the
   * id is invalid, NULL is returned.
   *
   * @param sid the id attribute of the layout object.
   *
   * @return the Layout object with the given id attribute. NULL
   * is returned if the given id is invalid. 
   */
  Layout* getLayout (const std::string& sid);


  /**
   * Returns the layout object with the given id attribute. If the
   * id is invalid, NULL is returned.
   *
   * @param sid the id attribute of the layout object.
   *
   * @return the Layout object with the given id attribute. NULL
   * is returned if the given id is invalid. 
   */
  const Layout* getLayout (const std::string& sid) const;


  /**
   * Adds a copy of the layout object to the list of layouts.
   *
   * @param layout the layout object to be added.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li LIBSBML_OPERATION_SUCCESS
   */ 
  int addLayout (const Layout* layout);


  /**
   * Creates a new layout object and adds it to the list of layout objects
   * and returns it.
   *
   * @return a new layout object.
   */
  Layout* createLayout();


  /**
   * Removes the nth Layout object from this Model object and
   * returns a pointer to it.
   *
   * The caller owns the returned object and is responsible for deleting it.
   *
   * @param n the index of the Layout object to remove
   *
   * @return the Layout object removed.  As mentioned above, the caller owns the
   * returned object. NULL is returned if the given index is out of range.
   */
  Layout* removeLayout (unsigned int n);


  /**
   * Returns the number of layout objects.
   *
   * @return the number of layout objects.
   */
  int getNumLayouts() const;


  // ---------------------------------------------------------
  //
  // virtual functions (internal implementation) which should
  // be overridden by subclasses.
  //
  // ---------------------------------------------------------


  /** @cond doxygen-libsbml-internal */
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
   * or more child elements. Also, SBasePlugin::connectToParent()
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
   *  SBase::enablePakcageInternal() function)
   *
   * @note Subclasses in which one or more SBase derived elements are
   * defined must override this function.
   *
   * @see setSBMLDocument
   * @see connectToParent
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);
  /** @endcond */


protected:
  /** @cond doxygen-libsbml-internal */

  /*-- data members --*/

  ListOfLayouts mLayouts;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */
#endif  /* LayoutModelPlugin_h */
