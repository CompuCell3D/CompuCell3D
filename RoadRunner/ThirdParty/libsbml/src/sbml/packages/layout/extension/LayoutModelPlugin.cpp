/**
 * @file    LayoutModelPlugin.cpp
 * @brief   Implementation of LayoutModelPlugin, the plugin class of 
 *          layout package for the Model element.
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

#include <sbml/packages/layout/extension/LayoutModelPlugin.h>
#include <sbml/packages/layout/util/LayoutAnnotation.h>

#include <iostream>
using namespace std;


#ifdef __cplusplus

LIBSBML_CPP_NAMESPACE_BEGIN

/*
 * 
 */
LayoutModelPlugin::LayoutModelPlugin (const std::string &uri, 
                                      const std::string &prefix,
                                      LayoutPkgNamespaces *layoutns)
  : SBasePlugin(uri,prefix,layoutns)
   ,mLayouts(layoutns)
{
}


/*
 * Copy constructor. Creates a copy of this SBase object.
 */
LayoutModelPlugin::LayoutModelPlugin(const LayoutModelPlugin& orig)
  : SBasePlugin(orig)
  , mLayouts(orig.mLayouts)
{
}


/*
 * Destroy this object.
 */
LayoutModelPlugin::~LayoutModelPlugin () {}

/*
 * Assignment operator for LayoutModelPlugin.
 */
LayoutModelPlugin& 
LayoutModelPlugin::operator=(const LayoutModelPlugin& orig)
{
  if(&orig!=this)
  {
    this->SBasePlugin::operator =(orig);
    mLayouts    = orig.mLayouts;
  }    

  return *this;
}


/*
 * Creates and returns a deep copy of this LayoutModelPlugin object.
 * 
 * @return a (deep) copy of this SBase object
 */
LayoutModelPlugin* 
LayoutModelPlugin::clone () const
{
  return new LayoutModelPlugin(*this);  
}


/*
 *
 */
SBase*
LayoutModelPlugin::createObject(XMLInputStream& stream)
{
  SBase*        object = 0;

  const std::string&   name   = stream.peek().getName();
  const XMLNamespaces& xmlns  = stream.peek().getNamespaces();
  const std::string&   prefix = stream.peek().getPrefix();

  const std::string& targetPrefix = (xmlns.hasURI(mURI)) ? xmlns.getPrefix(mURI) : mPrefix;
  
  if (prefix == targetPrefix)
  {
    if ( name == "listOfLayouts" ) 
    {
      //cout << "[DEBUG] LayoutModelPlugin::createObject create listOfLayouts" << endl;
      object = &mLayouts;
    
      if (targetPrefix.empty())
      {
        //
        // prefix is empty when writing elements in layout extension.
        //
        mLayouts.getSBMLDocument()->enableDefaultNS(mURI,true);
      }
    }          
  }    

  return object;
}


/*
 * This function is used only for Layout Extension in SBML Level 2.
 */
bool 
LayoutModelPlugin::readOtherXML (SBase* parentObject, XMLInputStream& stream)
{
  bool readAnnotationFromStream = false;
  const string& name = stream.peek().getName();

  if (!(name.empty()) && name != "annotation")
   {
     return readAnnotationFromStream;
   }

  //
  // This function is used only for SBML Level 2.
  //
  if ( getURI() != LayoutExtension::getXmlnsL2() ) return false;

  XMLNode *pAnnotation = parentObject->getAnnotation();

  if (!pAnnotation)
  {
    //
    // (NOTES)
    //
    // annotation element has not been parsed by the parent element
    // (Model) of this plugin object, thus annotation element is
    // parsed via the given XMLInputStream object in this block. 
    //
  
    const string& name = stream.peek().getName();

    if (name == "annotation")
    {
      pAnnotation = new XMLNode(stream); 

      parseLayoutAnnotation(pAnnotation, mLayouts);

      if (mLayouts.size() > 0)
      {
        //
        // Removes the annotation for layout extension from the annotation
        // of parent element (pAnnotation) and then set the new annotation 
        // (newAnnotation) to the parent element.
        //
        XMLNode *newAnnotation = deleteLayoutAnnotation(pAnnotation);
        parentObject->setAnnotation(newAnnotation);
        delete newAnnotation;
      }
      else
      {
        //
        // No layout annotation is included in the read annotation 
        // (pAnnotation) and thus just set the annotation to the parent
        // element.
        //
        parentObject->setAnnotation(pAnnotation);
      }

      delete pAnnotation;

      readAnnotationFromStream = true;
    }
    
  }
  else if (mLayouts.size() == 0)
  {
    //
    // (NOTES)
    //
    // annotation element has been parsed by the parent element
    // (Model) of this plugin object, thus the annotation element 
    // set to the above pAnnotation variable is parsed in this block.
    //
    parseLayoutAnnotation(pAnnotation, mLayouts);

    if (mLayouts.size() > 0)
    {
      //
      // Removes the annotation for layout extension from the annotation
      // of parent element (pAnnotation) and then set the new annotation 
      // (newAnnotation) to the parent element.
      //
      XMLNode *newAnnotation = deleteLayoutAnnotation(pAnnotation);
      parentObject->setAnnotation(newAnnotation);
    }

    readAnnotationFromStream = true;
  }

  return readAnnotationFromStream;
}


void 
LayoutModelPlugin::writeAttributes (XMLOutputStream& stream) const
{
  //
  // This function is used only for SBML Level 2.
  //
  if ( getURI() != LayoutExtension::getXmlnsL2() ) return;

  Model *parent = static_cast<Model*>(const_cast<SBase*>(getParentSBMLObject()));
  if (parent == NULL) return;


  XMLNode *parentAnnotation = parent->getAnnotation();
  if (parentAnnotation != NULL && parentAnnotation->getNumChildren() > 0)
  {
    deleteLayoutAnnotation(parentAnnotation);
  }

  XMLNode *annt = parseLayouts(parent);
  if (annt && annt->getNumChildren() > 0)
  {
    parent->appendAnnotation(annt);
    delete annt;
  }
}


/*
 *
 */
void
LayoutModelPlugin::writeElements (XMLOutputStream& stream) const
{
  //
  // This function is not used for SBML Level 2.
  //
  if ( getURI() == LayoutExtension::getXmlnsL2() ) return;

  if (mLayouts.size() > 0)
  {
    mLayouts.write(stream);
  }    
  // do nothing.  
}


/* default for components that have no required elements */
bool
LayoutModelPlugin::hasRequiredElements() const
{
  bool allPresent = true;

  if ( mLayouts.size() < 1)
  {
    allPresent = false;    
  }
  
  return allPresent;
}



/*
 *
 *  (EXTENSION) Additional public functions
 *
 */  


/*
 * Returns the ListOf Layouts for this Model.
 */
const ListOfLayouts*
LayoutModelPlugin::getListOfLayouts () const
{
  return &this->mLayouts;
}


/*
 * Returns the ListOf Layouts for this Model.
 */
ListOfLayouts*
LayoutModelPlugin::getListOfLayouts ()
{
  return &this->mLayouts;
}


/*
 * Returns the layout object that belongs to the given index. If the index
 * is invalid, NULL is returned.
 */
const Layout*
LayoutModelPlugin::getLayout (unsigned int index) const
{
  return static_cast<const Layout*>( mLayouts.get(index) );
}


/*
 * Returns the layout object that belongs to the given index. If the index
 * is invalid, NULL is returned.
 */
Layout*
LayoutModelPlugin::getLayout (unsigned int index)
{
  return static_cast<Layout*>( mLayouts.get(index) );
}


/*
 * Returns the layout object with the given id attribute. If the
 * id is invalid, NULL is returned.
 */
const Layout*
LayoutModelPlugin::getLayout (const std::string& sid) const
{
  return static_cast<const Layout*>( mLayouts.get(sid) );
}


/*
 * Returns the layout object with the given id attribute. If the
 * id is invalid, NULL is returned.
 */
Layout*
LayoutModelPlugin::getLayout (const std::string& sid)
{
  return static_cast<Layout*>( mLayouts.get(sid) );
}


int 
LayoutModelPlugin::getNumLayouts() const
{
  return mLayouts.size();
}


/*
 * Adds a copy of the layout object to the list of layouts.
 */ 
int
LayoutModelPlugin::addLayout (const Layout* layout)
{
  if (layout == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  //
  // (TODO) Layout::hasRequiredAttributes() and 
  //       Layout::hasRequiredElements() should be implemented.
  //
  else if (!(layout->hasRequiredAttributes()) || !(layout->hasRequiredElements()))
  {
    return LIBSBML_INVALID_OBJECT;
  }
  else if (getLevel() != layout->getLevel())
  {
    return LIBSBML_LEVEL_MISMATCH;
  }
  else if (getVersion() != layout->getVersion())
  {
    return LIBSBML_VERSION_MISMATCH;
  }
  else if (getPackageVersion() != layout->getPackageVersion())
  {
    return LIBSBML_PKG_VERSION_MISMATCH;
  }
  else if (getLayout(layout->getId()) != NULL)
  {
    // an object with this id already exists
    return LIBSBML_DUPLICATE_OBJECT_ID;
  }
  else
  {
    mLayouts.append(layout);
  }

  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Creates a new layout object and adds it to the list of layout objects.
 * A reference to the newly created object is returned.
 */
Layout*
LayoutModelPlugin::createLayout ()
{
  Layout* l = 0;
  try
  {  
    LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
    l = new Layout(layoutns);
  }
  catch(...)
  {
    /* 
     * NULL will be returned if the mSBMLNS is invalid (basically this
     * should not happen) or some exception is thrown (e.g. std::bad_alloc)
     *
     * (Maybe this should be changed so that caller can detect what kind 
     *  of error happened in this function.)
     */
  }    
  
  if (l) mLayouts.appendAndOwn(l);

  return l;
}


/*
 * Removes the nth Layout object from this Model object and
 * returns a pointer to it.
 */
Layout* 
LayoutModelPlugin::removeLayout (unsigned int n)
{
  return static_cast<Layout*>(mLayouts.remove(n));
}


/*
 * Sets the parent SBMLDocument of this SBML object.
 *
 * @param d the SBMLDocument object to use
 */
void 
LayoutModelPlugin::setSBMLDocument (SBMLDocument* d)
{
  SBasePlugin::setSBMLDocument(d);

  mLayouts.setSBMLDocument(d);  
}


/*
 * Sets the parent SBML object of this plugin object to
 * this object and child elements (if any).
 * (Creates a child-parent relationship by this plugin object)
 */
void
LayoutModelPlugin::connectToParent (SBase* sbase)
{
  SBasePlugin::connectToParent(sbase);

  mLayouts.connectToParent(sbase);
}


/*
 * Enables/Disables the given package with child elements in this plugin
 * object (if any).
 */
void
LayoutModelPlugin::enablePackageInternal(const std::string& pkgURI,
                                         const std::string& pkgPrefix, bool flag)
{
  mLayouts.enablePackageInternal(pkgURI,pkgPrefix,flag);
}


LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */
