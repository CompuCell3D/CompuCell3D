/**
 * Filename    : Layout.cpp
 * Description : SBML Layout Layout source
 * Organization: European Media Laboratories Research gGmbH
 * Created     : 2004-07-15
 *
 * Copyright 2004 European Media Laboratories Research gGmbH
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * European Media Laboratory Research gGmbH have no obligations to provide
 * maintenance, support, updates, enhancements or modifications.  In no
 * event shall the European Media Laboratories Research gGmbH be liable to
 * any party for direct, indirect, special, incidental or consequential
 * damages, including lost profits, arising out of the use of this software
 * and its documentation, even if the European Media Laboratory Research
 * gGmbH have been advised of the possibility of such damage.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Ralph Gauges
 *     Bioinformatics Group
 *     European Media Laboratories Research gGmbH
 *     Schloss-Wolfsbrunnenweg 31c
 *     69118 Heidelberg
 *     Germany
 *
 *     http://www.eml-research.de/english/Research/BCB/
 *     mailto:ralph.gauges@eml-r.villa-bosch.de
 *
 * Contributor(s):
 *
 *     Akiya Jouraku <jouraku@bio.keio.ac.jp>
 *     Modified this file for package extension in libSBML5
 *
 */


#include <climits> 
#include <iostream>
#include <limits>
#include <assert.h>
#include <memory>

#include <sbml/packages/layout/sbml/Layout.h>
#include <sbml/packages/layout/sbml/GeneralGlyph.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * Creates a new Layout with the given level, version, and package version.
 */
Layout::Layout (unsigned int level, unsigned int version, unsigned int pkgVersion) 
  : SBase (level,version)
   ,mId("")
   ,mDimensions(level,version,pkgVersion)
   ,mCompartmentGlyphs(level,version,pkgVersion)
   ,mSpeciesGlyphs(level,version,pkgVersion)
   ,mReactionGlyphs(level,version,pkgVersion)
   ,mTextGlyphs(level,version,pkgVersion)
   ,mAdditionalGraphicalObjects(level,version,pkgVersion)
{
  // set an SBMLNamespaces derived object (LayoutPkgNamespaces) of this package.
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  

  // connect child elements to this element.
  connectToChild();
}


/**
 * Creates a new Layout with the given namespaces, id, and dimensions.
 */
Layout::Layout (LayoutPkgNamespaces* layoutns, const std::string& id, const Dimensions* dimensions)
  : SBase (layoutns)
   ,mId (id)
   ,mDimensions(layoutns)
   ,mCompartmentGlyphs(layoutns)
   ,mSpeciesGlyphs(layoutns)
   ,mReactionGlyphs(layoutns)
   ,mTextGlyphs(layoutns)
   ,mAdditionalGraphicalObjects(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  if(dimensions)
  {
    this->mDimensions=*dimensions;
  }

  // connect child elements to this element.
  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Ctor.
 */
Layout::Layout(LayoutPkgNamespaces* layoutns)
 : SBase(layoutns)
  ,mId("")
  ,mDimensions(layoutns)
  ,mCompartmentGlyphs(layoutns)
  ,mSpeciesGlyphs(layoutns)
  ,mReactionGlyphs(layoutns)
  ,mTextGlyphs(layoutns)
  ,mAdditionalGraphicalObjects(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  // connect child elements to this element.
  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}



/**
 * Creates a new Layout from the given XMLNode
 */
Layout::Layout(const XMLNode& node, unsigned int l2version)
 : SBase(2,l2version)
  ,mId ("")
  ,mDimensions(2,l2version)
  ,mCompartmentGlyphs(2,l2version)
  ,mSpeciesGlyphs(2,l2version)
  ,mReactionGlyphs(2,l2version)
  ,mTextGlyphs(2,l2version)
  ,mAdditionalGraphicalObjects(2,l2version)
{

  LayoutPkgNamespaces *layoutNS = new LayoutPkgNamespaces(2,l2version);

  setSBMLNamespacesAndOwn(layoutNS);  
    
  // load plugins
  loadPlugins(mSBMLNamespaces);


    const XMLAttributes& attributes=node.getAttributes();
    const XMLNode* child;
    //ExpectedAttributes ea(getElementName());
    ExpectedAttributes ea;
    addExpectedAttributes(ea);
    this->readAttributes(attributes,ea);
    unsigned int n=0,nMax = node.getNumChildren();
    while(n<nMax)
    {
        child=&node.getChild(n);
        const std::string& childName=child->getName();
        if(childName=="dimensions")
        {
            this->mDimensions=Dimensions(*child);
        }
        else if(childName=="annotation")
        {
            this->setAnnotation(child);
        }
        else if(childName=="notes")
        {
            this->mNotes=new XMLNode(*child);
        }
        else if(childName=="listOfCompartmentGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                ListOf& list=this->mCompartmentGlyphs;
                if(innerChildName=="compartmentGlyph")
                {
                    list.appendAndOwn(new CompartmentGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    list.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    list.setNotes(new XMLNode(*innerChild));
                }
                else
                {
                    // throw
                }
                ++i;
            }
        }
        else if(childName=="listOfSpeciesGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                ListOf& list=this->mSpeciesGlyphs;
                if(innerChildName=="speciesGlyph")
                {
                    list.appendAndOwn(new SpeciesGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    list.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    list.setNotes(new XMLNode(*innerChild));
                }
                else
                {
                    // throw
                }
                ++i;
            }
        }
        else if(childName=="listOfReactionGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                ListOf& list=this->mReactionGlyphs;
                if(innerChildName=="reactionGlyph")
                {
                    list.appendAndOwn(new ReactionGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    list.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    list.setNotes(new XMLNode(*innerChild));
                }
                else
                {
                    // throw
                }
                ++i;
            }
        }
        else if(childName=="listOfTextGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                ListOf& list=this->mTextGlyphs;
                if(innerChildName=="textGlyph")
                {
                    list.appendAndOwn(new TextGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    list.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    list.setNotes(new XMLNode(*innerChild));
                }
                else
                {
                    // throw
                }
                ++i;
            }
        }
        else if(childName=="listOfAdditionalGraphicalObjects")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                ListOf& list=this->mAdditionalGraphicalObjects;
                if(innerChildName=="graphicalObject")
                {
                    list.appendAndOwn(new GraphicalObject(*innerChild));
                }
                else if(innerChildName=="generalGlyph")
                {
                  list.appendAndOwn(new GeneralGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    list.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    list.setNotes(new XMLNode(*innerChild));
                }
                else
                {
                    // throw
                }
                ++i;
            }
        }
        else
        {
            //throw;
        }
        
        ++n;
    }


  // connect child elements to this element.
  connectToChild();

}

/**
 * Copy constructor.
 */
Layout::Layout(const Layout& source):SBase(source)
{
    this->mId=source.getId();
    this->mDimensions=*source.getDimensions();
    this->mCompartmentGlyphs=*source.getListOfCompartmentGlyphs();
    this->mSpeciesGlyphs=*source.getListOfSpeciesGlyphs();
    this->mReactionGlyphs=*source.getListOfReactionGlyphs();
    this->mTextGlyphs=*source.getListOfTextGlyphs();
    this->mAdditionalGraphicalObjects=*source.getListOfAdditionalGraphicalObjects();

    // connect child elements to this element.
    connectToChild();
}

/**
 * Assignment operator.
 */
Layout& Layout::operator=(const Layout& source)
{
  if(&source!=this)
  {
    this->SBase::operator=(source);
    this->mId = source.mId;
    this->mDimensions=*source.getDimensions();
    this->mCompartmentGlyphs=*source.getListOfCompartmentGlyphs();
    this->mSpeciesGlyphs=*source.getListOfSpeciesGlyphs();
    this->mReactionGlyphs=*source.getListOfReactionGlyphs();
    this->mTextGlyphs=*source.getListOfTextGlyphs();
    this->mAdditionalGraphicalObjects=*source.getListOfAdditionalGraphicalObjects();

    // connect child elements to this element.
    connectToChild();
  }
  
  return *this;
}


/**
 * Destructor.
 */ 
Layout::~Layout ()
{
}


/**
 * Does nothing since no defaults are defined for Layout.
 */ 
void
Layout::initDefaults ()
{
}


/**
  * Returns the value of the "id" attribute of this Layout.
  */
const std::string& Layout::getId () const
{
  return mId;
}


/**
  * Predicate returning @c true or @c false depending on whether this
  * Layout's "id" attribute has been set.
  */
bool Layout::isSetId () const
{
  return (mId.empty() == false);
}

/**
  * Sets the value of the "id" attribute of this Layout.
  */
int Layout::setId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id,mId);
}


/**
  * Unsets the value of the "id" attribute of this Layout.
  */
int Layout::unsetId ()
{
  mId.erase();
  if (mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/**
 * Returns the dimensions of the layout.
 */ 
const Dimensions*
Layout::getDimensions() const
{
  return &this->mDimensions;
}


/**
 * Returns the dimensions of the layout.
 */ 
Dimensions*
Layout::getDimensions()
{
  return &this->mDimensions;
}


/**
 * Sets the dimensions of the layout.
 */ 
void
Layout::setDimensions (const Dimensions* dimensions)
{
  if(dimensions==NULL) return;  
  this->mDimensions = *dimensions;
  this->mDimensions.connectToParent(this);
}


/**
 * Returns the ListOf object that holds all compartment glyphs.
 */ 
const ListOfCompartmentGlyphs*
Layout::getListOfCompartmentGlyphs () const
{
  return &this->mCompartmentGlyphs;
}


/**
 * Returns the ListOf object that holds all species glyphs.
 */ 
const ListOfSpeciesGlyphs*
Layout::getListOfSpeciesGlyphs () const
{
  return &this->mSpeciesGlyphs;
}


/**
 * Returns the ListOf object that holds all reaction glyphs.
 */ 
const ListOfReactionGlyphs*
Layout::getListOfReactionGlyphs () const
{
  return &this->mReactionGlyphs;
}


/**
 * Returns the ListOf object that holds all text glyphs.
 */ 
const ListOfTextGlyphs*
Layout::getListOfTextGlyphs () const
{
  return &this->mTextGlyphs;
}


/**
 * Returns the ListOf object that holds all additonal graphical objects.
 */ 
const ListOfGraphicalObjects*
Layout::getListOfAdditionalGraphicalObjects () const
{
  return &this->mAdditionalGraphicalObjects;
}


/**
 * Returns the ListOf object that holds all compartment glyphs.
 */ 
ListOfCompartmentGlyphs*
Layout::getListOfCompartmentGlyphs ()
{
  return &this->mCompartmentGlyphs;
}


/**
 * Returns the ListOf object that holds all species glyphs.
 */ 
ListOfSpeciesGlyphs*
Layout::getListOfSpeciesGlyphs ()
{
  return &this->mSpeciesGlyphs;
}


/**
 * Returns the ListOf object that holds all reaction glyphs.
 */ 
ListOfReactionGlyphs*
Layout::getListOfReactionGlyphs ()
{
  return &this->mReactionGlyphs;
}


/**
 * Returns the ListOf object that holds all text glyphs.
 */ 
ListOfTextGlyphs*
Layout::getListOfTextGlyphs ()
{
  return &this->mTextGlyphs;
}


/**
 * Returns the ListOf object that holds all additional graphical objects.
 */ 
ListOfGraphicalObjects*
Layout::getListOfAdditionalGraphicalObjects ()
{
  return &this->mAdditionalGraphicalObjects;
}


/**
 * Returns the compartment glyph with the given index.  If the index is
 * invalid, NULL is returned.
 */ 
CompartmentGlyph*
Layout::getCompartmentGlyph (unsigned int index) 
{
  return static_cast<CompartmentGlyph*>( this->mCompartmentGlyphs.get(index) );
}

/**
 * Returns the compartment glyph with the given index.  If the index is
 * invalid, NULL is returned.
 */ 
const CompartmentGlyph*
Layout::getCompartmentGlyph (unsigned int index) const
{
  return static_cast<const CompartmentGlyph*>( this->mCompartmentGlyphs.get(index) );
}


/**
 * Returns the species glyph with the given index.  If the index is
 * invalid, NULL is returned.
 */ 
SpeciesGlyph*
Layout::getSpeciesGlyph (unsigned int index) 
{
  return static_cast<SpeciesGlyph*>( this->mSpeciesGlyphs.get(index) );
}

/**
 * Returns the species glyph with the given index.  If the index is
 * invalid, NULL is returned.
 */ 
const SpeciesGlyph*
Layout::getSpeciesGlyph (unsigned int index) const
{
  return static_cast<const SpeciesGlyph*>( this->mSpeciesGlyphs.get(index) );
}


/**
 * Returns the reaction glyph with the given index.  If the index is
 * invalid, NULL is returned.
 */ 
ReactionGlyph*
Layout::getReactionGlyph (unsigned int index) 
{
  return static_cast<ReactionGlyph*>( this->mReactionGlyphs.get(index) );
}

/**
 * Returns the reaction glyph with the given index.  If the index is
 * invalid, NULL is returned.
 */ 
const ReactionGlyph*
Layout::getReactionGlyph (unsigned int index) const
{
  return static_cast<const ReactionGlyph*>( this->mReactionGlyphs.get(index) );
}


/**
 * Returns the text glyph with the given index.  If the index is invalid,
 * NULL is returned.
 */ 
TextGlyph*
Layout::getTextGlyph (unsigned int index) 
{
  return static_cast<TextGlyph*>( this->mTextGlyphs.get(index) );
}

/**
 * Returns the text glyph with the given index.  If the index is invalid,
 * NULL is returned.
 */ 
const TextGlyph*
Layout::getTextGlyph (unsigned int index) const
{
  return static_cast<const TextGlyph*>( this->mTextGlyphs.get(index) );
}


/**
 * Returns the additional graphical object with the given index.
 * If the index is invalid, NULL is returned.
 */ 
GraphicalObject*
Layout::getAdditionalGraphicalObject (unsigned int index) 
{
  return static_cast<GraphicalObject*>
  ( 
    this->mAdditionalGraphicalObjects.get(index)
  );
}

/**
 * Returns the additional graphical object with the given index.
 * If the index is invalid, NULL is returned.
 */ 
const GraphicalObject*
Layout::getAdditionalGraphicalObject (unsigned int index) const
{
  return static_cast<const GraphicalObject*>
  ( 
    this->mAdditionalGraphicalObjects.get(index)
  );
}

/**
 * Returns the GeneralGlyph with the given index.
 * If the index is invalid, NULL is returned.
 */
GeneralGlyph*
Layout::getGeneralGlyph (unsigned int index)
{
  unsigned int count = 0;
  for (unsigned int i = 0; i < this->mAdditionalGraphicalObjects.size(); ++i)
  {
    if (mAdditionalGraphicalObjects.get(i)->getTypeCode() == SBML_LAYOUT_GENERALGLYPH)
    {
      if (count == index)
        return static_cast<GeneralGlyph*>(mAdditionalGraphicalObjects.get(i));
      ++count;
    }
  }
  return NULL;
}

/**
 * Returns the GeneralGlyph with the given index.
 * If the index is invalid, NULL is returned.
 */
const GeneralGlyph*
Layout::getGeneralGlyph (unsigned int index) const
{
  unsigned int count = 0;
  for (unsigned int i = 0; i < this->mAdditionalGraphicalObjects.size(); ++i)
  {
    if (mAdditionalGraphicalObjects.get(i)->getTypeCode() == SBML_LAYOUT_GENERALGLYPH)
    {
      if (count == index)
        return static_cast<const GeneralGlyph*>(mAdditionalGraphicalObjects.get(i));
      ++count;
    }
  }
  return NULL;

}


const GraphicalObject*
Layout::getObjectWithId (const ListOf* list,const std::string& id) const
{
  const GraphicalObject* object=NULL;
  unsigned int counter=0;
  while(counter < list->size()) {
    const GraphicalObject* tmp=dynamic_cast<const GraphicalObject*>(list->get(counter));
    if(tmp->getId()==id){
      object=tmp;
      break;
    }
    ++counter;
  }    
  return object;
}

GraphicalObject*
Layout::getObjectWithId (ListOf* list,const std::string& id) 
{
  GraphicalObject* object=NULL;
  unsigned int counter=0;
  while(counter < list->size()) {
    GraphicalObject* tmp=dynamic_cast<GraphicalObject*>(list->get(counter));
    if(tmp->getId()==id){
      object=tmp;
      break;
    }
    ++counter;
  }    
  return object;
}

GraphicalObject*
Layout::removeObjectWithId (ListOf* list,const std::string& id)
{
  GraphicalObject* object=NULL;
  unsigned int counter=0;
  while(counter < list->size()) {
    GraphicalObject* tmp=dynamic_cast<GraphicalObject*>(list->get(counter));
    if(tmp->getId()==id){
      object=tmp;
      list->remove(counter);
      break;
    }
    ++counter;
  }    
  return object;
}

/**
 * Removes the compartment glyph with the given index from the layout.
 * A pointer to the compartment glyph that was removed is returned.
 * If no compartment glyph has been removed, NULL is returned.
 */
CompartmentGlyph* Layout::removeCompartmentGlyph(unsigned int index)
{
    CompartmentGlyph* glyph=NULL;
    if(index < this->getNumCompartmentGlyphs())
    {
      glyph=dynamic_cast<CompartmentGlyph*>(this->getListOfCompartmentGlyphs()->remove(index));
    }
    return glyph;
}

/**
 * Removes the species glyph with the given index from the layout.
 * A pointer to the species glyph that was removed is returned.
 * If no species glyph has been removed, NULL is returned.
 */
SpeciesGlyph* Layout::removeSpeciesGlyph(unsigned int index)
{
    SpeciesGlyph* glyph=NULL;
    if(index < this->getNumSpeciesGlyphs())
    {
      glyph=dynamic_cast<SpeciesGlyph*>(this->getListOfSpeciesGlyphs()->remove(index));
    }
    return glyph;
}

/**
 * Removes the reaction glyph with the given index from the layout.
 * A pointer to the reaction glyph that was removed is returned.
 * If no reaction glyph has been removed, NULL is returned.
 */
ReactionGlyph* Layout::removeReactionGlyph(unsigned int index)
{
    ReactionGlyph* glyph=NULL;
    if(index < this->getNumReactionGlyphs())
    {
      glyph=dynamic_cast<ReactionGlyph*>(this->getListOfReactionGlyphs()->remove(index));
    }
    return glyph;
}

/**
 * Removes the text glyph with the given index from the layout.
 * A pointer to the text glyph that was removed is returned.
 * If no text glyph has been removed, NULL is returned.
 */
TextGlyph* Layout::removeTextGlyph(unsigned int index)
{
    TextGlyph* glyph=NULL;
    if(index < this->getNumTextGlyphs())
    {
      glyph=dynamic_cast<TextGlyph*>(this->getListOfTextGlyphs()->remove(index));
    }
    return glyph;
}

/**
 * Removes the graphical object with the given index from the layout.
 * A pointer to the graphical object that was removed is returned.
 * If no graphical object has been removed, NULL is returned.
 */
GraphicalObject* Layout::removeAdditionalGraphicalObject(unsigned int index)
{
    GraphicalObject* go=NULL;
    if(index < this->getNumAdditionalGraphicalObjects())
    {
      go=dynamic_cast<GraphicalObject*>(this->getListOfAdditionalGraphicalObjects()->remove(index));
    }
    return go;
}

/**
 * Remove the compartment glyph with the given id.
 * A pointer to the removed compartment glyph is returned.
 * If no compartment glyph has been removed, NULL is returned.
 */
CompartmentGlyph*
Layout::removeCompartmentGlyph(const std::string id)
{
    return dynamic_cast<CompartmentGlyph*>(this->removeObjectWithId(this->getListOfCompartmentGlyphs(),id));
}

/**
 * Remove the species glyph with the given id.
 * A pointer to the removed species glyph is returned.
 * If no species glyph has been removed, NULL is returned.
 */
SpeciesGlyph*
Layout::removeSpeciesGlyph(const std::string id)
{
    return dynamic_cast<SpeciesGlyph*>(this->removeObjectWithId(this->getListOfSpeciesGlyphs(),id));
}

/**
 * Remove the species reference glyph with the given id.
 * A pointer to the removed species glyph is returned.
 * If no species glyph has been removed, NULL is returned.
 */
SpeciesReferenceGlyph*
Layout::removeSpeciesReferenceGlyph(const std::string id)
{
    SpeciesReferenceGlyph *srg=NULL;
    unsigned int i,iMax=this->getNumReactionGlyphs();
    for(i=0;i<iMax;++i)
    {
        ReactionGlyph* rg=this->getReactionGlyph(i);
        unsigned int index=rg->getIndexForSpeciesReferenceGlyph(id);
        if(index!=std::numeric_limits<unsigned int>::max())
        {
            srg=rg->removeSpeciesReferenceGlyph(index);
            break;
        }
    }
    return srg;
}

/**
 * Remove the reaction glyph with the given id.
 * A pointer to the removed reaction glyph is returned.
 * If no reaction glyph has been removed, NULL is returned.
 */
ReactionGlyph*
Layout::removeReactionGlyph(const std::string id)
{
    return dynamic_cast<ReactionGlyph*>(this->removeObjectWithId(this->getListOfReactionGlyphs(),id));
}

/**
 * Remove the text glyph with the given id.
 * A pointer to the removed text glyph is returned.
 * If no text glyph has been removed, NULL is returned.
 */
TextGlyph*
Layout::removeTextGlyph(const std::string id)
{
    return dynamic_cast<TextGlyph*>(this->removeObjectWithId(this->getListOfTextGlyphs(),id));
}

/**
 * Remove the graphical object with the given id.
 * A pointer to the removed graphical object is returned.
 * If no graphical object has been removed, NULL is returned.
 */
GraphicalObject*
Layout::removeAdditionalGraphicalObject(const std::string id)
{
    return this->removeObjectWithId(this->getListOfAdditionalGraphicalObjects(),id);
}

/**
 * Returns the compartment glyph that has the given id, or NULL if no
 * compartment glyph has the id.
 */
CompartmentGlyph*
Layout::getCompartmentGlyph (const std::string& id) 
{
  return (CompartmentGlyph*) this->getObjectWithId(&this->mCompartmentGlyphs, id);
}

/**
 * Returns the compartment glyph that has the given id, or NULL if no
 * compartment glyph has the id.
 */
const CompartmentGlyph*
Layout::getCompartmentGlyph (const std::string& id) const
{
  return (const CompartmentGlyph*) this->getObjectWithId(&this->mCompartmentGlyphs, id);
}


/**
 * Returns the species glyph that has the given id, or NULL if no
 * species glyph has the id.
 */
const SpeciesGlyph*
Layout::getSpeciesGlyph (const std::string& id) const
{
  return (const SpeciesGlyph*) this->getObjectWithId(&this->mSpeciesGlyphs, id);
}

/**
 * Returns the species glyph that has the given id, or NULL if no
 * species glyph has the id.
 */
SpeciesGlyph*
Layout::getSpeciesGlyph (const std::string& id) 
{
  return (SpeciesGlyph*) this->getObjectWithId(&this->mSpeciesGlyphs, id);
}


/**
 * Returns the reaction glyph that has the given id, or NULL if no
 * reaction glyph has the id.
 */
const ReactionGlyph*
Layout::getReactionGlyph (const std::string& id) const
{
  return (const ReactionGlyph*) this->getObjectWithId(&this->mReactionGlyphs, id);
}

/**
 * Returns the reaction glyph that has the given id, or NULL if no
 * reaction glyph has the id.
 */
ReactionGlyph*
Layout::getReactionGlyph (const std::string& id) 
{
  return (ReactionGlyph*) this->getObjectWithId(&this->mReactionGlyphs, id);
}


/**
 * Returns the text glyph that has the given id, or NULL if no compartment
 * glyph has the id.
 */
const TextGlyph*
Layout::getTextGlyph (const std::string& id) const
{
  return (const TextGlyph*) this->getObjectWithId(&this->mTextGlyphs, id);
}


/**
 * Returns the text glyph that has the given id, or NULL if no compartment
 * glyph has the id.
 */
TextGlyph*
Layout::getTextGlyph (const std::string& id) 
{
  return (TextGlyph*) this->getObjectWithId(&this->mTextGlyphs, id);
}


/**
 * Returns the GeneralGlyph that has the given id, or NULL
 * if no general glyph has the id.
 */
const GeneralGlyph*
Layout::getGeneralGlyph (const std::string& id) const
{
  return static_cast<const GeneralGlyph*>(this->getObjectWithId(&this->mAdditionalGraphicalObjects, id));
}

/**
 * Returns the GeneralGlyph that has the given id, or NULL
 * if no general glyph has the id.
 */
GeneralGlyph*
Layout::getGeneralGlyph (const std::string& id)
{
  return static_cast<GeneralGlyph*>(this->getObjectWithId(&this->mAdditionalGraphicalObjects, id));
}

/**
 * Returns the additional graphicalo object that has the given id, or NULL
 * if no additional glyph has the id.
 */
const GraphicalObject*
Layout::getAdditionalGraphicalObject (const std::string& id) const
{
  return this->getObjectWithId(&this->mAdditionalGraphicalObjects, id);
}

/**
 * Returns the additional graphicalo object that has the given id, or NULL
 * if no additional glyph has the id.
 */
GraphicalObject*
Layout::getAdditionalGraphicalObject (const std::string& id) 
{
  return this->getObjectWithId(&this->mAdditionalGraphicalObjects, id);
}


/**
 * Adds a new compartment glyph.
 */
void
Layout::addCompartmentGlyph (const CompartmentGlyph* glyph)
{
  this->mCompartmentGlyphs.append(glyph);
}


/**
 * Adds a new species glyph.
 */
void
Layout::addSpeciesGlyph (const SpeciesGlyph* glyph)
{
  this->mSpeciesGlyphs.append(glyph);
}


/**
 * Adds a new reaction glyph.
 */
void
Layout::addReactionGlyph (const ReactionGlyph* glyph)
{
  this->mReactionGlyphs.append(glyph);
}


/**
 * Adds a new text glyph.
 */
void
Layout::addTextGlyph (const TextGlyph* glyph)
{
  this->mTextGlyphs.append(glyph);
}


/**
 * Adds a new additional graphical object glyph.
 */
void
Layout::addAdditionalGraphicalObject (const GraphicalObject* glyph)
{
  this->mAdditionalGraphicalObjects.append(glyph);
}

/**
 * Adds a new general glyph.
 */
void
Layout::addGeneralGlyph (const GeneralGlyph* glyph)
{
  addAdditionalGraphicalObject(glyph);
}


/**
 * Returns the number of compartment glyphs for the layout.
 */
unsigned int
Layout::getNumCompartmentGlyphs () const
{
  return this->mCompartmentGlyphs.size();
}


/**
 * Returns the number of species glyphs for the layout.
 */
unsigned int
Layout::getNumSpeciesGlyphs () const
{
  return this->mSpeciesGlyphs.size();
}


/**
 * Returns the number of reaction glyphs for the layout.
 */
unsigned int
Layout::getNumReactionGlyphs () const
{
  return this->mReactionGlyphs.size();
}


/**
 * Returns the number of text glyphs for the layout.
 */
unsigned int
Layout::getNumTextGlyphs () const
{
  return this->mTextGlyphs.size();
}


/**
 * Returns the number of additional graphical objects for the layout.
 */
unsigned int
Layout::getNumAdditionalGraphicalObjects () const
{
  return this->mAdditionalGraphicalObjects.size();
}

/**
 * Returns the number of general glyphs for the layout.
 */
unsigned int
Layout::getNumGeneralGlyphs() const
{
  unsigned int count = 0;
  for (unsigned int i = 0; i < this->mAdditionalGraphicalObjects.size(); ++i)
  {
    if (mAdditionalGraphicalObjects.get(i)->getTypeCode() == SBML_LAYOUT_GENERALGLYPH)
      ++count;
  }
  return count;
}


/**
 * Creates a CompartmentGlyph object, adds it to the end of the compartment
 * glyph objects list and returns a reference to the newly created object.
 */
CompartmentGlyph* 
Layout::createCompartmentGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  CompartmentGlyph* p = new CompartmentGlyph(layoutns);

  this->mCompartmentGlyphs.appendAndOwn(p);
  return p;
}


/**
 * Creates a SpeciesGlyph object, adds it to the end of the species glyph
 * objects list and returns a reference to the newly created object.
 */
SpeciesGlyph* 
Layout::createSpeciesGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  SpeciesGlyph* p = new SpeciesGlyph(layoutns);

  this->mSpeciesGlyphs.appendAndOwn(p);
  return p;
}


/**
 * Creates a ReactionGlyph object, adds it to the end of the reaction glyph
 * objects list and returns a reference to the newly created object.
 */
ReactionGlyph* 
Layout::createReactionGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  ReactionGlyph* p = new ReactionGlyph(layoutns);

  this->mReactionGlyphs.appendAndOwn(p);  
  return p;
}

/**
 * Creates a GeneralGlyph object, adds it to the end of the additional 
 * objects list and returns a reference to the newly created object.
 */
GeneralGlyph* 
Layout::createGeneralGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  GeneralGlyph* g = new GeneralGlyph(layoutns);

  this->mAdditionalGraphicalObjects.appendAndOwn(g);
  return g;
}


/**
 * Creates a TextGlyph object, adds it to the end of the text glyph objects
 * list and returns a reference to the newly created object.
 */
TextGlyph* 
Layout::createTextGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  TextGlyph* p = new TextGlyph(layoutns);

  this->mTextGlyphs.appendAndOwn(p);
  return p;
}


/**
 * Creates a GraphicalObject object, adds it to the end of the additional
 * graphical objects list and returns a reference to the newly created
 * object.
 */
GraphicalObject* 
Layout::createAdditionalGraphicalObject ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  GraphicalObject* p = new GraphicalObject(layoutns);

  this->mAdditionalGraphicalObjects.appendAndOwn(p);
  return p;
}


/**
 * Creates a new SpeciesReferenceGlyph for the last ReactionGlyph and adds
 * it to its list of SpeciesReferenceGlyph objects.  A pointer to the newly
 * created object is returned.
 */
SpeciesReferenceGlyph* 
Layout::createSpeciesReferenceGlyph ()
{
  int size = this->mReactionGlyphs.size();
  if (size == 0) return NULL;

  ReactionGlyph* r =dynamic_cast<ReactionGlyph*> (this->getReactionGlyph(size - 1));
  return r->createSpeciesReferenceGlyph();
}


/**
 * Creates a new LineSegment for the Curve object of the last ReactionGlyph
 * or the last SpeciesReferenceGlyph in the last ReactionGlyph and adds it
 * to its list of SpeciesReferenceGlyph objects.  A pointer to the newly
 * created object is returned.
 */
LineSegment* 
Layout::createLineSegment()
{
  int size = this->mReactionGlyphs.size();
  if (size == 0) return NULL;

  LineSegment*   ls = NULL;
  ReactionGlyph* r  = dynamic_cast<ReactionGlyph*> (this->getReactionGlyph(size - 1));

  size = r->getListOfSpeciesReferenceGlyphs()->size();
  if(size > 0)
  {
    SpeciesReferenceGlyph* srg = r->getSpeciesReferenceGlyph(size-1);
    ls = srg->createLineSegment();
  }
  else
  {
    ls = r->createLineSegment();
  }

  return ls;
}        


/**
 * Creates a new CubicBezier for the Curve object of the last ReactionGlyph
 * or the last SpeciesReferenceGlyph in the last ReactionGlyph and adds it
 * to its list of SpeciesReferenceGlyph objects.  A pointer to the newly
 * created object is returned.
 */
CubicBezier* 
Layout::createCubicBezier ()
{
  int size = this->mReactionGlyphs.size();
  if (size == 0) return NULL;

  CubicBezier*   cb = NULL;
  ReactionGlyph* r  = dynamic_cast<ReactionGlyph*>( this->getReactionGlyph(size - 1));

  size = r->getListOfSpeciesReferenceGlyphs()->size();
  if(size > 0)
  {
    SpeciesReferenceGlyph* srg = r->getSpeciesReferenceGlyph(size-1);
    cb = srg->createCubicBezier();
  }
  else
  {
    cb = r->createCubicBezier();
  }

  return cb;
}    

/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
Layout::getElementName () const
{
  static const std::string name = "layout";
  return name;
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
Layout::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;

  if (name == "listOfCompartmentGlyphs")
  {
    object = &mCompartmentGlyphs;
  }

  else if ( name == "listOfSpeciesGlyphs"      ) object = &mSpeciesGlyphs;
  else if ( name == "listOfReactionGlyphs"       ) object = &mReactionGlyphs;
  else if ( name == "listOfTextGlyphs"            ) object = &mTextGlyphs;
  else if ( name == "listOfAdditionalGraphicalObjects") object = &mAdditionalGraphicalObjects;
  else if ( name == "dimensions"               ) object = &mDimensions;

  return object;
}

/**
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
Layout::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  attributes.add("id");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */
void
Layout::readAttributes (const XMLAttributes& attributes,
                        const ExpectedAttributes& expectedAttributes)
{
  SBase::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("id", mId, getErrorLog(), true, getLine(), getColumn());
  if (assigned && mId.empty())
  {
    logEmptyString(mId, sbmlLevel, sbmlVersion, "<layout>");
  }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);
}


/**
 * Subclasses should override this method to write their XML attributes
 * to the XMLOutputStream.  Be sure to call your parents implementation
 * of this method as well.
 */
void
Layout::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  stream.writeAttribute("id", getPrefix(), mId);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}


/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
Layout::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  mDimensions.write(stream);
  if (getNumCompartmentGlyphs() > 0)
  {
    mCompartmentGlyphs.write(stream);
  }

  if ( getNumSpeciesGlyphs() > 0 ) mSpeciesGlyphs.write(stream);

  if ( getNumReactionGlyphs() > 0 ) mReactionGlyphs.write(stream);
  if ( getNumTextGlyphs    () > 0 ) mTextGlyphs.write(stream);

  if ( getNumAdditionalGraphicalObjects() > 0 )
  {
      mAdditionalGraphicalObjects.write(stream);
  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}


/**
 * Creates an XMLNode object from this.
 */
XMLNode Layout::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple layout_triple = XMLTriple("layout", "", "");
  XMLAttributes id_att = XMLAttributes();
  id_att.add("id", this->mId);
  // add the SBase Ids
  addSBaseAttributes(*this,id_att);
  XMLToken layout_token = XMLToken(layout_triple, id_att, xmlns); 
  XMLNode layout_node(layout_token);
  // add the notes and annotations
  if(this->mNotes) layout_node.addChild(*this->mNotes);
  
  Layout *self = const_cast<Layout*>(this);
  XMLNode *annot = self->getAnnotation();
  if(annot)
  {
    layout_node.addChild(*annot);
  }
  
  // add the dimensions
  layout_node.addChild(this->mDimensions.toXML());
  // add the list of compartment glyphs
  if(this->mCompartmentGlyphs.size()>0)
  {
    layout_node.addChild(this->mCompartmentGlyphs.toXML());
  }
  // add the list of species glyphs
  if(this->mSpeciesGlyphs.size()>0)
  {
    layout_node.addChild(this->mSpeciesGlyphs.toXML());
  }
  // add the list of reaction glyphs
  if(this->mReactionGlyphs.size()>0)
  {
    layout_node.addChild(this->mReactionGlyphs.toXML());
  }
  // add the list of text glyphs
  if(this->mTextGlyphs.size()>0)
  {
    layout_node.addChild(this->mTextGlyphs.toXML());
  }
  // add the list of additional graphical objects
  if(this->mAdditionalGraphicalObjects.size()>0)
  {
    layout_node.addChild(this->mAdditionalGraphicalObjects.toXML());
  }
  return layout_node;
}


/**
 * Returns the package type code for this object.
 */
int
Layout::getTypeCode () const
{
  return SBML_LAYOUT_LAYOUT;
}


Layout*
Layout::clone() const
{
    return new Layout(*this);
}


/**
 * Accepts the given SBMLVisitor.
 */
bool
Layout::accept (SBMLVisitor& v) const
{
  /*  
  bool result=v.visit(*this);
  this->mDimensions.accept(v);
  this->mCompartmentGlyphs.accept(v);
  this->mSpeciesGlyphs.accept(v);
  this->mReactionGlyphs.accept(v);
  this->mTextGlyphs.accept(v);
  this->mAdditionalGraphicalObjects.accept(v);
  v.leave(*this);*/
  return false;
}


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Layout::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);

  mDimensions.setSBMLDocument(d);
  mCompartmentGlyphs.setSBMLDocument(d);
  mSpeciesGlyphs.setSBMLDocument(d);
  mReactionGlyphs.setSBMLDocument(d);
  mTextGlyphs.setSBMLDocument(d);
  mAdditionalGraphicalObjects.setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
Layout::connectToChild()
{
  mDimensions.connectToParent(this);
  mCompartmentGlyphs.connectToParent(this);
  mSpeciesGlyphs.connectToParent(this);
  mReactionGlyphs.connectToParent(this);
  mTextGlyphs.connectToParent(this);
  mAdditionalGraphicalObjects.connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
Layout::enablePackageInternal(const std::string& pkgURI,
                              const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mDimensions.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mCompartmentGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mSpeciesGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mReactionGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mTextGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mAdditionalGraphicalObjects.enablePackageInternal(pkgURI,pkgPrefix,flag);
}


/**
 * Ctor.
 */
ListOfLayouts::ListOfLayouts(LayoutPkgNamespaces* layoutns)
 : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  loadPlugins(layoutns);

}


/**
 * Ctor.
 */
ListOfLayouts::ListOfLayouts(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));

  
};


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfLayouts*
ListOfLayouts::clone () const
{
  return new ListOfLayouts(*this);
}


/* return nth item in list */
Layout *
ListOfLayouts::get(unsigned int n)
{
  return static_cast<Layout*>(ListOf::get(n));
}


/* return nth item in list */
const Layout *
ListOfLayouts::get(unsigned int n) const
{
  return static_cast<const Layout*>(ListOf::get(n));
}


/* return item by id */
Layout*
ListOfLayouts::get (const std::string& sid)
{
  return const_cast<Layout*>( 
    static_cast<const ListOfLayouts&>(*this).get(sid) );
}


/* return item by id */
const Layout*
ListOfLayouts::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<Layout>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <Layout*> (*result);
}


/* Removes the nth item from this list */
Layout*
ListOfLayouts::remove (unsigned int n)
{
   return static_cast<Layout*>(ListOf::remove(n));
}


/* Removes item in this list by id */
Layout*
ListOfLayouts::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<Layout>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <Layout*> (item);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfLayouts::getItemTypeCode () const
{
  return SBML_LAYOUT_LAYOUT;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfLayouts::getElementName () const
{
  static const std::string name = "listOfLayouts";
  return name;
}


void
ListOfLayouts::resetElementNamespace(const std::string& uri)
{
  setElementNamespace(uri);
  SBMLNamespaces *sbmlns = getSBMLNamespaces();
  sbmlns->removeNamespace(LayoutExtension::getXmlnsL2());
  sbmlns->addNamespace(LayoutExtension::getXmlnsL3V1V1(), "layout");

}
/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfLayouts::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "layout")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new Layout(layoutns);
    appendAndOwn(object);
    //mItems.push_back(object);
  }

  return object;
}


void 
ListOfLayouts::writeXMLNS (XMLOutputStream& stream) const
{
  XMLNamespaces xmlns;
  xmlns.add(LayoutExtension::getXmlnsXSI(), "xsi");

  std::string prefix = getPrefix();

  if (prefix.empty())
  {
    if (getNamespaces()->hasURI(LayoutExtension::getXmlnsL3V1V1()))
    {
      xmlns.add(LayoutExtension::getXmlnsL3V1V1(),prefix);
    }
  }

  stream << xmlns;
}


/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfLayouts::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  xmlns.add("http://projects.eml.org/bcb/sbml/level2", "");
  xmlns.add("http://www.w3.org/2001/XMLSchema-instance", "xsi");
  XMLTriple triple = XMLTriple("listOfLayouts", "http://projects.eml.org/bcb/sbml/level2", "");
  XMLAttributes att = XMLAttributes();
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  bool end=true;
  if(this->mNotes)
  {
      node.addChild(*this->mNotes);
      end=false;
  }
  ListOfLayouts *self = const_cast<ListOfLayouts*>(this);
  XMLNode *annot = self->getAnnotation();
  if(annot)
  {
    
    node.addChild(*annot);
      end=false;
  }
  unsigned int i,iMax=this->size();
  const Layout* layout=NULL;
  for(i=0;i<iMax;++i)
  {
    layout=dynamic_cast<const Layout*>(this->get(i));
    assert(layout);
    node.addChild(layout->toXML());
  }  
  if(end==true && iMax==0) node.setEnd();
  return node;
}


/**
 * Ctor.
 */
ListOfCompartmentGlyphs::ListOfCompartmentGlyphs(LayoutPkgNamespaces* layoutns)
 : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());
}


/**
 * Ctor.
 */
ListOfCompartmentGlyphs::ListOfCompartmentGlyphs(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfCompartmentGlyphs*
ListOfCompartmentGlyphs::clone () const
{
  return new ListOfCompartmentGlyphs(*this);
}


/* return nth item in list */
CompartmentGlyph *
ListOfCompartmentGlyphs::get(unsigned int n)
{
  return static_cast<CompartmentGlyph*>(ListOf::get(n));
}


/* return nth item in list */
const CompartmentGlyph *
ListOfCompartmentGlyphs::get(unsigned int n) const
{
  return static_cast<const CompartmentGlyph*>(ListOf::get(n));
}


/* return item by id */
CompartmentGlyph*
ListOfCompartmentGlyphs::get (const std::string& sid)
{
  return const_cast<CompartmentGlyph*>( 
    static_cast<const ListOfCompartmentGlyphs&>(*this).get(sid) );
}


/* return item by id */
const CompartmentGlyph*
ListOfCompartmentGlyphs::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<CompartmentGlyph>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <CompartmentGlyph*> (*result);
}


/* Removes the nth item from this list */
CompartmentGlyph*
ListOfCompartmentGlyphs::remove (unsigned int n)
{
   return static_cast<CompartmentGlyph*>(ListOf::remove(n));
}


/* Removes item in this list by id */
CompartmentGlyph*
ListOfCompartmentGlyphs::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<CompartmentGlyph>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <CompartmentGlyph*> (item);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfCompartmentGlyphs::getItemTypeCode () const
{
  return SBML_LAYOUT_COMPARTMENTGLYPH;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfCompartmentGlyphs::getElementName () const
{
  static const std::string name = "listOfCompartmentGlyphs";
  return name;
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfCompartmentGlyphs::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "compartmentGlyph")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new CompartmentGlyph(layoutns);
    appendAndOwn(object);
//    mItems.push_back(object);
  }

  return object;
}




/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfCompartmentGlyphs::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfCompartmentGlyphs", "http://projects.eml.org/bcb/sbml/level2", "");
  XMLAttributes att = XMLAttributes();
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  bool end=true;
  if(this->mNotes)
  {
      node.addChild(*this->mNotes);
      end=false;
  }
  if(this->mAnnotation)
  {
      node.addChild(*this->mAnnotation);
      end=false;
  }
  unsigned int i,iMax=this->size();
  const CompartmentGlyph* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const CompartmentGlyph*>(this->get(i));
    assert(object);
    node.addChild(object->toXML());
  }
  if(end==true && iMax==0)
  {
    node.setEnd();
  }
  return node;
}


/**
 * Ctor.
 */
ListOfSpeciesGlyphs::ListOfSpeciesGlyphs(LayoutPkgNamespaces* layoutns)
  : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());
}


/**
 * Ctor.
 */
ListOfSpeciesGlyphs::ListOfSpeciesGlyphs(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfSpeciesGlyphs*
ListOfSpeciesGlyphs::clone () const
{
  return new ListOfSpeciesGlyphs(*this);
}


/* return nth item in list */
SpeciesGlyph *
ListOfSpeciesGlyphs::get(unsigned int n)
{
  return static_cast<SpeciesGlyph*>(ListOf::get(n));
}


/* return nth item in list */
const SpeciesGlyph *
ListOfSpeciesGlyphs::get(unsigned int n) const
{
  return static_cast<const SpeciesGlyph*>(ListOf::get(n));
}


/* return item by id */
SpeciesGlyph*
ListOfSpeciesGlyphs::get (const std::string& sid)
{
  return const_cast<SpeciesGlyph*>( 
    static_cast<const ListOfSpeciesGlyphs&>(*this).get(sid) );
}


/* return item by id */
const SpeciesGlyph*
ListOfSpeciesGlyphs::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<SpeciesGlyph>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <SpeciesGlyph*> (*result);
}


/* Removes the nth item from this list */
SpeciesGlyph*
ListOfSpeciesGlyphs::remove (unsigned int n)
{
   return static_cast<SpeciesGlyph*>(ListOf::remove(n));
}


/* Removes item in this list by id */
SpeciesGlyph*
ListOfSpeciesGlyphs::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<SpeciesGlyph>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <SpeciesGlyph*> (item);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfSpeciesGlyphs::getItemTypeCode () const
{
  return SBML_LAYOUT_SPECIESGLYPH;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfSpeciesGlyphs::getElementName () const
{
  static const std::string name = "listOfSpeciesGlyphs";
  return name;
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfSpeciesGlyphs::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "speciesGlyph")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new SpeciesGlyph(layoutns);
    appendAndOwn(object);
//    mItems.push_back(object);
  }

  return object;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfSpeciesGlyphs::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfSpeciesGlyphs", "http://projects.eml.org/bcb/sbml/level2", "");
  XMLAttributes att = XMLAttributes();
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  bool end=true;
  if(this->mNotes)
  { 
      node.addChild(*this->mNotes);
      end=false;
  }
  if(this->mAnnotation)
  {
      node.addChild(*this->mAnnotation);
      end=false;
  }
  unsigned int i,iMax=this->size();
  const SpeciesGlyph* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const SpeciesGlyph*>(this->get(i));
    assert(object);
    node.addChild(object->toXML());
  }
  if(end==true && iMax==0)
  {
    node.setEnd();
  }
  return node;
}


/**
 * Ctor.
 */
ListOfReactionGlyphs::ListOfReactionGlyphs(LayoutPkgNamespaces* layoutns)
  : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());
}


/**
 * Ctor.
 */
ListOfReactionGlyphs::ListOfReactionGlyphs(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfReactionGlyphs*
ListOfReactionGlyphs::clone () const
{
  return new ListOfReactionGlyphs(*this);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfReactionGlyphs::getItemTypeCode () const
{
  return SBML_LAYOUT_REACTIONGLYPH;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfReactionGlyphs::getElementName () const
{
  static const std::string name = "listOfReactionGlyphs";
  return name;
}


/* return nth item in list */
ReactionGlyph *
ListOfReactionGlyphs::get(unsigned int n)
{
  return static_cast<ReactionGlyph*>(ListOf::get(n));
}


/* return nth item in list */
const ReactionGlyph *
ListOfReactionGlyphs::get(unsigned int n) const
{
  return static_cast<const ReactionGlyph*>(ListOf::get(n));
}


/* return item by id */
ReactionGlyph*
ListOfReactionGlyphs::get (const std::string& sid)
{
  return const_cast<ReactionGlyph*>( 
    static_cast<const ListOfReactionGlyphs&>(*this).get(sid) );
}


/* return item by id */
const ReactionGlyph*
ListOfReactionGlyphs::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<ReactionGlyph>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <ReactionGlyph*> (*result);
}


/* Removes the nth item from this list */
ReactionGlyph*
ListOfReactionGlyphs::remove (unsigned int n)
{
   return static_cast<ReactionGlyph*>(ListOf::remove(n));
}


/* Removes item in this list by id */
ReactionGlyph*
ListOfReactionGlyphs::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<ReactionGlyph>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <ReactionGlyph*> (item);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfReactionGlyphs::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "reactionGlyph")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new ReactionGlyph(layoutns);
    appendAndOwn(object);
//    mItems.push_back(object);
  }

  return object;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfReactionGlyphs::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfReactionGlyphs", "http://projects.eml.org/bcb/sbml/level2", "");
  XMLAttributes att = XMLAttributes();
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  bool end=true;
  if(this->mNotes)
  {
      node.addChild(*this->mNotes);
      end=false;
  }
  if(this->mAnnotation)
  {
      node.addChild(*this->mAnnotation);
      end=false;
  }
  unsigned int i,iMax=this->size();
  const ReactionGlyph* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const ReactionGlyph*>(this->get(i));
    assert(object);
    node.addChild(object->toXML());
  }
  if(end==true && iMax==0)
  {
    node.setEnd();
  }
  return node;
}


/**
 * Ctor.
 */
ListOfTextGlyphs::ListOfTextGlyphs(LayoutPkgNamespaces* layoutns)
  : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());
}


/**
 * Ctor.
 */
ListOfTextGlyphs::ListOfTextGlyphs(unsigned int level, unsigned int version, unsigned int pkgVersion)
  : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfTextGlyphs*
ListOfTextGlyphs::clone () const
{
  return new ListOfTextGlyphs(*this);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfTextGlyphs::getItemTypeCode () const
{
  return SBML_LAYOUT_TEXTGLYPH;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfTextGlyphs::getElementName () const
{
  static const std::string name = "listOfTextGlyphs";
  return name;
}


/* return nth item in list */
TextGlyph *
ListOfTextGlyphs::get(unsigned int n)
{
  return static_cast<TextGlyph*>(ListOf::get(n));
}


/* return nth item in list */
const TextGlyph *
ListOfTextGlyphs::get(unsigned int n) const
{
  return static_cast<const TextGlyph*>(ListOf::get(n));
}


/* return item by id */
TextGlyph*
ListOfTextGlyphs::get (const std::string& sid)
{
  return const_cast<TextGlyph*>( 
    static_cast<const ListOfTextGlyphs&>(*this).get(sid) );
}


/* return item by id */
const TextGlyph*
ListOfTextGlyphs::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<TextGlyph>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <TextGlyph*> (*result);
}


/* Removes the nth item from this list */
TextGlyph*
ListOfTextGlyphs::remove (unsigned int n)
{
   return static_cast<TextGlyph*>(ListOf::remove(n));
}


/* Removes item in this list by id */
TextGlyph*
ListOfTextGlyphs::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<TextGlyph>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <TextGlyph*> (item);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfTextGlyphs::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "textGlyph")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new TextGlyph(layoutns);
    appendAndOwn(object);
//    mItems.push_back(object);
  }

  return object;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfTextGlyphs::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfTextGlyphs", "http://projects.eml.org/bcb/sbml/level2", "");
  XMLAttributes att = XMLAttributes();
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  bool end=true;
  if(this->mNotes)
  {
      node.addChild(*this->mNotes);
      end=false;
  }
  if(this->mAnnotation)
  {
      node.addChild(*this->mAnnotation);
      end=false;
  }
  unsigned int i,iMax=this->size();
  const TextGlyph* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const TextGlyph*>(this->get(i));
    assert(object);
    node.addChild(object->toXML());
  }
  if(end==true && iMax==0)
  {
    node.setEnd();
  }
  return node;
}















/**
 * Creates a new Layout and returns a pointer to it.
 */
LIBSBML_EXTERN
Layout_t *
Layout_create (void)
{
  return new(std::nothrow) Layout;
}


/**
 * Creates a new Layout with the given id and returns a pointer to it.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createWith (const char *sid)
{
  LayoutPkgNamespaces layoutns;

  Dimensions* d=new Dimensions(&layoutns);
  Layout_t* l=new(std::nothrow) Layout(&layoutns, sid ? sid : "", d);
  delete d;
  return l;
}


/**
 * Creates a Layout object from a template.
 */
LIBSBML_EXTERN
    Layout_t *
Layout_createFrom (const Layout_t *temp)
{
  return new(std::nothrow) Layout(*temp);
}


/**
 * Creates a new Layout with the given width, height and depth and returns
 * a pointer to it.  The depth value defaults to 0.0.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createWithSize (const char *id,
                       double width, double height, double depth)
{
    LayoutPkgNamespaces layoutns;
    Dimensions* d=new Dimensions(&layoutns, width,height,depth);
    Layout_t* l=new (std::nothrow) Layout(&layoutns, id ? id : "", d);
    delete d;
    return l;
}


/**
 * Creates a new Layout with the given Dimensions and returns a pointer to
 * it.
 */
LIBSBML_EXTERN
Layout_t *
Layout_createWithDimensions (const char *id, const Dimensions_t *dimensions)
{
  LayoutPkgNamespaces layoutns;  
  return new (std::nothrow) Layout(&layoutns, id ? id : "", dimensions);
}


/** 
 * Frees the memory for the given layout.
 */
LIBSBML_EXTERN
void
Layout_free (Layout_t *l)
{
  delete l;
}



/**
 * Sets the dimensions of the layout.
 */
LIBSBML_EXTERN
void
Layout_setDimensions (Layout_t *l, const Dimensions_t *dimensions)
{
    static_cast<Layout*>(l)->setDimensions( dimensions );
}


/**
 * Adds a new compartment glyph to the list of compartment glyphs.
 */
LIBSBML_EXTERN
void
Layout_addCompartmentGlyph (Layout_t *l, CompartmentGlyph_t *cg)
{
  l->addCompartmentGlyph(cg);
}


/**
 * Adds a new species glyph to the list of species glyphs.
 */
LIBSBML_EXTERN
void
Layout_addSpeciesGlyph (Layout_t *l, SpeciesGlyph_t *sg)
{
  l->addSpeciesGlyph(sg);
}


/**
 * Adds a new reaction glyph to the list of reaction glyphs.
 */
LIBSBML_EXTERN
void
Layout_addReactionGlyph (Layout_t *l, ReactionGlyph_t *rg)
{
  l->addReactionGlyph(rg);
}


/**
 * Adds a new TextGlyph to the list of text glyphs.
 */
LIBSBML_EXTERN
void
Layout_addTextGlyph (Layout_t *l, TextGlyph_t *tg)
{
  l->addTextGlyph(tg);
}


/**
 * Adds a new GraphicalObject to the list of additional graphical objects.
 */
LIBSBML_EXTERN
void
Layout_addAdditionalGraphicalObject (Layout_t *l, GraphicalObject_t *go)
{
  l->addAdditionalGraphicalObject(go);
}


/**
 * Returns a pointer to the CompartmentGlyph with the given index.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
Layout_getCompartmentGlyph (Layout_t *l, unsigned int index)
{
  return l->getCompartmentGlyph(index);
}


/**
 * Returns a pointer to the SpeciesGlyph with the given index.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
Layout_getSpeciesGlyph (Layout_t *l, unsigned int index)
{
  return l->getSpeciesGlyph(index);
}


/**
 * Returns a pointer to the ReactionGlyph with the given index.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
Layout_getReactionGlyph (Layout_t *l, unsigned int index)
{
  return l->getReactionGlyph(index);
}


/**
 * Returns a pointer to the AdditionalGraphicalObject with the given index.
 */
LIBSBML_EXTERN
TextGlyph_t *
Layout_getTextGlyph (Layout_t *l, unsigned int index)
{
  return l->getTextGlyph(index);
}



/**
 * Returns a pointer to the GraphicalObject with the given index.
 */
LIBSBML_EXTERN
GraphicalObject_t *
Layout_getAdditionalGraphicalObject (Layout_t *l, unsigned int index)
{
  return l->getAdditionalGraphicalObject(index);
}


/**
 * Returns a pointer to the list of CompartmentGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfCompartmentGlyphs (Layout_t *l)
{
  return l->getListOfCompartmentGlyphs();
}


/**
 * Returns a pointer to the list of SpeciesGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfSpeciesGlyphs (Layout_t *l)
{
  return l->getListOfSpeciesGlyphs();
}


/**
 * Returns a pointer to the list of ReactionGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfReactionGlyphs (Layout_t *l)
{
  return l->getListOfReactionGlyphs();
}


/**
 * Returns a pointer to the list of TextGlyphs.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfTextGlyphs (Layout_t *l)
{
  return l->getListOfTextGlyphs();
}


/**
 * Returns a pointer to the list of additional GraphicalObjects.
 */
LIBSBML_EXTERN
ListOf_t *
Layout_getListOfAdditionalGraphicalObjects (Layout_t *l)
{
  return l->getListOfAdditionalGraphicalObjects();
}



/**
 * @return the dimensions of the layout
 */
LIBSBML_EXTERN
Dimensions_t*
Layout_getDimensions(Layout_t *l)
{
  return l->getDimensions();
}


/**
 * Returns the number of CompartmentGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumCompartmentGlyphs (const Layout_t *l)
{
  return l->getNumCompartmentGlyphs();
}


/**
 * Returns the number of SpeciesGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumSpeciesGlyphs (const Layout_t *l)
{
  return l->getNumSpeciesGlyphs();
}


/**
 * Returns the number of ReactionGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumReactionGlyphs (const Layout_t *l)
{
  return l->getNumReactionGlyphs();
}


/**
 * Returns the number of TextGlyphs.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumTextGlyphs (const Layout_t *l)
{
  return l->getNumTextGlyphs();
}


/**
 * Returns the number of additional GraphicalObject.
 */
LIBSBML_EXTERN
unsigned int
Layout_getNumAdditionalGraphicalObjects (const Layout_t *l)
{
  return l->getNumAdditionalGraphicalObjects();
}


/**
 * Does nothing since no defaults are defined for Layout.
 */ 
LIBSBML_EXTERN
void
Layout_initDefaults (Layout_t *l)
{
  l->initDefaults();
}



/**
 * Creates a ComparmentGlyph_t object, adds it to the end of the
 * compartment glyphs objects list and returns a pointer to the newly
 * created object.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
Layout_createCompartmentGlyph (Layout_t *l)
{
  return l->createCompartmentGlyph();
}


/**
 * Creates a SpeciesGlyph object, adds it to the end of the species glyphs
 * objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
SpeciesGlyph_t *
Layout_createSpeciesGlyph (Layout_t *l)
{
  return l->createSpeciesGlyph();
}


/**
 * Creates a ReactionGlyph_t object, adds it to the end of the reaction
 * glyphs objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
Layout_createReactionGlyph (Layout_t *l)
{
  return l->createReactionGlyph();
}

/**
 * Creates a GeneralGlyph_t object, adds it to the end of the additional
 * objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
GeneralGlyph_t *
Layout_createGeneralGlyph (Layout_t *l)
{
  return l->createGeneralGlyph();
}



/**
 * Creates a TextGlyph_t object, adds it to the end of the text glyphs
 * objects list and returns a pointer to the newly created object.
 */
LIBSBML_EXTERN
TextGlyph_t *
Layout_createTextGlyph (Layout_t *l)
{
  return l->createTextGlyph();
}


/**
 * Creates a GraphicalObject object, adds it to the end of the additional
 * graphical objects list and returns a pointer to the newly created
 * object.
 */
LIBSBML_EXTERN
GraphicalObject_t *
Layout_createAdditionalGraphicalObject (Layout_t *l)
{
  return l->createAdditionalGraphicalObject();
}

/**
 * Remove the compartment glyph with the given index.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
CompartmentGlyph_t*
Layout_removeCompartmentGlyph(Layout_t* l, unsigned int index)
{
    return l->removeCompartmentGlyph(index);
}

/**
 * Remove the species glyph with the given index.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
SpeciesGlyph_t*
Layout_removeSpeciesGlyph(Layout_t* l, unsigned int index)
{
    return l->removeSpeciesGlyph(index);
}

/**
 * Remove the reaction glyph with the given index.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
ReactionGlyph_t*
Layout_removeReactionGlyph(Layout_t* l, unsigned int index)
{
    return l->removeReactionGlyph(index);
}

/**
 * Remove the text glyph with the given index.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
TextGlyph_t*
Layout_removeTextGlyph(Layout_t* l, unsigned int index)
{
    return l->removeTextGlyph(index);
}

/**
 * Remove the graphical object with the given index.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
GraphicalObject_t*
Layout_removeAdditionalGraphicalObject(Layout_t* l, unsigned int index)
{
    return l->removeAdditionalGraphicalObject(index);
}

/**
 * Remove the compartment glyph with the given id.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
CompartmentGlyph_t*
Layout_removeCompartmentGlyphWithId(Layout_t* l, const char* id)
{
    return l->removeCompartmentGlyph(id);
}

/**
 * Remove the species glyph with the given id.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
SpeciesGlyph_t*
Layout_removeSpeciesGlyphWithId(Layout_t* l, const char* id)
{
    return l->removeSpeciesGlyph(id);
}

/**
 * Remove the reaction glyph with the given id.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
ReactionGlyph_t*
Layout_removeReactionGlyphWithId(Layout_t* l, const char* id)
{
    return l->removeReactionGlyph(id);
}

/**
 * Remove the text glyph with the given id.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
TextGlyph_t*
Layout_removeTextGlyphWithId(Layout_t* l, const char* id)
{
    return l->removeTextGlyph(id);
}

/**
 * Remove the species reference glyph with the given id.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t*
Layout_removeSpeciesReferenceGlyphWithId(Layout_t* l, const char* id)
{
    return l->removeSpeciesReferenceGlyph(id);
}

/**
 * Remove the graphical object with the given id.
 * A pointer to the removed object is returned. If no object was removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
GraphicalObject_t*
Layout_removeAdditionalGraphicalObjectWithId(Layout_t* l, const char* id)
{
    return l->removeAdditionalGraphicalObject(id);
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
Layout_t *
Layout_clone (const Layout_t *m)
{
  return static_cast<Layout*>( m->clone() );
}


/**
 * Returns non-zero if the id is set
 */
LIBSBML_EXTERN
int
Layout_isSetId (const Layout_t *l)
{
  return static_cast <int> (l->isSetId());
}

/**
 * Returns the id
 */
LIBSBML_EXTERN
const char *
Layout_getId (const Layout_t *l)
{
  return l->isSetId() ? l->getId().c_str() : NULL;
}

/**
 * Sets the id
 */
LIBSBML_EXTERN
int
Layout_setId (Layout_t *l, const char *sid)
{
  return (sid == NULL) ? l->setId("") : l->setId(sid);
}

/**
 * Unsets the id
 */
LIBSBML_EXTERN
void
Layout_unsetId (Layout_t *l)
{
  l->unsetId();
}

LIBSBML_CPP_NAMESPACE_END

