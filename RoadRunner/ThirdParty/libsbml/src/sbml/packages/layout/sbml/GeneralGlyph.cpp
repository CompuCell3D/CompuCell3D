/**
 * Filename    : GeneralGlyph.cpp
 * Description : SBML Layout GeneralGlyph source
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
 * European Media Laboratories Research gGmbH have no obligations to
 * provide maintenance, support, updates, enhancements or modifications.
 * In no event shall the European Media Laboratories Research gGmbH be
 * liable to any party for direct, indirect, special, incidental or
 * consequential damages, including lost profits, arising out of the use of
 * this software and its documentation, even if the European Media
 * Laboratories Research gGmbH have been advised of the possibility of such
 * damage.  See the GNU Lesser General Public License for more details.
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

#include <assert.h>
#include <limits>
#include <sbml/packages/layout/sbml/GeneralGlyph.h>
#include <sbml/packages/layout/sbml/ReferenceGlyph.h>
#include <sbml/packages/layout/sbml/SpeciesGlyph.h>
#include <sbml/packages/layout/sbml/TextGlyph.h>
#include <sbml/packages/layout/sbml/CompartmentGlyph.h>
#include <sbml/packages/layout/sbml/ReactionGlyph.h>
#include <sbml/packages/layout/sbml/GraphicalObject.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

LIBSBML_CPP_NAMESPACE_BEGIN


/**
 * Creates a new GeneralGlyph.  The list of reference and sub glyph is
 * empty and the id of the associated element is set to the empty string.
 */
GeneralGlyph::GeneralGlyph(unsigned int level, unsigned int version, unsigned int pkgVersion) 
 : GraphicalObject (level,version,pkgVersion)
  ,mReference("")
  ,mReferenceGlyphs(level,version,pkgVersion)
  ,mSubGlyphs(level,version,pkgVersion)
  ,mCurve(level,version,pkgVersion)
{
  mSubGlyphs.setElementName("listOfSubGlyphs");

  connectToChild();
  //
  // (NOTE) Developers don't have to invoke setSBMLNamespacesAndOwn function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (GraphicalObject).
  //
  //setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
}


/**
 * Creates a new GeneralGlyph with the given LayoutPkgNamespaces
 */
GeneralGlyph::GeneralGlyph(LayoutPkgNamespaces* layoutns)
 : GraphicalObject (layoutns)
  ,mReference("")
  ,mReferenceGlyphs(layoutns)
  ,mSubGlyphs(layoutns)
  ,mCurve(layoutns)
{
  mSubGlyphs.setElementName("listOfSubGlyphs");
  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());


  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Creates a GeneralGlyph with the given id.
 */
GeneralGlyph::GeneralGlyph (LayoutPkgNamespaces* layoutns, const std::string& id)
  : GraphicalObject(layoutns, id)
   ,mReference("")
   ,mReferenceGlyphs(layoutns)
   ,mSubGlyphs(layoutns)
   ,mCurve(layoutns)
{
  mSubGlyphs.setElementName("listOfSubGlyphs");

  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Creates a GeneralGlyph with the given id and set the id of the
 * associated reaction to the second argument.
 */
GeneralGlyph::GeneralGlyph (LayoutPkgNamespaces* layoutns, const std::string& id,
                              const std::string& referenceId) 
  : GraphicalObject( layoutns, id  )
   ,mReference      ( referenceId  )
   ,mReferenceGlyphs(layoutns)
   ,mSubGlyphs(layoutns)
   ,mCurve(layoutns)
{
  mSubGlyphs.setElementName("listOfSubGlyphs");

  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Creates a new GeneralGlyph from the given XMLNode
 */
GeneralGlyph::GeneralGlyph(const XMLNode& node, unsigned int l2version)
  : GraphicalObject(2,l2version)
   ,mReference      ("")
   ,mReferenceGlyphs(2,l2version)
   ,mSubGlyphs(2,l2version)
   ,mCurve(2,l2version)
{
  mSubGlyphs.setElementName("listOfSubGlyphs");
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
        if(childName=="boundingBox")
        {
            this->mBoundingBox=BoundingBox(*child);
        }
        else if(childName=="annotation")
        {
            this->mAnnotation=new XMLNode(*child);
        }
        else if(childName=="notes")
        {
            this->mNotes=new XMLNode(*child);
        }
        else if(childName=="curve")
        {
            // since the copy constructor of ListOf does not make deep copies
            // of the objects, we have to add the individual curveSegments to the 
            // curve instead of just copying the whole curve.
            Curve* pTmpCurve=new Curve(*child);
            unsigned int i,iMax=pTmpCurve->getNumCurveSegments();
            for(i=0;i<iMax;++i)
            {
                this->mCurve.addCurveSegment(pTmpCurve->getCurveSegment(i));
            }
            // we also have to copy mAnnotations, mNotes, mCVTerms and mHistory
            if(pTmpCurve->isSetNotes()) this->mCurve.setNotes(new XMLNode(*pTmpCurve->getNotes()));
            if(pTmpCurve->isSetAnnotation()) this->mCurve.setAnnotation(new XMLNode(*pTmpCurve->getAnnotation()));
            if(pTmpCurve->getCVTerms()!=NULL)
            {
              iMax=pTmpCurve->getCVTerms()->getSize(); 
              for(i=0;i<iMax;++i)
              {
                this->mCurve.getCVTerms()->add(static_cast<CVTerm*>(pTmpCurve->getCVTerms()->get(i))->clone());
              }
            }
            delete pTmpCurve;
        }
        else if(childName=="listOfReferenceGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                if(innerChildName=="referenceGlyph")
                {
                    this->mReferenceGlyphs.appendAndOwn(new ReferenceGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    this->mReferenceGlyphs.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    this->mReferenceGlyphs.setNotes(new XMLNode(*innerChild));
                }
                else
                {
                    // throw
                }
                ++i;
            }
        }
        else if(childName=="listOfSubGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                ListOf& list=this->mSubGlyphs;
                if(innerChildName=="graphicalObject")
                {
                    list.appendAndOwn(new GraphicalObject(*innerChild));
                }
                else if(innerChildName=="textGlyph")
                {
                    list.appendAndOwn(new TextGlyph(*innerChild));
                }
                else if(innerChildName=="reactionGlyph")
                {
                    list.appendAndOwn(new ReactionGlyph(*innerChild));
                }
                else if(innerChildName=="speciesGlyph")
                {
                    list.appendAndOwn(new SpeciesGlyph(*innerChild));
                }
                else if(innerChildName=="compartmentGlyph")
                {
                    list.appendAndOwn(new CompartmentGlyph(*innerChild));
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

  connectToChild();
}

/**
 * Copy constructor.
 */
GeneralGlyph::GeneralGlyph(const GeneralGlyph& source):GraphicalObject(source)
{
    this->mReference=source.getReferenceId();
    this->mCurve=*source.getCurve();
    this->mReferenceGlyphs=*source.getListOfReferenceGlyphs();
    this->mSubGlyphs=*source.getListOfSubGlyphs();

    connectToChild();
}

/**
 * Assignment operator.
 */
GeneralGlyph& GeneralGlyph::operator=(const GeneralGlyph& source)
{
  if(&source!=this)
  {
    GraphicalObject::operator=(source);
    this->mReference=source.mReference;
    this->mCurve=*source.getCurve();
    this->mReferenceGlyphs=*source.getListOfReferenceGlyphs();
    this->mSubGlyphs=*source.getListOfSubGlyphs();
    connectToChild();
  }
  
  return *this;
}



/**
 * Destructor.
 */ 
GeneralGlyph::~GeneralGlyph ()
{
} 


/**
 * Returns the id of the associated reaction.
 */  
const std::string&
GeneralGlyph::getReferenceId () const
{
  return this->mReference;
}


/**
 * Sets the id of the associated reaction.
 */ 
int
GeneralGlyph::setReferenceId (const std::string& id)
{
  if (!(SyntaxChecker::isValidInternalSId(id)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mReference = id;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/**
 * Returns true if the id of the associated reaction is not the empty
 * string.
 */ 
bool
GeneralGlyph::isSetReferenceId() const
{
  return ! this->mReference.empty();
}


/**
 * Returns the ListOf object that hold the reference glyphs.
 */  
const ListOfReferenceGlyphs*
GeneralGlyph::getListOfReferenceGlyphs () const
{
  return &this->mReferenceGlyphs;
}


/**
 * Returns the ListOf object that hold the reference glyphs.
 */  
ListOfReferenceGlyphs*
GeneralGlyph::getListOfReferenceGlyphs ()
{
  return &this->mReferenceGlyphs;
}

/**
 * Returns the ListOf object that hold the subglyphs.
 */  
const ListOfGraphicalObjects*
GeneralGlyph::getListOfSubGlyphs () const
{
  return &this->mSubGlyphs;
}


/**
 * Returns the ListOf object that hold the subglyphs.
 */  
ListOfGraphicalObjects*
GeneralGlyph::getListOfSubGlyphs ()
{
  return &this->mSubGlyphs;
}


/**
 * Returns the reference glyph with the given index.  If the index
 * is invalid, NULL is returned.
 */ 
ReferenceGlyph*
GeneralGlyph::getReferenceGlyph (unsigned int index) 
{
  return static_cast<ReferenceGlyph*>
  (
    this->mReferenceGlyphs.get(index)
  );
}


/**
 * Returns the reference glyph with the given index.  If the index
 * is invalid, NULL is returned.
 */ 
const ReferenceGlyph*
GeneralGlyph::getReferenceGlyph (unsigned int index) const
{
  return static_cast<const ReferenceGlyph*>
  (
    this->mReferenceGlyphs.get(index)
  );
}

/**
 * Returns the reference glyph with the given index.  If the index
 * is invalid, NULL is returned.
 */ 
GraphicalObject*
GeneralGlyph::getSubGlyph (unsigned int index) 
{
  return static_cast<GraphicalObject*>
  (
    this->mSubGlyphs.get(index)
  );
}


/**
 * Returns the reference glyph with the given index.  If the index
 * is invalid, NULL is returned.
 */ 
const GraphicalObject*
GeneralGlyph::getSubGlyph (unsigned int index) const
{
  return static_cast<const GraphicalObject*>
  (
    this->mSubGlyphs.get(index)
  );
}

/**
 * Adds a new reference glyph to the list.
 */
void
GeneralGlyph::addReferenceGlyph (const ReferenceGlyph* glyph)
{
  this->mReferenceGlyphs.append(glyph);
}

/**
 * Adds a new subglyph to the list.
 */
void
  GeneralGlyph::addSubGlyph (const GraphicalObject* glyph)
{
  this->mSubGlyphs.append(glyph);
}


/**
 * Returns the number of reference glyph objects.
 */ 
unsigned int
GeneralGlyph::getNumReferenceGlyphs () const
{
  return this->mReferenceGlyphs.size();
}


/**
 * Returns the number of subglyph objects.
 */ 
unsigned int
GeneralGlyph::getNumSubGlyphs () const
{
  return this->mSubGlyphs.size();
}

/**
 * Calls initDefaults from GraphicalObject.
 */ 
void GeneralGlyph::initDefaults ()
{
  GraphicalObject::initDefaults();
}


/**
 * Returns the curve object for the glyph
 */ 
const Curve*
GeneralGlyph::getCurve () const
{
  return &this->mCurve;
}

/**
 * Returns the curve object for the glyph
 */ 
Curve*
GeneralGlyph::getCurve () 
{
  return &this->mCurve;
}


/**
 * Sets the curve object for the reaction glyph.
 */ 
void GeneralGlyph::setCurve (const Curve* curve)
{
  if(!curve) return;
  this->mCurve = *curve;
  this->mCurve.connectToParent(this);
}


/**
 * Returns true if the curve consists of one or more segments.
 */ 
bool GeneralGlyph::isSetCurve () const
{
  return this->mCurve.getNumCurveSegments() > 0;
}


/**
 * Creates a new ReferenceGlyph object, adds it to the end of the
 * list of reference objects and returns a reference to the newly
 * created object.
 */
ReferenceGlyph*
GeneralGlyph::createReferenceGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  ReferenceGlyph* srg = new ReferenceGlyph(layoutns);

  this->mReferenceGlyphs.appendAndOwn(srg);
  return srg;
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LineSegment*
GeneralGlyph::createLineSegment ()
{
  return this->mCurve.createLineSegment();
}

 
/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
CubicBezier*
GeneralGlyph::createCubicBezier ()
{
  return this->mCurve.createCubicBezier();
}

/**
 * Remove the reference glyph with the given index.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
ReferenceGlyph*
GeneralGlyph::removeReferenceGlyph(unsigned int index)
{
    ReferenceGlyph* srg=NULL;
    if(index < this->getNumReferenceGlyphs())
    {
        srg=dynamic_cast<ReferenceGlyph*>(this->getListOfReferenceGlyphs()->remove(index));
    }
    return srg;
}

/**
 * Remove the reference glyph with the given id.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
ReferenceGlyph*
GeneralGlyph::removeReferenceGlyph(const std::string& id)
{
    ReferenceGlyph* srg=NULL;
    unsigned int index=this->getIndexForReferenceGlyph(id);
    if(index!=std::numeric_limits<unsigned int>::max())
    {
        srg=this->removeReferenceGlyph(index);
    }
    return srg;
}


/**
 * Remove the subglyph with the given index.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
GraphicalObject*
GeneralGlyph::removeSubGlyph(unsigned int index)
{
    GraphicalObject* srg=NULL;
    if(index < this->getNumSubGlyphs())
    {
        srg=dynamic_cast<GraphicalObject*>(this->getListOfSubGlyphs()->remove(index));
    }
    return srg;
}

/**
 * Remove the subglyph with the given id.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
GraphicalObject*
GeneralGlyph::removeSubGlyph(const std::string& id)
{
    GraphicalObject* srg=NULL;
    unsigned int index=this->getIndexForSubGlyph(id);
    if(index!=std::numeric_limits<unsigned int>::max())
    {
        srg=this->removeSubGlyph(index);
    }
    return srg;
}

/**
 * Returns the index of the reference glyph with the given id.
 * If the reaction glyph does not contain a reference glyph with this
 * id, numreic_limits<int>::max() is returned.
 */
unsigned int
GeneralGlyph::getIndexForReferenceGlyph(const std::string& id) const
{
    unsigned int i,iMax=this->getNumReferenceGlyphs();
    unsigned int index=std::numeric_limits<unsigned int>::max();
    for(i=0;i<iMax;++i)
    {
        const ReferenceGlyph* srg=this->getReferenceGlyph(i);
        if(srg->getId()==id)
        {
            index=i;
            break;
        }
    }
    return index;
}



/**
 * Returns the index of the subglyph with the given id.
 * If the reaction glyph does not contain a subglyph with this
 * id, numreic_limits<int>::max() is returned.
 */
unsigned int
GeneralGlyph::getIndexForSubGlyph(const std::string& id) const
{
    unsigned int i,iMax=this->getNumSubGlyphs();
    unsigned int index=std::numeric_limits<unsigned int>::max();
    for(i=0;i<iMax;++i)
    {
      const GraphicalObject* srg=this->getSubGlyph(i);
        if(srg->getId()==id)
        {
            index=i;
            break;
        }
    }
    return index;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& GeneralGlyph::getElementName () const 
{
  static const std::string name = "generalGlyph";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
GeneralGlyph* 
GeneralGlyph::clone () const
{
    return new GeneralGlyph(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
GeneralGlyph::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  
  SBase*        object = 0;

  if (name == "listOfReferenceGlyphs")
  {
    object = &mReferenceGlyphs;
  }
  else if (name == "listOfSubGlyphs")
  {
    object = &mSubGlyphs;
  }
  else if(name=="curve")
  {
    object = &mCurve;
  }
  else
  {
    object=GraphicalObject::createObject(stream);
  }
  
  return object;
}

/**
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
GeneralGlyph::addExpectedAttributes(ExpectedAttributes& attributes)
{
  GraphicalObject::addExpectedAttributes(attributes);

  attributes.add("reference");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void GeneralGlyph::readAttributes (const XMLAttributes& attributes,
                                    const ExpectedAttributes& expectedAttributes)
{
  GraphicalObject::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("reference", mReference, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mReference.empty())
  {
    logEmptyString(mReference, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mReference)) logError(InvalidIdSyntax);
}

/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
GeneralGlyph::writeElements (XMLOutputStream& stream) const
{
  if(this->isSetCurve())
  {
    SBase::writeElements(stream);
    mCurve.write(stream);
    //
    // BoundingBox is to be ignored if a curve element defined.
    //
  }
  else
  {
    //
    // SBase::writeElements(stream) is invoked in the function below.
    //
    GraphicalObject::writeElements(stream);
  }

  if ( getNumReferenceGlyphs() > 0 ) mReferenceGlyphs.write(stream);
  if ( getNumSubGlyphs() > 0 ) mSubGlyphs.write(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}



/**
 * Subclasses should override this method to write their XML attributes
 * to the XMLOutputStream.  Be sure to call your parents implementation
 * of this method as well.  For example:
 *
 *   SBase::writeAttributes(stream);
 *   stream.writeAttribute( "id"  , mId   );
 *   stream.writeAttribute( "name", mName );
 *   ...
 */
void GeneralGlyph::writeAttributes (XMLOutputStream& stream) const
{
  GraphicalObject::writeAttributes(stream);
  if(this->isSetReferenceId())
  {
    stream.writeAttribute("reference", getPrefix(), mReference);
  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}


/**
 * Returns the package type code for this object.
 */
int
GeneralGlyph::getTypeCode () const
{
  return SBML_LAYOUT_GENERALGLYPH;
}


/**
 * Creates an XMLNode object from this.
 */
XMLNode GeneralGlyph::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("generalGlyph", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  addGraphicalObjectAttributes(*this,att);
  if(this->isSetReferenceId()) att.add("reference",this->mReference);
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  if(this->mNotes) node.addChild(*this->mNotes);
  if(this->mAnnotation) node.addChild(*this->mAnnotation);
  if(this->mCurve.getNumCurveSegments()==0)
  {
    // write the bounding box
    node.addChild(this->mBoundingBox.toXML());
  }
  else
  {
    // add the curve
    node.addChild(this->mCurve.toXML());
  }
  // add the list of reference glyphs
  if(this->mReferenceGlyphs.size()>0)
  {
    node.addChild(this->mReferenceGlyphs.toXML());
  }
  // add the list of reference glyphs
  if(this->mSubGlyphs.size()>0)
  {
    node.addChild(this->mSubGlyphs.toXML());
  }
  return node;
}



/**
 * Ctor.
 */
ListOfReferenceGlyphs::ListOfReferenceGlyphs(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * Ctor.
 */
ListOfReferenceGlyphs::ListOfReferenceGlyphs(LayoutPkgNamespaces* layoutns)
 : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());
}


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfReferenceGlyphs*
ListOfReferenceGlyphs::clone () const
{
  return new ListOfReferenceGlyphs(*this);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfReferenceGlyphs::getItemTypeCode () const
{
  return SBML_LAYOUT_REFERENCEGLYPH;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfReferenceGlyphs::getElementName () const
{
  static const std::string name = "listOfReferenceGlyphs";
  return name;
}


/* return nth item in list */
ReferenceGlyph *
ListOfReferenceGlyphs::get(unsigned int n)
{
  return static_cast<ReferenceGlyph*>(ListOf::get(n));
}


/* return nth item in list */
const ReferenceGlyph *
ListOfReferenceGlyphs::get(unsigned int n) const
{
  return static_cast<const ReferenceGlyph*>(ListOf::get(n));
}


/* return item by id */
ReferenceGlyph*
ListOfReferenceGlyphs::get (const std::string& sid)
{
  return const_cast<ReferenceGlyph*>( 
    static_cast<const ListOfReferenceGlyphs&>(*this).get(sid) );
}


/* return item by id */
const ReferenceGlyph*
ListOfReferenceGlyphs::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<ReferenceGlyph>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <ReferenceGlyph*> (*result);
}


/* Removes the nth item from this list */
ReferenceGlyph*
ListOfReferenceGlyphs::remove (unsigned int n)
{
   return static_cast<ReferenceGlyph*>(ListOf::remove(n));
}


/* Removes item in this list by id */
ReferenceGlyph*
ListOfReferenceGlyphs::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<ReferenceGlyph>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <ReferenceGlyph*> (item);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfReferenceGlyphs::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*             object = NULL;


  if (name == "referenceGlyph")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new ReferenceGlyph(layoutns);
    appendAndOwn(object);
  }

  return object;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfReferenceGlyphs::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfReferenceGlyphs", "http://projects.eml.org/bcb/sbml/level2", "");
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
  const ReferenceGlyph* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const ReferenceGlyph*>(this->get(i));
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
 * Accepts the given SBMLVisitor.
 
bool
GeneralGlyph::accept (SBMLVisitor& v) const
{
  bool result=v.visit(*this);
  if(this->mCurve.getNumCurveSegments()>0)
  {
    this->mCurve.accept(v);
  }
  else
  {
    this->mBoundingBox.accept(v);
  }
  this->mReferenceGlyphs.accept(this);
  v.leave(*this);
  return result;
}
*/


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
GeneralGlyph::setSBMLDocument (SBMLDocument* d)
{
  GraphicalObject::setSBMLDocument(d);

  mReferenceGlyphs.setSBMLDocument(d);
  mSubGlyphs.setSBMLDocument(d);
  mCurve.setSBMLDocument(d);
}

/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
GeneralGlyph::connectToChild()
{
  mReferenceGlyphs.connectToParent(this);
  mCurve.connectToParent(this);
}

/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
GeneralGlyph::enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mReferenceGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mSubGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mCurve.enablePackageInternal(pkgURI,pkgPrefix,flag);
}



/**
 * Creates a new GeneralGlyph and returns the pointer to it.
 */
LIBSBML_EXTERN
GeneralGlyph_t *
GeneralGlyph_create (void)
{
  return new(std::nothrow) GeneralGlyph;
}


/**
 * Creates a new GeneralGlyph with the given id
 */
LIBSBML_EXTERN
GeneralGlyph_t *
GeneralGlyph_createWith (const char *sid)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) GeneralGlyph(&layoutns, sid ? sid : "", "");
}


/**
 * Creates a new GeneralGlyph referencing the give reaction.
 */
LIBSBML_EXTERN
GeneralGlyph_t *
GeneralGlyph_createWithReferenceId (const char *id, const char *referenceId)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) GeneralGlyph(&layoutns, id ? id : "", referenceId ? referenceId : "");
}


/**
 * Creates a new GeneralGlyph object from a template.
 */
LIBSBML_EXTERN
GeneralGlyph_t *
GeneralGlyph_createFrom (const GeneralGlyph_t *temp)
{
  return new(std::nothrow) GeneralGlyph(*temp);
}


/**
 * Frees the memory taken up by the attributes.
 */
LIBSBML_EXTERN
void
GeneralGlyph_free (GeneralGlyph_t *rg)
{
  delete rg;
}


/**
 * Sets the reference reaction for the reaction glyph.
 */
LIBSBML_EXTERN
void
GeneralGlyph_setReferenceId (GeneralGlyph_t *rg,const char *id)
{
    static_cast<GeneralGlyph*>(rg)->setReferenceId( id ? id : "" );
}


/**
 * Gets the reference reactions id for the given reaction glyph.
 */
LIBSBML_EXTERN
const char *
GeneralGlyph_getReferenceId (const GeneralGlyph_t *rg)
{
    return rg->isSetReferenceId() ? rg->getReferenceId().c_str() : NULL;
}


/**
 * Returns 0 if the reference reaction has not been set for this glyph and
 * 1 otherwise.
 */
LIBSBML_EXTERN
int
GeneralGlyph_isSetReferenceId (const GeneralGlyph_t *rg)
{
  return static_cast<int>( rg->isSetReferenceId() );
}


/**
 * Add a ReferenceGlyph object to the list of
 * ReferenceGlyphs.
 */
LIBSBML_EXTERN
void
GeneralGlyph_addReferenceGlyph (GeneralGlyph_t         *rg,
                                        ReferenceGlyph_t *srg)
{
  rg->addReferenceGlyph(srg);
}


/**
 * Returns the number of ReferenceGlyphs for the GeneralGlyph.
 */
LIBSBML_EXTERN
unsigned int
GeneralGlyph_getNumReferenceGlyphs (const GeneralGlyph_t *rg)
{
  return rg->getNumReferenceGlyphs();
}


/**
 * Returns the pointer to the ReferenceGlyphs for the given index.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
GeneralGlyph_getReferenceGlyph (GeneralGlyph_t *rg,
                                        unsigned int           index)
{
  return rg->getReferenceGlyph(index);
}


/**
 * Returns the list object that holds all reference glyphs.
 */ 
LIBSBML_EXTERN
ListOf_t *
GeneralGlyph_getListOfReferenceGlyphs (GeneralGlyph_t *rg)
{
  return rg->getListOfReferenceGlyphs();
}


/**
 * Calls initDefaults from GraphicalObject.
 */ 
LIBSBML_EXTERN
void
GeneralGlyph_initDefaults (GeneralGlyph_t *rg)
{
  rg->initDefaults();
}


/**
 * Sets the curve for the reaction glyph.
 */
LIBSBML_EXTERN
void
GeneralGlyph_setCurve (GeneralGlyph_t *rg, Curve_t *c)
{
  rg->setCurve(c);
}


/**
 * Gets the Curve for the given reaction glyph.
 */
LIBSBML_EXTERN
Curve_t *
GeneralGlyph_getCurve (GeneralGlyph_t *rg)
{
  return rg->getCurve();
}


/**
 * Returns true if the Curve has one or more LineSegment.
 */
LIBSBML_EXTERN
int
GeneralGlyph_isSetCurve (GeneralGlyph_t *rg)
{
  return static_cast<int>( rg->isSetCurve() );
}


/**
 * Creates a new ReferenceGlyph_t object, adds it to the end of the
 * list of reference objects and returns a pointer to the newly
 * created object.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
GeneralGlyph_createReferenceGlyph (GeneralGlyph_t *rg)
{
  return rg->createReferenceGlyph();
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
LineSegment_t *
GeneralGlyph_createLineSegment (GeneralGlyph_t *rg)
{
  return rg->getCurve()->createLineSegment();
}


/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
CubicBezier_t *
GeneralGlyph_createCubicBezier (GeneralGlyph_t *rg)
{
  return rg->getCurve()->createCubicBezier();
}


/**
 * Remove the reference glyph with the given index.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
ReferenceGlyph_t*
GeneralGlyph_removeReferenceGlyph(GeneralGlyph_t* rg,unsigned int index)
{
    return rg->removeReferenceGlyph(index);
}

/**
 * Remove the reference glyph with the given id.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
ReferenceGlyph_t*
GeneralGlyph_removeReferenceGlyphWithId(GeneralGlyph_t* rg,const char* id)
{
    return rg->removeReferenceGlyph(id);
}

/**
 * Returns the index of the reference glyph with the given id.
 * If the reaction glyph does not contain a reference glyph with this
 * id, UINT_MAX from limits.h is returned.
 */
LIBSBML_EXTERN
unsigned int
GeneralGlyph_getIndexForReferenceGlyph(GeneralGlyph_t* rg,const char* id)
{
    return rg->getIndexForReferenceGlyph(id);
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
GeneralGlyph_t *
GeneralGlyph_clone (const GeneralGlyph_t *m)
{
  return static_cast<GeneralGlyph*>( m->clone() );
}


LIBSBML_CPP_NAMESPACE_END
