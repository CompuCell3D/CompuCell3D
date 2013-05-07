/**
 * Filename    : ReactionGlyph.cpp
 * Description : SBML Layout ReactionGlyph source
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
#include <sbml/packages/layout/sbml/ReactionGlyph.h>
#include <sbml/packages/layout/sbml/SpeciesReferenceGlyph.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

LIBSBML_CPP_NAMESPACE_BEGIN


/**
 * Creates a new ReactionGlyph.  The list of species reference glyph is
 * empty and the id of the associated reaction is set to the empty string.
 */
ReactionGlyph::ReactionGlyph(unsigned int level, unsigned int version, unsigned int pkgVersion) 
 : GraphicalObject (level,version,pkgVersion)
  ,mReaction("")
  ,mSpeciesReferenceGlyphs(level,version,pkgVersion)
  ,mCurve(level,version,pkgVersion)
{
  connectToChild();
  //
  // (NOTE) Developers don't have to invoke setSBMLNamespacesAndOwn function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (GraphicalObject).
  //
  //setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
}


/**
 * Creates a new ReactionGlyph with the given LayoutPkgNamespaces
 */
ReactionGlyph::ReactionGlyph(LayoutPkgNamespaces* layoutns)
 : GraphicalObject (layoutns)
  ,mReaction("")
  ,mSpeciesReferenceGlyphs(layoutns)
  ,mCurve(layoutns)
{
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
 * Creates a ReactionGlyph with the given id.
 */
ReactionGlyph::ReactionGlyph (LayoutPkgNamespaces* layoutns, const std::string& id)
  : GraphicalObject(layoutns, id)
   ,mReaction("")
   ,mSpeciesReferenceGlyphs(layoutns)
   ,mCurve(layoutns)
{
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
 * Creates a ReactionGlyph with the given id and set the id of the
 * associated reaction to the second argument.
 */
ReactionGlyph::ReactionGlyph (LayoutPkgNamespaces* layoutns, const std::string& id,
                              const std::string& reactionId) 
  : GraphicalObject( layoutns, id  )
   ,mReaction      ( reactionId  )
   ,mSpeciesReferenceGlyphs(layoutns)
   ,mCurve(layoutns)
{
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
 * Creates a new ReactionGlyph from the given XMLNode
 */
ReactionGlyph::ReactionGlyph(const XMLNode& node, unsigned int l2version)
  : GraphicalObject(2,l2version)
   ,mReaction      ("")
   ,mSpeciesReferenceGlyphs(2,l2version)
   ,mCurve(2,l2version)
{
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
        else if(childName=="listOfSpeciesReferenceGlyphs")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                if(innerChildName=="speciesReferenceGlyph")
                {
                    this->mSpeciesReferenceGlyphs.appendAndOwn(new SpeciesReferenceGlyph(*innerChild));
                }
                else if(innerChildName=="annotation")
                {
                    this->mSpeciesReferenceGlyphs.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    this->mSpeciesReferenceGlyphs.setNotes(new XMLNode(*innerChild));
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
ReactionGlyph::ReactionGlyph(const ReactionGlyph& source):GraphicalObject(source)
{
    this->mReaction=source.getReactionId();
    this->mCurve=*source.getCurve();
    this->mSpeciesReferenceGlyphs=*source.getListOfSpeciesReferenceGlyphs();

    connectToChild();
}

/**
 * Assignment operator.
 */
ReactionGlyph& ReactionGlyph::operator=(const ReactionGlyph& source)
{
  if(&source!=this)
  {
    GraphicalObject::operator=(source);
    this->mReaction=source.getReactionId();
    this->mCurve=*source.getCurve();
    this->mSpeciesReferenceGlyphs=*source.getListOfSpeciesReferenceGlyphs();
    connectToChild();
  }
  
  return *this;
}



/**
 * Destructor.
 */ 
ReactionGlyph::~ReactionGlyph ()
{
} 


/**
 * Returns the id of the associated reaction.
 */  
const std::string&
ReactionGlyph::getReactionId () const
{
  return this->mReaction;
}


/**
 * Sets the id of the associated reaction.
 */ 
int
ReactionGlyph::setReactionId (const std::string& id)
{
  if (!(SyntaxChecker::isValidInternalSId(id)))
  {
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
  else
  {
    mReaction = id;
    return LIBSBML_OPERATION_SUCCESS;
  }
}


/**
 * Returns true if the id of the associated reaction is not the empty
 * string.
 */ 
bool
ReactionGlyph::isSetReactionId() const
{
  return ! this->mReaction.empty();
}


/**
 * Returns the ListOf object that hold the species reference glyphs.
 */  
const ListOfSpeciesReferenceGlyphs*
ReactionGlyph::getListOfSpeciesReferenceGlyphs () const
{
  return &this->mSpeciesReferenceGlyphs;
}


/**
 * Returns the ListOf object that hold the species reference glyphs.
 */  
ListOfSpeciesReferenceGlyphs*
ReactionGlyph::getListOfSpeciesReferenceGlyphs ()
{
  return &this->mSpeciesReferenceGlyphs;
}

/**
 * Returns the species reference glyph with the given index.  If the index
 * is invalid, NULL is returned.
 */ 
SpeciesReferenceGlyph*
ReactionGlyph::getSpeciesReferenceGlyph (unsigned int index) 
{
  return static_cast<SpeciesReferenceGlyph*>
  (
    this->mSpeciesReferenceGlyphs.get(index)
  );
}


/**
 * Returns the species reference glyph with the given index.  If the index
 * is invalid, NULL is returned.
 */ 
const SpeciesReferenceGlyph*
ReactionGlyph::getSpeciesReferenceGlyph (unsigned int index) const
{
  return static_cast<const SpeciesReferenceGlyph*>
  (
    this->mSpeciesReferenceGlyphs.get(index)
  );
}


/**
 * Adds a new species reference glyph to the list.
 */
void
ReactionGlyph::addSpeciesReferenceGlyph (const SpeciesReferenceGlyph* glyph)
{
  this->mSpeciesReferenceGlyphs.append(glyph);
}


/**
 * Returns the number of species reference glyph objects.
 */ 
unsigned int
ReactionGlyph::getNumSpeciesReferenceGlyphs () const
{
  return this->mSpeciesReferenceGlyphs.size();
}


/**
 * Calls initDefaults from GraphicalObject.
 */ 
void ReactionGlyph::initDefaults ()
{
  GraphicalObject::initDefaults();
}


/**
 * Returns the curve object for the reaction glyph
 */ 
const Curve*
ReactionGlyph::getCurve () const
{
  return &this->mCurve;
}

/**
 * Returns the curve object for the reaction glyph
 */ 
Curve*
ReactionGlyph::getCurve () 
{
  return &this->mCurve;
}


/**
 * Sets the curve object for the reaction glyph.
 */ 
void ReactionGlyph::setCurve (const Curve* curve)
{
  if(!curve) return;
  this->mCurve = *curve;
  this->mCurve.connectToParent(this);
}


/**
 * Returns true if the curve consists of one or more segments.
 */ 
bool ReactionGlyph::isSetCurve () const
{
  return this->mCurve.getNumCurveSegments() > 0;
}


/**
 * Creates a new SpeciesReferenceGlyph object, adds it to the end of the
 * list of species reference objects and returns a reference to the newly
 * created object.
 */
SpeciesReferenceGlyph*
ReactionGlyph::createSpeciesReferenceGlyph ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  SpeciesReferenceGlyph* srg = new SpeciesReferenceGlyph(layoutns);

  this->mSpeciesReferenceGlyphs.appendAndOwn(srg);
  return srg;
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LineSegment*
ReactionGlyph::createLineSegment ()
{
  return this->mCurve.createLineSegment();
}

 
/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
CubicBezier*
ReactionGlyph::createCubicBezier ()
{
  return this->mCurve.createCubicBezier();
}

/**
 * Remove the species reference glyph with the given index.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
SpeciesReferenceGlyph*
ReactionGlyph::removeSpeciesReferenceGlyph(unsigned int index)
{
    SpeciesReferenceGlyph* srg=NULL;
    if(index < this->getNumSpeciesReferenceGlyphs())
    {
        srg=dynamic_cast<SpeciesReferenceGlyph*>(this->getListOfSpeciesReferenceGlyphs()->remove(index));
    }
    return srg;
}

/**
 * Remove the species reference glyph with the given id.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
SpeciesReferenceGlyph*
ReactionGlyph::removeSpeciesReferenceGlyph(const std::string& id)
{
    SpeciesReferenceGlyph* srg=NULL;
    unsigned int index=this->getIndexForSpeciesReferenceGlyph(id);
    if(index!=std::numeric_limits<unsigned int>::max())
    {
        srg=this->removeSpeciesReferenceGlyph(index);
    }
    return srg;
}

/**
 * Returns the index of the species reference glyph with the given id.
 * If the reaction glyph does not contain a species reference glyph with this
 * id, numreic_limits<int>::max() is returned.
 */
unsigned int
ReactionGlyph::getIndexForSpeciesReferenceGlyph(const std::string& id) const
{
    unsigned int i,iMax=this->getNumSpeciesReferenceGlyphs();
    unsigned int index=std::numeric_limits<unsigned int>::max();
    for(i=0;i<iMax;++i)
    {
        const SpeciesReferenceGlyph* srg=this->getSpeciesReferenceGlyph(i);
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
const std::string& ReactionGlyph::getElementName () const 
{
  static const std::string name = "reactionGlyph";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
ReactionGlyph* 
ReactionGlyph::clone () const
{
    return new ReactionGlyph(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ReactionGlyph::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  
  SBase*        object = 0;

  if (name == "listOfSpeciesReferenceGlyphs")
  {
    object = &mSpeciesReferenceGlyphs;
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
ReactionGlyph::addExpectedAttributes(ExpectedAttributes& attributes)
{
  GraphicalObject::addExpectedAttributes(attributes);

  attributes.add("reaction");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void ReactionGlyph::readAttributes (const XMLAttributes& attributes,
                                    const ExpectedAttributes& expectedAttributes)
{
  GraphicalObject::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("reaction", mReaction, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mReaction.empty())
  {
    logEmptyString(mReaction, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mReaction)) logError(InvalidIdSyntax);
}

/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
ReactionGlyph::writeElements (XMLOutputStream& stream) const
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

  if ( getNumSpeciesReferenceGlyphs() > 0 ) mSpeciesReferenceGlyphs.write(stream);

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
void ReactionGlyph::writeAttributes (XMLOutputStream& stream) const
{
  GraphicalObject::writeAttributes(stream);
  if(this->isSetReactionId())
  {
    stream.writeAttribute("reaction", getPrefix(), mReaction);
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
ReactionGlyph::getTypeCode () const
{
  return SBML_LAYOUT_REACTIONGLYPH;
}


/**
 * Creates an XMLNode object from this.
 */
XMLNode ReactionGlyph::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("reactionGlyph", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  addGraphicalObjectAttributes(*this,att);
  if(this->isSetReactionId()) att.add("reaction",this->mReaction);
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
  // add the list of species reference glyphs
  if(this->mSpeciesReferenceGlyphs.size()>0)
  {
    node.addChild(this->mSpeciesReferenceGlyphs.toXML());
  }
  return node;
}



/**
 * Ctor.
 */
ListOfSpeciesReferenceGlyphs::ListOfSpeciesReferenceGlyphs(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * Ctor.
 */
ListOfSpeciesReferenceGlyphs::ListOfSpeciesReferenceGlyphs(LayoutPkgNamespaces* layoutns)
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
ListOfSpeciesReferenceGlyphs*
ListOfSpeciesReferenceGlyphs::clone () const
{
  return new ListOfSpeciesReferenceGlyphs(*this);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfSpeciesReferenceGlyphs::getItemTypeCode () const
{
  return SBML_LAYOUT_SPECIESREFERENCEGLYPH;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfSpeciesReferenceGlyphs::getElementName () const
{
  static const std::string name = "listOfSpeciesReferenceGlyphs";
  return name;
}


/* return nth item in list */
SpeciesReferenceGlyph *
ListOfSpeciesReferenceGlyphs::get(unsigned int n)
{
  return static_cast<SpeciesReferenceGlyph*>(ListOf::get(n));
}


/* return nth item in list */
const SpeciesReferenceGlyph *
ListOfSpeciesReferenceGlyphs::get(unsigned int n) const
{
  return static_cast<const SpeciesReferenceGlyph*>(ListOf::get(n));
}


/* return item by id */
SpeciesReferenceGlyph*
ListOfSpeciesReferenceGlyphs::get (const std::string& sid)
{
  return const_cast<SpeciesReferenceGlyph*>( 
    static_cast<const ListOfSpeciesReferenceGlyphs&>(*this).get(sid) );
}


/* return item by id */
const SpeciesReferenceGlyph*
ListOfSpeciesReferenceGlyphs::get (const std::string& sid) const
{
  std::vector<SBase*>::const_iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<SpeciesReferenceGlyph>(sid) );
  return (result == mItems.end()) ? 0 : static_cast <SpeciesReferenceGlyph*> (*result);
}


/* Removes the nth item from this list */
SpeciesReferenceGlyph*
ListOfSpeciesReferenceGlyphs::remove (unsigned int n)
{
   return static_cast<SpeciesReferenceGlyph*>(ListOf::remove(n));
}


/* Removes item in this list by id */
SpeciesReferenceGlyph*
ListOfSpeciesReferenceGlyphs::remove (const std::string& sid)
{
  SBase* item = 0;
  std::vector<SBase*>::iterator result;

  result = std::find_if( mItems.begin(), mItems.end(), IdEq<SpeciesReferenceGlyph>(sid) );

  if (result != mItems.end())
  {
    item = *result;
    mItems.erase(result);
  }

  return static_cast <SpeciesReferenceGlyph*> (item);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfSpeciesReferenceGlyphs::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "speciesReferenceGlyph")
  {
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    object = new SpeciesReferenceGlyph(layoutns);
    appendAndOwn(object);
//    mItems.push_back(object);
  }

  return object;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfSpeciesReferenceGlyphs::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfSpeciesReferenceGlyphs", "http://projects.eml.org/bcb/sbml/level2", "");
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
  const SpeciesReferenceGlyph* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const SpeciesReferenceGlyph*>(this->get(i));
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
ReactionGlyph::accept (SBMLVisitor& v) const
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
  this->mSpeciesReferenceGlyphs.accept(this);
  v.leave(*this);
  return result;
}
*/


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
ReactionGlyph::setSBMLDocument (SBMLDocument* d)
{
  GraphicalObject::setSBMLDocument(d);

  mSpeciesReferenceGlyphs.setSBMLDocument(d);
  mCurve.setSBMLDocument(d);
}

/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
ReactionGlyph::connectToChild()
{
  mSpeciesReferenceGlyphs.connectToParent(this);
  mCurve.connectToParent(this);
}

/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
ReactionGlyph::enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mSpeciesReferenceGlyphs.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mCurve.enablePackageInternal(pkgURI,pkgPrefix,flag);
}



/**
 * Creates a new ReactionGlyph and returns the pointer to it.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
ReactionGlyph_create (void)
{
  return new(std::nothrow) ReactionGlyph;
}


/**
 * Creates a new ReactionGlyph with the given id
 */
LIBSBML_EXTERN
ReactionGlyph_t *
ReactionGlyph_createWith (const char *sid)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) ReactionGlyph(&layoutns, sid ? sid : "", "");
}


/**
 * Creates a new ReactionGlyph referencing the give reaction.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
ReactionGlyph_createWithReactionId (const char *id, const char *reactionId)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) ReactionGlyph(&layoutns, id ? id : "", reactionId ? reactionId : "");
}


/**
 * Creates a new ReactionGlyph object from a template.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
ReactionGlyph_createFrom (const ReactionGlyph_t *temp)
{
  return new(std::nothrow) ReactionGlyph(*temp);
}


/**
 * Frees the memory taken up by the attributes.
 */
LIBSBML_EXTERN
void
ReactionGlyph_free (ReactionGlyph_t *rg)
{
  delete rg;
}


/**
 * Sets the reference reaction for the reaction glyph.
 */
LIBSBML_EXTERN
void
ReactionGlyph_setReactionId (ReactionGlyph_t *rg,const char *id)
{
    static_cast<ReactionGlyph*>(rg)->setReactionId( id ? id : "" );
}


/**
 * Gets the reference reactions id for the given reaction glyph.
 */
LIBSBML_EXTERN
const char *
ReactionGlyph_getReactionId (const ReactionGlyph_t *rg)
{
    return rg->isSetReactionId() ? rg->getReactionId().c_str() : NULL;
}


/**
 * Returns 0 if the reference reaction has not been set for this glyph and
 * 1 otherwise.
 */
LIBSBML_EXTERN
int
ReactionGlyph_isSetReactionId (const ReactionGlyph_t *rg)
{
  return static_cast<int>( rg->isSetReactionId() );
}


/**
 * Add a SpeciesReferenceGlyph object to the list of
 * SpeciesReferenceGlyphs.
 */
LIBSBML_EXTERN
void
ReactionGlyph_addSpeciesReferenceGlyph (ReactionGlyph_t         *rg,
                                        SpeciesReferenceGlyph_t *srg)
{
  rg->addSpeciesReferenceGlyph(srg);
}


/**
 * Returns the number of SpeciesReferenceGlyphs for the ReactionGlyph.
 */
LIBSBML_EXTERN
unsigned int
ReactionGlyph_getNumSpeciesReferenceGlyphs (const ReactionGlyph_t *rg)
{
  return rg->getNumSpeciesReferenceGlyphs();
}


/**
 * Returns the pointer to the SpeciesReferenceGlyphs for the given index.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
ReactionGlyph_getSpeciesReferenceGlyph (ReactionGlyph_t *rg,
                                        unsigned int           index)
{
  return rg->getSpeciesReferenceGlyph(index);
}


/**
 * Returns the list object that holds all species reference glyphs.
 */ 
LIBSBML_EXTERN
ListOf_t *
ReactionGlyph_getListOfSpeciesReferenceGlyphs (ReactionGlyph_t *rg)
{
  return rg->getListOfSpeciesReferenceGlyphs();
}


/**
 * Calls initDefaults from GraphicalObject.
 */ 
LIBSBML_EXTERN
void
ReactionGlyph_initDefaults (ReactionGlyph_t *rg)
{
  rg->initDefaults();
}


/**
 * Sets the curve for the reaction glyph.
 */
LIBSBML_EXTERN
void
ReactionGlyph_setCurve (ReactionGlyph_t *rg, Curve_t *c)
{
  rg->setCurve(c);
}


/**
 * Gets the Curve for the given reaction glyph.
 */
LIBSBML_EXTERN
Curve_t *
ReactionGlyph_getCurve (ReactionGlyph_t *rg)
{
  return rg->getCurve();
}


/**
 * Returns true if the Curve has one or more LineSegment.
 */
LIBSBML_EXTERN
int
ReactionGlyph_isSetCurve (ReactionGlyph_t *rg)
{
  return static_cast<int>( rg->isSetCurve() );
}


/**
 * Creates a new SpeciesReferenceGlyph_t object, adds it to the end of the
 * list of species reference objects and returns a pointer to the newly
 * created object.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
ReactionGlyph_createSpeciesReferenceGlyph (ReactionGlyph_t *rg)
{
  return rg->createSpeciesReferenceGlyph();
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
LineSegment_t *
ReactionGlyph_createLineSegment (ReactionGlyph_t *rg)
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
ReactionGlyph_createCubicBezier (ReactionGlyph_t *rg)
{
  return rg->getCurve()->createCubicBezier();
}


/**
 * Remove the species reference glyph with the given index.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t*
ReactionGlyph_removeSpeciesReferenceGlyph(ReactionGlyph_t* rg,unsigned int index)
{
    return rg->removeSpeciesReferenceGlyph(index);
}

/**
 * Remove the species reference glyph with the given id.
 * A pointer to the object is returned. If no object has been removed, NULL
 * is returned.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t*
ReactionGlyph_removeSpeciesReferenceGlyphWithId(ReactionGlyph_t* rg,const char* id)
{
    return rg->removeSpeciesReferenceGlyph(id);
}

/**
 * Returns the index of the species reference glyph with the given id.
 * If the reaction glyph does not contain a species reference glyph with this
 * id, UINT_MAX from limits.h is returned.
 */
LIBSBML_EXTERN
unsigned int
ReactionGlyph_getIndexForSpeciesReferenceGlyph(ReactionGlyph_t* rg,const char* id)
{
    return rg->getIndexForSpeciesReferenceGlyph(id);
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
ReactionGlyph_t *
ReactionGlyph_clone (const ReactionGlyph_t *m)
{
  return static_cast<ReactionGlyph*>( m->clone() );
}


LIBSBML_CPP_NAMESPACE_END
