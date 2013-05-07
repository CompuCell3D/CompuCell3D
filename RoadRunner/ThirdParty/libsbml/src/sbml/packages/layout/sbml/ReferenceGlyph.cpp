/**
 * Filename    : ReferenceGlyph.cpp
 * Description : SBML Layout ReferenceGlyph source
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


#include <sbml/packages/layout/sbml/ReferenceGlyph.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

LIBSBML_CPP_NAMESPACE_BEGIN



/**
 * Creates a new ReferenceGlyph.  The id if the associated 
 * reference and the id of the associated glyph are set to the
 * empty string.  The role is set to empty.
 */
ReferenceGlyph::ReferenceGlyph (unsigned int level, unsigned int version, unsigned int pkgVersion)
 : GraphicalObject(level,version,pkgVersion)
   ,mReference("")
   ,mGlyph("")
   ,mRole  ( "" )
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


ReferenceGlyph::ReferenceGlyph(LayoutPkgNamespaces* layoutns)
 : GraphicalObject(layoutns)
   ,mReference("")
   ,mGlyph    ("")
   ,mRole     ("")
   ,mCurve(layoutns)
{
  connectToChild();
  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Creates a new ReferenceGlyph.  The id is given as the first
 * argument, the id of the associated reference is given as the
 * second argument.  The third argument is the id of the associated
 * glpyh and the fourth argument is the role.
 */ 
ReferenceGlyph::ReferenceGlyph
(
  LayoutPkgNamespaces* layoutns,
  const std::string& sid,
  const std::string& glyphId,
  const std::string& referenceId,
  const std::string& role
) :
  GraphicalObject    ( layoutns, sid      )
  , mReference       ( referenceId )
  , mGlyph           ( glyphId     )
  , mRole            ( role               )
  ,mCurve            ( layoutns)
{
  connectToChild();

  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Creates a new ReferenceGlyph from the given XMLNode
 */
ReferenceGlyph::ReferenceGlyph(const XMLNode& node, unsigned int l2version)
 :  GraphicalObject  (2, l2version)
   ,mReference("")
   ,mGlyph    ("")
   ,mRole     ("")
  , mCurve           (2, l2version)
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
ReferenceGlyph::ReferenceGlyph(const ReferenceGlyph& source) :
    GraphicalObject(source)
{
    this->mReference=source.mReference;
    this->mGlyph=source.mGlyph;
    this->mRole=source.mRole;
    this->mCurve=*source.getCurve();

    connectToChild();
}

/**
 * Assignment operator.
 */
ReferenceGlyph& ReferenceGlyph::operator=(const ReferenceGlyph& source)
{
  if(&source!=this)
  {
    GraphicalObject::operator=(source);
    this->mReference=source.mReference;
    this->mGlyph=source.mGlyph;
    this->mRole=source.mRole;
    this->mCurve=*source.getCurve();

    connectToChild();
  }
  
  return *this;
}

/**
 * Destructor.
 */ 
ReferenceGlyph::~ReferenceGlyph ()
{
}


/**
 * Returns the id of the associated glyph.
 */ 
const std::string&
ReferenceGlyph::getGlyphId () const
{
  return this->mGlyph;
}


/**
 * Sets the id of the associated glyph.
 */ 
void
ReferenceGlyph::setGlyphId (const std::string& glyphId)
{
  this->mGlyph = glyphId;
}


/**
 * Returns the id of the associated reference.
 */ 
const std::string&
ReferenceGlyph::getReferenceId () const
{
  return this->mReference;
}


/**
 * Sets the id of the associated reference.
 */ 
void
ReferenceGlyph::setReferenceId (const std::string& id)
{
  this->mReference=id;
}



/**
 * Returns a string representation for the role
 */
const std::string& ReferenceGlyph::getRole() const{
    return this->mRole;
}

/**
 * Sets the role based on a string.
 */ 
void
ReferenceGlyph::setRole (const std::string& role)
{
  this->mRole = role;
}



/**
 * Returns the curve object for the reference glyph
 */ 
Curve* ReferenceGlyph::getCurve() 
{
  return &this->mCurve;
}

/**
 * Returns the curve object for the reference glyph
 */ 
const Curve* ReferenceGlyph::getCurve() const
{
  return &this->mCurve;
}


/**
 * Sets the curve object for the reference glyph.
 */ 
void
ReferenceGlyph::setCurve (const Curve* curve)
{
  if(!curve) return;
  this->mCurve = *curve;
  this->mCurve.connectToParent(this);
}


/**
 * Returns true if the curve consists of one or more segments.
 */ 
bool
ReferenceGlyph::isSetCurve () const
{
  return this->mCurve.getNumCurveSegments() > 0;
}


/**
 * Returns true if the id of the associated glpyh is not the empty
 * string.
 */ 
bool
ReferenceGlyph::isSetGlyphId () const
{
  return ! this->mGlyph.empty();
}


/**
 * Returns true if the id of the associated reference is not the
 * empty string.
 */ 
bool
ReferenceGlyph::isSetReferenceId () const
{
  return ! this->mReference.empty();
}


/**
 * Returns true of role is different from the empty string.
 */ 
bool ReferenceGlyph::isSetRole () const
{
  return ! this->mRole.empty();
}


/**
 * Calls initDefaults on GraphicalObject 
 */ 
void
ReferenceGlyph::initDefaults ()
{
    GraphicalObject::initDefaults();    
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LineSegment*
ReferenceGlyph::createLineSegment ()
{
  return this->mCurve.createLineSegment();
}


/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
CubicBezier*
ReferenceGlyph::createCubicBezier ()
{
  return this->mCurve.createCubicBezier();
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& ReferenceGlyph::getElementName () const 
{
  static const std::string name = "referenceGlyph";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
ReferenceGlyph* 
ReferenceGlyph::clone () const
{
    return new ReferenceGlyph(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ReferenceGlyph::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  
  SBase*        object = 0;

  if (name == "curve")
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
ReferenceGlyph::addExpectedAttributes(ExpectedAttributes& attributes)
{
  GraphicalObject::addExpectedAttributes(attributes);

  attributes.add("reference");
  attributes.add("glyph");
  attributes.add("role");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void ReferenceGlyph::readAttributes (const XMLAttributes& attributes,
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

  assigned = attributes.readInto("glyph", mGlyph, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mGlyph.empty())
  {
    logEmptyString(mGlyph, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mGlyph)) logError(InvalidIdSyntax);  
  
  std::string role;
  if(attributes.readInto("role", role, getErrorLog(), false, getLine(), getColumn()))
  {
    this->setRole(role);
  }
 
  
}

/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
ReferenceGlyph::writeElements (XMLOutputStream& stream) const
{
  if(this->isSetCurve())
  {
      SBase::writeElements(stream);
      mCurve.write(stream);
  }
  else
  {
    GraphicalObject::writeElements(stream);
  }

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
void ReferenceGlyph::writeAttributes (XMLOutputStream& stream) const
{
  GraphicalObject::writeAttributes(stream);
  if(this->isSetReferenceId())
  {
    stream.writeAttribute("reference", getPrefix(), mReference);
  }
  if(this->isSetGlyphId())
  {
    stream.writeAttribute("glyph", getPrefix(), mGlyph);
  }
  if(this->isSetRole())
  {
    stream.writeAttribute("role", getPrefix(), this->mRole );
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
ReferenceGlyph::getTypeCode () const
{
  return SBML_LAYOUT_REFERENCEGLYPH;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ReferenceGlyph::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("referenceGlyph", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  addGraphicalObjectAttributes(*this,att);
  if(this->isSetReferenceId()) att.add("reference",this->mReference);
  if(this->isSetGlyphId()) att.add("glyph",this->mGlyph);
  if(this->isSetRole()) att.add("role",this->mRole);
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
  return node;
}


/**
 * Accepts the given SBMLVisitor.

bool
ReferenceGlyph::accept (SBMLVisitor& v) const
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
  v.leave(*this);
  return result;
}
*/


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
ReferenceGlyph::setSBMLDocument (SBMLDocument* d)
{
  GraphicalObject::setSBMLDocument(d);

  mCurve.setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
ReferenceGlyph::connectToChild()
{
  mCurve.connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
ReferenceGlyph::enablePackageInternal(const std::string& pkgURI,
                                             const std::string& pkgPrefix, 
                                             bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mCurve.enablePackageInternal(pkgURI,pkgPrefix,flag);
}




/**
 * Creates a new ReferenceGlyph object and returns a pointer to it.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_create(void)
{
  return new(std::nothrow) ReferenceGlyph;
}


/**
 * Creates a new ReferenceGlyph from a template.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_createFrom (const ReferenceGlyph_t *temp)
{
  return new(std::nothrow) ReferenceGlyph(*temp);
}


/**
 * Creates a new ReferenceGlyph object with the given id and returns
 * a pointer to it.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_createWith (const char *sid,
                                  const char *referenceId,
                                  const char *glyphId,
                                  const char* role)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow)
    ReferenceGlyph(&layoutns, sid ? sid : "", referenceId ? referenceId : "", glyphId ? glyphId : "", role ? role : "");
}


/**
 * Frees the memory for the ReferenceGlyph
 */
LIBSBML_EXTERN
void
ReferenceGlyph_free(ReferenceGlyph_t *srg)
{
  delete srg;
}


/**
 * Sets the reference for the glyph.
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setReferenceId (ReferenceGlyph_t *srg,
                                             const char *id)
{
    srg->setReferenceId( id ? id : "" );
}


/**
 * Gets the reference id for the given glyph.
 */
LIBSBML_EXTERN
const char *
ReferenceGlyph_getReferenceId (const ReferenceGlyph_t *srg)
{
    return srg->isSetReferenceId() ? srg->getReferenceId().c_str() : NULL;
}


/**
 * Returns 0 if the reference reference has not been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetReferenceId
  (const ReferenceGlyph_t *srg)
{
    return (int)srg->isSetReferenceId();
}


/**
 * Sets the glyph reference for this glyph.
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setGlyphId (ReferenceGlyph_t *srg,
                                         const char *id)
{
    srg->setGlyphId( id ? id : "" );
}


/**
 * Gets the reference id for the given glyph.
 */
LIBSBML_EXTERN
const char *
ReferenceGlyph_getGlyphId (const ReferenceGlyph_t *srg)
{
    return srg->isSetGlyphId() ? srg->getGlyphId().c_str() : NULL;
}


/**
 * Returns 0 if  the reference reference has not  been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetGlyphId (const ReferenceGlyph_t *srg)
{
  return static_cast<int>( srg->isSetGlyphId() );
}


/**
 * Sets the curve for the reference glyph.
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setCurve(ReferenceGlyph_t *srg, Curve_t *c)
{
  srg->setCurve(c);
}


/**
 * Gets the Curve for the given reference glyph.
 */
LIBSBML_EXTERN
Curve_t *
ReferenceGlyph_getCurve (ReferenceGlyph_t *srg)
{
  return srg->getCurve();
}


/**
 * Returns true if the Curve has one or more LineSegment.
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetCurve (ReferenceGlyph_t *srg)
{
  return static_cast<int>( srg->isSetCurve() );
}


/**
 * Sets the role of the reference glyph based on the string.  
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setRole (ReferenceGlyph_t *srg,
                               const char *r)
{
  srg->setRole(r);
}


/**
 * Returns a string representation of the role of the reference.
 */ 
LIBSBML_EXTERN
const char*
ReferenceGlyph_getRole(const ReferenceGlyph_t* srg){
    return srg->getRole().empty() ? NULL : srg->getRole().c_str();
}



/**
 * Returns true if the role is not empty.
 */ 
LIBSBML_EXTERN
int
ReferenceGlyph_isSetRole (const ReferenceGlyph_t *srg)
{
  return static_cast<int>( srg->isSetRole() );
}


/**
 * Calls initDefaults on GraphicalObject 
 */ 
LIBSBML_EXTERN
void
ReferenceGlyph_initDefaults (ReferenceGlyph_t *srg)
{
  srg->initDefaults();
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
LineSegment_t *
ReferenceGlyph_createLineSegment (ReferenceGlyph_t *srg)
{
  return srg->getCurve()->createLineSegment();
}  


/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
CubicBezier_t *
ReferenceGlyph_createCubicBezier (ReferenceGlyph_t *srg)
{
  return srg->getCurve()->createCubicBezier();
}


/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_clone (const ReferenceGlyph_t *m)
{
  return static_cast<ReferenceGlyph*>( m->clone() );
}

/**
 * Returns non-zero if the id is set
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetId (const ReferenceGlyph_t *srg)
{
  return static_cast <int> (srg->isSetId());
}

/**
 * Returns the id
 */
LIBSBML_EXTERN
const char *
ReferenceGlyph_getId (const ReferenceGlyph_t *srg)
{
  return srg->isSetId() ? srg->getId().c_str() : NULL;
}

/**
 * Sets the id
 */
LIBSBML_EXTERN
int
ReferenceGlyph_setId (ReferenceGlyph_t *srg, const char *sid)
{
  return (sid == NULL) ? srg->setId("") : srg->setId(sid);
}

/**
 * Unsets the id
 */
LIBSBML_EXTERN
void
ReferenceGlyph_unsetId (ReferenceGlyph_t *srg)
{
  srg->unsetId();
}

LIBSBML_CPP_NAMESPACE_END

