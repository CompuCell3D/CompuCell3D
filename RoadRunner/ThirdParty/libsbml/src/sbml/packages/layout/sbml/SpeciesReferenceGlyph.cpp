/**
 * Filename    : SpeciesReferenceGlyph.cpp
 * Description : SBML Layout SpeciesReferenceGlyph source
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


#include <sbml/packages/layout/sbml/SpeciesReferenceGlyph.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

LIBSBML_CPP_NAMESPACE_BEGIN

const std::string SpeciesReferenceGlyph::SPECIES_REFERENCE_ROLE_STRING[]={
    "undefined" 
   ,"substrate"
   ,"product"
   ,"sidesubstrate"
   ,"sideproduct"
   ,"modifier"
   ,"activator"
   ,"inhibitor"
   ,""
};



/**
 * Creates a new SpeciesReferenceGlyph.  The id if the associated species
 * reference and the id of the associated species glyph are set to the
 * empty string.  The role is set to SPECIES_ROLE_UNDEFINED.
 */
SpeciesReferenceGlyph::SpeciesReferenceGlyph (unsigned int level, unsigned int version, unsigned int pkgVersion)
 : GraphicalObject(level,version,pkgVersion)
   ,mSpeciesReference("")
   ,mSpeciesGlyph("")
   ,mRole  ( SPECIES_ROLE_UNDEFINED )
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


SpeciesReferenceGlyph::SpeciesReferenceGlyph(LayoutPkgNamespaces* layoutns)
 : GraphicalObject(layoutns)
   ,mSpeciesReference("")
   ,mSpeciesGlyph("")
   ,mRole  ( SPECIES_ROLE_UNDEFINED )
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
 * Creates a new SpeciesReferenceGlyph.  The id is given as the first
 * argument, the id of the associated species reference is given as the
 * second argument.  The third argument is the id of the associated species
 * glpyh and the fourth argument is the role.
 */ 
SpeciesReferenceGlyph::SpeciesReferenceGlyph
(
  LayoutPkgNamespaces* layoutns,
  const std::string& sid,
  const std::string& speciesGlyphId,
  const std::string& speciesReferenceId,
  SpeciesReferenceRole_t role
) :
  GraphicalObject    ( layoutns, sid      )
  , mSpeciesReference( speciesReferenceId )
  , mSpeciesGlyph    ( speciesGlyphId     )
  , mRole            ( role               )
  ,mCurve            (layoutns)
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
 * Creates a new SpeciesReferenceGlyph from the given XMLNode
 */
SpeciesReferenceGlyph::SpeciesReferenceGlyph(const XMLNode& node, unsigned int l2version)
 :  GraphicalObject  (2, l2version)
  , mSpeciesReference("")
  , mSpeciesGlyph    ("")
  , mRole            (SPECIES_ROLE_UNDEFINED)
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
SpeciesReferenceGlyph::SpeciesReferenceGlyph(const SpeciesReferenceGlyph& source) :
    GraphicalObject(source)
{
    this->mSpeciesReference=source.getSpeciesReferenceId();
    this->mSpeciesGlyph=source.getSpeciesGlyphId();
    this->mRole=source.getRole();
    this->mCurve=*source.getCurve();

    connectToChild();
}

/**
 * Assignment operator.
 */
SpeciesReferenceGlyph& SpeciesReferenceGlyph::operator=(const SpeciesReferenceGlyph& source)
{
  if(&source!=this)
  {
    GraphicalObject::operator=(source);
    this->mSpeciesReference=source.getSpeciesReferenceId();
    this->mSpeciesGlyph=source.getSpeciesGlyphId();
    this->mRole=source.getRole();
    this->mCurve=*source.getCurve();

    connectToChild();
  }
  
  return *this;
}

/**
 * Destructor.
 */ 
SpeciesReferenceGlyph::~SpeciesReferenceGlyph ()
{
}


/**
 * Returns the id of the associated SpeciesGlyph.
 */ 
const std::string&
SpeciesReferenceGlyph::getSpeciesGlyphId () const
{
  return this->mSpeciesGlyph;
}


/**
 * Sets the id of the associated species glyph.
 */ 
void
SpeciesReferenceGlyph::setSpeciesGlyphId (const std::string& speciesGlyphId)
{
  this->mSpeciesGlyph = speciesGlyphId;
}


/**
 * Returns the id of the associated species reference.
 */ 
const std::string&
SpeciesReferenceGlyph::getSpeciesReferenceId () const
{
  return this->mSpeciesReference;
}


/**
 * Sets the id of the associated species reference.
 */ 
void
SpeciesReferenceGlyph::setSpeciesReferenceId (const std::string& id)
{
  this->mSpeciesReference=id;
}


/**
 * Returns the role.
 */ 
SpeciesReferenceRole_t
SpeciesReferenceGlyph::getRole() const
{
  return this->mRole;
}

/**
 * Returns a string representation for the role
 */
const std::string& SpeciesReferenceGlyph::getRoleString() const{
    return SpeciesReferenceGlyph::SPECIES_REFERENCE_ROLE_STRING[this->mRole];
}

/**
 * Sets the role based on a string.
 * The String can be one of
 * SUBSTRATE
 * PRODUCT
 * SIDESUBSTRATE
 * SIDEPRODUCT
 * MODIFIER
 * ACTIVATOR
 * INHIBITOR    
 */ 
void
SpeciesReferenceGlyph::setRole (const std::string& role)
{
       if ( role == "substrate"     ) this->mRole = SPECIES_ROLE_SUBSTRATE;
  else if ( role == "product"       ) this->mRole = SPECIES_ROLE_PRODUCT;
  else if ( role == "sidesubstrate" ) this->mRole = SPECIES_ROLE_SIDESUBSTRATE;
  else if ( role == "sideproduct"   ) this->mRole = SPECIES_ROLE_SIDEPRODUCT;
  else if ( role == "modifier"      ) this->mRole = SPECIES_ROLE_MODIFIER;
  else if ( role == "activator"     ) this->mRole = SPECIES_ROLE_ACTIVATOR;
  else if ( role == "inhibitor"     ) this->mRole = SPECIES_ROLE_INHIBITOR;
  else                                this->mRole = SPECIES_ROLE_UNDEFINED;
}


/**
 * Sets the role.
 */ 
void
SpeciesReferenceGlyph::setRole (SpeciesReferenceRole_t role)
{
  this->mRole=role;
}


/**
 * Returns the curve object for the species reference glyph
 */ 
Curve* SpeciesReferenceGlyph::getCurve() 
{
  return &this->mCurve;
}

/**
 * Returns the curve object for the species reference glyph
 */ 
const Curve* SpeciesReferenceGlyph::getCurve() const
{
  return &this->mCurve;
}


/**
 * Sets the curve object for the species reference glyph.
 */ 
void
SpeciesReferenceGlyph::setCurve (const Curve* curve)
{
  if(!curve) return;
  this->mCurve = *curve;
  this->mCurve.connectToParent(this);
}


/**
 * Returns true if the curve consists of one or more segments.
 */ 
bool
SpeciesReferenceGlyph::isSetCurve () const
{
  return this->mCurve.getNumCurveSegments() > 0;
}


/**
 * Returns true if the id of the associated species glpyh is not the empty
 * string.
 */ 
bool
SpeciesReferenceGlyph::isSetSpeciesGlyphId () const
{
  return ! this->mSpeciesGlyph.empty();
}


/**
 * Returns true if the id of the associated species reference is not the
 * empty string.
 */ 
bool
SpeciesReferenceGlyph::isSetSpeciesReferenceId () const
{
  return ! this->mSpeciesReference.empty();
}


/**
 * Returns true of role is different from SPECIES_ROLE_UNDEFINED.
 */ 
bool SpeciesReferenceGlyph::isSetRole () const
{
  return ! (this->mRole == SPECIES_ROLE_UNDEFINED);
}


/**
 * Calls initDefaults on GraphicalObject and sets role to
 * SPECIES_ROLE_UNDEFINED.
 */ 
void
SpeciesReferenceGlyph::initDefaults ()
{
    GraphicalObject::initDefaults();
    this->mRole = SPECIES_ROLE_UNDEFINED;
}


/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LineSegment*
SpeciesReferenceGlyph::createLineSegment ()
{
  return this->mCurve.createLineSegment();
}


/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
CubicBezier*
SpeciesReferenceGlyph::createCubicBezier ()
{
  return this->mCurve.createCubicBezier();
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& SpeciesReferenceGlyph::getElementName () const 
{
  static const std::string name = "speciesReferenceGlyph";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
SpeciesReferenceGlyph* 
SpeciesReferenceGlyph::clone () const
{
    return new SpeciesReferenceGlyph(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
SpeciesReferenceGlyph::createObject (XMLInputStream& stream)
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
SpeciesReferenceGlyph::addExpectedAttributes(ExpectedAttributes& attributes)
{
  GraphicalObject::addExpectedAttributes(attributes);

  attributes.add("speciesReference");
  attributes.add("speciesGlyph");
  attributes.add("role");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void SpeciesReferenceGlyph::readAttributes (const XMLAttributes& attributes,
                                            const ExpectedAttributes& expectedAttributes)
{
  GraphicalObject::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("speciesReference", mSpeciesReference, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mSpeciesReference.empty())
  {
    logEmptyString(mSpeciesReference, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mSpeciesReference)) logError(InvalidIdSyntax);

  assigned = attributes.readInto("speciesGlyph", mSpeciesGlyph, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mSpeciesGlyph.empty())
  {
    logEmptyString(mSpeciesGlyph, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mSpeciesGlyph)) logError(InvalidIdSyntax);  
  
  std::string role;
  if(attributes.readInto("role", role, getErrorLog(), false, getLine(), getColumn()))
  {
    this->setRole(role);
  }
  else
  {
    this->setRole(SPECIES_ROLE_UNDEFINED);
  }
  
}

/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
SpeciesReferenceGlyph::writeElements (XMLOutputStream& stream) const
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
void SpeciesReferenceGlyph::writeAttributes (XMLOutputStream& stream) const
{
  GraphicalObject::writeAttributes(stream);
  if(this->isSetSpeciesReferenceId())
  {
    stream.writeAttribute("speciesReference", getPrefix(), mSpeciesReference);
  }
  if(this->isSetSpeciesGlyphId())
  {
    stream.writeAttribute("speciesGlyph", getPrefix(), mSpeciesGlyph);
  }
  if(this->isSetRole())
  {
    stream.writeAttribute("role", getPrefix(), this->getRoleString().c_str() );
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
SpeciesReferenceGlyph::getTypeCode () const
{
  return SBML_LAYOUT_SPECIESREFERENCEGLYPH;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode SpeciesReferenceGlyph::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("speciesReferenceGlyph", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  addGraphicalObjectAttributes(*this,att);
  if(this->isSetSpeciesReferenceId()) att.add("speciesReference",this->mSpeciesReference);
  if(this->isSetSpeciesGlyphId()) att.add("speciesGlyph",this->mSpeciesGlyph);
  if(this->isSetRole()) att.add("role",this->getRoleString());
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
SpeciesReferenceGlyph::accept (SBMLVisitor& v) const
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
SpeciesReferenceGlyph::setSBMLDocument (SBMLDocument* d)
{
  GraphicalObject::setSBMLDocument(d);

  mCurve.setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
SpeciesReferenceGlyph::connectToChild()
{
  mCurve.connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
SpeciesReferenceGlyph::enablePackageInternal(const std::string& pkgURI,
                                             const std::string& pkgPrefix, 
                                             bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mCurve.enablePackageInternal(pkgURI,pkgPrefix,flag);
}




/**
 * Creates a new SpeciesReferenceGlyph object and returns a pointer to it.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_create(void)
{
  return new(std::nothrow) SpeciesReferenceGlyph;
}


/**
 * Creates a new SpeciesReferenceGlyph from a template.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_createFrom (const SpeciesReferenceGlyph_t *temp)
{
  return new(std::nothrow) SpeciesReferenceGlyph(*temp);
}


/**
 * Creates a new SpeciesReferenceGlyph object with the given id and returns
 * a pointer to it.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_createWith (const char *sid,
                                  const char *speciesReferenceId,
                                  const char *speciesGlyphId,
                                  SpeciesReferenceRole_t role)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow)
    SpeciesReferenceGlyph(&layoutns, sid ? sid : "", speciesReferenceId ? speciesReferenceId : "", speciesGlyphId ? speciesGlyphId : "", role);
}


/**
 * Frees the memory for the SpeciesReferenceGlyph
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_free(SpeciesReferenceGlyph_t *srg)
{
  delete srg;
}


/**
 * Sets the reference species for the species glyph.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setSpeciesReferenceId (SpeciesReferenceGlyph_t *srg,
                                             const char *id)
{
    srg->setSpeciesReferenceId( id ? id : "" );
}


/**
 * Gets the reference species id for the given species glyph.
 */
LIBSBML_EXTERN
const char *
SpeciesReferenceGlyph_getSpeciesReferenceId (const SpeciesReferenceGlyph_t *srg)
{
    return srg->isSetSpeciesReferenceId() ? srg->getSpeciesReferenceId().c_str() : NULL;
}


/**
 * Returns 0 if the reference species reference has not been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetSpeciesReferenceId
  (const SpeciesReferenceGlyph_t *srg)
{
    return (int)srg->isSetSpeciesReferenceId();
}


/**
 * Sets the species glyph reference for the species glyph.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setSpeciesGlyphId (SpeciesReferenceGlyph_t *srg,
                                         const char *id)
{
    srg->setSpeciesGlyphId( id ? id : "" );
}


/**
 * Gets the reference speciess id for the given species glyph.
 */
LIBSBML_EXTERN
const char *
SpeciesReferenceGlyph_getSpeciesGlyphId (const SpeciesReferenceGlyph_t *srg)
{
    return srg->isSetSpeciesGlyphId() ? srg->getSpeciesGlyphId().c_str() : NULL;
}


/**
 * Returns 0 if  the reference species reference has not  been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetSpeciesGlyphId (const SpeciesReferenceGlyph_t *srg)
{
  return static_cast<int>( srg->isSetSpeciesGlyphId() );
}


/**
 * Sets the curve for the species reference glyph.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setCurve(SpeciesReferenceGlyph_t *srg, Curve_t *c)
{
  srg->setCurve(c);
}


/**
 * Gets the Curve for the given species reference glyph.
 */
LIBSBML_EXTERN
Curve_t *
SpeciesReferenceGlyph_getCurve (SpeciesReferenceGlyph_t *srg)
{
  return srg->getCurve();
}


/**
 * Returns true if the Curve has one or more LineSegment.
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetCurve (SpeciesReferenceGlyph_t *srg)
{
  return static_cast<int>( srg->isSetCurve() );
}


/**
 * Sets the role of the species reference glyph based on the string.  The
 * string can be one of UNDEFINED, SUBSTRATE, PRODUCT, SIDESUBSTRATE,
 * SIDEPRODUCT, MODIFIER, INHIBITOR or ACTIVATOR.  If it is none of those,
 * the role is set to SPECIES_ROLE_UNDEFINED.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setRole (SpeciesReferenceGlyph_t *srg,
                               const char *r)
{
  srg->setRole(r);
}


/**
 * Returns the role of the species reference.
 */ 
LIBSBML_EXTERN
SpeciesReferenceRole_t
SpeciesReferenceGlyph_getRole (const SpeciesReferenceGlyph_t *srg)
{
  return srg->getRole();
}

/**
 * Returns a string representation of the role of the species reference.
 */ 
LIBSBML_EXTERN
const char*
SpeciesReferenceGlyph_getRoleString(const SpeciesReferenceGlyph_t* srg){
    return srg->getRoleString().empty() ? NULL : srg->getRoleString().c_str();
}



/**
 * Returns true if the role is not SPECIES_ROLE_UNDEFINED.
 */ 
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetRole (const SpeciesReferenceGlyph_t *srg)
{
  return static_cast<int>( srg->isSetRole() );
}


/**
 * Calls initDefaults on GraphicalObject and sets role to
 * SPECIES_ROLE_UNDEFINED.
 */ 
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_initDefaults (SpeciesReferenceGlyph_t *srg)
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
SpeciesReferenceGlyph_createLineSegment (SpeciesReferenceGlyph_t *srg)
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
SpeciesReferenceGlyph_createCubicBezier (SpeciesReferenceGlyph_t *srg)
{
  return srg->getCurve()->createCubicBezier();
}


/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_clone (const SpeciesReferenceGlyph_t *m)
{
  return static_cast<SpeciesReferenceGlyph*>( m->clone() );
}

/**
 * Returns non-zero if the id is set
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetId (const SpeciesReferenceGlyph_t *srg)
{
  return static_cast <int> (srg->isSetId());
}

/**
 * Returns the id
 */
LIBSBML_EXTERN
const char *
SpeciesReferenceGlyph_getId (const SpeciesReferenceGlyph_t *srg)
{
  return srg->isSetId() ? srg->getId().c_str() : NULL;
}

/**
 * Sets the id
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_setId (SpeciesReferenceGlyph_t *srg, const char *sid)
{
  return (sid == NULL) ? srg->setId("") : srg->setId(sid);
}

/**
 * Unsets the id
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_unsetId (SpeciesReferenceGlyph_t *srg)
{
  srg->unsetId();
}

LIBSBML_CPP_NAMESPACE_END

