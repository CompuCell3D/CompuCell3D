/**
 * Filename    : Point.cpp
 * Description : SBML Layout Point source
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

#include <sstream>

#include <sbml/packages/layout/sbml/Point.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/SBMLErrorLog.h>
#include <sbml/SBMLVisitor.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * Creates a new point with x,y and z set  to 0.0.
 */ 
Point::Point(unsigned int level, unsigned int version, unsigned int pkgVersion) 
 :  SBase(level,version)
  , mXOffset(0.0)
  , mYOffset(0.0)
  , mZOffset(0.0)
  , mElementName("point")
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
}



/**
 * Constructor
 */ 
Point::Point(LayoutPkgNamespaces* layoutns)
 : SBase(layoutns)
  , mXOffset(0.0)
  , mYOffset(0.0)
  , mZOffset(0.0)
  , mElementName("point")
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Copy constructor.
 */
Point::Point(const Point& orig):SBase(orig)
{
    this->mXOffset=orig.mXOffset;
    this->mYOffset=orig.mYOffset;
    this->mZOffset=orig.mZOffset;
    this->mElementName=orig.mElementName;

    // attributes of SBase
    //this->mId=orig.mId;
    //this->mName=orig.mName;
    this->mMetaId=orig.mMetaId;
    if(orig.mNotes) this->mNotes=new XMLNode(*const_cast<Point&>(orig).getNotes());
    if(orig.mAnnotation) this->mAnnotation=new XMLNode(*const_cast<Point&>(orig).mAnnotation);
    this->mSBML=orig.mSBML;
    this->mSBOTerm=orig.mSBOTerm;
    this->mLine=orig.mLine;
    this->mColumn=orig.mColumn;

    if(orig.mCVTerms)
    {
      this->mCVTerms=new List();
      unsigned int i,iMax=orig.mCVTerms->getSize();
      for(i=0;i<iMax;++i)
      {
        this->mCVTerms->add(static_cast<CVTerm*>(orig.mCVTerms->get(i))->clone());
      }
    }
}

Point& Point::operator=(const Point& orig)
{
  if(&orig!=this)
  {
    this->mXOffset=orig.mXOffset;
    this->mYOffset=orig.mYOffset;
    this->mZOffset=orig.mZOffset;
    this->mElementName=orig.mElementName;

    this->mMetaId=orig.mMetaId;
    delete this->mNotes;
    this->mNotes=NULL;
    if(orig.mNotes)
    {
        this->mNotes=new XMLNode(*const_cast<Point&>(orig).getNotes());
    }
    delete this->mAnnotation;
    this->mAnnotation=NULL;
    if(orig.mAnnotation)
    {
        this->mAnnotation=new XMLNode(*const_cast<Point&>(orig).mAnnotation);
    }
    this->mSBML=orig.mSBML;
    this->mSBOTerm=orig.mSBOTerm;
    this->mLine=orig.mLine;
    this->mColumn=orig.mColumn;
    delete this->mCVTerms;
    this->mCVTerms=NULL;
    if(orig.mCVTerms)
    {
      this->mCVTerms=new List();
      unsigned int i,iMax=orig.mCVTerms->getSize();
      for(i=0;i<iMax;++i)
      {
        this->mCVTerms->add(static_cast<CVTerm*>(orig.mCVTerms->get(i))->clone());
      }
    }
  }
  
  return *this;
}


/**
 * Creates a new point with the given ccordinates.
 */ 
Point::Point(LayoutPkgNamespaces* layoutns, double x, double y, double z)
  : SBase  (layoutns)
  , mXOffset(x)
  , mYOffset(y)
  , mZOffset(z)
  , mElementName("point")  
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Sets the Z offset to 0.0.
 */
void Point::initDefaults ()
{
  this->setZOffset(0.0);
}

/**
 * Creates a new Point from the given XMLNode
 */
Point::Point(const XMLNode& node, unsigned int l2version) 
 : SBase(2,l2version)
  , mXOffset(0.0)
  , mYOffset(0.0)
  , mZOffset(0.0)
  , mElementName(node.getName())
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
        if(childName=="annotation")
        {
            this->mAnnotation=new XMLNode(node);
        }
        else if(childName=="notes")
        {
            this->mNotes=new XMLNode(node);
        }
        else
        {
            //throw;
        }
        ++n;
    }    

  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(2,l2version));  
}


/**
 * Destructor.
 */ 
Point::~Point()
{
}


/**
  * Returns the value of the "id" attribute of this Point.
  */
const std::string& Point::getId () const
{
  return mId;
}


/**
  * Predicate returning @c true or @c false depending on whether this
  * Point's "id" attribute has been set.
  */
bool Point::isSetId () const
{
  return (mId.empty() == false);
}

/**
  * Sets the value of the "id" attribute of this Point.
  */
int Point::setId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id,mId);
}


/**
  * Unsets the value of the "id" attribute of this Point.
  */
int Point::unsetId ()
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
 * Sets the coordinates to the given values.
 */ 
void
Point::setOffsets (double x, double y, double z)
{
  this->setXOffset(x);
  this->setYOffset(y);
  this->setZOffset(z);
}


/**
 * Sets the x offset.
 */ 
void
Point::setXOffset (double x)
{
  this->setX(x);
}


/**
 * Sets the y offset.
 */ 
void
Point::setYOffset (double y)
{
  this->setY(y);
}


/**
 * Sets the z offset.
 */ 
void
Point::setZOffset (double z)
{
  this->setZ(z);
}


/**
 * Sets the x offset.
 */ 
void
Point::setX (double x)
{
  this->mXOffset = x;
}


/**
 * Sets the y offset.
 */ 
void
Point::setY (double y)
{
  this->mYOffset = y;
}


/**
 * Sets the z offset.
 */ 
void
Point::setZ (double z)
{
  this->mZOffset = z;
}


/**
 * Returns the x offset.
 */ 
double
Point::getXOffset () const
{
  return this->x();
}


/**
 * Returns the y offset.
 */ 
double
Point::getYOffset () const
{
  return this->y();
}


/**
 * Returns the z offset.
 */ 
double
Point::getZOffset () const
{
  return this->z();
}

/**
 * Returns the x offset.
 */ 
double
Point::x () const
{
  return this->mXOffset;
}


/**
 * Returns the y offset.
 */ 
double
Point::y () const
{
  return this->mYOffset;
}


/**
 * Returns the z offset.
 */ 
double
Point::z () const
{
  return this->mZOffset;
}


/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.  For example:
 *
 *   SBase::writeElements(stream);
 *   mReactans.write(stream);
 *   mProducts.write(stream);
 *   ...
 */
void Point::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}

/**
 * Sets the element name to be returned by getElementName.
 */
void Point::setElementName(const std::string& name)
{
    this->mElementName=name;
}
 
/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& Point::getElementName () const 
{
  return this->mElementName;
}

/**
 * @return a (deep) copy of this Model.
 */
Point* 
Point::clone () const
{
    return new Point(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
Point::createObject (XMLInputStream& stream)
{
  SBase*        object = 0;

  object=SBase::createObject(stream);
  
  return object;
}

/**
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
Point::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);

  attributes.add("id");
  attributes.add("x");
  attributes.add("y");
  attributes.add("z");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void Point::readAttributes (const XMLAttributes& attributes,
                            const ExpectedAttributes& expectedAttributes)
{
  SBase::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("id", mId, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mId.empty())
    {
      logEmptyString(mId, sbmlLevel, sbmlVersion, "<point>");
    }
  if (!SyntaxChecker::isValidInternalSId(mId)) logError(InvalidIdSyntax);

  attributes.readInto("x", mXOffset,getErrorLog(),true, getLine(), getColumn());
  attributes.readInto("y", mYOffset,getErrorLog(),true, getLine(), getColumn());
  if(!attributes.readInto("z", mZOffset, getErrorLog(), false, getLine(), getColumn()))
  {
      this->mZOffset=0.0;
  }
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
void Point::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);
  if (isSetId())
  {
    stream.writeAttribute("id", getPrefix(), mId);
  }
  stream.writeAttribute("x", getPrefix(), mXOffset);
  stream.writeAttribute("y", getPrefix(), mYOffset);

  //
  // (TODO) default value should be allowd in package of Level 3?
  //
  if(this->mZOffset!=0.0)
  {
    stream.writeAttribute("z", getPrefix(), mZOffset);
  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode Point::toXML(const std::string& name) const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple(name, "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  std::ostringstream os;
  os << this->mXOffset;
  att.add("x",os.str());
  os.str("");
  os << this->mYOffset;
  att.add("y",os.str());
  if(this->mZOffset!=0.0)
  {
    os.str("");
    os << this->mZOffset;
    att.add("z",os.str());
  }
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

  if(end==true) node.setEnd();
  return node;
}


/**
 * Returns the package type code for this object.
 */
int
Point::getTypeCode () const
{
  return SBML_LAYOUT_POINT;
}


/**
 * Accepts the given SBMLVisitor.
 *
 * @return the result of calling <code>v.visit()</code>, which indicates
 * whether or not the Visitor would like to visit the SBML object's next
 * sibling object (if available).
 */
bool Point::accept (SBMLVisitor& v) const
{
    //v.visit(*this);
    return false;
}




/**
 * Creates a new point with the coordinates (0.0,0.0,0.0).
 */ 
LIBSBML_EXTERN
Point_t *
Point_create (void)
{
  return new(std::nothrow) Point; 
}


/**
 * Creates a new Point with the given coordinates.
 */ 
LIBSBML_EXTERN
Point_t *
Point_createWithCoordinates (double x, double y, double z)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) Point(&layoutns, x, y, z);
}


/**
 * Frees all memory for the Point.
 */ 
LIBSBML_EXTERN
void
Point_free (Point_t *p)
{
  delete p;
}


/**
 * Sets the Z offset to 0.0
 */ 
LIBSBML_EXTERN
void
Point_initDefaults (Point_t *p)
{
  p->initDefaults();
}


/**
 * Sets the coordinates to the given values.
 */ 
LIBSBML_EXTERN
void
Point_setOffsets (Point_t *p, double x, double y, double z)
{
  p->setOffsets(x, y, z);
}


/**
 * Sets the x offset.
 */ 
LIBSBML_EXTERN
void
Point_setXOffset (Point_t *p, double x)
{
  p->setX(x);
}


/**
 * Sets the y offset.
 */ 
LIBSBML_EXTERN
void
Point_setYOffset (Point_t *p, double y)
{
  p->setY(y);
}


/**
 * Sets the z offset.
 */ 
LIBSBML_EXTERN
void
Point_setZOffset (Point_t *p, double z)
{
  p->setZ(z);
}


/**
 * Gets the x offset.
 */ 
LIBSBML_EXTERN
double
Point_getXOffset (const Point_t *p)
{
  return p->x();
}


/**
 * Gets the y offset.
 */ 
LIBSBML_EXTERN
double
Point_getYOffset (const Point_t *p)
{
  return p->y();
}


/**
 * Gets the z offset.
 */ 
LIBSBML_EXTERN
double
Point_getZOffset (const Point_t *p)
{
  return p->z();
}


/**
 * Sets the x offset.
 */ 
LIBSBML_EXTERN
void
Point_setX (Point_t *p, double x)
{
  p->setX(x);
}


/**
 * Sets the y offset.
 */ 
LIBSBML_EXTERN
void
Point_setY (Point_t *p, double y)
{
  p->setY(y);
}


/**
 * Sets the z offset.
 */ 
LIBSBML_EXTERN
void
Point_setZ (Point_t *p, double z)
{
  p->setZ(z);
}


/**
 * Gets the x offset.
 */ 
LIBSBML_EXTERN
double
Point_x (const Point_t *p)
{
  return p->x();
}


/**
 * Gets the y offset.
 */ 
LIBSBML_EXTERN
double
Point_y (const Point_t *p)
{
  return p->y();
}


/**
 * Gets the z offset.
 */ 
LIBSBML_EXTERN
double
Point_z (const Point_t *p)
{
  return p->z();
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
Point_t *
Point_clone (const Point_t *m)
{
  return static_cast<Point*>( m->clone() );
}

LIBSBML_CPP_NAMESPACE_END

