/**
 * Filename    : CubicBezier.cpp
 * Description : SBML Layout CubicBezier source
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


#include <sbml/packages/layout/sbml/CubicBezier.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * Creates a CubicBezier and returns the pointer.
 */
CubicBezier::CubicBezier(unsigned int level, unsigned int version, unsigned int pkgVersion) 
 : LineSegment(level,version,pkgVersion)
  ,mBasePoint1(level,version,pkgVersion)
  ,mBasePoint2(level,version,pkgVersion)
{
  this->mStartPoint.setElementName("start");
  this->mEndPoint.setElementName("end");
  this->mBasePoint1.setElementName("basePoint1");
  this->mBasePoint2.setElementName("basePoint2");

  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
  connectToChild();
}


/**
 * Creates a CubicBezier and returns the pointer.
 */
CubicBezier::CubicBezier(LayoutPkgNamespaces* layoutns)
 : LineSegment(layoutns)
  ,mBasePoint1(layoutns)
  ,mBasePoint2(layoutns)
{
  this->mStartPoint.setElementName("start");
  this->mEndPoint.setElementName("end");
  this->mBasePoint1.setElementName("basePoint1");
  this->mBasePoint2.setElementName("basePoint2");

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
 * Creates a CubicBezier with the given 2D coordinates and returns the
 * pointer.
 */
CubicBezier::CubicBezier (LayoutPkgNamespaces* layoutns, double x1, double y1, double x2, double y2)
  : LineSegment(layoutns, x1, y1, 0.0, x2, y2, 0.0 )
  ,mBasePoint1(layoutns)
  ,mBasePoint2(layoutns)
{
  this->straighten();
  this->mBasePoint1.setElementName("basePoint1");
  this->mBasePoint2.setElementName("basePoint2");

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
 * Creates a CubicBezier with the given 3D coordinates and returns the
 * pointer.
 */
CubicBezier::CubicBezier (LayoutPkgNamespaces* layoutns, double x1, double y1, double z1,
                          double x2, double y2, double z2)
 : LineSegment(layoutns, x1, y1, z1, x2, y2, z2 )
  ,mBasePoint1(layoutns)
  ,mBasePoint2(layoutns)
{
  this->straighten();
  this->mBasePoint1.setElementName("basePoint1");
  this->mBasePoint2.setElementName("basePoint2");

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
 * Copy constructor.
 */
CubicBezier::CubicBezier(const CubicBezier& orig):LineSegment(orig)
{
  this->mBasePoint1=orig.mBasePoint1;
  this->mBasePoint2=orig.mBasePoint2;

  connectToChild();
}


/**
 * Assignment operator.
 */
CubicBezier& CubicBezier::operator=(const CubicBezier& orig)
{
  if(&orig!=this)
  {
    LineSegment::operator=(orig);
    this->mBasePoint1=orig.mBasePoint1;
    this->mBasePoint2=orig.mBasePoint2;

    connectToChild();
  }

  return *this;
}



/**
 * Makes a line from a CubicBezier by setting both base points into the
 * middle between the start and the end point.
 */
void CubicBezier::straighten ()
{
  double x = (this->mEndPoint.getXOffset()+this->mStartPoint.getXOffset()) / 2.0;
  double y = (this->mEndPoint.getYOffset()+this->mStartPoint.getYOffset()) / 2.0;
  double z = (this->mEndPoint.getZOffset()+this->mStartPoint.getZOffset()) / 2.0;

  this->mBasePoint1.setOffsets(x, y, z);
  this->mBasePoint2.setOffsets(x, y, z);
}


/**
 * Creates a CubicBezier with the given points and returns the pointer.
 */
CubicBezier::CubicBezier (LayoutPkgNamespaces* layoutns, const Point* start, const Point* end)
 : LineSegment(layoutns, start, end)
  ,mBasePoint1(layoutns)
  ,mBasePoint2(layoutns)
{
  this->straighten();
  this->mBasePoint1.setElementName("basePoint1");
  this->mBasePoint2.setElementName("basePoint2");

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
 * Creates a CubicBezier with the given points and returns the pointer.
 */
CubicBezier::CubicBezier (LayoutPkgNamespaces* layoutns, const Point* start, const Point* base1,
                          const Point* base2, const Point* end)
 : LineSegment(layoutns, start ,end )
  ,mBasePoint1(layoutns)
  ,mBasePoint2(layoutns)
{
    if(base1 && base2 && start && end)
    {
      this->mBasePoint1=*base1;
      this->mBasePoint1.setElementName("basePoint1");
      this->mBasePoint2=*base2;
      this->mBasePoint2.setElementName("basePoint2");
    }
    else
    {
        this->mStartPoint=Point(layoutns);
        this->mEndPoint=Point(layoutns);
    }

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
 * Creates a new CubicBezier from the given XMLNode
 */
CubicBezier::CubicBezier(const XMLNode& node, unsigned int l2version)
 : LineSegment(2, l2version)
  ,mBasePoint1(2, l2version)
  ,mBasePoint2(2, l2version)
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
        if(childName=="start")
        {
            this->mStartPoint=Point(*child);
        }
        else if(childName=="end")
        {
            this->mEndPoint=Point(*child);
        }
        else if(childName=="basePoint1")
        {
            this->mBasePoint1=Point(*child);
        }
        else if(childName=="basePoint2")
        {
            this->mBasePoint2=Point(*child);
        }
        else if(childName=="annotation")
        {
            this->mAnnotation=new XMLNode(*child);
        }
        else if(childName=="notes")
        {
            this->mNotes=new XMLNode(*child);
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
 * Destructor.
 */ 
CubicBezier::~CubicBezier ()
{
}


/**
 * Calls initDefaults from LineSegment.
 */ 
void
CubicBezier::initDefaults()
{
  LineSegment::initDefaults();
}


/**
 * Returns the first base point of the curve (the one closer to the
 * starting point).
 */ 
const Point*
CubicBezier::getBasePoint1() const
{
  return &this->mBasePoint1;
}


/**
 * Returns the first base point of the curve (the one closer to the
 * starting point).
 */ 
Point*
CubicBezier::getBasePoint1 ()
{
  return &this->mBasePoint1;
}


/**
 * Initializes first base point with a copy of the given point.
 */
void
CubicBezier::setBasePoint1 (const Point* p)
{
  if(p)
  {  
    this->mBasePoint1 = *p;
    this->mBasePoint1.setElementName("basePoint1");
    this->mBasePoint1.connectToParent(this);
  }
}


/**
 * Initializes first base point with the given ccordinates.
 */
void
CubicBezier::setBasePoint1 (double x, double y, double z)
{
  this->mBasePoint1.setOffsets(x, y ,z);
  this->mBasePoint1.connectToParent(this);
}


/**
 * Returns the second base point of the curve (the one closer to the
 * starting point).
 */ 
const Point*
CubicBezier::getBasePoint2 () const
{
  return &this->mBasePoint2;
}


/**
 * Returns the second base point of the curve (the one closer to the
 * starting point).
 */ 
Point*
CubicBezier::getBasePoint2 ()
{
  return &this->mBasePoint2;
}


/**
 * Initializes second base point with a copy of the given point.
 */
void CubicBezier::setBasePoint2 (const Point* p)
{
  if(p)
  {  
    this->mBasePoint2 = *p;
    this->mBasePoint2.setElementName("basePoint2");
    this->mBasePoint2.connectToParent(this);
  }
}


/**
 * Initializes second base point with the given ccordinates.
 */
void
CubicBezier::setBasePoint2 (double x, double y, double z)
{
  this->mBasePoint2.setOffsets(x, y, z);
  this->mBasePoint2.connectToParent(this);
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& CubicBezier::getElementName () const 
{
  static const std::string name = "curveSegment";
  return name;
}


/**
 * @return a (deep) copy of this Model.
 */
CubicBezier* 
CubicBezier::clone () const
{
    return new CubicBezier(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
CubicBezier::createObject (XMLInputStream& stream)
{

  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;

  if (name == "basePoint1")
  {
    object = &mBasePoint1;
  }
  else if(name == "basePoint2")
  {
    object = &mBasePoint2;
  }
  else
  {
      object = LineSegment::createObject(stream);
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
CubicBezier::addExpectedAttributes(ExpectedAttributes& attributes)
{
  LineSegment::addExpectedAttributes(attributes);
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void CubicBezier::readAttributes (const XMLAttributes& attributes,
                                  const ExpectedAttributes& expectedAttributes)
{
  LineSegment::readAttributes(attributes,expectedAttributes);
}

/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
CubicBezier::writeElements (XMLOutputStream& stream) const
{
  LineSegment::writeElements(stream);
  mBasePoint1.write(stream);
  mBasePoint2.write(stream);

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
void CubicBezier::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);
  stream.writeAttribute("type", "xsi", "CubicBezier");

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}


/**
 * Returns the package type code for this object.
 */
int
CubicBezier::getTypeCode () const
{
  return SBML_LAYOUT_CUBICBEZIER;
}


/**
 * Accepts the given SBMLVisitor.

bool
CubicBezier::accept (SBMLVisitor& v) const
{
  bool result=v.visit(*this);
  this->mStartPoint.accept(v);
  this->mBasePoint1.accept(v);
  this->mBasePoint2.accept(v);
  this->mEndPoint.accept(v);
  v.leave(*this);
  return result;
}
*/


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
CubicBezier::setSBMLDocument (SBMLDocument* d)
{
  LineSegment::setSBMLDocument(d);

  mBasePoint1.setSBMLDocument(d);
  mBasePoint2.setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
CubicBezier::connectToChild()
{
  LineSegment::connectToChild();
  mBasePoint1.connectToParent(this);
  mBasePoint2.connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
CubicBezier::enablePackageInternal(const std::string& pkgURI,
                                   const std::string& pkgPrefix, bool flag)
{
  LineSegment::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mBasePoint1.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mBasePoint2.enablePackageInternal(pkgURI,pkgPrefix,flag);
}



/**
 * Creates an XMLNode object from this.
 */
XMLNode CubicBezier::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("curveSegment", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  att.add("type","CubicBezier","http://www.w3.org/2001/XMLSchema-instance","xsi");  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  if(this->mNotes) node.addChild(*this->mNotes);
  if(this->mAnnotation) node.addChild(*this->mAnnotation);
  // add start point
  node.addChild(this->mStartPoint.toXML("start"));
  // add end point
  node.addChild(this->mEndPoint.toXML("end"));
  // add start point
  node.addChild(this->mBasePoint1.toXML("basePoint1"));
  // add end point
  node.addChild(this->mBasePoint2.toXML("basePoint2"));
  return node;
}



/**
 * Creates a CubicBezier and returns the pointer.
 */
LIBSBML_EXTERN
CubicBezier_t *
CubicBezier_create (void)
{
  return new(std::nothrow) CubicBezier;
}


/**
 * Creates a CubicBezier with the given points and returns the pointer.
 */
LIBSBML_EXTERN
CubicBezier_t *
CubicBezier_createWithPoints (const Point_t *start, const Point_t *base1,
                              const Point_t *base2, const Point_t *end)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow)CubicBezier(&layoutns, start , base1, base2 , end );
}


/**
 * Creates a CubicBezier with the given coordinates and returns the
 * pointer.
 */
LIBSBML_EXTERN
CubicBezier_t *
CubicBezier_createWithCoordinates (double x1, double y1, double z1,
                                   double x2, double y2, double z2,
                                   double x3, double y3, double z3,
                                   double x4, double y4, double z4)
{
  LayoutPkgNamespaces layoutns;

  Point* p1=new Point(&layoutns,x1,y1,z1);  
  Point* p2=new Point(&layoutns,x2,y2,z2);  
  Point* p3=new Point(&layoutns,x3,y3,z3);  
  Point* p4=new  Point(&layoutns,x4,y4,z4);  
  CubicBezier* cb=new(std::nothrow)CubicBezier(&layoutns, p1,p2,p3,p4);
  delete p1;
  delete p2;
  delete p3;
  delete p4;
  return cb;
}


/**
 * Creates a CubicBezier object from a template.
 */
LIBSBML_EXTERN
CubicBezier_t *
CubicBezier_createFrom (const CubicBezier_t *temp)
{
  return new(std::nothrow) CubicBezier(temp ? *temp : CubicBezier());
}


/**
 * Frees the memory for the cubic bezier.
 */
LIBSBML_EXTERN
void
CubicBezier_free (CubicBezier_t *cb)
{
  delete cb;
}


/**
 * Initializes start point with a copy of the given point.
 */
LIBSBML_EXTERN
void
CubicBezier_setStart (CubicBezier_t *cb, const Point_t *start)
{
  LineSegment_setStart((LineSegment_t*)cb, start);
}


/**
 * Returns the starting point of the curve.
 */ 
LIBSBML_EXTERN
Point_t *
CubicBezier_getStart (CubicBezier_t *cb)
{
  return LineSegment_getStart(cb);
}


/**
 * Initializes end point with a copy of the given point.
 */
LIBSBML_EXTERN
void
CubicBezier_setEnd (CubicBezier_t *cb, const Point_t *end)
{
  LineSegment_setEnd((LineSegment_t*)cb, end);
}


/**
 * Returns the end point of the curve.
 */ 
LIBSBML_EXTERN
Point_t *
CubicBezier_getEnd (CubicBezier_t *cb)
{
  return LineSegment_getEnd(cb);
}


/**
 * Initializes the first base point with a copy of the given point.
 */
LIBSBML_EXTERN
void
CubicBezier_setBasePoint1 (CubicBezier_t *cb, const Point_t *point)
{
  cb->setBasePoint1(point);
}


/**
 * Returns the first base point of the curve (the one closer to the
 * starting point).
 */ 
LIBSBML_EXTERN
Point_t *
CubicBezier_getBasePoint1 (CubicBezier_t *cb)
{
  return cb->getBasePoint1();
}


/**
 * Initializes the second base point with a copy of the given point.
 */
LIBSBML_EXTERN
void
CubicBezier_setBasePoint2 (CubicBezier_t *cb, const Point_t *point)
{
  cb->setBasePoint2(point );
}


/**
 * Returns the second base point of the curve (the one closer to the
 * starting point).
 */ 
LIBSBML_EXTERN
Point_t *
CubicBezier_getBasePoint2 (CubicBezier_t *cb)
{
  return cb->getBasePoint2();
}


/**
 * Calls initDefaults from LineSegment.
 */ 
LIBSBML_EXTERN
void
CubicBezier_initDefaults (CubicBezier_t *cb)
{
  cb->initDefaults();
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
CubicBezier_t *
CubicBezier_clone (const CubicBezier_t *m)
{
  return static_cast<CubicBezier*>( m->clone() );
}


LIBSBML_CPP_NAMESPACE_END

