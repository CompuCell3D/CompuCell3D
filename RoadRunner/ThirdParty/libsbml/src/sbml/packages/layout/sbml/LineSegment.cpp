/**
 * Filename    : LineSegment.cpp
 * Description : SBML Layout LineSegment source
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


#include <sbml/packages/layout/sbml/LineSegment.h>
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
 * Creates a line segment with the given SBML level, version, and package version
 * and both points set to (0.0,0.0,0.0)
 */ 
LineSegment::LineSegment (unsigned int level, unsigned int version, unsigned int pkgVersion)
 :  SBase (level,version)
  , mStartPoint(level,version,pkgVersion)
  , mEndPoint  (level,version,pkgVersion)
{
  this->mStartPoint.setElementName("start");
  this->mEndPoint.setElementName("end");

  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
  connectToChild();
}


/**
 * Creates a new line segment with the given LayoutPkgNamespaces
 */ 
LineSegment::LineSegment (LayoutPkgNamespaces* layoutns)
 : SBase (layoutns)
 , mStartPoint(layoutns)
 , mEndPoint (layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  this->mStartPoint.setElementName("start");
  this->mEndPoint.setElementName("end");

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Creates a new line segment with the given 2D coordinates.
 */ 
LineSegment::LineSegment (LayoutPkgNamespaces* layoutns, double x1, double y1, double x2, double y2) 
 : SBase (layoutns)
 , mStartPoint(layoutns, x1, y1, 0.0 )
 , mEndPoint (layoutns, x2, y2, 0.0 )
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  this->mStartPoint.setElementName("start");
  this->mEndPoint.setElementName("end");

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Creates a new line segment with the given 3D coordinates.
 */ 
LineSegment::LineSegment (LayoutPkgNamespaces* layoutns, double x1, double y1, double z1,
                          double x2, double y2, double z2) 
 : SBase(layoutns)
  , mStartPoint(layoutns, x1, y1, z1)
  , mEndPoint  (layoutns, x2, y2, z2)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  this->mStartPoint.setElementName("start");
  this->mEndPoint.setElementName("end");

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Copy constructor.
 */
LineSegment::LineSegment(const LineSegment& orig):SBase(orig)
{
  this->mStartPoint=orig.mStartPoint;
  this->mEndPoint=orig.mEndPoint;

  connectToChild();
}


/**
 * Assignment operator.
 */
LineSegment& LineSegment::operator=(const LineSegment& orig)
{
  if(&orig!=this)
  {
    this->SBase::operator=(orig);
    this->mStartPoint=orig.mStartPoint;
    this->mEndPoint=orig.mEndPoint;
    connectToChild();
  }
  
  return *this;
}


/**
 * Creates a new line segment with the two given points.
 */ 
LineSegment::LineSegment (LayoutPkgNamespaces* layoutns, const Point* start, const Point* end) 
 : SBase (layoutns)
 , mStartPoint(layoutns)
 , mEndPoint  (layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  if(start && end)
  {  
    this->mStartPoint=*start;  
    this->mStartPoint.setElementName("start");
    this->mEndPoint=*end;  
    this->mEndPoint.setElementName("end");
  }

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Creates a new LineSegment from the given XMLNode
 */
LineSegment::LineSegment(const XMLNode& node, unsigned int l2version)
 : SBase (2, l2version)
 , mStartPoint(2, l2version)
 , mEndPoint  (2, l2version)
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
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(2,l2version));  
}


/**
 * Destructor.
 */ 
LineSegment::~LineSegment ()
{
}


/**
 * Does nothing since no defaults are defined for LineSegment.
 */ 
void LineSegment::initDefaults ()
{
}


/**
 * Returns the start point of the line.
 */ 
const Point*
LineSegment::getStart () const
{
  return &this->mStartPoint;
}


/**
 * Returns the start point of the line.
 */ 
Point*
LineSegment::getStart()
{
  return &this->mStartPoint;
}


/**
 * Initializes the start point with a copy of the given Point object.
 */
void
LineSegment::setStart (const Point* start)
{
  if(start)
  {  
    this->mStartPoint=*start;
    this->mStartPoint.setElementName("start");
    this->mStartPoint.connectToParent(this);
  }
}


/**
 * Initializes the start point with the given coordinates.
 */
void
LineSegment::setStart (double x, double y, double z)
{
  this->mStartPoint.setOffsets(x, y, z);
}


/**
 * Returns the end point of the line.
 */ 
const Point*
LineSegment::getEnd () const
{
  return &this->mEndPoint;
}


/**
 * Returns the end point of the line.
 */ 
Point*
LineSegment::getEnd ()
{
  return &this->mEndPoint;
}


/**
 * Initializes the end point with a copy of the given Point object.
 */
void
LineSegment::setEnd (const Point* end)
{
  if(end)
  {  
    this->mEndPoint = *end;
    this->mEndPoint.setElementName("end");
    this->mEndPoint.connectToParent(this);
  }
}


/**
 * Initializes the end point with the given coordinates.
 */
void
LineSegment::setEnd (double x, double y, double z)
{
  this->mEndPoint.setOffsets(x, y, z);
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& LineSegment::getElementName () const 
{
  static const std::string name = "curveSegment";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
LineSegment* 
LineSegment::clone () const
{
    return new LineSegment(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
LineSegment::createObject (XMLInputStream& stream)
{

  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;

  if (name == "start")
  {
    object = &mStartPoint;
  }
  else if(name == "end")
  {
    object = &mEndPoint;
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
LineSegment::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);
  attributes.add("xsi:type");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void LineSegment::readAttributes (const XMLAttributes& attributes,
                                  const ExpectedAttributes& expectedAttributes)
{
  SBase::readAttributes(attributes,expectedAttributes);

}

/**
 * Subclasses should override this method to write out their contained
 * SBML objects as XML elements.  Be sure to call your parents
 * implementation of this method as well.
 */
void
LineSegment::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);
  mStartPoint.write(stream);
  mEndPoint.write(stream);

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
void LineSegment::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);
  stream.writeAttribute("type", "xsi", "LineSegment");

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}


/**
 * Returns the package type code for this object.
 */
int
LineSegment::getTypeCode () const
{
  return SBML_LAYOUT_LINESEGMENT;
}

/**
 * Accepts the given SBMLVisitor.
 */
bool
LineSegment::accept (SBMLVisitor& v) const
{
   /*
  bool result=v.visit(*this);
  this->mStartPoint.accept(v);
  this->mEndPoint.accept(v);
  v.leave(*this);*/
  return false;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode LineSegment::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("curveSegment", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  att.add("type","LineSegment","http://www.w3.org/2001/XMLSchema-instance","xsi");
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  if(this->mNotes) node.addChild(*this->mNotes);
  if(this->mAnnotation) node.addChild(*this->mAnnotation);
  // add start point
  node.addChild(this->mStartPoint.toXML("start"));
  // add end point
  node.addChild(this->mEndPoint.toXML("end"));
  return node;
}


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
LineSegment::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);

  mStartPoint.setSBMLDocument(d);
  mEndPoint.setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
LineSegment::connectToChild()
{
  mStartPoint.connectToParent(this);
  mEndPoint.connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
LineSegment::enablePackageInternal(const std::string& pkgURI,
                                   const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mStartPoint.enablePackageInternal(pkgURI,pkgPrefix,flag);
  mEndPoint.enablePackageInternal(pkgURI,pkgPrefix,flag);
}




/**
 * Creates a LineSegment and returns the pointer.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_create (void)
{
  return new(std::nothrow) LineSegment;
}


/**
 * Creates a LineSegment from a template.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_createFrom (const LineSegment_t *temp)
{
  return new(std::nothrow) LineSegment(temp ? *temp : LineSegment());
}


/**
 * Creates a LineSegment with the given points and returns the pointer.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_createWithPoints (const Point_t *start, const Point_t *end)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) LineSegment (&layoutns, start, end );
}


/**
 * Creates a LineSegment with the given coordinates and returns the pointer.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_createWithCoordinates (double x1, double y1, double z1,
                                   double x2, double y2, double z2)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) LineSegment(&layoutns, x1, y1, z1, x2, y2, z2);
}


/**
 * Frees the memory for the line segment.
 */
LIBSBML_EXTERN
void
LineSegment_free (LineSegment_t *ls)
{
  delete ls;
}


/**
 * Initializes the start point with a copy of the given Point object.
 */
LIBSBML_EXTERN
void
LineSegment_setStart (LineSegment_t *ls, const Point_t *start)
{
  ls->setStart(start);
}


/**
 * Initializes the end point with a copy of the given Point object.
 */
LIBSBML_EXTERN
void
LineSegment_setEnd (LineSegment_t *ls, const Point_t *end)
{
  ls->setEnd(end);
}


/**
 * Returns the start point of the line.
 */ 
LIBSBML_EXTERN
Point_t *
LineSegment_getStart (LineSegment_t *ls)
{
  return ls->getStart();
}


/**
 * Returns the end point of the line.
 */ 
LIBSBML_EXTERN
Point_t *
LineSegment_getEnd (LineSegment_t *ls)
{
  return ls->getEnd();
}


/**
 * Does nothing since no defaults are defined for LineSegment.
 */ 
LIBSBML_EXTERN
void
LineSegment_initDefaults (LineSegment_t *ls)
{
  ls->initDefaults();
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_clone (const LineSegment_t *m)
{
  return static_cast<LineSegment*>( m->clone() );
}

LIBSBML_CPP_NAMESPACE_END

