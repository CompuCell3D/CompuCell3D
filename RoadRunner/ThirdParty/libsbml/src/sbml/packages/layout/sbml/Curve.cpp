/**
 * Filename    : Curve.cpp
 * Description : SBML Layout Curve source
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
#include <iostream>

#include <sbml/packages/layout/sbml/Curve.h>
#include <sbml/packages/layout/sbml/LineSegment.h>
#include <sbml/packages/layout/sbml/CubicBezier.h>
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
 * Creates a curve with the given SBML level, version and package version and 
 * an empty list of segments.
 */ 
Curve::Curve (unsigned int level, unsigned int version, unsigned int pkgVersion) 
 : SBase (level,version)
  ,mCurveSegments(level,version,pkgVersion)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
  connectToChild();
}


/**
 * Creates a curve with the given LayoutPkgNamespaces and an empty list of segments.
 */ 
Curve::Curve (LayoutPkgNamespaces *layoutns)
 : SBase (layoutns)
  ,mCurveSegments(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());

  connectToChild();

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Creates a new ReactionGlyph from the given XMLNode
 */
Curve::Curve(const XMLNode& node, unsigned int l2version)
 : SBase (2,l2version)
  ,mCurveSegments(2,l2version)
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
            this->mAnnotation=new XMLNode(*child);
        }
        else if(childName=="notes")
        {
            this->mNotes=new XMLNode(*child);
        }
        else if(childName=="listOfCurveSegments")
        {
            const XMLNode* innerChild;
            unsigned int i=0,iMax=child->getNumChildren();
            while(i<iMax)
            {
                innerChild=&child->getChild(i);
                const std::string innerChildName=innerChild->getName();
                if(innerChildName=="curveSegment")
                {
                    // get the type
                    const XMLAttributes& innerAttributes=innerChild->getAttributes();
                    int typeIndex=innerAttributes.getIndex("type");
                    if(typeIndex==-1 || innerAttributes.getURI(typeIndex)!="http://www.w3.org/2001/XMLSchema-instance")
                    {
                        // throw
                        ++i;
                        continue;
                    }
                    if(innerAttributes.getValue(typeIndex)=="LineSegment")
                    {
                      this->mCurveSegments.appendAndOwn(new LineSegment(*innerChild));
                    }
                    else if(innerAttributes.getValue(typeIndex)=="CubicBezier")
                    {
                      this->mCurveSegments.appendAndOwn(new CubicBezier(*innerChild));
                    }
                    else
                    {
                        // throw
                    }
                }
                else if(innerChildName=="annotation")
                {
                    this->mCurveSegments.setAnnotation(new XMLNode(*innerChild));
                }
                else if(innerChildName=="notes")
                {
                    this->mCurveSegments.setNotes(new XMLNode(*innerChild));
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
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(2,l2version));
  connectToChild();
}



/**
 * Destructor.
 */ 
Curve::~Curve ()
{
}


/**
 * Does nothing since no defaults are defined for Curve.
 */ 
void Curve::initDefaults ()
{
}


/**
 * Ctor.
 */
ListOfLineSegments::ListOfLineSegments(unsigned int level, unsigned int version, unsigned int pkgVersion)
 : ListOf(level,version)
{
  setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));
};


/**
 * Ctor.
 */
ListOfLineSegments::ListOfLineSegments(LayoutPkgNamespaces* layoutns)
 : ListOf(layoutns)
{
  //
  // set the element namespace of this object
  //
  setElementNamespace(layoutns->getURI());
}


/* return nth item in list */
LineSegment *
ListOfLineSegments::get(unsigned int n)
{
  return static_cast<LineSegment*>(ListOf::get(n));
}


/* return nth item in list */
const LineSegment *
ListOfLineSegments::get(unsigned int n) const
{
  return static_cast<const LineSegment*>(ListOf::get(n));
}


/* Removes the nth item from this list */
LineSegment*
ListOfLineSegments::remove (unsigned int n)
{
   return static_cast<LineSegment*>(ListOf::remove(n));
}


bool 
ListOfLineSegments::isValidTypeForList(SBase * item)
{
  int tc = item->getTypeCode();
  return ((tc == SBML_LAYOUT_CUBICBEZIER )
    ||    (tc == SBML_LAYOUT_LINESEGMENT ) );
}


/**
 * Returns a reference to the ListOf object that holds all the curve
 * segments.
 */
const ListOfLineSegments*
Curve::getListOfCurveSegments () const
{
  return & this->mCurveSegments;
}


/**
 * Returns a reference to the ListOf object that holds all the curve
 * segments.
 */
ListOfLineSegments*
Curve::getListOfCurveSegments ()
{
  return &this->mCurveSegments;
}



/**
 * Returns a pointer to the curve segment with the given index.  If the
 * index is invalid, NULL is returned.
 */  
const LineSegment*
Curve::getCurveSegment (unsigned int index) const
{
  return dynamic_cast<const LineSegment*>( this->mCurveSegments.get(index) );
}


/**
 * Returns a pointer to the curve segment with the given index.  If the
 * index is invalid, NULL is returned.
 */  
LineSegment*
Curve::getCurveSegment (unsigned int index)
{
  return static_cast<LineSegment*>( this->mCurveSegments.get(index) );
}


/**
 * Adds a new CurveSegment to the end of the list.
 */ 
void
Curve::addCurveSegment (const LineSegment* segment)
{
  this->mCurveSegments.append(segment);
}


/**
 * Returns the number of curve segments.
 */ 
unsigned int
Curve::getNumCurveSegments () const
{
  return this->mCurveSegments.size();
}


/**
 * Creates a new LineSegment and adds it to the end of the list.  A
 * reference to the new LineSegment object is returned.
 */
LineSegment*
Curve::createLineSegment ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  LineSegment* ls = new LineSegment(layoutns);

  this->mCurveSegments.appendAndOwn(ls);
  return ls;
}


/**
 * Creates a new CubicBezier and adds it to the end of the list.  A
 * reference to the new CubicBezier object is returned.
 */
CubicBezier* Curve::createCubicBezier ()
{
  LAYOUT_CREATE_NS(layoutns,getSBMLNamespaces());
  CubicBezier* cb = new CubicBezier(layoutns);

  this->mCurveSegments.appendAndOwn(cb);
  return cb;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& Curve::getElementName () const 
{
  static const std::string name = "curve";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
Curve* 
Curve::clone () const
{
    return new Curve(*this);
}


/**
 * Copy constructor.
 */
Curve::Curve(const Curve& source):SBase(source)
{
    // copy the line segments
    this->mCurveSegments=*source.getListOfCurveSegments();

    connectToChild();
}

/**
 * Assignment operator.
 */
Curve& Curve::operator=(const Curve& source)
{
  if(&source!=this)
  {
    this->SBase::operator=(source);
    // copy the line segments
    this->mCurveSegments=*source.getListOfCurveSegments();

    connectToChild();  
  }
  
  return *this;
}



/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
Curve::createObject (XMLInputStream& stream)
{

  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;

  if (name == "listOfCurveSegments")
  {
    object = &mCurveSegments;
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
Curve::addExpectedAttributes(ExpectedAttributes& attributes)
{
  SBase::addExpectedAttributes(attributes);
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void Curve::readAttributes (const XMLAttributes& attributes,
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
Curve::writeElements (XMLOutputStream& stream) const
{
  SBase::writeElements(stream);

  mCurveSegments.write(stream);

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
void Curve::writeAttributes (XMLOutputStream& stream) const
{
  SBase::writeAttributes(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode Curve::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("curve", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
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
  // add the list of line segments
  if(this->mCurveSegments.size()>0)
  {
      node.addChild(this->mCurveSegments.toXML());
      end=false;
  }
  if(end==true) node.setEnd();
  return node;
}


/*
 * Sets the parent SBMLDocument of this SBML object.
 */
void
Curve::setSBMLDocument (SBMLDocument* d)
{
  SBase::setSBMLDocument(d);

  mCurveSegments.setSBMLDocument(d);
}


/*
 * Sets this SBML object to child SBML objects (if any).
 * (Creates a child-parent relationship by the parent)
 */
void
Curve::connectToChild()
{
  mCurveSegments.connectToParent(this);
}


/**
 * Enables/Disables the given package with this element and child
 * elements (if any).
 * (This is an internal implementation for enablePakcage function)
 */
void
Curve::enablePackageInternal(const std::string& pkgURI,
                                      const std::string& pkgPrefix, bool flag)
{
  SBase::enablePackageInternal(pkgURI,pkgPrefix,flag);

  mCurveSegments.enablePackageInternal(pkgURI,pkgPrefix,flag);
}


/**
 * @return a (deep) copy of this ListOfUnitDefinitions.
 */
ListOfLineSegments*
ListOfLineSegments::clone () const
{
  return new ListOfLineSegments(*this);
}


/**
 * @return the typecode (int) of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
int
ListOfLineSegments::getItemTypeCode () const
{
  return SBML_LAYOUT_LINESEGMENT;
}


/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string&
ListOfLineSegments::getElementName () const
{
  static const std::string name = "listOfCurveSegments";
  return name;
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
ListOfLineSegments::createObject (XMLInputStream& stream)
{
  const std::string& name   = stream.peek().getName();
  SBase*        object = 0;


  if (name == "curveSegment")
  {
    std::string type = "LineSegment";
    XMLTriple triple("type","http://www.w3.org/2001/XMLSchema-instance","xsi");

    if (!stream.peek().getAttributes().readInto(triple, type))
    {
      //std::cout << "[DEBUG] ListOfLineSegments::createObject () : Failed to read xsi:type" << std::endl;
      return object;
    }

    //std::cout << "[DEBUG] ListOfLineSegments::createObject () : type " << type << std::endl;
    LAYOUT_CREATE_NS(layoutns,this->getSBMLNamespaces());
    if(type=="LineSegment")
    {
      object = new LineSegment(layoutns);
    }
    else if(type=="CubicBezier")
    {
      object = new CubicBezier(layoutns);
    }
  }
  
  if(object) appendAndOwn(object);

  return object;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode ListOfLineSegments::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("listOfCurveSegments", "http://projects.eml.org/bcb/sbml/level2", "");
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
  const LineSegment* object=NULL;
  for(i=0;i<iMax;++i)
  {
    object=dynamic_cast<const LineSegment*>(this->get(i));
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
 * Returns the package type code  for this object.
 */
int
Curve::getTypeCode () const
{
  return SBML_LAYOUT_CURVE;
}


/**
 * Accepts the given SBMLVisitor.
 */
bool
Curve::accept (SBMLVisitor& v) const
{
    
  /*bool result=v.visit(*this);
  mCurveSegments.accept(v);
  v.leave(*this);*/
  return false;
}








/**
 * Creates a new curve and returns the pointer to it.
 */
LIBSBML_EXTERN
Curve_t *
Curve_create (void)
{
  return new(std::nothrow) Curve;
}


/**
 * Creates a new Curve object from a template.
 */
LIBSBML_EXTERN
Curve_t *
Curve_createFrom (const Curve_t *temp)
{
  return new(std::nothrow) Curve(temp ? *temp : Curve());
}


/**
 * Frees the memory taken by the Curve.
 */
LIBSBML_EXTERN
void
Curve_free (Curve_t *c)
{
  delete c;
}


/**
 * Adds a LineSegment.
 */
LIBSBML_EXTERN
void
Curve_addCurveSegment (Curve_t *c, LineSegment_t *ls)
{
  c->addCurveSegment(ls);
}


/**
 * Returns the number of line segments.
 */
LIBSBML_EXTERN
unsigned int
Curve_getNumCurveSegments (const Curve_t *c)
{
  return c->getNumCurveSegments();
}


/**
 * Returns the line segment with the given index.
 */
LIBSBML_EXTERN
LineSegment_t *
Curve_getCurveSegment (const Curve_t *c, unsigned int index)
{
  return const_cast<LineSegment*>(c->getCurveSegment(index));
}


/**
 * Returns the ListOf object that holds all the curve segments.
 */ 
LIBSBML_EXTERN
ListOf_t *
Curve_getListOfCurveSegments (Curve_t *c)
{
  return c->getListOfCurveSegments();
}


/**
 * Does nothing since no defaults are defined for Curve.
 */ 
LIBSBML_EXTERN
void
Curve_initDefaults (Curve_t *c)
{
  c->initDefaults();
}


/**
 * Creates a new LineSegment and adds it to the end of the list.  A pointer
 * to the new LineSegment object is returned.
 */
LIBSBML_EXTERN
LineSegment_t *
Curve_createLineSegment (Curve_t *c)
{
  return c->createLineSegment();
}


/**
 * Creates a new CubicBezier and adds it to the end of the list.  A pointer
 * to the new CubicBezier object is returned.
 */
LIBSBML_EXTERN
CubicBezier_t *
Curve_createCubicBezier (Curve_t *c)
{
  return c->createCubicBezier();
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
Curve_t *
Curve_clone (const Curve_t *m)
{
  return static_cast<Curve*>( m->clone() );
}

LIBSBML_CPP_NAMESPACE_END

