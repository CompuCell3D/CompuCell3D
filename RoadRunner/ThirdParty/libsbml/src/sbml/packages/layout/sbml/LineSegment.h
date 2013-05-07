/**
 * Filename    : LineSegment.h
 * Description : SBML Layout LineSegment C++ Header
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


#ifndef LineSegment_H__
#define LineSegment_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/packages/layout/sbml/Point.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN LineSegment : public SBase
{
protected:

  Point mStartPoint;
  Point mEndPoint;


public:

  /**
   * Creates a line segment with the given SBML level, version, and package version
   * and both points set to (0.0,0.0,0.0)
   */ 
  LineSegment (unsigned int level      = LayoutExtension::getDefaultLevel(),
               unsigned int version    = LayoutExtension::getDefaultVersion(),
               unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());

  /**
   * Creates a line segment with the LayoutPkgNamespaces and both points set to (0.0,0.0,0.0)
   */ 
  LineSegment (LayoutPkgNamespaces* layoutns);


  /**
   * Creates a new line segment with the given 2D coordinates.
   */ 
  
  LineSegment (LayoutPkgNamespaces* layoutns, double x1, double y1, double x2, double y2);

  /**
   * Copy constructor.
   */
  LineSegment(const LineSegment& orig);

  /**
   * Creates a new line segment with the given 3D coordinates.
   */ 
  LineSegment(LayoutPkgNamespaces* layoutns, double x1, double y1, double z1, double x2, double y2, double z2);

  /**
   * Creates a new line segment with the two given points.
   */ 
  
  LineSegment (LayoutPkgNamespaces* layoutns, const Point* start, const Point* end);


  /**
   * Creates a new LineSegment from the given XMLNode
   */
  LineSegment(const XMLNode& node, unsigned int l2version=4);

  /**
   * Destructor.
   */ 
  
  virtual ~LineSegment ();

  /**
   * Assignment operator
   */
  virtual LineSegment& operator=(const LineSegment& orig);


  /**
   * Returns the start point of the line.
   */ 
  
  const Point* getStart () const;

  /**
   * Returns the start point of the line.
   */ 
  
  Point* getStart ();

  /**
   * Initializes the start point with a copy of the given Point object.
   */
  
  void setStart (const Point* start);

  /**
   * Initializes the start point with the given coordinates.
   */
  
  void setStart (double x, double y, double z = 0.0);

  /**
   * Returns the end point of the line.
   */ 
  
  const Point* getEnd () const;

  /**
   * Returns the end point of the line.
   */ 
  
  Point* getEnd ();

  /**
   * Initializes the end point with a copy of the given Point object.
   */
  
  void setEnd (const Point* end);

  /**
   * Initializes the end point with the given coordinates.
   */
  
  void setEnd (double x, double y, double z = 0.0);

  /**
   * Does noting since no defaults are defined for LineSegment.
   */ 
  
  void initDefaults ();

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
  virtual void writeElements (XMLOutputStream& stream) const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const ;

  /**
   * @return a (deep) copy of this LineSegment.
   */
  virtual LineSegment* clone () const;


  /**
   * Returns the libSBML type code for this object.
   *
   * This method MAY return the typecode of this SBML object or it MAY
   * return SBML_UNKNOWN.  That is, subclasses of SBase are not required to
   * implement this method to return a typecode.  This method is meant
   * primarily for the LibSBML C interface where class and subclass
   * information is not readily available.
   *
   * @note In libSBML 5, the type of return value has been changed from
   *       SBMLTypeCode_t to int. The return value is one of enum values defined
   *       for each package. For example, return values will be one of
   *       SBMLTypeCode_t if this object is defined in SBML core package,
   *       return values will be one of SBMLLayoutTypeCode_t if this object is
   *       defined in Layout extension (i.e. similar enum types are defined in
   *       each pacakge extension for each SBase subclass)
   *       The value of each typecode can be duplicated between those of
   *       different packages. Thus, to distinguish the typecodes of different
   *       packages, not only the return value of getTypeCode() but also that of
   *       getPackageName() must be checked.
   *
   * @return the typecode (int value) of this SBML object or SBML_UNKNOWN
   * (default).
   *
   * @see getElementName()
   * @see getPackageName()
   */
  virtual int getTypeCode () const;


  /**
   * Accepts the given SBMLVisitor.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether or not the Visitor would like to visit the SBML object's next
   * sibling object (if available).
   */
  virtual bool accept (SBMLVisitor& v) const;
   

   /**
    * Creates an XMLNode object from this.
    */
    virtual XMLNode toXML() const;

  /** @cond doxygen-libsbml-internal */
  /**
   * Sets the parent SBMLDocument of this SBML object.
   *
   * @param d the SBMLDocument object to use
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets this SBML object to child SBML objects (if any).
   * (Creates a child-parent relationship by the parent)
   *
   * Subclasses must override this function if they define
   * one ore more child elements.
   * Basically, this function needs to be called in
   * constructor, copy constructor, assignment operator.
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToChild ();


  /**
   * Enables/Disables the given package with this element and child
   * elements (if any).
   * (This is an internal implementation for enablePakcage function)
   *
   * @note Subclasses in which one or more child elements are defined
   * must override this function.
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);
  /** @endcond */
    
protected:
  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase*
  createObject (XMLInputStream& stream);

  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);


  /**
   * Subclasses should override this method to read values from the given
   * XMLAttributes set into their specific fields.  Be sure to call your
   * parents implementation of this method as well.
   */
  virtual void readAttributes (const XMLAttributes& attributes, 
                               const ExpectedAttributes& expectedAttributes);

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
  virtual void writeAttributes (XMLOutputStream& stream) const;


};

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS



/**
 * Creates a LineSegment and returns the pointer.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_create (void);


/**
 * Creates a LineSegment from a template.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_createFrom (const LineSegment_t *temp);

/**
 * Creates a LineSegment with the given points and returns the pointer.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_createWithPoints (const Point_t *start, const Point_t *end);

/**
 * Creates a LineSegment with the given coordinates and returns the
 * pointer.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_createWithCoordinates (double x1, double y1, double z1,
                                   double x2, double y2, double z2);

/**
 * Frees the memory for the line segment.
 */
LIBSBML_EXTERN
void
LineSegment_free (LineSegment_t *ls);


/**
 * Initializes the start point with a copy of the given Point object.
 */
LIBSBML_EXTERN
void 
LineSegment_setStart (LineSegment_t *ls, const Point_t *start);

/**
 * Initializes the end point with a copy of the given Point object.
 */
LIBSBML_EXTERN
void 
LineSegment_setEnd (LineSegment_t *ls, const Point_t *end);


/**
 * Returns the start point of the line.
 */ 
LIBSBML_EXTERN
Point_t *
LineSegment_getStart (LineSegment_t *ls);

/**
 * Returns the end point of the line.
 */ 
LIBSBML_EXTERN
Point_t *
LineSegment_getEnd (LineSegment_t *ls);

/**
 * Does noting since no defaults are defined for LineSegment.
 */ 
LIBSBML_EXTERN
void
LineSegment_initDefaults (LineSegment_t *ls);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
LineSegment_t *
LineSegment_clone (const LineSegment_t *m);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* LineSegment_H__ */
