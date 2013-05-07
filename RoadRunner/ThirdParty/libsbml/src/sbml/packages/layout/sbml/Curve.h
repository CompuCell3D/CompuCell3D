/**
 * Filename    : Curve.h
 * Description : SBML Layout Curve C++ Header
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


#ifndef Curve_H__
#define Curve_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/ListOf.h>
#include <sbml/packages/layout/sbml/LineSegment.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN ListOfLineSegments : public ListOf
{
 public:

  /**
   * @return a (deep) copy of this ListOfLineSegments.
   */
  virtual ListOfLineSegments* clone () const;


  /**
   * Ctor.
   */
   ListOfLineSegments(unsigned int level      = LayoutExtension::getDefaultLevel(), 
                      unsigned int version    = LayoutExtension::getDefaultVersion(), 
                      unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());

  /**
   * Ctor.
   */
   ListOfLineSegments(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;


  /**
   * Get a LineSegment from the ListOfLineSegments.
   *
   * @param n the index number of the LineSegment to get.
   * 
   * @return the nth LineSegment in this ListOfLineSegments.
   *
   * @see size()
   */
  virtual LineSegment * get(unsigned int n); 


  /**
   * Get a LineSegment from the ListOfLineSegments.
   *
   * @param n the index number of the LineSegment to get.
   * 
   * @return the nth LineSegment in this ListOfLineSegments.
   *
   * @see size()
   */
  virtual const LineSegment * get(unsigned int n) const; 


  /**
   * Removes the nth item from this ListOfLineSegments items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual LineSegment* remove (unsigned int n);


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;
    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
  
  
  virtual bool isValidTypeForList(SBase * item);
};
  
class LIBSBML_EXTERN Curve : public SBase
{
protected:

  ListOfLineSegments mCurveSegments;


public:

  /**
   * Creates a curve with an empty list of segments.
   */ 
  
  Curve (unsigned int level      = LayoutExtension::getDefaultLevel(),
         unsigned int version    = LayoutExtension::getDefaultVersion(),
         unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Creates a new Curve with the given LayoutPkgNamespaces object.
   */
  Curve (LayoutPkgNamespaces* layoutns);


  /**
   * Creates a new Curve from the given XMLNode
   */
  Curve(const XMLNode& node, unsigned int l2version=4);


  /**
   * Copy constructor.
   */
   Curve(const Curve& source);

  /**
   * Assignment operator.
   */
   Curve& operator=(const Curve& source);

  /**
   * Destructor.
   */ 
  virtual ~Curve ();

  /**
   * Does nothing since no defaults are defined for Curve.
   */ 
  
  void initDefaults ();

  /**
   * Returns a reference to the ListOf object that holds all the curve
   * segments.
   */
  
  const ListOfLineSegments* getListOfCurveSegments () const;
       
  /**
   * Returns a refernce to the ListOf object That holds all the curve
   * segments.
   */
  
  ListOfLineSegments* getListOfCurveSegments ();

  /**
   * Returns a pointer to the curve segment with the given index.
   * If the index is invalid, NULL is returned.
   */  
  const LineSegment* getCurveSegment (unsigned int index) const;

  /**
   * Returns a pointer to the curve segment with the given index.
   * If the index is invalid, NULL is returned.
   */  
  LineSegment* getCurveSegment (unsigned int index);

  /**
   * Adds a new CurveSegment to the end of the list.
   */ 
  
  void addCurveSegment (const LineSegment* segment);
  
  /**
   * Returns the number of curve segments.
   */ 
  
  unsigned int getNumCurveSegments () const;


  /**
   * Creates a new LineSegment and adds it to the end of the list.  A
   * reference to the new LineSegment object is returned.
   */
  
  LineSegment* createLineSegment ();

  /**
   * Creates a new CubicBezier and adds it to the end of the list.  A
   * reference to the new CubicBezier object is returned.
   */
  
  CubicBezier* createCubicBezier ();

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
   * @return a (deep) copy of this Curve.
   */
  virtual Curve* clone () const;


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
    XMLNode toXML() const;

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
 * Creates a new curve and returns the pointer to it.
 */
LIBSBML_EXTERN
Curve_t *
Curve_create ();

/**
 * Creates a new Curve object from a template.
 */
LIBSBML_EXTERN
Curve_t *
Curve_createFrom (const Curve_t *c);

/**
 * Frees the memory taken by the Curve.
 */
LIBSBML_EXTERN
void
Curve_free (Curve_t *c);


/**
 * Adds a LineSegment.
 */
LIBSBML_EXTERN
void
Curve_addCurveSegment (Curve_t *c, LineSegment_t *ls);

/**
 * Returns the number of line segments.
 */
LIBSBML_EXTERN
unsigned int
Curve_getNumCurveSegments (const Curve_t *c);

/**
 * Returns the line segment with the given index.
 */
LIBSBML_EXTERN
LineSegment_t *
Curve_getCurveSegment (const Curve_t *c, unsigned int index);

/**
 * Returns the ListOf object that holds all the curve segments.
 */ 
LIBSBML_EXTERN
ListOf_t *
Curve_getListOfCurveSegments (Curve_t *curve);

/**
 * Removes the curve segment with the given index.  If the index is
 * invalid, nothing is done.
 */ 
LIBSBML_EXTERN
LineSegment_t *
Curve_removeCurveSegment (Curve_t *c, unsigned int index);

/**
 * Does nothing since no defaults are defined for Curve.
 */ 
LIBSBML_EXTERN
void
Curve_initDefaults (Curve_t *c);

/**
 * Creates a new LineSegment and adds it to the end of the list.  A pointer
 * to the new LineSegment object is returned.
 */
LIBSBML_EXTERN
LineSegment_t *
Curve_createLineSegment (Curve_t *c);

/**
 * Creates a new CubicBezier and adds it to the end of the list.  A pointer
 * to the new CubicBezier object is returned.
 */
LIBSBML_EXTERN
CubicBezier_t *
Curve_createCubicBezier (Curve_t *c);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
Curve_t *
Curve_clone (const Curve_t *m);



END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* Curve_H__ */
