/**
 * Filename    : BoundingBox.h
 * Description : SBML Layout BoundingBox C++ Header
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


#ifndef BoundingBox_H__
#define BoundingBox_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>

#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>

#include <sbml/packages/layout/sbml/Point.h>
#include <sbml/packages/layout/sbml/Dimensions.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN BoundingBox : public SBase
{
protected:

  std::string mId;

  Point mPosition;
  Dimensions mDimensions;


public:
        
  /**
   * Default Constructor set position and dimensions to (0.0,0.0,0.0) and
   * the id to an empty string.
   */ 
  
  BoundingBox (unsigned int level      = LayoutExtension::getDefaultLevel(),
               unsigned int version    = LayoutExtension::getDefaultVersion(),
               unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Creates a new BoundingBox object with the given LayoutPkgNamespaces object.
   */
  BoundingBox(LayoutPkgNamespaces* layoutns);

  
  /**
   * Copy constructor.
   */
  BoundingBox(const BoundingBox& orig); 

  /**
   * Constructor set position and dimensions to (0.0,0.0,0.0) and the id to
   * a copy of the given string.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */ 
  
  BoundingBox (LayoutPkgNamespaces* layoutns, const std::string id);
        
  /**
   * Constructor which sets the id, the coordinates and the dimensions to
   * the given 2D values.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */ 
  
  BoundingBox (LayoutPkgNamespaces* layoutns, const std::string id, double x, double y,
               double width, double height);
        
  /**
   * Constructor which sets the id, the coordinates and the dimensions to
   * the given 3D values.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */ 
  
  BoundingBox (LayoutPkgNamespaces* layoutns, const std::string id, double x, double y, double z,
               double width, double height, double depth);
  
  /**
   * Constructor which sets the id, the coordinates and the dimensions to
   * the given values.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */ 
  
  BoundingBox (LayoutPkgNamespaces* layoutns, const std::string id, const Point* p, const Dimensions* d);

  /**
   * Creates a new BoundingBox from the given XMLNode
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
   BoundingBox(const XMLNode& node, unsigned int l2version=4);

  /**
   * Destructor which does nothing.
   */ 
  
  virtual ~BoundingBox ();
        
  /**
   * Assignment operator
   */
  BoundingBox& operator=(const BoundingBox& orig);

  /**
   * Returns the value of the "id" attribute of this BoundingBox.
   */
  virtual const std::string& getId () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * BoundingBox's "id" attribute has been set.
   */
  virtual bool isSetId () const;

  
  /**
   * Sets the value of the "id" attribute of this BoundingBox.
   */
  virtual int setId (const std::string& id);


  /**
   * Unsets the value of the "id" attribute of this BoundingBox.
   */
  virtual int unsetId ();


  /**
   * Returns the position of the BoundingBox as const referece to a Point
   * object.
   */ 
  
  const Point* getPosition () const;

  /**
   * Returns the dimensions of the BoundingBox as const referece to a
   * Dimensions object.
   */ 
  
  const Dimensions* getDimensions () const;
        
  /**
   * Returns the position of the BoundingBox as referece to a Point object.
   */ 
  
  Point* getPosition ();
        
  /**
   * Returns the dimensions of the BoundingBox as referece to a Dimensions
   * object.
   */ 
  
  Dimensions* getDimensions ();
        
  /**
   * Sets the position to a copy of the Point object given.
   */ 
  
  void setPosition (const Point* p);
        
  /**
   * Sets the dimensions to a copy of the Dimensions object given.
   */ 
  
  void setDimensions (const Dimensions* d);  
        
  /**
   * Does nothing yet since there are no defaults fo a BoundingBox. 
   */ 
  
  void initDefaults ();


  /**
   * Get the x offset of the bounding box.
   */
  
  double x() const;
  
  /**
   * Get the y offset of the bounding box.
   */
  
  double y() const;
  
  /**
   * Get the z offset of the bounding box.
   */
  
  double z() const;
  
  /**
   * Get the width of the bounding box.
   */
  
  double width() const;
  
  /**
   * Get the height of the bounding box.
   */
  
  double height() const;
  
  /**
   * Get the depth of the bounding box.
   */
  
  double depth() const;

  /**
   * Set x offset of the bounding box
   */
  
  void setX(double x);

  /**
   * Set y offset of the bounding box
   */
  
  void setY(double y);

  /**
   * Set z offset of the bounding box
   */
  
  void setZ(double z);

  /**
   * Set width of the bounding box
   */
  
  void setWidth(double width);

  /**
   * Set height of the bounding box
   */
  
  void setHeight(double height);

  /**
   * Set depth of the bounding box
   */
  
  void setDepth(double depth);

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
   * @return a (deep) copy of this BoundingBox.
   */
  virtual BoundingBox* clone () const;


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
 * Function that creates a BoundingBox_t object with position set to
 * (0.0,0.0,0.0) and dimensions set to (0.0,0.0,0.0). The id is set to the
 * empty string.
 */
LIBSBML_EXTERN
BoundingBox_t *
BoundingBox_create (void);

/**
 * ZFunction that creates a BoundingBox_t object with position set to
 * (0.0,0.0,0.0) and dimensions set to (0.0,0.0,0.0).  The id is set to the
 * given string.
 */
LIBSBML_EXTERN
BoundingBox_t *
BoundingBox_createWith (const char *id);

/**
 * Function that creates a BoundingBox_t object with the coordinates and
 * sizes given as arguments. The id is set to the empty string.
 */ 
LIBSBML_EXTERN
BoundingBox_t *
BoundingBox_createWithCoordinates (const char *id, double x, double y, double z,
                                   double width, double height, double depth);

/**
 * Frees all memory taken by the given BoundingBox_t object.
 */ 
LIBSBML_EXTERN
void
BoundingBox_free (BoundingBox_t *bb);

/**
 * Does nothing since no defaults are defined for BoundingBox.
  */
LIBSBML_EXTERN
void
BoundingBox_initDefaults (BoundingBox_t *bb);

/**
 * Returns the position as a Point_t object.
 */ 
LIBSBML_EXTERN
Point_t *
BoundingBox_getPosition (BoundingBox_t *bb);

/**
 * Returns the dimensions as a Dimensions_t object.
 */ 
LIBSBML_EXTERN
Dimensions_t *
BoundingBox_getDimensions (BoundingBox_t *bb);

/**
 * Sets the position to a copy of the Point_t object given as argument.
  */
LIBSBML_EXTERN
void
BoundingBox_setPosition (BoundingBox_t *bb, const Point_t *p);

/**
 * Sets the dimensions to a copy of the Dimensions_t object given.
 */ 
LIBSBML_EXTERN
void
BoundingBox_setDimensions (BoundingBox_t *bb, const Dimensions_t *d);

/**
 * Sets the x offset of the bounding box.
 */
LIBSBML_EXTERN
void
BoundingBox_setX(BoundingBox_t* bb,double x);


/**
 * Sets the y offset of the bounding box.
 */
LIBSBML_EXTERN
void
BoundingBox_setY(BoundingBox_t* bb,double y);


/**
 * Sets the z offset of the bounding box.
 */
LIBSBML_EXTERN
void
BoundingBox_setZ(BoundingBox_t* bb,double z);


/**
 * Sets the width of the bounding box.
 */
LIBSBML_EXTERN
void
BoundingBox_setWidth(BoundingBox_t* bb,double width);


/**
 * Sets the height of the bounding box.
 */
LIBSBML_EXTERN
void
BoundingBox_setHeight(BoundingBox_t* bb,double height);


/**
 * Sets the depth of the bounding box.
 */
LIBSBML_EXTERN
void
BoundingBox_setDepth(BoundingBox_t* bb,double depth);

/**
 * Returns the x offset of the bounding box.
 */
LIBSBML_EXTERN
double
BoundingBox_x(BoundingBox_t* bb);


/**
 * Returns the y offset of the bounding box.
 */
LIBSBML_EXTERN
double
BoundingBox_y(BoundingBox_t* bb);


/**
 * Returns the z offset of the bounding box.
 */
LIBSBML_EXTERN
double
BoundingBox_z(BoundingBox_t* bb);


/**
 * Returns the width of the bounding box.
 */
LIBSBML_EXTERN
double
BoundingBox_width(BoundingBox_t* bb);

/**
 * Returns the height of the bounding box.
 */
LIBSBML_EXTERN
double
BoundingBox_height(BoundingBox_t* bb);

/**
 * Returns the depth of the bounding box.
 */
LIBSBML_EXTERN
double
BoundingBox_depth(BoundingBox_t* bb);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
BoundingBox_t *
BoundingBox_clone (const BoundingBox_t *m);


LIBSBML_EXTERN
int
BoundingBox_isSetId (const BoundingBox_t *bb);

LIBSBML_EXTERN
const char *
BoundingBox_getId (const BoundingBox_t *bb);


LIBSBML_EXTERN
int
BoundingBox_setId (BoundingBox_t *bb, const char *sid);


LIBSBML_EXTERN
void
BoundingBox_unsetId (BoundingBox_t *bb);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* BoundingBox_H__ */
